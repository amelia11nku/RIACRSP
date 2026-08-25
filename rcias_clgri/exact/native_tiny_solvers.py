"""Native Gurobi MILP and OR-Tools CP-SAT validation for tiny_03.

The audited profile fixes one eligible island per operation and one operation
per island. Both models optimize W routing with sequence-dependent empty travel,
F round-trip capacity, product precedence, configuration readiness, and
multi-source synchronization. Extracted assignments are always replayed by the
production decoder and checked independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.objective import ObjectiveBreakdown, calculate_objective
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.env.schedule import Schedule


@dataclass(frozen=True)
class NativeExactResult:
    backend: str
    status: str
    solver_version: str
    solver_makespan: float
    best_bound: float
    gap: float
    runtime_seconds: float
    actions: tuple[Action, ...]
    schedule: Schedule
    objective: ObjectiveBreakdown
    replay_makespan: float
    replay_feasible: bool
    native_operation_times: Mapping[str, Mapping[str, float | str]]
    w_assignments: Mapping[str, str]
    f_assignments: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "solver_version": self.solver_version,
            "solver_makespan": self.solver_makespan,
            "best_bound": self.best_bound,
            "gap": self.gap,
            "runtime_seconds": self.runtime_seconds,
            "replay_makespan": self.replay_makespan,
            "replay_feasible": self.replay_feasible,
            "objective_breakdown": self.objective.to_dict(),
            "actions": [action.__dict__ for action in self.actions],
            "native_operation_times": self.native_operation_times,
            "w_assignments": dict(self.w_assignments),
            "f_assignments": dict(self.f_assignments),
        }


@dataclass(frozen=True)
class _Profile:
    island: Mapping[str, str]
    predecessor: Mapping[str, str | None]
    pickup: Mapping[str, str]
    processing: Mapping[str, int]
    setup: Mapping[str, int]
    horizon: int


def _native_profile(instance: Instance) -> _Profile:
    if len(instance.islands) != 4 or len(instance.agvs_w) != 2 or len(instance.agvs_f) != 2:
        raise ValueError("native Tiny profile requires exactly four islands, two W-AGVs, and two F-AGVs")
    island: dict[str, str] = {}
    for op_id in instance.operations:
        eligible = instance.operation_data[op_id].eligible_islands
        if len(eligible) != 1:
            raise ValueError("native Tiny profile requires one fixed eligible island per operation")
        island[op_id] = eligible[0]
    if len(set(island.values())) != len(instance.operations):
        raise ValueError("native Tiny profile requires at most one operation per island")
    predecessor: dict[str, str | None] = {}
    pickup: dict[str, str] = {}
    for product_id in instance.products:
        operations = instance.product_data[product_id].operations
        expected = tuple(zip(operations, operations[1:]))
        if instance.product_data[product_id].precedence != expected:
            raise ValueError("native Tiny profile requires a fixed adjacent product chain")
        for index, op_id in enumerate(operations):
            predecessor[op_id] = None if index == 0 else operations[index - 1]
            pickup[op_id] = "WH" if index == 0 else island[operations[index - 1]]
    if any(pickup[op_id] == island[op_id] for op_id in instance.operations):
        raise ValueError("native Tiny profile expects one W transport for every operation")
    processing = {
        op_id: int(instance.processing_time[(op_id, island[op_id])])
        for op_id in instance.operations
    }
    setup = {
        op_id: int(instance.reconfiguration_time[(
            island[op_id], instance.island_data[island[op_id]].initial_config,
            instance.operation_data[op_id].required_config,
        )])
        for op_id in instance.operations
    }
    return _Profile(
        island=island,
        predecessor=predecessor,
        pickup=pickup,
        processing=processing,
        setup=setup,
        horizon=max(100, int(math.ceil(instance.horizon * 2.0))),
    )


def _replay(
    instance: Instance,
    profile: _Profile,
    starts: Mapping[str, float],
    w_assignment: Mapping[str, str],
    f_assignment: Mapping[str, str],
) -> tuple[tuple[Action, ...], Schedule, ObjectiveBreakdown, bool]:
    order_index = {op_id: index for index, op_id in enumerate(instance.operations)}
    ordered = sorted(instance.operations, key=lambda op_id: (starts[op_id], order_index[op_id]))
    env = RCIASConstructionEnv(instance)
    actions: list[Action] = []
    for op_id in ordered:
        action = Action(
            op_id, profile.island[op_id], w_assignment[op_id], f_assignment[op_id]
        )
        env.step(action)
        actions.append(action)
    audit = check_schedule(instance, env.schedule)
    objective = calculate_objective(instance, env.schedule)
    return tuple(actions), env.schedule, objective, bool(audit["feasible"])


def _finish_result(
    *, instance: Instance, profile: _Profile, backend: str, status: str,
    version: str, makespan: float, best_bound: float, gap: float, runtime: float,
    starts: Mapping[str, float], completions: Mapping[str, float],
    f_departures: Mapping[str, float], w_starts: Mapping[str, float],
    w_assignment: Mapping[str, str], f_assignment: Mapping[str, str],
) -> NativeExactResult:
    actions, schedule, objective, feasible = _replay(
        instance, profile, starts, w_assignment, f_assignment
    )
    if not feasible:
        raise RuntimeError(f"{backend} assignments produced an infeasible decoder replay")
    if abs(objective.makespan - makespan) > 1e-6:
        raise RuntimeError(
            f"{backend} native/replay makespan mismatch: {makespan} != {objective.makespan}"
        )
    records = {
        op_id: {
            "island": profile.island[op_id],
            "process_start": float(starts[op_id]),
            "process_end": float(completions[op_id]),
            "w_loaded_start": float(w_starts[op_id]),
            "f_departure_wh": float(f_departures[op_id]),
        }
        for op_id in instance.operations
    }
    return NativeExactResult(
        backend=backend,
        status=status,
        solver_version=version,
        solver_makespan=float(makespan),
        best_bound=float(best_bound),
        gap=float(gap),
        runtime_seconds=float(runtime),
        actions=actions,
        schedule=schedule,
        objective=objective,
        replay_makespan=objective.makespan,
        replay_feasible=True,
        native_operation_times=records,
        w_assignments=dict(w_assignment),
        f_assignments=dict(f_assignment),
    )


def solve_with_gurobi(
    instance: Instance, *, time_limit_seconds: float = 60.0, seed: int = 23,
) -> NativeExactResult:
    """Solve the audited Tiny profile as a native Gurobi MILP."""

    import gurobipy as gp
    from gurobipy import GRB

    profile = _native_profile(instance)
    operations = instance.operations
    model = gp.Model(f"{instance.instance_id}_gurobi")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = seed
    model.Params.TimeLimit = time_limit_seconds
    model.Params.MIPGap = 0.0
    horizon = profile.horizon

    start = {op: model.addVar(lb=0, ub=horizon, name=f"start[{op}]") for op in operations}
    end = {op: model.addVar(lb=0, ub=horizon, name=f"end[{op}]") for op in operations}
    f_depart = {op: model.addVar(lb=0, ub=horizon, name=f"f_depart[{op}]") for op in operations}
    f_arrive = {op: model.addVar(lb=0, ub=horizon, name=f"f_arrive[{op}]") for op in operations}
    f_return = {op: model.addVar(lb=0, ub=horizon, name=f"f_return[{op}]") for op in operations}
    w_start = {op: model.addVar(lb=0, ub=horizon, name=f"w_start[{op}]") for op in operations}
    w_arrive = {op: model.addVar(lb=0, ub=horizon, name=f"w_arrive[{op}]") for op in operations}
    x_f = {(op, f): model.addVar(vtype=GRB.BINARY, name=f"xF[{op},{f}]")
           for op in operations for f in instance.agvs_f}
    x_w = {(op, w): model.addVar(vtype=GRB.BINARY, name=f"xW[{op},{w}]")
           for op in operations for w in instance.agvs_w}

    for op in operations:
        island = profile.island[op]
        model.addConstr(gp.quicksum(x_f[(op, f)] for f in instance.agvs_f) == 1)
        model.addConstr(gp.quicksum(x_w[(op, w)] for w in instance.agvs_w) == 1)
        model.addConstr(end[op] == start[op] + profile.processing[op])
        model.addConstr(f_arrive[op] == f_depart[op] + gp.quicksum(
            instance.f_outbound_time[(f, island)] * x_f[(op, f)] for f in instance.agvs_f
        ))
        model.addConstr(f_return[op] == f_depart[op] + gp.quicksum(
            (instance.f_outbound_time[(f, island)] + instance.f_return_time[(f, island)])
            * x_f[(op, f)] for f in instance.agvs_f
        ))
        model.addConstr(w_arrive[op] == w_start[op] + gp.quicksum(
            instance.w_loaded_time[(w, profile.pickup[op], island)] * x_w[(op, w)]
            for w in instance.agvs_w
        ))
        predecessor = profile.predecessor[op]
        if predecessor is not None:
            model.addConstr(w_start[op] >= end[predecessor])
            model.addConstr(start[op] >= end[predecessor])
        model.addConstr(start[op] >= profile.setup[op])
        model.addConstr(start[op] >= f_arrive[op])
        model.addConstr(start[op] >= w_arrive[op])

    pairs = [(operations[i], operations[j]) for i in range(len(operations))
             for j in range(i + 1, len(operations))]
    for f in instance.agvs_f:
        for left, right in pairs:
            left_first = model.addVar(vtype=GRB.BINARY, name=f"fOrder[{f},{left},{right}]")
            right_first = model.addVar(vtype=GRB.BINARY, name=f"fOrder[{f},{right},{left}]")
            for order in (left_first, right_first):
                model.addConstr(order <= x_f[(left, f)])
                model.addConstr(order <= x_f[(right, f)])
            model.addConstr(left_first + right_first >= x_f[(left, f)] + x_f[(right, f)] - 1)
            model.addConstr(f_depart[right] >= f_return[left] - horizon * (1 - left_first))
            model.addConstr(f_depart[left] >= f_return[right] - horizon * (1 - right_first))

    depot = "DEPOT"
    nodes = (depot, *operations)
    arc: dict[tuple[str, str, str], Any] = {}
    for w in instance.agvs_w:
        used = model.addVar(vtype=GRB.BINARY, name=f"wUsed[{w}]")
        for source in nodes:
            for target in nodes:
                if source != target:
                    arc[(w, source, target)] = model.addVar(
                        vtype=GRB.BINARY, name=f"wArc[{w},{source},{target}]"
                    )
        model.addConstr(gp.quicksum(arc[(w, depot, op)] for op in operations) == used)
        model.addConstr(gp.quicksum(arc[(w, op, depot)] for op in operations) == used)
        model.addConstr(gp.quicksum(x_w[(op, w)] for op in operations) >= used)
        model.addConstr(gp.quicksum(x_w[(op, w)] for op in operations) <= len(operations) * used)
        for op in operations:
            model.addConstr(gp.quicksum(
                arc[(w, source, op)] for source in nodes if source != op
            ) == x_w[(op, w)])
            model.addConstr(gp.quicksum(
                arc[(w, op, target)] for target in nodes if target != op
            ) == x_w[(op, w)])
            model.addConstr(w_start[op] >= instance.w_empty_time[(w, "WH", profile.pickup[op])]
                            - horizon * (1 - arc[(w, depot, op)]))
        for source in operations:
            for target in operations:
                if source == target:
                    continue
                empty = instance.w_empty_time[(w, profile.island[source], profile.pickup[target])]
                model.addConstr(
                    w_start[target] >= w_arrive[source] + empty
                    - horizon * (1 - arc[(w, source, target)])
                )

    makespan = model.addVar(lb=0, ub=horizon, name="makespan")
    for op in operations:
        model.addConstr(makespan >= end[op])
    model.setObjective(makespan, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not prove optimality: status={model.Status}")
    starts = {op: start[op].X for op in operations}
    completions = {op: end[op].X for op in operations}
    w_assignment = {
        op: next(w for w in instance.agvs_w if x_w[(op, w)].X > 0.5) for op in operations
    }
    f_assignment = {
        op: next(f for f in instance.agvs_f if x_f[(op, f)].X > 0.5) for op in operations
    }
    return _finish_result(
        instance=instance, profile=profile, backend="gurobi-milp", status="OPTIMAL",
        version=".".join(map(str, gp.gurobi.version())), makespan=model.ObjVal,
        best_bound=model.ObjBound, gap=model.MIPGap, runtime=model.Runtime,
        starts=starts, completions=completions,
        f_departures={op: f_depart[op].X for op in operations},
        w_starts={op: w_start[op].X for op in operations},
        w_assignment=w_assignment, f_assignment=f_assignment,
    )


def solve_with_cp_sat(
    instance: Instance, *, time_limit_seconds: float = 60.0, seed: int = 23,
) -> NativeExactResult:
    """Solve the audited Tiny profile as a native OR-Tools CP-SAT model."""

    import ortools
    from ortools.sat.python import cp_model

    profile = _native_profile(instance)
    operations = instance.operations
    horizon = profile.horizon
    model = cp_model.CpModel()
    start = {op: model.NewIntVar(0, horizon, f"start[{op}]") for op in operations}
    end = {op: model.NewIntVar(0, horizon, f"end[{op}]") for op in operations}
    f_depart = {op: model.NewIntVar(0, horizon, f"f_depart[{op}]") for op in operations}
    f_arrive = {op: model.NewIntVar(0, horizon, f"f_arrive[{op}]") for op in operations}
    f_return = {op: model.NewIntVar(0, horizon, f"f_return[{op}]") for op in operations}
    w_start = {op: model.NewIntVar(0, horizon, f"w_start[{op}]") for op in operations}
    w_arrive = {op: model.NewIntVar(0, horizon, f"w_arrive[{op}]") for op in operations}
    x_f = {(op, f): model.NewBoolVar(f"xF[{op},{f}]")
           for op in operations for f in instance.agvs_f}
    x_w = {(op, w): model.NewBoolVar(f"xW[{op},{w}]")
           for op in operations for w in instance.agvs_w}
    f_intervals = {f: [] for f in instance.agvs_f}

    for op in operations:
        island = profile.island[op]
        model.AddExactlyOne(x_f[(op, f)] for f in instance.agvs_f)
        model.AddExactlyOne(x_w[(op, w)] for w in instance.agvs_w)
        model.Add(end[op] == start[op] + profile.processing[op])
        for f in instance.agvs_f:
            outbound = instance.f_outbound_time[(f, island)]
            duration = outbound + instance.f_return_time[(f, island)]
            optional_end = model.NewIntVar(0, horizon, f"f_end[{op},{f}]")
            f_intervals[f].append(model.NewOptionalIntervalVar(
                f_depart[op], duration, optional_end, x_f[(op, f)], f"fTask[{op},{f}]"
            ))
            model.Add(f_arrive[op] == f_depart[op] + outbound).OnlyEnforceIf(x_f[(op, f)])
            model.Add(f_return[op] == optional_end).OnlyEnforceIf(x_f[(op, f)])
        for w in instance.agvs_w:
            duration = instance.w_loaded_time[(w, profile.pickup[op], island)]
            model.Add(w_arrive[op] == w_start[op] + duration).OnlyEnforceIf(x_w[(op, w)])
        predecessor = profile.predecessor[op]
        if predecessor is not None:
            model.Add(w_start[op] >= end[predecessor])
            model.Add(start[op] >= end[predecessor])
        model.Add(start[op] >= profile.setup[op])
        model.Add(start[op] >= f_arrive[op])
        model.Add(start[op] >= w_arrive[op])
    for f in instance.agvs_f:
        model.AddNoOverlap(f_intervals[f])

    for w in instance.agvs_w:
        used = model.NewBoolVar(f"wUsed[{w}]")
        not_used = model.NewBoolVar(f"wNotUsed[{w}]")
        model.Add(used + not_used == 1)
        model.Add(sum(x_w[(op, w)] for op in operations) >= used)
        model.Add(sum(x_w[(op, w)] for op in operations) <= len(operations) * used)
        node_index = {op: index + 1 for index, op in enumerate(operations)}
        arcs: list[tuple[int, int, Any]] = [(0, 0, not_used)]
        for op in operations:
            arcs.append((node_index[op], node_index[op], x_w[(op, w)].Not()))
            first = model.NewBoolVar(f"wArc[{w},DEPOT,{op}]")
            last = model.NewBoolVar(f"wArc[{w},{op},DEPOT]")
            arcs.append((0, node_index[op], first))
            arcs.append((node_index[op], 0, last))
            empty = instance.w_empty_time[(w, "WH", profile.pickup[op])]
            model.Add(w_start[op] >= empty).OnlyEnforceIf(first)
        for source in operations:
            for target in operations:
                if source == target:
                    continue
                follows = model.NewBoolVar(f"wArc[{w},{source},{target}]")
                arcs.append((node_index[source], node_index[target], follows))
                empty = instance.w_empty_time[(w, profile.island[source], profile.pickup[target])]
                model.Add(w_start[target] >= w_arrive[source] + empty).OnlyEnforceIf(follows)
        model.AddCircuit(arcs)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [end[op] for op in operations])
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status_code = solver.Solve(model)
    if status_code != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT did not prove optimality: {solver.StatusName(status_code)}")
    starts = {op: float(solver.Value(start[op])) for op in operations}
    completions = {op: float(solver.Value(end[op])) for op in operations}
    w_assignment = {
        op: next(w for w in instance.agvs_w if solver.BooleanValue(x_w[(op, w)]))
        for op in operations
    }
    f_assignment = {
        op: next(f for f in instance.agvs_f if solver.BooleanValue(x_f[(op, f)]))
        for op in operations
    }
    objective = float(solver.ObjectiveValue())
    bound = float(solver.BestObjectiveBound())
    gap = abs(objective - bound) / max(1.0, abs(objective))
    return _finish_result(
        instance=instance, profile=profile, backend="ortools-cp-sat", status="OPTIMAL",
        version=ortools.__version__, makespan=objective, best_bound=bound, gap=gap,
        runtime=solver.WallTime(), starts=starts, completions=completions,
        f_departures={op: float(solver.Value(f_depart[op])) for op in operations},
        w_starts={op: float(solver.Value(w_start[op])) for op in operations},
        w_assignment=w_assignment, f_assignment=f_assignment,
    )
