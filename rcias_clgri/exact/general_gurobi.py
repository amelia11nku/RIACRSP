"""General small-instance Gurobi MILP for the full RCIAS-2.0 semantics.

Unlike :mod:`native_tiny_solvers`, this formulation does not assume fixed
operation/island assignments, one operation per island, or chain precedence.
It builds sparse source-task-sink paths for products, islands, W-AGVs, and
F-AGVs.  The intended use is exact validation on tiny/small instances; model
size grows quadratically (and for W routing also with location pairs).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.objective import ObjectiveBreakdown, calculate_objective
from rcias_clgri.env.objective import ObjectiveBreakdown, calculate_objective
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.env.schedule import FTask, OperationSchedule, Schedule, WTask
from rcias_clgri.env.schedule import FTask, OperationSchedule, Schedule, WTask
from rcias_clgri.heuristic.dispatching import solve_dispatching


@dataclass(frozen=True)
class GeneralGurobiResult:
    backend: str
    status: str
    solver_version: str
    solver_makespan: float
    best_bound: float
    gap: float
    runtime_seconds: float
    total_runtime_seconds: float
    optimality_proven: bool
    actions: tuple[Action, ...]
    schedule: Schedule
    objective: ObjectiveBreakdown
    replay_makespan: float
    replay_feasible: bool
    action_replay_makespan: float
    action_replay_feasible: bool
    action_replay_matches_solver: bool
    action_replay_schedule: Schedule
    action_replay_makespan: float
    action_replay_feasible: bool
    action_replay_matches_solver: bool
    action_replay_schedule: Schedule
    island_assignments: Mapping[str, str]
    w_assignments: Mapping[str, str | None]
    f_assignments: Mapping[str, str]
    product_sequences: Mapping[str, tuple[str, ...]]
    island_sequences: Mapping[str, tuple[str, ...]]
    w_sequences: Mapping[str, tuple[str, ...]]
    f_sequences: Mapping[str, tuple[str, ...]]
    variable_count: int
    constraint_count: int
    node_count: float
    node_count: float
    h1_upper_bound: float
    h1_mip_start_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "solver_version": self.solver_version,
            "solver_makespan": self.solver_makespan,
            "best_bound": self.best_bound,
            "gap": self.gap,
            "runtime_seconds": self.runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "optimality_proven": self.optimality_proven,
            "replay_makespan": self.replay_makespan,
            "replay_feasible": self.replay_feasible,
            "action_replay_makespan": self.action_replay_makespan,
            "action_replay_feasible": self.action_replay_feasible,
            "action_replay_matches_solver": self.action_replay_matches_solver,
            "action_replay_makespan": self.action_replay_makespan,
            "action_replay_feasible": self.action_replay_feasible,
            "action_replay_matches_solver": self.action_replay_matches_solver,
            "objective_breakdown": self.objective.to_dict(),
            "actions": [action.__dict__ for action in self.actions],
            "island_assignments": dict(self.island_assignments),
            "w_assignments": dict(self.w_assignments),
            "f_assignments": dict(self.f_assignments),
            "product_sequences": {
                key: list(value) for key, value in self.product_sequences.items()
            },
            "island_sequences": {
                key: list(value) for key, value in self.island_sequences.items()
            },
            "w_sequences": {key: list(value) for key, value in self.w_sequences.items()},
            "f_sequences": {key: list(value) for key, value in self.f_sequences.items()},
            "variable_count": self.variable_count,
            "constraint_count": self.constraint_count,
            "node_count": self.node_count,
            "node_count": self.node_count,
            "h1_upper_bound": self.h1_upper_bound,
            "h1_mip_start_used": self.h1_mip_start_used,
            "h1_mip_start_used": self.h1_mip_start_used,
            "schedule": self.schedule.to_dict(),
            "action_replay_schedule": self.action_replay_schedule.to_dict(),
            "action_replay_schedule": self.action_replay_schedule.to_dict(),
        }


def _status_name(status: int, grb) -> str:
    names = {
        grb.OPTIMAL: "OPTIMAL",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.NODE_LIMIT: "NODE_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
    }
    return names.get(status, f"STATUS_{status}")


def _selected_path(
    source: str,
    sink: str,
    arcs: Mapping[tuple[str, str], Any],
) -> tuple[str, ...]:
    successor = {
        left: right for (left, right), variable in arcs.items() if variable.X > 0.5
    }
    sequence: list[str] = []
    current = source
    seen = {source}
    while current in successor:
        current = successor[current]
        if current == sink:
            return tuple(sequence)
        if current in seen:
            raise RuntimeError("selected MILP route contains a cycle")
        seen.add(current)
        sequence.append(current)
    if sequence:
        raise RuntimeError("selected MILP route does not reach its sink")
    return ()


def _clean_time(value: float) -> float:
    """Remove ordinary MILP feasibility-tolerance noise from integral event times."""

    nearest = round(float(value))
    return float(nearest) if math.isclose(value, nearest, abs_tol=1e-5) else float(value)


def _clean_time(value: float) -> float:
    """Remove ordinary MILP feasibility-tolerance noise from integral event times."""

    nearest = round(float(value))
    return float(nearest) if math.isclose(value, nearest, abs_tol=1e-5) else float(value)


def solve_general_gurobi(
    instance: Instance,
    *,
    time_limit_seconds: float = 600.0,
    seed: int = 23,
    threads: int = 1,
    mip_gap: float = 0.0,
    output_flag: bool = False,
    log_file: str | None = None,
    use_h1_mip_start: bool = True,
) -> GeneralGurobiResult:
    """Build and solve the general makespan MILP, then replay via production actions.

    The model accepts arbitrary technological DAGs and eligible-island sets.  A
    feasible H1 schedule supplies a safe incumbent upper bound; it does not fix
    any MILP decision.  Model loading time is included in ``runtime_seconds``.
    """
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if not 0 <= mip_gap < 1:
        raise ValueError("mip_gap must be in [0, 1)")

    import gurobipy as gp
    from gurobipy import GRB

    started = time.perf_counter()
    operations = tuple(instance.operations)
    h1 = solve_dispatching(instance, "H1")
    makespan_upper = float(h1.objective.makespan)
    max_f_return = max(instance.f_return_time.values(), default=0)
    event_upper = max(1.0, makespan_upper + float(max_f_return))
    max_empty = max(instance.w_empty_time.values(), default=0)
    big_m = 2.0 * event_upper + float(max_empty) + 1.0

    model = gp.Model(f"{instance.instance_id}_general_rcias")
    model.Params.OutputFlag = int(output_flag or log_file is not None)
    if log_file is not None:
        model.Params.LogToConsole = int(output_flag)
        model.Params.LogFile = log_file
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)

    # Operation assignment and processing times.
    assign = {
        (op, island): model.addVar(vtype=GRB.BINARY, name=f"assign[{op},{island}]")
        for op in operations
        for island in instance.operation_data[op].eligible_islands
    }
    start = {
        op: model.addVar(lb=0.0, ub=makespan_upper, name=f"start[{op}]")
        for op in operations
    }
    completion = {
        op: model.addVar(lb=0.0, ub=makespan_upper, name=f"completion[{op}]")
        for op in operations
    }
    for op in operations:
        eligible = instance.operation_data[op].eligible_islands
        model.addConstr(gp.quicksum(assign[(op, island)] for island in eligible) == 1)
        model.addConstr(
            completion[op] == start[op] + gp.quicksum(
                instance.processing_time[(op, island)] * assign[(op, island)]
                for island in eligible
            )
        )

    # A product is an indivisible workpiece following one total order that
    # extends its technological DAG.  Path arcs identify the direct pickup predecessor.
    product_arcs: dict[str, dict[tuple[str, str], Any]] = {}
    product_rank: dict[str, Any] = {}
    for product in instance.products:
        product_ops = tuple(instance.product_data[product].operations)
        source, sink = f"PRODUCT_SOURCE::{product}", f"PRODUCT_SINK::{product}"
        arcs: dict[tuple[str, str], Any] = {}
        for target in product_ops:
            arcs[(source, target)] = model.addVar(
                vtype=GRB.BINARY, name=f"productArc[{product},SOURCE,{target}]"
            )
        for left in product_ops:
            arcs[(left, sink)] = model.addVar(
                vtype=GRB.BINARY, name=f"productArc[{product},{left},SINK]"
            )
            for right in product_ops:
                if left != right:
                    arcs[(left, right)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"productArc[{product},{left},{right}]",
                    )
        model.addConstr(gp.quicksum(arcs[(source, op)] for op in product_ops) == 1)
        model.addConstr(gp.quicksum(arcs[(op, sink)] for op in product_ops) == 1)
        for op in product_ops:
            model.addConstr(
                arcs[(source, op)]
                + gp.quicksum(arcs[(left, op)] for left in product_ops if left != op)
                == 1
            )
            model.addConstr(
                arcs[(op, sink)]
                + gp.quicksum(arcs[(op, right)] for right in product_ops if right != op)
                == 1
            )
            product_rank[op] = model.addVar(
                lb=1.0, ub=float(len(product_ops)), vtype=GRB.INTEGER,
                name=f"productRank[{op}]",
            )
            model.addConstr((arcs[(source, op)] == 1) >> (product_rank[op] == 1))
        for left in product_ops:
            for right in product_ops:
                if left == right:
                    continue
                model.addConstr(
                    (arcs[(left, right)] == 1)
                    >> (product_rank[right] == product_rank[left] + 1)
                )
                model.addConstr(
                    (arcs[(left, right)] == 1) >> (start[right] >= completion[left])
                )
        for left, right in instance.product_data[product].precedence:
            model.addConstr(product_rank[right] >= product_rank[left] + 1)
            model.addConstr(start[right] >= completion[left])
        product_arcs[product] = arcs

    # Island paths jointly select assignments and sequence-dependent setups.
    island_arcs: dict[str, dict[tuple[str, str], Any]] = {}
    for island in instance.islands:
        eligible_ops = tuple(
            op for op in operations if (op, island) in assign
        )
        source, sink = f"ISLAND_SOURCE::{island}", f"ISLAND_SINK::{island}"
        arcs: dict[tuple[str, str], Any] = {}
        rank = {
            op: model.addVar(
                lb=0.0, ub=float(len(eligible_ops)), name=f"islandRank[{island},{op}]"
            )
            for op in eligible_ops
        }
        for target in eligible_ops:
            arcs[(source, target)] = model.addVar(
                vtype=GRB.BINARY, name=f"islandArc[{island},SOURCE,{target}]"
            )
        for left in eligible_ops:
            arcs[(left, sink)] = model.addVar(
                vtype=GRB.BINARY, name=f"islandArc[{island},{left},SINK]"
            )
            for right in eligible_ops:
                if left != right:
                    arcs[(left, right)] = model.addVar(
                        vtype=GRB.BINARY, name=f"islandArc[{island},{left},{right}]"
                    )
        if eligible_ops:
            model.addConstr(
                gp.quicksum(arcs[(source, op)] for op in eligible_ops)
                == gp.quicksum(arcs[(op, sink)] for op in eligible_ops)
            )
            model.addConstr(gp.quicksum(arcs[(source, op)] for op in eligible_ops) <= 1)
        for op in eligible_ops:
            model.addConstr(
                arcs[(source, op)]
                + gp.quicksum(arcs[(left, op)] for left in eligible_ops if left != op)
                == assign[(op, island)]
            )
            model.addConstr(
                arcs[(op, sink)]
                + gp.quicksum(arcs[(op, right)] for right in eligible_ops if right != op)
                == assign[(op, island)]
            )
            initial = instance.island_data[island].initial_config
            required = instance.operation_data[op].required_config
            setup = instance.reconfiguration_time[(island, initial, required)]
            model.addConstr((arcs[(source, op)] == 1) >> (start[op] >= setup))
            model.addConstr((arcs[(source, op)] == 1) >> (rank[op] >= 1))
        for left in eligible_ops:
            left_config = instance.operation_data[left].required_config
            for right in eligible_ops:
                if left == right:
                    continue
                right_config = instance.operation_data[right].required_config
                setup = instance.reconfiguration_time[(island, left_config, right_config)]
                model.addConstr(
                    (arcs[(left, right)] == 1)
                    >> (start[right] >= completion[left] + setup)
                )
                model.addConstr(
                    (arcs[(left, right)] == 1) >> (rank[right] >= rank[left] + 1)
                )
        island_arcs[island] = arcs

    # F kits: island and F assignment are selected together, then sequenced as
    # non-overlapping warehouse round trips.
    f_pair = {
        (op, vehicle, island): model.addVar(
            vtype=GRB.BINARY, name=f"fPair[{op},{vehicle},{island}]"
        )
        for op in operations
        for vehicle in instance.agvs_f
        for island in instance.operation_data[op].eligible_islands
    }
    f_assigned = {
        (op, vehicle): model.addVar(
            vtype=GRB.BINARY, name=f"fAssigned[{op},{vehicle}]"
        )
        for op in operations for vehicle in instance.agvs_f
    }
    f_depart = {
        op: model.addVar(lb=0.0, ub=event_upper, name=f"fDepart[{op}]")
        for op in operations
    }
    f_arrive = {
        op: model.addVar(lb=0.0, ub=event_upper, name=f"fArrive[{op}]")
        for op in operations
    }
    f_return = {
        op: model.addVar(lb=0.0, ub=event_upper, name=f"fReturn[{op}]")
        for op in operations
    }
    for op in operations:
        eligible = instance.operation_data[op].eligible_islands
        for island in eligible:
            model.addConstr(
                gp.quicksum(f_pair[(op, vehicle, island)] for vehicle in instance.agvs_f)
                == assign[(op, island)]
            )
        for vehicle in instance.agvs_f:
            model.addConstr(
                f_assigned[(op, vehicle)]
                == gp.quicksum(f_pair[(op, vehicle, island)] for island in eligible)
            )
        model.addConstr(
            f_arrive[op] == f_depart[op] + gp.quicksum(
                instance.f_outbound_time[(vehicle, island)]
                * f_pair[(op, vehicle, island)]
                for vehicle in instance.agvs_f for island in eligible
            )
        )
        model.addConstr(
            f_return[op] == f_depart[op] + gp.quicksum(
                (
                    instance.f_outbound_time[(vehicle, island)]
                    + instance.f_return_time[(vehicle, island)]
                ) * f_pair[(op, vehicle, island)]
                for vehicle in instance.agvs_f for island in eligible
            )
        )
        model.addConstr(start[op] >= f_arrive[op])

    f_arcs: dict[str, dict[tuple[str, str], Any]] = {}
    for vehicle in instance.agvs_f:
        source, sink = f"F_SOURCE::{vehicle}", f"F_SINK::{vehicle}"
        arcs: dict[tuple[str, str], Any] = {}
        rank = {
            op: model.addVar(lb=0.0, ub=float(len(operations)), name=f"fRank[{vehicle},{op}]")
            for op in operations
        }
        for target in operations:
            arcs[(source, target)] = model.addVar(
                vtype=GRB.BINARY, name=f"fArc[{vehicle},SOURCE,{target}]"
            )
        for left in operations:
            arcs[(left, sink)] = model.addVar(
                vtype=GRB.BINARY, name=f"fArc[{vehicle},{left},SINK]"
            )
            for right in operations:
                if left != right:
                    arcs[(left, right)] = model.addVar(
                        vtype=GRB.BINARY, name=f"fArc[{vehicle},{left},{right}]"
                    )
        model.addConstr(
            gp.quicksum(arcs[(source, op)] for op in operations)
            == gp.quicksum(arcs[(op, sink)] for op in operations)
        )
        model.addConstr(gp.quicksum(arcs[(source, op)] for op in operations) <= 1)
        for op in operations:
            model.addConstr(
                arcs[(source, op)]
                + gp.quicksum(arcs[(left, op)] for left in operations if left != op)
                == f_assigned[(op, vehicle)]
            )
            model.addConstr(
                arcs[(op, sink)]
                + gp.quicksum(arcs[(op, right)] for right in operations if right != op)
                == f_assigned[(op, vehicle)]
            )
            model.addConstr((arcs[(source, op)] == 1) >> (rank[op] >= 1))
        for left in operations:
            for right in operations:
                if left == right:
                    continue
                model.addConstr(
                    (arcs[(left, right)] == 1) >> (f_depart[right] >= f_return[left])
                )
                model.addConstr(
                    (arcs[(left, right)] == 1) >> (rank[right] >= rank[left] + 1)
                )
        f_arcs[vehicle] = arcs

    # Realized product adjacency and island choices define one route type for
    # each potential W task. Same-island adjacency creates no W task.
    details_by_operation: dict[str, list[tuple[tuple[str, str, str, str], Any]]] = {
        op: [] for op in operations
    }
    for product in instance.products:
        product_ops = tuple(instance.product_data[product].operations)
        source = f"PRODUCT_SOURCE::{product}"
        arcs = product_arcs[product]
        for op in product_ops:
            for destination in instance.operation_data[op].eligible_islands:
                key = (source, op, "WH", destination)
                detail = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"productLocation[SOURCE,{op},WH,{destination}]",
                )
                model.addConstr(detail <= arcs[(source, op)])
                model.addConstr(detail <= assign[(op, destination)])
                model.addConstr(detail >= arcs[(source, op)] + assign[(op, destination)] - 1)
                details_by_operation[op].append((key, detail))
            for predecessor in product_ops:
                if predecessor == op:
                    continue
                for pickup in instance.operation_data[predecessor].eligible_islands:
                    for destination in instance.operation_data[op].eligible_islands:
                        key = (predecessor, op, pickup, destination)
                        detail = model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"productLocation[{predecessor},{op},"
                                f"{pickup},{destination}]"
                            ),
                        )
                        model.addConstr(detail <= arcs[(predecessor, op)])
                        model.addConstr(detail <= assign[(predecessor, pickup)])
                        model.addConstr(detail <= assign[(op, destination)])
                        model.addConstr(
                            detail >= arcs[(predecessor, op)]
                            + assign[(predecessor, pickup)]
                            + assign[(op, destination)] - 2
                        )
                        details_by_operation[op].append((key, detail))
    for op in operations:
        model.addConstr(gp.quicksum(value for _, value in details_by_operation[op]) == 1)

    cross_by_route: dict[tuple[str, str, str], list[Any]] = {}
    for op, details in details_by_operation.items():
        for (_, _, pickup, destination), detail in details:
            if pickup == destination:
                continue
            else:
                cross_by_route.setdefault((op, pickup, destination), []).append(detail)
    need_w = {
        op: model.addVar(vtype=GRB.BINARY, name=f"needW[{op}]") for op in operations
    }
    w_type: dict[tuple[str, str, str, str], Any] = {}
    w_assigned = {
        (op, vehicle): model.addVar(
            vtype=GRB.BINARY, name=f"wAssigned[{op},{vehicle}]"
        )
        for op in operations for vehicle in instance.agvs_w
    }
    for op in operations:
        cross_details = [
            detail for (route_op, _, _), values in cross_by_route.items()
            if route_op == op for detail in values
        ]
        model.addConstr(need_w[op] == gp.quicksum(cross_details))
        for (route_op, pickup, destination), details in cross_by_route.items():
            if route_op != op:
                continue
            for vehicle in instance.agvs_w:
                w_type[(op, vehicle, pickup, destination)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"wType[{op},{vehicle},{pickup},{destination}]",
                )
            model.addConstr(
                gp.quicksum(
                    w_type[(op, vehicle, pickup, destination)]
                    for vehicle in instance.agvs_w
                ) == gp.quicksum(details)
            )
        for vehicle in instance.agvs_w:
            model.addConstr(
                w_assigned[(op, vehicle)] == gp.quicksum(
                    variable for (route_op, route_vehicle, _, _), variable in w_type.items()
                    if route_op == op and route_vehicle == vehicle
                )
            )
        model.addConstr(
            gp.quicksum(w_assigned[(op, vehicle)] for vehicle in instance.agvs_w)
            == need_w[op]
        )

    w_loaded_start = {
        op: model.addVar(lb=0.0, ub=event_upper, name=f"wLoadedStart[{op}]")
        for op in operations
    }
    w_arrive = {
        op: model.addVar(lb=0.0, ub=event_upper, name=f"wArrive[{op}]")
        for op in operations
    }
    for op in operations:
        model.addConstr(w_loaded_start[op] <= event_upper * need_w[op])
        model.addConstr(w_arrive[op] <= event_upper * need_w[op])
        model.addConstr(
            w_arrive[op] == w_loaded_start[op] + gp.quicksum(
                instance.w_loaded_time[(vehicle, pickup, destination)] * variable
                for (route_op, vehicle, pickup, destination), variable in w_type.items()
                if route_op == op
            )
        )
        model.addConstr(start[op] >= w_arrive[op])
        for (predecessor, _, pickup, destination), detail in details_by_operation[op]:
            if predecessor.startswith("PRODUCT_SOURCE::") or pickup == destination:
                continue
            model.addConstr(
                (detail == 1) >> (w_loaded_start[op] >= completion[predecessor])
            )

    w_pickup: dict[tuple[str, str, str], Any] = {}
    w_destination: dict[tuple[str, str, str], Any] = {}
    for op in operations:
        for vehicle in instance.agvs_w:
            pickups = sorted({
                pickup for route_op, route_vehicle, pickup, _ in w_type
                if route_op == op and route_vehicle == vehicle
            })
            destinations = sorted({
                destination for route_op, route_vehicle, _, destination in w_type
                if route_op == op and route_vehicle == vehicle
            })
            for pickup in pickups:
                variable = model.addVar(
                    vtype=GRB.BINARY, name=f"wPickup[{op},{vehicle},{pickup}]"
                )
                model.addConstr(variable == gp.quicksum(
                    route_variable
                    for (route_op, route_vehicle, route_pickup, _), route_variable
                    in w_type.items()
                    if route_op == op and route_vehicle == vehicle and route_pickup == pickup
                ))
                w_pickup[(op, vehicle, pickup)] = variable
            for destination in destinations:
                variable = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"wDestination[{op},{vehicle},{destination}]",
                )
                model.addConstr(variable == gp.quicksum(
                    route_variable
                    for (route_op, route_vehicle, _, route_destination), route_variable
                    in w_type.items()
                    if route_op == op and route_vehicle == vehicle
                    and route_destination == destination
                ))
                w_destination[(op, vehicle, destination)] = variable

    w_arcs: dict[str, dict[tuple[str, str], Any]] = {}
    for vehicle in instance.agvs_w:
        source, sink = f"W_SOURCE::{vehicle}", f"W_SINK::{vehicle}"
        arcs: dict[tuple[str, str], Any] = {}
        rank = {
            op: model.addVar(lb=0.0, ub=float(len(operations)), name=f"wRank[{vehicle},{op}]")
            for op in operations
        }
        for target in operations:
            arcs[(source, target)] = model.addVar(
                vtype=GRB.BINARY, name=f"wArc[{vehicle},SOURCE,{target}]"
            )
        for left in operations:
            arcs[(left, sink)] = model.addVar(
                vtype=GRB.BINARY, name=f"wArc[{vehicle},{left},SINK]"
            )
            for right in operations:
                if left != right:
                    arcs[(left, right)] = model.addVar(
                        vtype=GRB.BINARY, name=f"wArc[{vehicle},{left},{right}]"
                    )
        model.addConstr(
            gp.quicksum(arcs[(source, op)] for op in operations)
            == gp.quicksum(arcs[(op, sink)] for op in operations)
        )
        model.addConstr(gp.quicksum(arcs[(source, op)] for op in operations) <= 1)
        for op in operations:
            model.addConstr(
                arcs[(source, op)]
                + gp.quicksum(arcs[(left, op)] for left in operations if left != op)
                == w_assigned[(op, vehicle)]
            )
            model.addConstr(
                arcs[(op, sink)]
                + gp.quicksum(arcs[(op, right)] for right in operations if right != op)
                == w_assigned[(op, vehicle)]
            )
            model.addConstr((arcs[(source, op)] == 1) >> (rank[op] >= 1))
            for (route_op, route_vehicle, pickup), pickup_variable in w_pickup.items():
                if route_op != op or route_vehicle != vehicle:
                    continue
                empty = instance.w_empty_time[(vehicle, "WH", pickup)]
                model.addConstr(
                    w_loaded_start[op] >= empty
                    - big_m * (2 - arcs[(source, op)] - pickup_variable)
                )
        for left in operations:
            for right in operations:
                if left == right:
                    continue
                arc = arcs[(left, right)]
                model.addConstr((arc == 1) >> (rank[right] >= rank[left] + 1))
                left_destinations = [
                    (destination, variable)
                    for (route_op, route_vehicle, destination), variable
                    in w_destination.items()
                    if route_op == left and route_vehicle == vehicle
                ]
                right_pickups = [
                    (pickup, variable)
                    for (route_op, route_vehicle, pickup), variable in w_pickup.items()
                    if route_op == right and route_vehicle == vehicle
                ]
                for destination, destination_variable in left_destinations:
                    for pickup, pickup_variable in right_pickups:
                        empty = instance.w_empty_time[(vehicle, destination, pickup)]
                        model.addConstr(
                            w_loaded_start[right] >= w_arrive[left] + empty
                            - big_m * (
                                3 - arc - destination_variable - pickup_variable
                            )
                        )
        w_arcs[vehicle] = arcs

    makespan = model.addVar(
        lb=0.0, ub=makespan_upper, name="makespan"
    )
    for op in operations:
        model.addConstr(makespan >= completion[op])
    model.setObjective(makespan, GRB.MINIMIZE)

    # The production H1 schedule is already known to be feasible.  Supplying it
    # as a complete discrete/temporal MIP start gives time-limited paper runs a
    # reproducible incumbent without restricting any MILP decision.
    if use_h1_mip_start:
        h1_schedule = h1.schedule

        def set_path_start(
            source: str,
            sink: str,
            arcs: Mapping[tuple[str, str], Any],
            sequence: tuple[str, ...] | list[str],
        ) -> None:
            selected = set(zip((source, *sequence), (*sequence, sink))) if sequence else set()
            for key, variable in arcs.items():
                variable.Start = float(key in selected)

        for op in operations:
            operation = h1_schedule.operation_schedules[op]
            for island in instance.operation_data[op].eligible_islands:
                assign[(op, island)].Start = float(operation.island_id == island)
            start[op].Start = float(operation.start_time)
            completion[op].Start = float(operation.completion_time)

        for product in instance.products:
            source, sink = f"PRODUCT_SOURCE::{product}", f"PRODUCT_SINK::{product}"
            sequence = h1_schedule.product_sequences[product]
            set_path_start(source, sink, product_arcs[product], sequence)
            for position, op in enumerate(sequence, 1):
                product_rank[op].Start = float(position)

        for island in instance.islands:
            set_path_start(
                f"ISLAND_SOURCE::{island}",
                f"ISLAND_SINK::{island}",
                island_arcs[island],
                h1_schedule.island_timelines[island],
            )

        f_task_by_operation = {
            task.operation_id: task
            for tasks in h1_schedule.f_timelines.values()
            for task in tasks
        }
        for op in operations:
            operation = h1_schedule.operation_schedules[op]
            task = f_task_by_operation[op]
            f_depart[op].Start = float(task.departure_wh)
            f_arrive[op].Start = float(task.arrival_island)
            f_return[op].Start = float(task.return_wh)
            for vehicle in instance.agvs_f:
                f_assigned[(op, vehicle)].Start = float(task.vehicle_id == vehicle)
                for island in instance.operation_data[op].eligible_islands:
                    f_pair[(op, vehicle, island)].Start = float(
                        task.vehicle_id == vehicle and operation.island_id == island
                    )
        for vehicle in instance.agvs_f:
            set_path_start(
                f"F_SOURCE::{vehicle}",
                f"F_SINK::{vehicle}",
                f_arcs[vehicle],
                [task.operation_id for task in h1_schedule.f_timelines[vehicle]],
            )

        w_task_by_operation = {
            task.operation_id: task
            for tasks in h1_schedule.w_timelines.values()
            for task in tasks
        }
        for op in operations:
            operation = h1_schedule.operation_schedules[op]
            predecessor = operation.product_predecessor
            pickup = (
                "WH" if predecessor is None
                else h1_schedule.operation_schedules[predecessor].island_id
            )
            detail_key = (
                f"PRODUCT_SOURCE::{operation.product_id}" if predecessor is None else predecessor,
                op,
                pickup,
                operation.island_id,
            )
            for key, variable in details_by_operation[op]:
                variable.Start = float(key == detail_key)

            task = w_task_by_operation.get(op)
            need_w[op].Start = float(task is not None)
            w_loaded_start[op].Start = 0.0 if task is None else float(task.loaded_start)
            w_arrive[op].Start = 0.0 if task is None else float(task.arrival_time)
            for vehicle in instance.agvs_w:
                w_assigned[(op, vehicle)].Start = float(
                    task is not None and task.vehicle_id == vehicle
                )
            for (route_op, vehicle, route_pickup, destination), variable in w_type.items():
                if route_op == op:
                    variable.Start = float(
                        task is not None
                        and task.vehicle_id == vehicle
                        and task.pickup == route_pickup
                        and task.destination == destination
                    )
            for (route_op, vehicle, route_pickup), variable in w_pickup.items():
                if route_op == op:
                    variable.Start = float(
                        task is not None
                        and task.vehicle_id == vehicle
                        and task.pickup == route_pickup
                    )
            for (route_op, vehicle, destination), variable in w_destination.items():
                if route_op == op:
                    variable.Start = float(
                        task is not None
                        and task.vehicle_id == vehicle
                        and task.destination == destination
                    )
        for vehicle in instance.agvs_w:
            set_path_start(
                f"W_SOURCE::{vehicle}",
                f"W_SINK::{vehicle}",
                w_arcs[vehicle],
                [task.operation_id for task in h1_schedule.w_timelines[vehicle]],
            )
        makespan.Start = makespan_upper

    model.optimize()

    status = _status_name(model.Status, GRB)
    if model.SolCount == 0:
        raise RuntimeError(f"general Gurobi produced no incumbent: status={status}")

    island_assignment = {
        op: next(
            island for island in instance.operation_data[op].eligible_islands
            if assign[(op, island)].X > 0.5
        )
        for op in operations
    }
    f_assignment = {
        op: next(vehicle for vehicle in instance.agvs_f if f_assigned[(op, vehicle)].X > 0.5)
        for op in operations
    }
    w_assignment: dict[str, str | None] = {}
    for op in operations:
        selected = [
            vehicle for vehicle in instance.agvs_w if w_assigned[(op, vehicle)].X > 0.5
        ]
        w_assignment[op] = selected[0] if selected else None

    product_sequences = {
        product: _selected_path(
            f"PRODUCT_SOURCE::{product}", f"PRODUCT_SINK::{product}", product_arcs[product]
        )
        for product in instance.products
    }
    island_sequences = {
        island: _selected_path(
            f"ISLAND_SOURCE::{island}", f"ISLAND_SINK::{island}", island_arcs[island]
        )
        for island in instance.islands
    }
    f_sequences = {
        vehicle: _selected_path(
            f"F_SOURCE::{vehicle}", f"F_SINK::{vehicle}", f_arcs[vehicle]
        )
        for vehicle in instance.agvs_f
    }
    w_sequences = {
        vehicle: _selected_path(
            f"W_SOURCE::{vehicle}", f"W_SINK::{vehicle}", w_arcs[vehicle]
        )
        for vehicle in instance.agvs_w
    }

    # Reconstruct the selected MILP schedule from all four independent path
    # systems.  A single operation-action ordering cannot, in general, encode
    # an F-kit order that differs from the operation order (or an independent
    # W route order).  The former start-time-only replay therefore changed
    # valid time-limited incumbents on Core-S.  The reconstructed Schedule is
    # the lossless representation audited for exact-solver evidence; the old
    # action projection is retained below as an explicit diagnostic only.
    product_predecessor: dict[str, str | None] = {}
    product_successor: dict[str, str | None] = {}
    for sequence in product_sequences.values():
        for index, op in enumerate(sequence):
            product_predecessor[op] = sequence[index - 1] if index else None
            product_successor[op] = sequence[index + 1] if index + 1 < len(sequence) else None

    selected_detail: dict[str, tuple[str, str, str, str]] = {}
    for op, details in details_by_operation.items():
        selected = [key for key, variable in details if variable.X > 0.5]
        if len(selected) != 1:
            raise RuntimeError(f"expected one selected product/location detail for {op}")
        selected_detail[op] = selected[0]

    f_timelines: dict[str, list[FTask]] = {vehicle: [] for vehicle in instance.agvs_f}
    f_task_by_operation: dict[str, FTask] = {}
    for vehicle, sequence in f_sequences.items():
        for op in sequence:
            island = island_assignment[op]
            departure = _clean_time(f_depart[op].X)
            outbound = float(instance.f_outbound_time[(vehicle, island)])
            return_duration = float(instance.f_return_time[(vehicle, island)])
            task = FTask(
                task_id=f"F:{op}",
                vehicle_id=vehicle,
                operation_id=op,
                island_id=island,
                departure_wh=departure,
                arrival_island=departure + outbound,
                return_wh=departure + outbound + return_duration,
                outbound_time=outbound,
                return_time=return_duration,
                outbound_distance=instance.distance[("WH", island)],
                return_distance=instance.distance[(island, "WH")],
            )
            f_timelines[vehicle].append(task)
            f_task_by_operation[op] = task

    w_timelines: dict[str, list[WTask]] = {vehicle: [] for vehicle in instance.agvs_w}
    w_task_by_operation: dict[str, WTask] = {}
    for vehicle, sequence in w_sequences.items():
        previous_location = "WH"
        previous_arrival = 0.0
        for op in sequence:
            _, _, pickup, destination = selected_detail[op]
            predecessor = product_predecessor[op]
            expected_pickup = (
                "WH" if predecessor is None else island_assignment[predecessor]
            )
            if pickup != expected_pickup or destination != island_assignment[op]:
                raise RuntimeError(f"selected W route disagrees with product path for {op}")
            empty_duration = float(
                instance.w_empty_time[(vehicle, previous_location, pickup)]
            )
            loaded_duration = float(
                instance.w_loaded_time[(vehicle, pickup, destination)]
            )
            loaded_start = _clean_time(w_loaded_start[op].X)
            release = (
                0.0 if predecessor is None
                else _clean_time(
                    start[predecessor].X
                    + instance.processing_time[(predecessor, island_assignment[predecessor])]
                )
            )
            task = WTask(
                task_id=f"W:{op}",
                vehicle_id=vehicle,
                product_id=instance.product_of[op],
                predecessor_op=predecessor,
                operation_id=op,
                pickup=pickup,
                destination=destination,
                release_time=release,
                empty_origin=previous_location,
                empty_start=previous_arrival,
                empty_arrival=previous_arrival + empty_duration,
                loaded_start=loaded_start,
                arrival_time=loaded_start + loaded_duration,
                empty_travel_time=empty_duration,
                loaded_travel_time=loaded_duration,
                empty_distance=instance.distance[(previous_location, pickup)],
                loaded_distance=instance.distance[(pickup, destination)],
            )
            w_timelines[vehicle].append(task)
            w_task_by_operation[op] = task
            previous_location = destination
            previous_arrival = task.arrival_time

    island_predecessor: dict[str, str | None] = {}
    accumulated_reconfiguration_cost = 0.0
    for island, sequence in island_sequences.items():
        previous: str | None = None
        previous_config = instance.island_data[island].initial_config
        for op in sequence:
            island_predecessor[op] = previous
            config = instance.operation_data[op].required_config
            accumulated_reconfiguration_cost += instance.reconfiguration_cost[
                (island, previous_config, config)
            ]
            previous = op
            previous_config = config

    operation_schedules: dict[str, OperationSchedule] = {}
    for op in operations:
        island = island_assignment[op]
        config = instance.operation_data[op].required_config
        op_start = _clean_time(start[op].X)
        op_completion = _clean_time(
            op_start + instance.processing_time[(op, island)]
        )
        product_previous = product_predecessor[op]
        product_ready = (
            0.0 if product_previous is None
            else _clean_time(
                start[product_previous].X
                + instance.processing_time[
                    (product_previous, island_assignment[product_previous])
                ]
            )
        )
        island_previous = island_predecessor[op]
        island_ready = (
            0.0 if island_previous is None
            else _clean_time(
                start[island_previous].X
                + instance.processing_time[
                    (island_previous, island_assignment[island_previous])
                ]
            )
        )
        previous_config = (
            instance.island_data[island].initial_config
            if island_previous is None
            else instance.operation_data[island_previous].required_config
        )
        setup = float(instance.reconfiguration_time[(island, previous_config, config)])
        config_ready = island_ready + setup
        w_task = w_task_by_operation.get(op)
        w_ready = product_ready if w_task is None else w_task.arrival_time
        f_ready = f_task_by_operation[op].arrival_island
        readiness = {
            "PRODUCT": product_ready,
            "ISLAND_CONFIG": config_ready,
            "W_AGV": w_ready,
            "F_AGV": f_ready,
        }
        binding = tuple(
            name for name, value in readiness.items()
            if math.isclose(value, op_start, rel_tol=0.0, abs_tol=1e-6)
        ) or ("MILP_SLACK",)
        operation_schedules[op] = OperationSchedule(
            op_id=op,
            product_id=instance.product_of[op],
            island_id=island,
            config_id=config,
            product_predecessor=product_previous,
            processing_time=instance.processing_time[(op, island)],
            product_ready_time=product_ready,
            island_ready_time=island_ready,
            config_ready_time=config_ready,
            w_ready_time=w_ready,
            f_ready_time=f_ready,
            reconfiguration_start=island_ready,
            reconfiguration_end=config_ready,
            start_time=op_start,
            completion_time=op_completion,
            binding_resource=binding,
            w_task_id=None if w_task is None else w_task.task_id,
            f_task_id=f_task_by_operation[op].task_id,
        )

    reconstructed_schedule = Schedule(
        instance_id=instance.instance_id,
        operation_schedules=operation_schedules,
        product_sequences={key: list(value) for key, value in product_sequences.items()},
        product_predecessor=product_predecessor,
        product_successor=product_successor,
        island_timelines={key: list(value) for key, value in island_sequences.items()},
        w_timelines=w_timelines,
        f_timelines=f_timelines,
        accumulated_reconfiguration_cost=accumulated_reconfiguration_cost,
    )
    audit = check_schedule(instance, reconstructed_schedule)
    if not audit["feasible"]:
        raise RuntimeError(
            f"reconstructed Gurobi schedule is infeasible: {audit['violations']}"
        )
    solver_makespan = _clean_time(makespan.X)
    reconstructed_objective = calculate_objective(instance, reconstructed_schedule)
    replay_makespan = float(reconstructed_objective.makespan)
    if not math.isclose(solver_makespan, replay_makespan, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            "general Gurobi reconstructed makespan mismatch: "
            f"{solver_makespan} != {replay_makespan}"
        )

    # Reconstruct the selected MILP schedule from all four independent path
    # systems.  A single operation-action ordering cannot, in general, encode
    # an F-kit order that differs from the operation order (or an independent
    # W route order).  The former start-time-only replay therefore changed
    # valid time-limited incumbents on Core-S.  The reconstructed Schedule is
    # the lossless representation audited for exact-solver evidence; the old
    # action projection is retained below as an explicit diagnostic only.
    product_predecessor: dict[str, str | None] = {}
    product_successor: dict[str, str | None] = {}
    for sequence in product_sequences.values():
        for index, op in enumerate(sequence):
            product_predecessor[op] = sequence[index - 1] if index else None
            product_successor[op] = sequence[index + 1] if index + 1 < len(sequence) else None

    selected_detail: dict[str, tuple[str, str, str, str]] = {}
    for op, details in details_by_operation.items():
        selected = [key for key, variable in details if variable.X > 0.5]
        if len(selected) != 1:
            raise RuntimeError(f"expected one selected product/location detail for {op}")
        selected_detail[op] = selected[0]

    f_timelines: dict[str, list[FTask]] = {vehicle: [] for vehicle in instance.agvs_f}
    f_task_by_operation: dict[str, FTask] = {}
    for vehicle, sequence in f_sequences.items():
        for op in sequence:
            island = island_assignment[op]
            departure = _clean_time(f_depart[op].X)
            outbound = float(instance.f_outbound_time[(vehicle, island)])
            return_duration = float(instance.f_return_time[(vehicle, island)])
            task = FTask(
                task_id=f"F:{op}",
                vehicle_id=vehicle,
                operation_id=op,
                island_id=island,
                departure_wh=departure,
                arrival_island=departure + outbound,
                return_wh=departure + outbound + return_duration,
                outbound_time=outbound,
                return_time=return_duration,
                outbound_distance=instance.distance[("WH", island)],
                return_distance=instance.distance[(island, "WH")],
            )
            f_timelines[vehicle].append(task)
            f_task_by_operation[op] = task

    w_timelines: dict[str, list[WTask]] = {vehicle: [] for vehicle in instance.agvs_w}
    w_task_by_operation: dict[str, WTask] = {}
    for vehicle, sequence in w_sequences.items():
        previous_location = "WH"
        previous_arrival = 0.0
        for op in sequence:
            _, _, pickup, destination = selected_detail[op]
            predecessor = product_predecessor[op]
            expected_pickup = (
                "WH" if predecessor is None else island_assignment[predecessor]
            )
            if pickup != expected_pickup or destination != island_assignment[op]:
                raise RuntimeError(f"selected W route disagrees with product path for {op}")
            empty_duration = float(
                instance.w_empty_time[(vehicle, previous_location, pickup)]
            )
            loaded_duration = float(
                instance.w_loaded_time[(vehicle, pickup, destination)]
            )
            loaded_start = _clean_time(w_loaded_start[op].X)
            release = (
                0.0 if predecessor is None
                else _clean_time(
                    start[predecessor].X
                    + instance.processing_time[(predecessor, island_assignment[predecessor])]
                )
            )
            task = WTask(
                task_id=f"W:{op}",
                vehicle_id=vehicle,
                product_id=instance.product_of[op],
                predecessor_op=predecessor,
                operation_id=op,
                pickup=pickup,
                destination=destination,
                release_time=release,
                empty_origin=previous_location,
                empty_start=previous_arrival,
                empty_arrival=previous_arrival + empty_duration,
                loaded_start=loaded_start,
                arrival_time=loaded_start + loaded_duration,
                empty_travel_time=empty_duration,
                loaded_travel_time=loaded_duration,
                empty_distance=instance.distance[(previous_location, pickup)],
                loaded_distance=instance.distance[(pickup, destination)],
            )
            w_timelines[vehicle].append(task)
            w_task_by_operation[op] = task
            previous_location = destination
            previous_arrival = task.arrival_time

    island_predecessor: dict[str, str | None] = {}
    accumulated_reconfiguration_cost = 0.0
    for island, sequence in island_sequences.items():
        previous: str | None = None
        previous_config = instance.island_data[island].initial_config
        for op in sequence:
            island_predecessor[op] = previous
            config = instance.operation_data[op].required_config
            accumulated_reconfiguration_cost += instance.reconfiguration_cost[
                (island, previous_config, config)
            ]
            previous = op
            previous_config = config

    operation_schedules: dict[str, OperationSchedule] = {}
    for op in operations:
        island = island_assignment[op]
        config = instance.operation_data[op].required_config
        op_start = _clean_time(start[op].X)
        op_completion = _clean_time(
            op_start + instance.processing_time[(op, island)]
        )
        product_previous = product_predecessor[op]
        product_ready = (
            0.0 if product_previous is None
            else _clean_time(
                start[product_previous].X
                + instance.processing_time[
                    (product_previous, island_assignment[product_previous])
                ]
            )
        )
        island_previous = island_predecessor[op]
        island_ready = (
            0.0 if island_previous is None
            else _clean_time(
                start[island_previous].X
                + instance.processing_time[
                    (island_previous, island_assignment[island_previous])
                ]
            )
        )
        previous_config = (
            instance.island_data[island].initial_config
            if island_previous is None
            else instance.operation_data[island_previous].required_config
        )
        setup = float(instance.reconfiguration_time[(island, previous_config, config)])
        config_ready = island_ready + setup
        w_task = w_task_by_operation.get(op)
        w_ready = product_ready if w_task is None else w_task.arrival_time
        f_ready = f_task_by_operation[op].arrival_island
        readiness = {
            "PRODUCT": product_ready,
            "ISLAND_CONFIG": config_ready,
            "W_AGV": w_ready,
            "F_AGV": f_ready,
        }
        binding = tuple(
            name for name, value in readiness.items()
            if math.isclose(value, op_start, rel_tol=0.0, abs_tol=1e-6)
        ) or ("MILP_SLACK",)
        operation_schedules[op] = OperationSchedule(
            op_id=op,
            product_id=instance.product_of[op],
            island_id=island,
            config_id=config,
            product_predecessor=product_previous,
            processing_time=instance.processing_time[(op, island)],
            product_ready_time=product_ready,
            island_ready_time=island_ready,
            config_ready_time=config_ready,
            w_ready_time=w_ready,
            f_ready_time=f_ready,
            reconfiguration_start=island_ready,
            reconfiguration_end=config_ready,
            start_time=op_start,
            completion_time=op_completion,
            binding_resource=binding,
            w_task_id=None if w_task is None else w_task.task_id,
            f_task_id=f_task_by_operation[op].task_id,
        )

    reconstructed_schedule = Schedule(
        instance_id=instance.instance_id,
        operation_schedules=operation_schedules,
        product_sequences={key: list(value) for key, value in product_sequences.items()},
        product_predecessor=product_predecessor,
        product_successor=product_successor,
        island_timelines={key: list(value) for key, value in island_sequences.items()},
        w_timelines=w_timelines,
        f_timelines=f_timelines,
        accumulated_reconfiguration_cost=accumulated_reconfiguration_cost,
    )
    audit = check_schedule(instance, reconstructed_schedule)
    if not audit["feasible"]:
        raise RuntimeError(
            f"reconstructed Gurobi schedule is infeasible: {audit['violations']}"
        )
    solver_makespan = _clean_time(makespan.X)
    reconstructed_objective = calculate_objective(instance, reconstructed_schedule)
    replay_makespan = float(reconstructed_objective.makespan)
    if not math.isclose(solver_makespan, replay_makespan, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            "general Gurobi reconstructed makespan mismatch: "
            f"{solver_makespan} != {replay_makespan}"
        )

    canonical_index = {op: index for index, op in enumerate(operations)}
    ordered_operations = sorted(
        operations,
        key=lambda op: (start[op].X, completion[op].X, canonical_index[op]),
    )
    actions = tuple(
        Action(op, island_assignment[op], w_assignment[op], f_assignment[op])
        for op in ordered_operations
    )
    action_replay = RCIASConstructionEnv(instance)
    for action in actions:
        action_replay.step(action)
    action_audit = check_schedule(instance, action_replay.schedule)
    action_replay_makespan = float(action_replay.objective().makespan)
    action_replay_matches_solver = math.isclose(
        solver_makespan, action_replay_makespan, rel_tol=0.0, abs_tol=1e-6
    )
    model.update()
    return GeneralGurobiResult(
        backend="gurobi-general-rcias-milp",
        status=status,
        solver_version=".".join(map(str, gp.gurobi.version())),
        solver_makespan=solver_makespan,
        best_bound=float(model.ObjBound),
        gap=float(model.MIPGap),
        runtime_seconds=float(model.Runtime),
        total_runtime_seconds=time.perf_counter() - started,
        optimality_proven=model.Status == GRB.OPTIMAL,
        actions=actions,
        schedule=reconstructed_schedule,
        objective=reconstructed_objective,
        replay_makespan=replay_makespan,
        replay_feasible=True,
        action_replay_makespan=action_replay_makespan,
        action_replay_feasible=bool(action_audit["feasible"]),
        action_replay_matches_solver=action_replay_matches_solver,
        action_replay_schedule=action_replay.schedule,
        island_assignments=island_assignment,
        w_assignments=w_assignment,
        f_assignments=f_assignment,
        product_sequences=product_sequences,
        island_sequences=island_sequences,
        w_sequences=w_sequences,
        f_sequences=f_sequences,
        variable_count=int(model.NumVars),
        constraint_count=int(model.NumConstrs + model.NumGenConstrs),
        node_count=float(model.NodeCount),
        h1_upper_bound=makespan_upper,
        h1_mip_start_used=bool(use_h1_mip_start),
        h1_mip_start_used=bool(use_h1_mip_start),
    )
