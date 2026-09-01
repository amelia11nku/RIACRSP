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
from rcias_clgri.env.objective import ObjectiveBreakdown
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.env.schedule import Schedule
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
    island_assignments: Mapping[str, str]
    w_assignments: Mapping[str, str | None]
    f_assignments: Mapping[str, str]
    product_sequences: Mapping[str, tuple[str, ...]]
    island_sequences: Mapping[str, tuple[str, ...]]
    w_sequences: Mapping[str, tuple[str, ...]]
    f_sequences: Mapping[str, tuple[str, ...]]
    variable_count: int
    constraint_count: int
    h1_upper_bound: float

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
            "h1_upper_bound": self.h1_upper_bound,
            "schedule": self.schedule.to_dict(),
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


def solve_general_gurobi(
    instance: Instance,
    *,
    time_limit_seconds: float = 600.0,
    seed: int = 23,
    threads: int = 1,
    mip_gap: float = 0.0,
    output_flag: bool = False,
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
    model.Params.OutputFlag = int(output_flag)
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

    canonical_index = {op: index for index, op in enumerate(operations)}
    ordered_operations = sorted(
        operations,
        key=lambda op: (start[op].X, completion[op].X, canonical_index[op]),
    )
    actions = tuple(
        Action(op, island_assignment[op], w_assignment[op], f_assignment[op])
        for op in ordered_operations
    )
    replay = RCIASConstructionEnv(instance)
    for action in actions:
        replay.step(action)
    audit = check_schedule(instance, replay.schedule)
    if not audit["feasible"]:
        raise RuntimeError(f"general Gurobi replay is infeasible: {audit['violations']}")
    solver_makespan = float(makespan.X)
    replay_makespan = float(replay.objective().makespan)
    if not math.isclose(solver_makespan, replay_makespan, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            "general Gurobi native/replay makespan mismatch: "
            f"{solver_makespan} != {replay_makespan}"
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
        schedule=replay.schedule,
        objective=replay.objective(),
        replay_makespan=replay_makespan,
        replay_feasible=True,
        island_assignments=island_assignment,
        w_assignments=w_assignment,
        f_assignments=f_assignment,
        product_sequences=product_sequences,
        island_sequences=island_sequences,
        w_sequences=w_sequences,
        f_sequences=f_sequences,
        variable_count=int(model.NumVars),
        constraint_count=int(model.NumConstrs + model.NumGenConstrs),
        h1_upper_bound=makespan_upper,
    )
