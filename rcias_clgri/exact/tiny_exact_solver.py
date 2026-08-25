"""Exhaustive active-schedule solver used when commercial solvers are absent."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.env.objective import ObjectiveBreakdown, calculate_objective
from rcias_clgri.env.schedule import Schedule


@dataclass(frozen=True)
class ExactResult:
    status: str
    backend: str
    objective_name: str
    best_value: float
    objective: ObjectiveBreakdown
    schedule: Schedule
    actions: tuple[Action, ...]
    explored_nodes: int
    runtime_seconds: float


def gurobi_available() -> bool:
    """Report availability without importing or licensing Gurobi."""

    return importlib.util.find_spec("gurobipy") is not None


def _ready(instance: Instance, schedule: Schedule) -> tuple[str, ...]:
    selected = schedule.scheduled_operations
    return tuple(
        op_id for op_id in instance.operations
        if op_id not in selected and instance.predecessors[op_id] <= selected
    )


def _actions(instance: Instance, schedule: Schedule, op_id: str) -> tuple[Action, ...]:
    product_id = instance.product_of[op_id]
    sequence = schedule.product_sequences[product_id]
    pickup = "WH" if not sequence else schedule.operation_schedules[sequence[-1]].island_id
    actions: list[Action] = []
    for island_id in instance.operation_data[op_id].eligible_islands:
        w_choices: tuple[str | None, ...] = (None,) if pickup == island_id else instance.agvs_w
        for w_agv_id in w_choices:
            for f_agv_id in instance.agvs_f:
                actions.append(Action(op_id, island_id, w_agv_id, f_agv_id))
    return tuple(actions)


def solve_tiny_exact(
    instance: Instance,
    *,
    objective: str = "makespan",
    cost_weight: float | None = None,
    max_operations: int = 8,
    node_limit: int = 1_000_000,
    time_limit_seconds: float = 60.0,
) -> ExactResult:
    """Enumerate precedence-feasible decisions with makespan branch-and-bound.

    For a regular makespan objective, an optimal active schedule exists; the
    insertion decoder left-shifts every chosen resource order, so enumerating all
    topological operation orders, island assignments, and W/F assignments covers
    the active-schedule decision space used for tiny correctness validation.
    """

    if instance.num_operations > max_operations:
        raise ValueError(
            f"tiny exact solver protects against large instances: {instance.num_operations}>{max_operations}"
        )
    normalized = objective.lower()
    if normalized not in {"makespan", "weighted"}:
        raise ValueError("objective must be 'makespan' or 'weighted'")
    weight = (
        float(cost_weight)
        if cost_weight is not None
        else float(instance.objective_parameters.get("cost_weight", 0.05))
    )
    decoder = InsertionDecoder(instance)
    initial = decoder.empty_schedule()
    incumbent_schedule: Schedule | None = None
    incumbent_actions: tuple[Action, ...] = ()
    incumbent_breakdown: ObjectiveBreakdown | None = None
    incumbent_value = float("inf")
    explored_nodes = 0
    started = time.perf_counter()
    truncated = False

    def score(breakdown: ObjectiveBreakdown) -> float:
        if normalized == "makespan":
            return breakdown.makespan
        return breakdown.makespan + weight * breakdown.total_cost

    def search(schedule: Schedule, action_path: tuple[Action, ...]) -> None:
        nonlocal incumbent_schedule, incumbent_actions, incumbent_breakdown
        nonlocal incumbent_value, explored_nodes, truncated
        if truncated:
            return
        explored_nodes += 1
        if explored_nodes > node_limit or time.perf_counter() - started > time_limit_seconds:
            truncated = True
            return
        current_makespan = max(
            (record.completion_time for record in schedule.operation_schedules.values()), default=0.0
        )
        if normalized == "makespan" and current_makespan >= incumbent_value:
            return
        if len(schedule.operation_schedules) == instance.num_operations:
            breakdown = calculate_objective(instance, schedule)
            value = score(breakdown)
            if value + 1e-9 < incumbent_value:
                audit = check_schedule(instance, schedule)
                if not audit["feasible"]:
                    raise RuntimeError(f"exact search produced an infeasible schedule: {audit['violations']}")
                incumbent_value = value
                incumbent_schedule = schedule
                incumbent_actions = action_path
                incumbent_breakdown = breakdown
            return

        candidates = []
        for op_id in _ready(instance, schedule):
            for action in _actions(instance, schedule, op_id):
                probe = decoder.probe_action(schedule, action)
                candidates.append((probe.predicted_completion, probe))
        candidates.sort(key=lambda item: (item[0], item[1].action.operation_id, item[1].action.island_id))
        for _, probe in candidates:
            child = schedule.clone()
            decoder.commit_action(child, probe)
            search(child, (*action_path, probe.action))
            if truncated:
                break

    search(initial, ())
    if incumbent_schedule is None or incumbent_breakdown is None:
        raise RuntimeError("no complete schedule found before the exact-search limit")
    return ExactResult(
        status="BEST_KNOWN_LIMIT" if truncated else "OPTIMAL",
        backend="exhaustive-active-schedule-bnb",
        objective_name=normalized,
        best_value=incumbent_value,
        objective=incumbent_breakdown,
        schedule=incumbent_schedule,
        actions=incumbent_actions,
        explored_nodes=explored_nodes,
        runtime_seconds=time.perf_counter() - started,
    )
