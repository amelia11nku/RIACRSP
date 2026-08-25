"""Three deterministic constructive heuristics for decoder validation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.insertion_decoder import Action, ActionProbe
from rcias_clgri.env.objective import ObjectiveBreakdown
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.env.schedule import Schedule


@dataclass(frozen=True)
class BaselineResult:
    method: str
    schedule: Schedule
    objective: ObjectiveBreakdown
    runtime_seconds: float
    actions: tuple[Action, ...]


def _candidate_probes(env: RCIASConstructionEnv, op_ids: tuple[str, ...]) -> list[ActionProbe]:
    probes: list[ActionProbe] = []
    for op_id in op_ids:
        for island_id in env.get_eligible_islands(op_id):
            for w_agv_id in env.get_feasible_w_agvs(op_id, island_id):
                for f_agv_id in env.get_feasible_f_agvs(op_id, island_id):
                    probes.append(env.probe(Action(op_id, island_id, w_agv_id, f_agv_id)))
    return probes


def _remaining_workload(instance: Instance, op_id: str) -> float:
    affected = {op_id, *instance.transitive_successors[op_id]}
    return float(sum(
        min(instance.processing_time[(operation, island)] for island in instance.operation_data[operation].eligible_islands)
        for operation in affected
    ))


def _h1_probe(env: RCIASConstructionEnv) -> ActionProbe:
    ready = env.get_ready_operations()
    product_ready = {}
    for op_id in ready:
        sequence = env.schedule.product_sequences[env.instance.product_of[op_id]]
        product_ready[op_id] = 0.0 if not sequence else env.schedule.operation_schedules[sequence[-1]].completion_time
    earliest = min(product_ready.values())
    selected = tuple(op_id for op_id in ready if product_ready[op_id] == earliest)
    probes = _candidate_probes(env, selected)
    return min(
        probes,
        key=lambda probe: (
            probe.predicted_completion,
            probe.w_ready,
            probe.f_probe.task.arrival_island,
            probe.action.operation_id,
            probe.action.island_id,
            probe.action.w_agv_id or "",
            probe.action.f_agv_id,
        ),
    )


def _h2_probe(env: RCIASConstructionEnv) -> ActionProbe:
    ready = env.get_ready_operations()
    priorities = {op_id: _remaining_workload(env.instance, op_id) for op_id in ready}
    highest = max(priorities.values())
    selected = tuple(op_id for op_id in ready if priorities[op_id] == highest)
    probes = _candidate_probes(env, selected)
    return min(
        probes,
        key=lambda probe: (
            env.instance.processing_time[(probe.action.operation_id, probe.action.island_id)]
            + probe.machine_probe.setup_before
            + (0.0 if probe.w_probe is None else probe.w_probe.task.loaded_travel_time),
            probe.predicted_completion,
            probe.action.operation_id,
            probe.action.island_id,
        ),
    )


def _h3_probe(env: RCIASConstructionEnv) -> ActionProbe:
    probes = _candidate_probes(env, env.get_ready_operations())
    return min(
        probes,
        key=lambda probe: (
            probe.machine_probe.setup_before > 0,
            probe.machine_probe.setup_before * 2.0
            + env.instance.processing_time[(probe.action.operation_id, probe.action.island_id)]
            + (0.0 if probe.w_probe is None else probe.w_probe.task.loaded_travel_time)
            + 0.25 * probe.f_probe.task.outbound_time,
            probe.predicted_completion,
            probe.action.operation_id,
            probe.action.island_id,
        ),
    )


def solve_dispatching(instance: Instance, method: str = "H1") -> BaselineResult:
    """Construct a complete schedule with H1, H2, or H3."""

    normalized = method.upper()
    selectors = {"H1": _h1_probe, "H2": _h2_probe, "H3": _h3_probe}
    if normalized not in selectors:
        raise ValueError(f"unknown dispatching method: {method}")
    started = time.perf_counter()
    env = RCIASConstructionEnv(instance)
    actions: list[Action] = []
    while not env.done:
        probe = selectors[normalized](env)
        env.decoder.commit_action(env.schedule, probe)
        actions.append(probe.action)
    runtime = time.perf_counter() - started
    return BaselineResult(normalized, env.schedule, env.objective(), runtime, tuple(actions))
