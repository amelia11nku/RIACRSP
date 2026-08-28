"""Shared representation, decoder, and result schema for Phase 5C search."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.objective import ObjectiveBreakdown
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.env.schedule import Schedule


@dataclass(frozen=True)
class Candidate:
    """Direct decision layers; priority order is precedence-safe at decode time."""

    operation_order: tuple[str, ...]
    island_assignment: tuple[str, ...]
    w_assignment: tuple[str, ...]
    f_assignment: tuple[str, ...]


@dataclass(frozen=True)
class DecodedCandidate:
    candidate: Candidate
    schedule: Schedule
    objective: ObjectiveBreakdown
    actions: tuple[Action, ...]
    feasible: bool

    @property
    def makespan(self) -> float:
        return float(self.objective.makespan)


@dataclass(frozen=True)
class TracePoint:
    elapsed_time: float
    decoder_evaluations: int
    current_best_makespan: float


@dataclass(frozen=True)
class SearchResult:
    method: str
    best: DecodedCandidate
    best_found_time: float
    runtime: float
    decoder_evaluations: int
    iterations: int
    generations_if_applicable: int | None
    convergence_trace: tuple[TracePoint, ...]
    diagnostics: Mapping[str, object]


def candidate_from_actions(instance: Instance, actions: tuple[Action, ...]) -> Candidate:
    by_operation = {action.operation_id: action for action in actions}
    return Candidate(
        tuple(action.operation_id for action in actions),
        tuple(by_operation[op].island_id for op in instance.operations),
        tuple((by_operation[op].w_agv_id or instance.agvs_w[0]) for op in instance.operations),
        tuple(by_operation[op].f_agv_id for op in instance.operations),
    )


def random_candidate(instance: Instance, rng: random.Random) -> Candidate:
    operations = list(instance.operations)
    rng.shuffle(operations)
    return Candidate(
        tuple(operations),
        tuple(rng.choice(instance.operation_data[op].eligible_islands) for op in instance.operations),
        tuple(rng.choice(instance.agvs_w) for _ in instance.operations),
        tuple(rng.choice(instance.agvs_f) for _ in instance.operations),
    )


def _cumulative_w_time(env: RCIASConstructionEnv, vehicle: str | None) -> float:
    if vehicle is None:
        return 0.0
    return sum(task.empty_travel_time + task.loaded_travel_time for task in env.schedule.w_timelines[vehicle])


def _cumulative_f_time(env: RCIASConstructionEnv, vehicle: str) -> float:
    return sum(task.outbound_time + task.return_time for task in env.schedule.f_timelines[vehicle])


def decode_candidate(
    instance: Instance,
    candidate: Candidate,
    transport_tiebreak: str | None = None,
) -> DecodedCandidate:
    """Decode only through the project's frozen ``RCIASConstructionEnv``."""

    if set(candidate.operation_order) != set(instance.operations):
        raise ValueError("operation_order must be a permutation of all operations")
    if not all(len(layer) == instance.num_operations for layer in (
        candidate.island_assignment, candidate.w_assignment, candidate.f_assignment
    )):
        raise ValueError("assignment layers must match the canonical operation count")
    position = {operation: index for index, operation in enumerate(instance.operations)}
    priority = {operation: index for index, operation in enumerate(candidate.operation_order)}
    env = RCIASConstructionEnv(instance)
    actions: list[Action] = []
    while not env.done:
        operation = min(env.get_ready_operations(), key=priority.__getitem__)
        index = position[operation]
        island = candidate.island_assignment[index]
        if island not in instance.operation_data[operation].eligible_islands:
            raise ValueError(f"ineligible encoded island {island} for {operation}")
        feasible_w = env.get_feasible_w_agvs(operation, island)
        if transport_tiebreak is None:
            encoded_w = candidate.w_assignment[index]
            w_agv = encoded_w if encoded_w in feasible_w else feasible_w[0]
            action = Action(operation, island, w_agv, candidate.f_assignment[index])
        else:
            if transport_tiebreak not in {"fixed", "cumulative"}:
                raise ValueError(f"unknown transport tiebreak: {transport_tiebreak}")
            probes = [
                env.probe(Action(operation, island, w_agv, f_agv))
                for w_agv in feasible_w for f_agv in env.get_feasible_f_agvs(operation, island)
            ]
            earliest_w = min(probe.w_ready for probe in probes)
            earliest_f = min(probe.f_probe.task.arrival_island for probe in probes)
            tied = [
                probe for probe in probes
                if probe.w_ready == earliest_w and probe.f_probe.task.arrival_island == earliest_f
            ]
            if transport_tiebreak == "fixed":
                selected = min(
                    tied,
                    key=lambda probe: (
                        instance.agvs_w.index(probe.action.w_agv_id) if probe.action.w_agv_id is not None else -1,
                        instance.agvs_f.index(probe.action.f_agv_id),
                    ),
                )
            else:
                selected = min(
                    tied,
                    key=lambda probe: (
                        _cumulative_w_time(env, probe.action.w_agv_id),
                        _cumulative_f_time(env, probe.action.f_agv_id),
                        probe.action.w_agv_id or "",
                        probe.action.f_agv_id,
                    ),
                )
            action = selected.action
        env.step(action)
        actions.append(action)
    audit = check_schedule(instance, env.schedule)
    feasible = bool(audit["feasible"])
    if not feasible:
        raise RuntimeError(f"common decoder produced infeasible schedule: {audit['violations']}")
    return DecodedCandidate(candidate, env.schedule, env.objective(), tuple(actions), feasible)
