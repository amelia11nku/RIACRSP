"""Problem-adapted dual-space artificial bee colony for RIACRSP."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any

from rcias_clgri.data.instance import Instance

from .common import Candidate, DecodedCandidate, SearchResult, TracePoint, decode_candidate, random_candidate
from .dabc_chdg import CriticalIslandBlock, build_generalized_chdg, critical_island_blocks, critical_path
from .dabc_chdg_rules import audit_intermachine_move, audit_intramachine_move
from .operators import pox_pair, uniform_pair


@dataclass(frozen=True)
class DABCConfig:
    population_size: int = 120
    neighborhood_search_intensity: int = 2
    max_nonimprovements: int = 40
    self_exploration_rate: float = 0.1
    first_level_fraction: float = 0.2
    restart_self_explorations: int = 10
    source_clipping_mode: str = "shadow"
    iteration_limit: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("DABC population_size must be at least two")
        if self.neighborhood_search_intensity < 1:
            raise ValueError("DABC neighborhood_search_intensity must be positive")
        if not 0.0 <= self.self_exploration_rate <= 1.0:
            raise ValueError("DABC self_exploration_rate must lie in [0, 1]")
        if not 0.0 < self.first_level_fraction <= 1.0:
            raise ValueError("DABC first_level_fraction must lie in (0, 1]")
        if self.source_clipping_mode != "shadow":
            raise ValueError("only audited source_clipping_mode='shadow' is implemented")


@dataclass
class DABCIndividual:
    individual_id: int
    decoded: DecodedCandidate
    unimproved_count: int = 0


class _EvaluationState:
    def __init__(self, instance: Instance, started: float, time_limit: float) -> None:
        self.instance = instance
        self.started = started
        self.time_limit = time_limit
        self.evaluations = 0
        self.best: DecodedCandidate | None = None
        self.best_time = 0.0
        self.trace: list[TracePoint] = []

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def budget_exhausted(self) -> bool:
        return self.elapsed >= self.time_limit

    def consider(self, decoded: DecodedCandidate) -> None:
        if self.best is None or decoded.makespan < self.best.makespan:
            self.best = decoded
            self.best_time = self.elapsed
            self.trace.append(TracePoint(self.best_time, self.evaluations, decoded.makespan))

    def evaluate(self, candidate: Candidate, *, consider: bool = True) -> DecodedCandidate:
        decoded = decode_candidate(self.instance, candidate)
        self.evaluations += 1
        if consider:
            self.consider(decoded)
        return decoded


def initialize_population(
    instance: Instance,
    rng: random.Random,
    evaluator: _EvaluationState,
    config: DABCConfig,
) -> list[DABCIndividual]:
    """Generate and fully decode the source-sized random population."""

    return [
        DABCIndividual(index, evaluator.evaluate(random_candidate(instance, rng)))
        for index in range(config.population_size)
    ]


def hierarchy_first_level(
    population: list[DABCIndividual], fraction: float
) -> tuple[DABCIndividual, ...]:
    """Select the stable best fraction, rounding a non-integer size upward."""

    count = max(1, math.ceil(len(population) * fraction))
    ranked = sorted(population, key=lambda item: (item.decoded.makespan, item.individual_id))
    return tuple(ranked[:count])


def _self_explore(
    instance: Instance,
    candidate: Candidate,
    rng: random.Random,
) -> tuple[Candidate, dict[str, object]]:
    """Apply the paper's equally likely OS-insertion or one-gene CS analog."""

    if rng.random() < 0.5:
        order = list(candidate.operation_order)
        changed = len(order) >= 2
        if changed:
            source, target = rng.sample(range(len(order)), 2)
            operation = order.pop(source)
            order.insert(target, operation)
        else:
            source = target = 0
        return Candidate(
            tuple(order), candidate.island_assignment, candidate.w_assignment, candidate.f_assignment
        ), {
            "branch": "sequence",
            "operator": "insertion",
            "source_position": source,
            "target_position": target,
            "changed": changed,
        }

    layer = rng.choice(("island", "W", "F"))
    islands = list(candidate.island_assignment)
    w_agvs = list(candidate.w_assignment)
    f_agvs = list(candidate.f_assignment)
    alternatives: list[tuple[int, tuple[str, ...]]] = []
    if layer == "island":
        for index, operation_id in enumerate(instance.operations):
            values = tuple(
                value for value in instance.operation_data[operation_id].eligible_islands
                if value != islands[index]
            )
            if values:
                alternatives.append((index, values))
    elif layer == "W":
        for index, current in enumerate(w_agvs):
            values = tuple(value for value in instance.agvs_w if value != current)
            if values:
                alternatives.append((index, values))
    else:
        for index, current in enumerate(f_agvs):
            values = tuple(value for value in instance.agvs_f if value != current)
            if values:
                alternatives.append((index, values))
    changed = bool(alternatives)
    operation_id: str | None = None
    if alternatives:
        index, values = rng.choice(alternatives)
        operation_id = instance.operations[index]
        replacement = rng.choice(values)
        if layer == "island":
            islands[index] = replacement
        elif layer == "W":
            w_agvs[index] = replacement
        else:
            f_agvs[index] = replacement
    return Candidate(
        candidate.operation_order, tuple(islands), tuple(w_agvs), tuple(f_agvs)
    ), {
        "branch": "assignment",
        "layer": layer,
        "operation_id": operation_id,
        "changed": changed,
    }


def _crossover_candidates(
    instance: Instance,
    left: Candidate,
    right: Candidate,
    rng: random.Random,
) -> tuple[Candidate, Candidate]:
    order = pox_pair(instance, left.operation_order, right.operation_order, rng)
    islands = uniform_pair(left.island_assignment, right.island_assignment, rng)
    w_agvs = uniform_pair(left.w_assignment, right.w_assignment, rng)
    f_agvs = uniform_pair(left.f_assignment, right.f_assignment, rng)
    return (
        Candidate(order[0], islands[0], w_agvs[0], f_agvs[0]),
        Candidate(order[1], islands[1], w_agvs[1], f_agvs[1]),
    )


def _select_mate(
    current: DABCIndividual,
    population: list[DABCIndividual],
    first_level: tuple[DABCIndividual, ...],
    rng: random.Random,
) -> DABCIndividual:
    choices = [item for item in first_level if item.individual_id != current.individual_id]
    if not choices:
        choices = [item for item in population if item.individual_id != current.individual_id]
    return rng.choice(choices)


def _project_relative_order(
    candidate: Candidate,
    requested_order: tuple[str, ...],
) -> Candidate:
    requested_set = set(requested_order)
    if len(requested_set) != len(requested_order):
        raise ValueError("requested graph order contains duplicate operations")
    positions = [
        index for index, operation_id in enumerate(candidate.operation_order)
        if operation_id in requested_set
    ]
    if len(positions) != len(requested_order):
        raise ValueError("requested graph order is not a candidate subset")
    order = list(candidate.operation_order)
    for position, operation_id in zip(positions, requested_order):
        order[position] = operation_id
    return Candidate(
        tuple(order), candidate.island_assignment, candidate.w_assignment, candidate.f_assignment
    )


def _replace_island(
    instance: Instance,
    candidate: Candidate,
    operation_id: str,
    island_id: str,
) -> Candidate:
    index = instance.operations.index(operation_id)
    islands = list(candidate.island_assignment)
    islands[index] = island_id
    return Candidate(
        candidate.operation_order, tuple(islands), candidate.w_assignment, candidate.f_assignment
    )


def _roundtrip_matches(
    decoded: DecodedCandidate,
    island_id: str,
    requested_order: tuple[str, ...],
) -> bool:
    requested = set(requested_order)
    timeline = decoded.schedule.island_timelines[island_id]
    realized = tuple(operation for operation in timeline if operation in requested)
    if realized != requested_order:
        return False
    positions = [timeline.index(operation) for operation in requested_order]
    return positions == list(range(positions[0], positions[-1] + 1))


def _record_rule_diagnostics(audit, diagnostics: dict[str, Any]) -> None:
    if not audit.full_dag_reachability_feasible:
        diagnostics["reachability_rejected_proposals"] += 1
    for rule in audit.feasibility_rules:
        if rule.triggered:
            diagnostics["source_theorem_feasibility_hits"] += 1
    for rule in audit.clipping_rules:
        if rule.triggered:
            diagnostics["source_clipping_theorem_hits"] += 1


def _cns1(
    instance: Instance,
    current: DecodedCandidate,
    blocks: tuple[CriticalIslandBlock, ...],
    rng: random.Random,
    evaluator: _EvaluationState,
    diagnostics: dict[str, Any],
) -> DecodedCandidate:
    diagnostics["cns1_calls"] += 1
    movable_blocks = [
        (index, block) for index, block in enumerate(blocks)
        if len(block.operation_ids) >= 2
    ]
    if not movable_blocks:
        return current
    block_index, block = rng.choice(movable_blocks)
    moved = rng.choice(block.operation_ids)
    remainder = [operation for operation in block.operation_ids if operation != moved]
    decoded_neighbors: list[DecodedCandidate] = []
    original_position = block.operation_ids.index(moved)
    for target in range(len(block.operation_ids)):
        if target == original_position or evaluator.budget_exhausted:
            continue
        requested = list(remainder)
        requested.insert(target, moved)
        requested_order = tuple(requested)
        audit = audit_intramachine_move(
            instance,
            current,
            blocks,
            block_index,
            moved,
            target,
            requested_order,
        )
        _record_rule_diagnostics(audit, diagnostics)
        if not audit.full_dag_reachability_feasible:
            continue
        projected = _project_relative_order(current.candidate, requested_order)
        decoded = evaluator.evaluate(projected, consider=False)
        diagnostics["cns1_generated_candidates"] += 1
        represented = _roundtrip_matches(decoded, block.island_id, requested_order)
        diagnostics["graph_move_records"].append({
            "move": "CNS1",
            "operation_id": moved,
            "island_from": block.island_id,
            "island_to": block.island_id,
            "requested_order": requested_order,
            "projected_order_changed": projected.operation_order != current.candidate.operation_order,
            "roundtrip_status": "REPRESENTED" if represented else "ROUNDTRIP_UNREPRESENTABLE",
            "objective_before": current.makespan,
            "objective_after": decoded.makespan,
            "source_clip_predicate": audit.source_clip_predicate,
        })
        if not represented:
            diagnostics["roundtrip_unrepresentable_proposals"] += 1
            continue
        evaluator.consider(decoded)
        if audit.source_clip_predicate:
            diagnostics["shadow_clipped_proposals"] += 1
            if decoded.makespan < current.makespan:
                diagnostics["shadow_false_clip_improvements"] += 1
        decoded_neighbors.append(decoded)
    return min(
        [current, *decoded_neighbors],
        key=lambda item: (item.makespan, 0 if item is current else 1),
    )


def _cns2(
    instance: Instance,
    current: DecodedCandidate,
    blocks: tuple[CriticalIslandBlock, ...],
    rng: random.Random,
    evaluator: _EvaluationState,
    diagnostics: dict[str, Any],
) -> DecodedCandidate:
    diagnostics["cns2_calls"] += 1
    eligible: list[tuple[CriticalIslandBlock, tuple[str, ...]]] = []
    for block in blocks:
        movable = tuple(
            operation for operation in block.operation_ids
            if any(
                island != block.island_id
                for island in instance.operation_data[operation].eligible_islands
            )
        )
        if movable:
            eligible.append((block, movable))
    if not eligible:
        return current
    block, movable = rng.choice(eligible)
    moved = rng.choice(movable)
    alternatives = tuple(
        island for island in instance.operation_data[moved].eligible_islands
        if island != block.island_id
    )
    target_island = rng.choice(alternatives)
    target_timeline = list(current.schedule.island_timelines[target_island])
    decoded_neighbors: list[DecodedCandidate] = []
    for target in range(len(target_timeline) + 1):
        if evaluator.budget_exhausted:
            break
        requested = list(target_timeline)
        requested.insert(target, moved)
        requested_order = tuple(requested)
        audit = audit_intermachine_move(
            instance, current, moved, target_island, requested_order
        )
        _record_rule_diagnostics(audit, diagnostics)
        if not audit.full_dag_reachability_feasible:
            continue
        reassigned = _replace_island(instance, current.candidate, moved, target_island)
        projected = _project_relative_order(reassigned, requested_order)
        decoded = evaluator.evaluate(projected, consider=False)
        diagnostics["cns2_generated_candidates"] += 1
        represented = tuple(decoded.schedule.island_timelines[target_island]) == requested_order
        diagnostics["graph_move_records"].append({
            "move": "CNS2",
            "operation_id": moved,
            "island_from": block.island_id,
            "island_to": target_island,
            "requested_order": requested_order,
            "projected_order_changed": projected.operation_order != current.candidate.operation_order,
            "roundtrip_status": "REPRESENTED" if represented else "ROUNDTRIP_UNREPRESENTABLE",
            "objective_before": current.makespan,
            "objective_after": decoded.makespan,
            "source_clip_predicate": audit.source_clip_predicate,
        })
        if not represented:
            diagnostics["roundtrip_unrepresentable_proposals"] += 1
            continue
        evaluator.consider(decoded)
        decoded_neighbors.append(decoded)
    return min(
        [current, *decoded_neighbors],
        key=lambda item: (item.makespan, 0 if item is current else 1),
    )


def _diagnostics(config: DABCConfig) -> dict[str, Any]:
    return {
        "fidelity": "PAPER_FAITHFUL_RMSSP_RULES_WITH_RIACRSP_PROJECTION",
        "source_clipping_mode": config.source_clipping_mode,
        "source_clipping_predicates": "THEOREMS_5_TO_10_EXACT_SHADOW_ONLY",
        "hierarchy_rounding": "ceil(population_size * first_level_fraction)",
        "employed_crossover_count": 0,
        "self_exploration_count": 0,
        "self_exploration_sequence_count": 0,
        "self_exploration_assignment_count": 0,
        "cns1_calls": 0,
        "cns2_calls": 0,
        "cns1_generated_candidates": 0,
        "cns2_generated_candidates": 0,
        "roundtrip_unrepresentable_proposals": 0,
        "reachability_rejected_proposals": 0,
        "source_theorem_feasibility_hits": 0,
        "source_clipping_theorem_hits": 0,
        "shadow_clipped_proposals": 0,
        "shadow_false_clip_improvements": 0,
        "restart_count": 0,
        "restart_self_exploration_decodes": 0,
        "graph_move_records": [],
    }


def solve_dabc(
    instance: Instance,
    time_limit: float,
    seed: int,
    config: DABCConfig = DABCConfig(),
) -> SearchResult:
    """Run DABC-RIACRSP with common-decoder fitness and a wall-clock stop."""

    rng = random.Random(seed)
    started = time.perf_counter()
    evaluator = _EvaluationState(instance, started, time_limit)
    diagnostics = _diagnostics(config)
    population = initialize_population(instance, rng, evaluator, config)
    initialization_seconds = evaluator.elapsed
    iterations = 0

    while (
        not evaluator.budget_exhausted
        and (config.iteration_limit is None or iterations < config.iteration_limit)
    ):
        first_level = hierarchy_first_level(population, config.first_level_fraction)
        for index, individual in enumerate(population):
            if evaluator.budget_exhausted:
                break
            if rng.random() < config.self_exploration_rate:
                candidate, detail = _self_explore(instance, individual.decoded.candidate, rng)
                decoded = evaluator.evaluate(candidate)
                diagnostics["self_exploration_count"] += 1
                diagnostics[f"self_exploration_{detail['branch']}_count"] += 1
                population[index] = DABCIndividual(
                    individual.individual_id, decoded, individual.unimproved_count
                )
            else:
                mate = _select_mate(individual, population, first_level, rng)
                candidates = _crossover_candidates(
                    instance, individual.decoded.candidate, mate.decoded.candidate, rng
                )
                children = [evaluator.evaluate(candidate) for candidate in candidates]
                diagnostics["employed_crossover_count"] += 1
                best = min(
                    [individual.decoded, *children],
                    key=lambda item: (item.makespan, 0 if item is individual.decoded else 1),
                )
                population[index] = DABCIndividual(
                    individual.individual_id, best, individual.unimproved_count
                )

        for index, individual in enumerate(population):
            if evaluator.budget_exhausted:
                break
            graph = build_generalized_chdg(instance, individual.decoded)
            path = critical_path(graph, rng)
            blocks = critical_island_blocks(individual.decoded, path)
            graph_neighbors: list[DecodedCandidate] = []
            for _ in range(config.neighborhood_search_intensity):
                if evaluator.budget_exhausted:
                    break
                graph_neighbors.append(
                    _cns1(instance, individual.decoded, blocks, rng, evaluator, diagnostics)
                )
                if evaluator.budget_exhausted:
                    break
                graph_neighbors.append(
                    _cns2(instance, individual.decoded, blocks, rng, evaluator, diagnostics)
                )
            best = min(
                [individual.decoded, *graph_neighbors],
                key=lambda item: (item.makespan, 0 if item is individual.decoded else 1),
            )
            improved = best.makespan < individual.decoded.makespan
            population[index] = DABCIndividual(
                individual.individual_id,
                best,
                0 if improved else individual.unimproved_count + 1,
            )

        for index, individual in enumerate(population):
            if evaluator.budget_exhausted:
                break
            if individual.unimproved_count < config.max_nonimprovements:
                continue
            current = individual.decoded
            diagnostics["restart_count"] += 1
            for _ in range(config.restart_self_explorations):
                candidate, detail = _self_explore(instance, current.candidate, rng)
                current = evaluator.evaluate(candidate)
                diagnostics["self_exploration_count"] += 1
                diagnostics[f"self_exploration_{detail['branch']}_count"] += 1
                diagnostics["restart_self_exploration_decodes"] += 1
            population[index] = DABCIndividual(individual.individual_id, current, 0)
        iterations += 1

    if evaluator.best is None:
        raise RuntimeError("DABC population initialization produced no decoded candidate")
    diagnostics.update({
        "initialization_seconds": initialization_seconds,
        "population_size": len(population),
        "first_level_size": len(hierarchy_first_level(population, config.first_level_fraction)),
        "final_unimproved_counts": {
            str(item.individual_id): item.unimproved_count for item in population
        },
        "chdg_implementation_version": "generalized-riacrsp-event-dag-v1",
    })
    return SearchResult(
        "DABC-RIACRSP",
        evaluator.best,
        evaluator.best_time,
        evaluator.elapsed,
        evaluator.evaluations,
        iterations,
        None,
        tuple(evaluator.trace),
        diagnostics,
    )
