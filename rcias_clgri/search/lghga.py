"""Single-objective learning-guided hybrid genetic algorithm for RIACRSP."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
import time
from typing import Any, Mapping

from rcias_clgri.data.instance import Instance

from .common import Candidate, DecodedCandidate, SearchResult, TracePoint, decode_candidate, random_candidate
from .lghga_learning import DTRBundle, improvement_rate_pct, predict_rates, select_neighborhood
from .lghga_neighborhoods import (
    NEIGHBORHOODS,
    inject_product_topology,
    propose_neighborhood,
    randomized_product_topology,
)
from .operators import pox_pair, uniform_pair


@dataclass(frozen=True)
class LGHGAConfig:
    max_generations: int = 100
    population_size: int = 40
    crossover_probability: float = 0.9
    mutation_probability: float = 0.4
    local_search_threshold_pct: float = 50.0
    local_search_population_size: int = 5
    local_search_max_iterations: int = 5
    neighborhood_size: int = 20
    knowledge_generation_runs: int = 20
    tournament_size: int = 2

    def __post_init__(self) -> None:
        if self.max_generations < 1 or self.population_size < 2:
            raise ValueError("LG_HGA requires positive generations and population >= 2")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must lie in [0, 1]")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1]")
        if self.local_search_population_size < 1:
            raise ValueError("local_search_population_size must be positive")
        if self.local_search_max_iterations < 1 or self.neighborhood_size < 1:
            raise ValueError("local-search iteration and neighborhood sizes must be positive")


@dataclass(frozen=True)
class KnowledgeGenerationResult:
    rows: tuple[Mapping[str, object], ...]
    best: DecodedCandidate
    runtime: float
    decoder_evaluations: int
    generations: int


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

    def evaluate(self, candidate: Candidate) -> DecodedCandidate:
        decoded = decode_candidate(self.instance, candidate)
        self.evaluations += 1
        if self.best is None or decoded.makespan < self.best.makespan:
            self.best = decoded
            self.best_time = self.elapsed
            self.trace.append(TracePoint(self.best_time, self.evaluations, decoded.makespan))
        return decoded


def _candidate_key(decoded: DecodedCandidate):
    candidate = decoded.candidate
    return (
        decoded.makespan,
        candidate.operation_order,
        candidate.island_assignment,
        candidate.w_assignment,
        candidate.f_assignment,
    )


def _retain_best(pool: list[DecodedCandidate], size: int) -> list[DecodedCandidate]:
    """Single-objective NSGA-II specialization with stable deterministic ties."""

    ranked = sorted(enumerate(pool), key=lambda item: (*_candidate_key(item[1]), item[0]))
    return [decoded for _, decoded in ranked[:size]]


def _tournament(
    population: list[DecodedCandidate],
    rng: random.Random,
    size: int,
) -> DecodedCandidate:
    return min(rng.sample(population, min(size, len(population))), key=_candidate_key)


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


def _topology_mutation(
    instance: Instance,
    candidate: Candidate,
    rng: random.Random,
) -> Candidate:
    product = rng.choice(instance.products)
    operations = instance.product_data[product].operations
    topology = randomized_product_topology(instance, product, candidate.operation_order, rng)
    return inject_product_topology(candidate, operations, topology)


def _island_mutation(
    instance: Instance,
    candidate: Candidate,
    rng: random.Random,
) -> Candidate:
    alternatives = []
    for index, operation in enumerate(instance.operations):
        values = tuple(
            island for island in instance.operation_data[operation].eligible_islands
            if island != candidate.island_assignment[index]
        )
        if values:
            alternatives.append((index, values))
    if not alternatives:
        return candidate
    index, values = rng.choice(alternatives)
    islands = list(candidate.island_assignment)
    islands[index] = rng.choice(values)
    return Candidate(
        candidate.operation_order, tuple(islands), candidate.w_assignment, candidate.f_assignment
    )


def _sequence_position_mutation(candidate: Candidate, rng: random.Random) -> Candidate:
    if len(candidate.operation_order) < 2:
        return candidate
    left, right = rng.sample(range(len(candidate.operation_order)), 2)
    order = list(candidate.operation_order)
    order[left], order[right] = order[right], order[left]
    return Candidate(
        tuple(order), candidate.island_assignment, candidate.w_assignment, candidate.f_assignment
    )


def _logistics_mutation(
    instance: Instance,
    candidate: Candidate,
    rng: random.Random,
) -> Candidate:
    w_agvs = list(candidate.w_assignment)
    f_agvs = list(candidate.f_assignment)
    if len(instance.agvs_w) > 1:
        index = rng.randrange(instance.num_operations)
        w_agvs[index] = rng.choice(tuple(value for value in instance.agvs_w if value != w_agvs[index]))
    if len(instance.agvs_f) > 1:
        index = rng.randrange(instance.num_operations)
        f_agvs[index] = rng.choice(tuple(value for value in instance.agvs_f if value != f_agvs[index]))
    return Candidate(
        candidate.operation_order,
        candidate.island_assignment,
        tuple(w_agvs),
        tuple(f_agvs),
    )


def _mutate_candidate(
    instance: Instance,
    candidate: Candidate,
    rng: random.Random,
) -> Candidate:
    """Apply the fixed SOURCE_GAP mutation-event interpretation to all layers."""

    candidate = _topology_mutation(instance, candidate, rng)
    candidate = _island_mutation(instance, candidate, rng)
    candidate = _sequence_position_mutation(candidate, rng)
    return _logistics_mutation(instance, candidate, rng)


def _initialize_population(
    instance: Instance,
    rng: random.Random,
    evaluator: _EvaluationState,
    size: int,
) -> list[DecodedCandidate]:
    return [evaluator.evaluate(random_candidate(instance, rng)) for _ in range(size)]


def _genetic_offspring(
    instance: Instance,
    population: list[DecodedCandidate],
    rng: random.Random,
    evaluator: _EvaluationState,
    config: LGHGAConfig,
) -> list[DecodedCandidate]:
    offspring: list[DecodedCandidate] = []
    while len(offspring) < config.population_size:
        if evaluator.budget_exhausted and offspring:
            break
        left = _tournament(population, rng, config.tournament_size).candidate
        right = _tournament(population, rng, config.tournament_size).candidate
        if rng.random() < config.crossover_probability:
            candidates = _crossover_candidates(instance, left, right, rng)
        else:
            candidates = (left, right)
        for candidate in candidates:
            if rng.random() < config.mutation_probability:
                candidate = _mutate_candidate(instance, candidate, rng)
            offspring.append(evaluator.evaluate(candidate))
            if len(offspring) == config.population_size:
                break
    return offspring


def _local_search(
    instance: Instance,
    population: list[DecodedCandidate],
    neighborhood_id: str,
    rng: random.Random,
    evaluator: _EvaluationState,
    config: LGHGAConfig,
    diagnostics: dict[str, Any],
) -> tuple[list[DecodedCandidate], float | None]:
    """Execute the frozen lsize/MaxIterNum/nsize SOURCE_GAP assumption."""

    current = _retain_best(population, min(config.local_search_population_size, len(population)))
    initial_best = current[0].makespan
    for _ in range(config.local_search_max_iterations):
        neighbors: list[DecodedCandidate] = []
        for proposal_index in range(config.neighborhood_size):
            if evaluator.budget_exhausted:
                break
            source = current[proposal_index % len(current)]
            proposal = propose_neighborhood(instance, source, neighborhood_id, rng)
            neighbors.append(evaluator.evaluate(proposal.candidate))
            diagnostics["neighborhood_proposal_counts"][neighborhood_id] += 1
            diagnostics["local_decoder_evaluations"] += 1
        if not neighbors:
            break
        current = _retain_best([*current, *neighbors], config.local_search_population_size)
    improvement = initial_best - current[0].makespan if current else None
    return current, improvement


def solve_lghga(
    instance: Instance,
    time_limit: float,
    seed: int,
    models: DTRBundle,
    config: LGHGAConfig = LGHGAConfig(),
) -> SearchResult:
    """Run online LG_HGA-RIACRSP without fitting or updating DTR models."""

    if set(models.models) != set(NEIGHBORHOODS):
        raise ValueError("LG_HGA online search requires four frozen DTR models")
    rng = random.Random(seed)
    started = time.perf_counter()
    evaluator = _EvaluationState(instance, started, time_limit)
    population = _initialize_population(instance, rng, evaluator, config.population_size)
    initialization_seconds = evaluator.elapsed
    diagnostics: dict[str, Any] = {
        "fidelity": "PAPER_FAITHFUL_SINGLE_OBJECTIVE_ADAPTATION",
        "formal_objective": "makespan",
        "source_gap_local_search": (
            "best lsize seeds; nsize total round-robin proposals; elitist lsize replacement; "
            "repeat MaxIterNum"
        ),
        "source_gap_mutation_event": "Lp/Lm/Lj analogs plus neutral W/F mutation in one Pm event",
        "dtr_features": ["normalized_generation_index"],
        "dtr_model_hashes": dict(models.model_hashes),
        "knowledge_manifest_hash": models.knowledge_manifest_hash,
        "genetic_offspring_count": 0,
        "neighborhood_proposal_counts": {name: 0 for name in NEIGHBORHOODS},
        "local_search_gate_passes": 0,
        "local_decoder_evaluations": 0,
        "generation_records": [],
    }
    generation = 0
    while not evaluator.budget_exhausted and generation < config.max_generations:
        generation_index = generation + 1
        offspring = _genetic_offspring(instance, population, rng, evaluator, config)
        diagnostics["genetic_offspring_count"] += len(offspring)
        rates = predict_rates(models, generation_index, config.max_generations)
        selected, gate_passed = select_neighborhood(
            rates, config.local_search_threshold_pct
        )
        local: list[DecodedCandidate] = []
        improvement: float | None = None
        before_local_evaluations = diagnostics["local_decoder_evaluations"]
        if gate_passed and not evaluator.budget_exhausted:
            diagnostics["local_search_gate_passes"] += 1
            local, improvement = _local_search(
                instance, population, selected, rng, evaluator, config, diagnostics
            )
        population = _retain_best(
            [*population, *offspring, *local], config.population_size
        )
        diagnostics["generation_records"].append({
            "generation": generation_index,
            "predicted_R_pct": rates,
            "selected_neighborhood": selected,
            "gate_passed": gate_passed,
            "local_decoder_evaluations": (
                diagnostics["local_decoder_evaluations"] - before_local_evaluations
            ),
            "best_improvement": improvement,
        })
        generation += 1
    if evaluator.best is None:
        raise RuntimeError("LG_HGA population initialization produced no candidate")
    diagnostics["initialization_seconds"] = initialization_seconds
    diagnostics["final_population_size"] = len(population)
    return SearchResult(
        "LG_HGA-RIACRSP",
        evaluator.best,
        evaluator.best_time,
        evaluator.elapsed,
        evaluator.evaluations,
        generation,
        generation,
        tuple(evaluator.trace),
        diagnostics,
    )


def generate_knowledge_run(
    instance: Instance,
    seed: int,
    config: LGHGAConfig = LGHGAConfig(),
    *,
    time_limit: float = math.inf,
) -> KnowledgeGenerationResult:
    """Run LG_HGA-KB, evaluating all four neighborhoods at every generation."""

    rng = random.Random(seed)
    started = time.perf_counter()
    evaluator = _EvaluationState(instance, started, time_limit)
    population = _initialize_population(instance, rng, evaluator, config.population_size)
    rows: list[Mapping[str, object]] = []
    generation = 0
    while not evaluator.budget_exhausted and generation < config.max_generations:
        generation_index = generation + 1
        offspring = _genetic_offspring(instance, population, rng, evaluator, config)
        seeds = _retain_best(
            population, min(config.local_search_population_size, len(population))
        )
        all_neighbors: list[DecodedCandidate] = []
        for neighborhood in NEIGHBORHOODS:
            references: list[float] = []
            outcomes: list[float] = []
            neighbors: list[DecodedCandidate] = []
            for proposal_index in range(config.neighborhood_size):
                if evaluator.budget_exhausted:
                    break
                source = seeds[proposal_index % len(seeds)]
                proposal = propose_neighborhood(instance, source, neighborhood, rng)
                decoded = evaluator.evaluate(proposal.candidate)
                references.append(source.makespan)
                outcomes.append(decoded.makespan)
                neighbors.append(decoded)
            better, generated, rate = improvement_rate_pct(references, outcomes)
            deltas = [outcome - reference for reference, outcome in zip(references, outcomes)]
            rows.append({
                "instance_id": instance.instance_id,
                "instance_scale_features": {
                    "num_operations": instance.num_operations,
                    "num_products": len(instance.products),
                    "num_islands": len(instance.islands),
                    "num_w_agvs": len(instance.agvs_w),
                    "num_f_agvs": len(instance.agvs_f),
                },
                "run_seed": seed,
                "generation_index": generation_index,
                "normalized_generation_index": generation_index / config.max_generations,
                "neighborhood_id": neighborhood,
                "num_generated": generated,
                "num_better": better,
                "R_pct": rate,
                "mean_delta_makespan": statistics.fmean(deltas) if deltas else None,
                "median_delta_makespan": statistics.median(deltas) if deltas else None,
            })
            all_neighbors.extend(neighbors)
        population = _retain_best(
            [*population, *offspring, *all_neighbors], config.population_size
        )
        generation += 1
    if evaluator.best is None:
        raise RuntimeError("LG_HGA-KB initialization produced no candidate")
    return KnowledgeGenerationResult(
        tuple(rows), evaluator.best, evaluator.elapsed, evaluator.evaluations, generation
    )
