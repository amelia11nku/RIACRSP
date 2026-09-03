"""Versioned LG_HGA variant with source-closer N4 and no-op diagnostics."""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any, Mapping

from rcias_clgri.data.instance import Instance

from .common import DecodedCandidate, SearchResult, TracePoint
from .lghga import (
    LGHGAConfig,
    KnowledgeGenerationResult,
    _EvaluationState,
    _genetic_offspring,
    _initialize_population,
    _retain_best,
)
from .lghga_learning import DTRBundle, improvement_rate_pct, predict_rates, select_neighborhood
from .lghga_neighborhoods import NEIGHBORHOODS
from .lghga_neighborhoods_v2 import propose_neighborhood


METHOD = "LG_HGA-RIACRSP-v2-N4M"


def _record_proposal(diagnostics: dict[str, Any], neighborhood: str, changed: bool) -> None:
    diagnostics["neighborhood_proposal_counts"][neighborhood] += 1
    key = "changed_proposal_counts" if changed else "noop_proposal_counts"
    diagnostics[key][neighborhood] += 1


def _local_search_v2(
    instance: Instance,
    population: list[DecodedCandidate],
    neighborhood_id: str,
    rng: random.Random,
    evaluator: _EvaluationState,
    config: LGHGAConfig,
    diagnostics: dict[str, Any],
) -> tuple[list[DecodedCandidate], float | None]:
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
            _record_proposal(
                diagnostics, neighborhood_id, bool(proposal.detail.get("changed"))
            )
            diagnostics["local_decoder_evaluations"] += 1
        if not neighbors:
            break
        current = _retain_best([*current, *neighbors], config.local_search_population_size)
    improvement = initial_best - current[0].makespan if current else None
    return current, improvement


def solve_lghga_v2(
    instance: Instance,
    time_limit: float,
    seed: int,
    models: DTRBundle,
    config: LGHGAConfig = LGHGAConfig(),
) -> SearchResult:
    """Run v2 without fitting, threshold changes, or synthetic objectives."""

    if set(models.models) != set(NEIGHBORHOODS):
        raise ValueError("LG_HGA v2 online search requires four frozen DTR models")
    rng = random.Random(seed)
    started = time.perf_counter()
    evaluator = _EvaluationState(instance, started, time_limit)
    population = _initialize_population(instance, rng, evaluator, config.population_size)
    initialization_seconds = evaluator.elapsed
    diagnostics: dict[str, Any] = {
        "fidelity": "VERSIONED_SINGLE_OBJECTIVE_N4_MINIMAL_PLURAL",
        "formal_objective": "makespan",
        "n4_selection_rule": "MINIMAL_PLURAL_TWO_EFFECTIVE_MOVES",
        "dtr_features": ["normalized_generation_index"],
        "dtr_model_hashes": dict(models.model_hashes),
        "knowledge_manifest_hash": models.knowledge_manifest_hash,
        "genetic_offspring_count": 0,
        "neighborhood_proposal_counts": {name: 0 for name in NEIGHBORHOODS},
        "changed_proposal_counts": {name: 0 for name in NEIGHBORHOODS},
        "noop_proposal_counts": {name: 0 for name in NEIGHBORHOODS},
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
            local, improvement = _local_search_v2(
                instance, population, selected, rng, evaluator, config, diagnostics
            )
        population = _retain_best([*population, *offspring, *local], config.population_size)
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
        raise RuntimeError("LG_HGA v2 population initialization produced no candidate")
    diagnostics["initialization_seconds"] = initialization_seconds
    diagnostics["final_population_size"] = len(population)
    return SearchResult(
        METHOD,
        evaluator.best,
        evaluator.best_time,
        evaluator.elapsed,
        evaluator.evaluations,
        generation,
        generation,
        tuple(evaluator.trace),
        diagnostics,
    )


def generate_knowledge_run_v2(
    instance: Instance,
    seed: int,
    config: LGHGAConfig = LGHGAConfig(),
    *,
    time_limit: float = math.inf,
) -> KnowledgeGenerationResult:
    """Generate Eq. (11)-style one-step R values with effective-move diagnostics."""

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
            changed = 0
            for proposal_index in range(config.neighborhood_size):
                if evaluator.budget_exhausted:
                    break
                source = seeds[proposal_index % len(seeds)]
                proposal = propose_neighborhood(instance, source, neighborhood, rng)
                decoded = evaluator.evaluate(proposal.candidate)
                references.append(source.makespan)
                outcomes.append(decoded.makespan)
                neighbors.append(decoded)
                changed += int(bool(proposal.detail.get("changed")))
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
                "num_changed": changed,
                "num_unchanged": generated - changed,
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
        raise RuntimeError("LG_HGA v2 knowledge initialization produced no candidate")
    return KnowledgeGenerationResult(
        tuple(rows), evaluator.best, evaluator.elapsed, evaluator.evaluations, generation
    )
