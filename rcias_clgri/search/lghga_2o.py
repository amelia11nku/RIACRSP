"""Budget-only adaptation of the frozen LG_HGA v2 online search."""

from __future__ import annotations

import random
import time
from typing import Any

from rcias_clgri.data.instance import Instance

from .common import SearchResult
from .lghga import LGHGAConfig, _EvaluationState, _genetic_offspring, _initialize_population, _retain_best
from .lghga_learning import DTRBundle, predict_rates, select_neighborhood
from .lghga_neighborhoods import NEIGHBORHOODS
from .lghga_v2 import _local_search_v2


METHOD = "LG_HGA-2O"
SOURCE_GENERATION_DENOMINATOR = 100


def operation_budget(instance: Instance) -> float:
    return 2.0 * instance.num_operations


def solve_lghga_2o(
    instance: Instance,
    time_limit: float,
    seed: int,
    models: DTRBundle,
    config: LGHGAConfig = LGHGAConfig(),
) -> SearchResult:
    """Retain canonical mechanics and DTR inputs, removing only MAXGEN stop.

    Explicit time_limit allows deterministic budget-accounting tests. Formal
    callers must pass operation_budget(instance), shared with every comparator.
    Canonical inner-loop deadline checks and generation finalization remain.
    """
    if config.max_generations != SOURCE_GENERATION_DENOMINATOR:
        raise ValueError("LG_HGA-2O requires the frozen source normalization of 100")
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
    while not evaluator.budget_exhausted:
        generation_index = generation + 1
        offspring = _genetic_offspring(instance, population, rng, evaluator, config)
        diagnostics["genetic_offspring_count"] += len(offspring)
        rates = predict_rates(models, generation_index, SOURCE_GENERATION_DENOMINATOR)
        selected, gate_passed = select_neighborhood(rates, config.local_search_threshold_pct)
        local = []
        improvement = None
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
            "local_decoder_evaluations": diagnostics["local_decoder_evaluations"] - before_local_evaluations,
            "best_improvement": improvement,
        })
        generation += 1
    if evaluator.best is None:
        raise RuntimeError("LG_HGA v2 population initialization produced no candidate")
    diagnostics["initialization_seconds"] = initialization_seconds
    diagnostics["final_population_size"] = len(population)
    return SearchResult(
        METHOD, evaluator.best, evaluator.best_time, evaluator.elapsed,
        evaluator.evaluations, generation, generation, tuple(evaluator.trace), diagnostics,
    )
