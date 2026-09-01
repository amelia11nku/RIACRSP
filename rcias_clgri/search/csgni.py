"""Live CSG-NI integration around the frozen H1-seeded ALNS baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.ni.live_policy import LiveInterventionPolicy

from .alns import (
    ALNSConfig, DESTROY, REPAIR, _destroy, _neighbor, _roulette, solve_alns,
)
from .common import SearchResult, TracePoint, candidate_from_actions, decode_candidate
from .counterfactual import stable_seed


@dataclass(frozen=True)
class CSGNIConfig:
    intervention_rate: int = 20
    proposal_seed_namespace: int = 670102
    ni_repair_seed_namespace: int = 670103
    acceptance_seed_namespace: int = 670104
    diagnostics_seed_namespace: int = 670105
    eligibility_offset: int = 0

    def __post_init__(self) -> None:
        if self.intervention_rate not in {0, 20, 50, 100}:
            raise ValueError("intervention_rate must be one of 0, 20, 50, 100")
        if self.eligibility_offset < 0:
            raise ValueError("eligibility_offset must be non-negative")


CSGNIObserver = Callable[[Mapping[str, object]], None]


def intervention_eligible(iteration: int, rate: int, offset: int = 0) -> bool:
    """Deterministic schedules with exact long-run rates and no RNG consumption."""
    if rate == 0:
        return False
    period, active = {20: (5, 1), 50: (2, 1), 100: (1, 1)}[rate]
    return (iteration + offset) % period < active


def _stage(progress: float) -> str:
    bounds = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
    return bounds[min(4, int(max(0.0, min(progress, .999999)) * 5))]


def solve_csgni(
    instance: Instance,
    time_limit: float,
    seed: int,
    policy: LiveInterventionPolicy | None,
    *,
    alns_config: ALNSConfig = ALNSConfig(),
    csgni_config: CSGNIConfig = CSGNIConfig(),
    observer: CSGNIObserver | None = None,
) -> SearchResult:
    """Run live CSG-NI; rate zero delegates byte-for-byte search semantics to ALNS."""
    if csgni_config.intervention_rate == 0:
        result = solve_alns(instance, time_limit, seed, alns_config, observer)
        return SearchResult(
            "CSG-NI-R0", result.best, result.best_found_time, result.runtime,
            result.decoder_evaluations, result.iterations,
            result.generations_if_applicable, result.convergence_trace,
            {**result.diagnostics, "zero_intervention_delegated_to_frozen_alns": True},
        )
    if policy is None:
        raise ValueError("a live intervention policy is required when intervention_rate > 0")

    baseline_rng = random.Random(stable_seed(seed, "baseline", namespace=0))
    acceptance_seed = stable_seed(
        seed, "acceptance", namespace=csgni_config.acceptance_seed_namespace
    )
    acceptance_rng = random.Random(acceptance_seed)
    started = time.perf_counter()
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    best = current
    evaluations = 1
    best_time = time.perf_counter() - started
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    weights = {name: 1.0 for name in (*DESTROY, *REPAIR)}
    selections, successes, improvements = Counter(), Counter(), Counter()
    temperature = alns_config.initial_temperature * max(1.0, current.makespan)
    iterations = 0
    eligible_count = intervention_count = fallback_count = 0

    while (
        time.perf_counter() - started < time_limit
        and (alns_config.iteration_limit is None or iterations < alns_config.iteration_limit)
    ):
        iteration_started = time.perf_counter()
        current_before, best_before = current, best
        evaluations_before = evaluations
        weights_before = dict(weights)
        count = min(
            max(2, round(instance.num_operations * alns_config.destroy_fraction)),
            instance.num_operations,
        )
        progress = (
            iterations / alns_config.iteration_limit
            if alns_config.iteration_limit else (time.perf_counter() - started) / time_limit
        )
        eligible = intervention_eligible(
            iterations, csgni_config.intervention_rate, csgni_config.eligibility_offset
        )
        decision = None
        if eligible:
            eligible_count += 1
            state_id = f"{instance.instance_id}__seed{seed}__it{iterations:07d}"
            decision = policy.decide(
                instance,
                current,
                state_id=state_id,
                destroy_count=count,
                search_progress=min(progress, .999999),
                search_stage=_stage(progress),
            )
        use_ni = bool(decision and decision.intervene)
        if use_ni:
            intervention_count += 1
            destroy, repair = "CSG_NI", "transport_aware"
            removed = set(decision.destroyed_operations)
            repair_seed = stable_seed(
                seed, iterations, "ni_repair",
                namespace=csgni_config.ni_repair_seed_namespace,
            )
            repair_rng = random.Random(repair_seed)
        else:
            fallback_count += int(eligible)
            destroy = _roulette(DESTROY, weights, baseline_rng)
            repair = _roulette(REPAIR, weights, baseline_rng)
            selections.update((destroy, repair))
            removed = _destroy(instance, current, destroy, count, baseline_rng)
            repair_seed = None
            repair_rng = baseline_rng

        candidates = []
        repair_started = time.perf_counter()
        for _ in range(alns_config.candidate_trials):
            if time.perf_counter() - started >= time_limit and candidates:
                break
            candidates.append(decode_candidate(
                instance,
                _neighbor(instance, current.candidate, removed, repair, repair_rng),
            ))
            evaluations += 1
        candidate = min(candidates, key=lambda item: item.makespan)
        repair_runtime = time.perf_counter() - repair_started
        delta = candidate.makespan - current.makespan
        accepted = delta <= 0 or acceptance_rng.random() < math.exp(
            -delta / max(temperature, 1e-12)
        )
        score = 0.0
        if accepted:
            current = candidate
            score = 1.0
            if not use_ni:
                successes.update((destroy, repair))
        if candidate.makespan < best.makespan:
            best = candidate
            best_time = time.perf_counter() - started
            trace.append(TracePoint(best_time, evaluations, best.makespan))
            score = 5.0
            if not use_ni:
                improvements.update((destroy, repair))
        if not use_ni:
            for operator in (destroy, repair):
                weights[operator] = (
                    (1 - alns_config.reaction_factor) * weights[operator]
                    + alns_config.reaction_factor * max(score, 0.1)
                )
        temperature_before = temperature
        temperature *= alns_config.cooling_rate
        if observer is not None:
            timing = dict(decision.timings_ms or {}) if decision else {}
            observer({
                "iteration": iterations,
                "elapsed_time": time.perf_counter() - started,
                "iteration_runtime": time.perf_counter() - iteration_started,
                "decoder_evaluations": evaluations,
                "current_before": current_before,
                "best_before": best_before,
                "candidate": candidate,
                "current_after": current,
                "best_after": best,
                "destroy_operator": destroy,
                "repair_operator": repair,
                "destroy_fraction": alns_config.destroy_fraction,
                "destroyed_operation_ids": tuple(sorted(removed)),
                "accepted": accepted,
                "new_global_best": candidate.makespan < best_before.makespan,
                "operator_weights_before": weights_before,
                "operator_weights_after": dict(weights),
                "repair_decoder_evaluations": evaluations - evaluations_before,
                "repair_runtime": repair_runtime,
                "candidate_trials_completed": len(candidates),
                "temperature_before": temperature_before,
                "ni_eligible": eligible,
                "ni_intervened": use_ni,
                "ni_fallback_reason": decision.fallback_reason if decision else "NOT_ELIGIBLE",
                "ni_state_id": decision.state_id if decision else None,
                "ni_target_set_id": decision.selected_target_set_id if decision else None,
                "ni_proposal_count": decision.proposal_count if decision else 0,
                "ni_requested_proposal_count": decision.requested_proposal_count if decision else 0,
                "ni_duplicate_proposal_count": decision.duplicate_proposal_count if decision else 0,
                "ni_selected_origin_family": decision.selected_origin_family if decision else None,
                "ni_selected_origin_operator": decision.selected_origin_operator if decision else None,
                "ni_selected_origin_rules": decision.selected_origin_rules if decision else (),
                "ni_calibrated_probability": decision.calibrated_probability if decision else None,
                "ni_calibrated_utility": decision.calibrated_utility if decision else None,
                "ni_decision_margin": decision.decision_margin if decision else None,
                "ni_graph_hash": decision.graph_hash if decision else None,
                "ni_state_feature_summary": (
                    decision.state_feature_summary if decision else None
                ),
                "ni_timing_ms": timing,
                "rng_baseline_namespace": 0,
                "rng_proposal_namespace": csgni_config.proposal_seed_namespace,
                "rng_ni_repair_namespace": csgni_config.ni_repair_seed_namespace,
                "rng_ni_repair_seed": repair_seed,
                "rng_acceptance_namespace": csgni_config.acceptance_seed_namespace,
                "rng_acceptance_seed": acceptance_seed,
                "rng_diagnostics_namespace": csgni_config.diagnostics_seed_namespace,
                "alns_weight_credit": not use_ni,
            })
        iterations += 1

    return SearchResult(
        f"CSG-NI-R{csgni_config.intervention_rate}",
        best, best_time, time.perf_counter() - started, evaluations, iterations,
        None, tuple(trace),
        {
            "operator_selections": dict(selections),
            "operator_successes": dict(successes),
            "operator_improvements": dict(improvements),
            "final_adaptive_weights": weights,
            "ni_eligible_iterations": eligible_count,
            "ni_interventions": intervention_count,
            "ni_fallbacks": fallback_count,
            "rng_namespaces": {
                "baseline": 0,
                "proposal": csgni_config.proposal_seed_namespace,
                "ni_repair": csgni_config.ni_repair_seed_namespace,
                "acceptance": csgni_config.acceptance_seed_namespace,
                "diagnostics": csgni_config.diagnostics_seed_namespace,
            },
        },
    )
