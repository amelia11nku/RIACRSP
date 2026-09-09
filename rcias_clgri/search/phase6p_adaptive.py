"""Phase 6P adaptive neural-shortlist portfolio on the frozen ALNS search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Mapping, Protocol

from rcias_clgri.analysis.phase6l_legacy_score import select_score_free_fallback
from rcias_clgri.data.instance import Instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.ni.phase6p_live_inference import (
    Phase6PBankScores,
    origin_destroy_operators,
)
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank

from .alns import ALNSConfig, DESTROY, REPAIR, _destroy, _neighbor, _roulette
from .common import SearchResult, TracePoint, candidate_from_actions, decode_candidate
from .counterfactual import stable_seed
from .csgni import intervention_eligible


PRIMARY_METHOD = "P1_CSG_ADAPTIVE_PORTFOLIO"
TOP1_METHOD = "PHASE6N_DETERMINISTIC_TOP1"


class Phase6PCritic(Protocol):
    def prepare_instance(self, instance: Instance, schedule) -> None: ...

    def score_bank(self, instance: Instance, current, **kwargs) -> Phase6PBankScores: ...


@dataclass(frozen=True)
class Phase6PConfig:
    intervention_rate: int = 20
    eligibility_offset: int = 0
    proposal_seed_namespace: int = 692000000
    acceptance_seed_namespace: int = 670104
    diagnostics_seed_namespace: int = 670105
    shortlist_k: int = 6

    def __post_init__(self) -> None:
        if self.intervention_rate != 20 or self.eligibility_offset != 0:
            raise ValueError("Phase 6P freezes the inherited 20% eligibility schedule")
        if self.proposal_seed_namespace != 692000000:
            raise ValueError("Phase 6P proposal namespace drifted")
        if self.acceptance_seed_namespace != 670104:
            raise ValueError("Phase 6P acceptance namespace drifted")
        if self.shortlist_k != 6:
            raise ValueError("Phase 6P freezes a top-6 shortlist")


Phase6PObserver = Callable[[Mapping[str, object]], None]


def _stage(progress: float) -> str:
    bounds = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
    return bounds[min(4, int(max(0.0, min(progress, .999999)) * 5))]


def mapped_destroy_weight(
    origins: tuple[str, ...], operator_weights: Mapping[str, float]
) -> float:
    unique = tuple(dict.fromkeys(origins))
    if not unique or not set(unique) <= set(DESTROY):
        raise ValueError("target has invalid Phase 6P destroy origins")
    values = [float(operator_weights[name]) for name in unique]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("ALNS destroy weights must be finite and positive")
    return math.fsum(values) / len(values)


def portfolio_distribution(
    ranked_target_ids: tuple[str, ...],
    origins_by_target: Mapping[str, tuple[str, ...]],
    operator_weights: Mapping[str, float],
    *,
    shortlist_k: int = 6,
) -> tuple[tuple[str, float, float], ...]:
    shortlist = ranked_target_ids[:shortlist_k]
    if len(shortlist) != shortlist_k or len(set(shortlist)) != shortlist_k:
        raise ValueError("Phase 6P requires six distinct shortlisted targets")
    raw = []
    for rank, target_id in enumerate(shortlist, 1):
        mapped = mapped_destroy_weight(origins_by_target[target_id], operator_weights)
        raw.append((target_id, mapped / rank, mapped))
    denominator = math.fsum(item[1] for item in raw)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid Phase 6P portfolio mass")
    return tuple((target_id, weight / denominator, mapped) for target_id, weight, mapped in raw)


def sample_portfolio_target(
    distribution: tuple[tuple[str, float, float], ...], rng: random.Random
) -> str:
    return rng.choices(
        [item[0] for item in distribution],
        weights=[item[1] for item in distribution],
        k=1,
    )[0]


def updated_operator_weight(current: float, score: float, reaction: float) -> float:
    return (1.0 - reaction) * current + reaction * max(score, 0.1)


def simulated_annealing_accept(
    delta: float, temperature: float, rng: random.Random
) -> bool:
    return delta <= 0 or rng.random() < math.exp(
        -delta / max(temperature, 1e-12)
    )


def _safe_fallback_bank(
    instance: Instance,
    current,
    *,
    state_id: str,
    destroy_count: int,
    proposal_seed_namespace: int,
):
    generated, _ = build_live_proposal_bank(
        instance,
        current,
        state_id=state_id,
        destroy_count=destroy_count,
        seed_namespace=proposal_seed_namespace,
    )
    fallback = select_score_free_fallback(generated)
    return generated, fallback, origin_destroy_operators(generated)


def solve_phase6p(
    instance: Instance,
    time_limit: float,
    seed: int,
    critic: Phase6PCritic,
    *,
    target_mode: str = "portfolio",
    alns_config: ALNSConfig = ALNSConfig(),
    phase6p_config: Phase6PConfig = Phase6PConfig(),
    observer: Phase6PObserver | None = None,
) -> SearchResult:
    if target_mode not in {"portfolio", "top1"}:
        raise ValueError("target_mode must be portfolio or top1")
    if alns_config.candidate_trials != 8:
        raise ValueError("Phase 6P freezes eight repair trials")
    baseline_rng = random.Random(stable_seed(seed, "baseline", namespace=0))
    acceptance_seed = stable_seed(
        seed, "acceptance", namespace=phase6p_config.acceptance_seed_namespace
    )
    acceptance_rng = random.Random(acceptance_seed)
    started = time.perf_counter()
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    preparation_started = time.perf_counter()
    critic.prepare_instance(instance, h1.schedule)
    critic_preparation_seconds = time.perf_counter() - preparation_started
    best = current
    evaluations = 1
    best_time = time.perf_counter() - started
    initialization_seconds = best_time
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    weights = {name: 1.0 for name in (*DESTROY, *REPAIR)}
    selections, successes, improvements = Counter(), Counter(), Counter()
    temperature = alns_config.initial_temperature * max(1.0, current.makespan)
    iterations = 0
    eligible_count = portfolio_count = safe_fallback_count = 0
    critic_seconds = 0.0

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
            if alns_config.iteration_limit
            else (time.perf_counter() - started) / time_limit
        )
        eligible = intervention_eligible(
            iterations,
            phase6p_config.intervention_rate,
            phase6p_config.eligibility_offset,
        )
        state_id = None
        decision = None
        selection_distribution = ()
        scoring_error = None
        safe_fallback = False
        if eligible:
            eligible_count += 1
            state_id = f"{instance.instance_id}__seed{seed}__it{iterations:07d}"
            critic_started = time.perf_counter()
            try:
                decision = critic.score_bank(
                    instance,
                    current,
                    state_id=state_id,
                    destroy_count=count,
                    search_progress=min(progress, .999999),
                    search_stage=_stage(progress),
                    proposal_seed_namespace=phase6p_config.proposal_seed_namespace,
                )
                if (
                    decision.generated.requested_arm_count != 24
                    or len(decision.target_scores) != decision.generated.unique_arm_count
                    or len(decision.top_target_ids) != phase6p_config.shortlist_k
                    or set(decision.target_scores)
                    != {arm.target_set_id for arm in decision.generated.arms}
                    or not all(math.isfinite(value) for value in decision.target_scores.values())
                ):
                    raise RuntimeError("incomplete or non-finite Phase 6P critic result")
                if target_mode == "portfolio":
                    selection_distribution = portfolio_distribution(
                        decision.ranked_target_ids,
                        decision.origin_destroy_operators,
                        weights,
                        shortlist_k=phase6p_config.shortlist_k,
                    )
                    target_id = sample_portfolio_target(
                        selection_distribution, baseline_rng
                    )
                else:
                    target_id = decision.ranked_target_ids[0]
                arms = {arm.target_set_id: arm for arm in decision.generated.arms}
                selected_arm = arms[target_id]
                selected_origins = decision.origin_destroy_operators[target_id]
                generated = decision.generated
                portfolio_count += 1
            except Exception as error:  # protocol-defined fail-closed path
                scoring_error = f"{type(error).__name__}: {error}"
                generated, selected_arm, origins = _safe_fallback_bank(
                    instance,
                    current,
                    state_id=state_id,
                    destroy_count=count,
                    proposal_seed_namespace=phase6p_config.proposal_seed_namespace,
                )
                target_id = selected_arm.target_set_id
                selected_origins = origins[target_id]
                safe_fallback = True
                safe_fallback_count += 1
            finally:
                critic_seconds += time.perf_counter() - critic_started
            repair = _roulette(REPAIR, weights, baseline_rng)
            removed = set(selected_arm.destroyed_operations)
            credited_destroy = tuple(dict.fromkeys(selected_origins))
            selections.update((*credited_destroy, repair))
        else:
            destroy = _roulette(DESTROY, weights, baseline_rng)
            repair = _roulette(REPAIR, weights, baseline_rng)
            removed = _destroy(instance, current, destroy, count, baseline_rng)
            target_id = None
            generated = None
            credited_destroy = (destroy,)
            selections.update((destroy, repair))

        candidates = []
        neighbor_runtime = 0.0
        decoder_runtime = 0.0
        repair_started = time.perf_counter()
        for _ in range(alns_config.candidate_trials):
            if time.perf_counter() - started >= time_limit and candidates:
                break
            neighbor_started = time.perf_counter()
            neighbor = _neighbor(
                instance, current.candidate, removed, repair, baseline_rng
            )
            neighbor_runtime += time.perf_counter() - neighbor_started
            decoder_started = time.perf_counter()
            candidates.append(decode_candidate(instance, neighbor))
            decoder_runtime += time.perf_counter() - decoder_started
            evaluations += 1
        candidate = min(candidates, key=lambda item: item.makespan)
        repair_runtime = time.perf_counter() - repair_started
        delta = candidate.makespan - current.makespan
        temperature_before = temperature
        accepted = simulated_annealing_accept(delta, temperature, acceptance_rng)
        score = 0.0
        if accepted:
            current = candidate
            successes.update((*credited_destroy, repair))
            score = 1.0
        new_global_best = candidate.makespan < best.makespan
        if new_global_best:
            best = candidate
            best_time = time.perf_counter() - started
            trace.append(TracePoint(best_time, evaluations, best.makespan))
            improvements.update((*credited_destroy, repair))
            score = 5.0
        credited_operators = (*credited_destroy, repair)
        for operator in credited_operators:
            weights[operator] = updated_operator_weight(
                weights[operator], score, alns_config.reaction_factor
            )
        temperature *= alns_config.cooling_rate
        if observer is not None:
            timing = dict(decision.timings_ms) if decision is not None else {}
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
                "repair_operator": repair,
                "destroyed_operation_ids": tuple(sorted(removed)),
                "credited_destroy_operators": credited_destroy,
                "accepted": accepted,
                "new_global_best": new_global_best,
                "reward_score": score,
                "operator_weights_before": weights_before,
                "operator_weights_after": dict(weights),
                "repair_decoder_evaluations": evaluations - evaluations_before,
                "repair_runtime": repair_runtime,
                "repair_excluding_decoder_runtime": neighbor_runtime,
                "decoder_runtime": decoder_runtime,
                "candidate_trials_completed": len(candidates),
                "temperature_before": temperature_before,
                "neural_eligible": eligible,
                "target_mode": target_mode,
                "state_id": state_id,
                "selected_target_set_id": target_id,
                "safe_fallback": safe_fallback,
                "scoring_error": scoring_error,
                "requested_proposal_count": (
                    generated.requested_arm_count if generated is not None else 0
                ),
                "unique_proposal_count": (
                    generated.unique_arm_count if generated is not None else 0
                ),
                "duplicate_proposal_count": (
                    generated.duplicate_arm_count if generated is not None else 0
                ),
                "ranked_target_ids": (
                    decision.ranked_target_ids if decision is not None else ()
                ),
                "top_target_ids": (
                    decision.top_target_ids if decision is not None else ()
                ),
                "portfolio_distribution": selection_distribution,
                "target_scores": dict(decision.target_scores) if decision else {},
                "held_fold": decision.held_fold if decision else None,
                "graph_hash": decision.graph_hash if decision else None,
                "critic_timing_ms": timing,
                "rng_baseline_namespace": 0,
                "rng_proposal_namespace": phase6p_config.proposal_seed_namespace,
                "rng_acceptance_namespace": phase6p_config.acceptance_seed_namespace,
                "rng_acceptance_seed": acceptance_seed,
                "rng_diagnostics_namespace": phase6p_config.diagnostics_seed_namespace,
            })
        iterations += 1

    method = PRIMARY_METHOD if target_mode == "portfolio" else TOP1_METHOD
    return SearchResult(
        method,
        best,
        best_time,
        time.perf_counter() - started,
        evaluations,
        iterations,
        None,
        tuple(trace),
        {
            "operator_selections": dict(selections),
            "operator_successes": dict(successes),
            "operator_improvements": dict(improvements),
            "final_adaptive_weights": weights,
            "neural_eligible_iterations": eligible_count,
            "neural_portfolio_decisions": portfolio_count,
            "safe_fallbacks": safe_fallback_count,
            "initialization_seconds": initialization_seconds,
            "critic_preparation_seconds": critic_preparation_seconds,
            "critic_seconds": critic_seconds,
            "target_mode": target_mode,
            "rng_namespaces": {
                "baseline": 0,
                "proposal": phase6p_config.proposal_seed_namespace,
                "acceptance": phase6p_config.acceptance_seed_namespace,
                "diagnostics": phase6p_config.diagnostics_seed_namespace,
            },
        },
    )
