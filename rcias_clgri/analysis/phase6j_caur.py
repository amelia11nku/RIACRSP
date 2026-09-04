"""Continuation-aware labels and outcome-blind candidate context for Phase 6J."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import statistics
import time
from typing import Any, Mapping, Sequence

from rcias_clgri.analysis.phase6i_mr import (
    ContinuationResult,
    continue_frozen_alns_from_candidate,
)
from rcias_clgri.analysis.phase6a import bottleneck_proxy, schedule_features
from rcias_clgri.data.instance import Instance
from rcias_clgri.search.alns import ALNSConfig
from rcias_clgri.search.alns import DESTROY, REPAIR, _destroy, _neighbor, _roulette
from rcias_clgri.search.common import DecodedCandidate, decode_candidate
from rcias_clgri.search.phase6c import ArmGenerationResult, Phase6CTargetArm


FULL_BANK_SCOPE = "TRUE_FULL_DEDUPLICATED_24_RULE_BANK"
REDUCED_AUDIT_SCOPE = "REDUCED_TOP8_AUDIT_ONLY"

CANDIDATE_INPUT_COLUMNS = (
    "primary_origin_rule",
    "origin_destroy_operator",
    "origin_family",
    "origin_rule_count",
    "origin_family_count",
    "destroy_target_cardinality",
    "destroy_target_fraction",
    "fallback_overlap_fraction",
    "fallback_jaccard",
    "critical_overlap_fraction",
    "bottleneck_overlap_fraction",
    "best_frozen_score_jaccard",
    "normalized_frozen_score_rank",
    "normalized_diversity_rank",
    "is_fallback",
)
LABEL_COLUMNS = (
    "continuation_advantage_mean",
    "beats_fallback",
    "immediate_utility",
    "continuation_best_makespan",
    "fallback_continuation_best_makespan",
)


@dataclass(frozen=True)
class PairedContinuationAdvantage:
    state_id: str
    target_set_id: str
    fallback_target_set_id: str
    continuation_seed: int
    horizon: int
    incumbent_makespan: float
    candidate_best_makespan: float
    fallback_best_makespan: float
    advantage: float
    candidate: ContinuationResult
    fallback: ContinuationResult


@dataclass(frozen=True)
class GateDecision:
    selected_target_set_id: str
    neural_target_set_id: str
    fallback_target_set_id: str
    intervened: bool
    reason: str
    lcb: float


@dataclass(frozen=True)
class ContinuationPrefix:
    result: ContinuationResult
    decoder_seconds: float
    neighbor_seconds: float


def continue_frozen_alns_at_horizons(
    instance: Instance,
    start: DecodedCandidate,
    *,
    state_id: str,
    continuation_seed: int,
    seed_namespace: int,
    horizons: Sequence[int],
    config: ALNSConfig,
) -> dict[int, ContinuationPrefix]:
    """Run one frozen continuation stream and capture deterministic prefixes.

    The transition loop is intentionally identical to the frozen Phase 6I-MR
    continuation implementation. A regression test compares its maximum-horizon
    scientific outputs with that implementation.
    """
    requested = tuple(sorted(set(int(value) for value in horizons)))
    if not requested or requested[0] <= 0:
        raise ValueError("positive continuation horizons are required")
    if config.candidate_trials <= 0:
        raise ValueError("candidate_trials must be positive")
    from rcias_clgri.search.counterfactual import stable_seed

    derived_seed = stable_seed(state_id, str(continuation_seed), namespace=seed_namespace)
    rng = random.Random(derived_seed)
    started = time.perf_counter()
    current = start
    best = start
    weights = {name: 1.0 for name in (*DESTROY, *REPAIR)}
    selections: Counter[str] = Counter()
    temperature = config.initial_temperature * max(1.0, current.makespan)
    evaluations = accepted_moves = improving_moves = 0
    results: dict[int, ContinuationPrefix] = {}
    decoder_seconds = neighbor_seconds = 0.0

    for iteration in range(1, requested[-1] + 1):
        destroy = _roulette(DESTROY, weights, rng)
        repair = _roulette(REPAIR, weights, rng)
        selections.update((destroy, repair))
        count = min(
            max(2, round(instance.num_operations * config.destroy_fraction)),
            instance.num_operations,
        )
        removed = _destroy(instance, current, destroy, count, rng)
        candidates = []
        for _ in range(config.candidate_trials):
            neighbor_started = time.perf_counter()
            neighbor = _neighbor(instance, current.candidate, removed, repair, rng)
            neighbor_seconds += time.perf_counter() - neighbor_started
            decoder_started = time.perf_counter()
            candidates.append(decode_candidate(instance, neighbor))
            decoder_seconds += time.perf_counter() - decoder_started
        candidates = tuple(candidates)
        evaluations += len(candidates)
        candidate = min(candidates, key=lambda item: item.makespan)
        delta = candidate.makespan - current.makespan
        accepted = delta <= 0 or rng.random() < math.exp(
            -delta / max(temperature, 1e-12)
        )
        score = 0.0
        if accepted:
            current = candidate
            accepted_moves += 1
            score = 1.0
        if candidate.makespan < best.makespan:
            best = candidate
            improving_moves += 1
            score = 5.0
        for operator in (destroy, repair):
            weights[operator] = (
                (1 - config.reaction_factor) * weights[operator]
                + config.reaction_factor * max(score, 0.1)
            )
        temperature *= config.cooling_rate
        if iteration in requested:
            results[iteration] = ContinuationPrefix(
                result=ContinuationResult(
                    candidate=best,
                    start_makespan=float(start.makespan),
                    best_makespan=float(best.makespan),
                    continuation_value=float((start.makespan - best.makespan) / start.makespan),
                    continuation_seed=int(continuation_seed),
                    derived_seed=int(derived_seed),
                    iterations=iteration,
                    decoder_evaluations=evaluations,
                    accepted_moves=accepted_moves,
                    improving_moves=improving_moves,
                    operator_selections=dict(selections),
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                ),
                decoder_seconds=decoder_seconds,
                neighbor_seconds=neighbor_seconds,
            )
    return results


def fallback_relative_advantage(
    fallback_best_makespan: float,
    candidate_best_makespan: float,
    incumbent_makespan: float,
) -> float:
    """Return positive-is-better continuation advantage over the fallback."""
    if not all(math.isfinite(value) for value in (
        fallback_best_makespan, candidate_best_makespan, incumbent_makespan
    )):
        raise ValueError("continuation makespans must be finite")
    if incumbent_makespan <= 0:
        raise ValueError("incumbent_makespan must be positive")
    return (fallback_best_makespan - candidate_best_makespan) / incumbent_makespan


def evaluate_paired_continuation_advantage(
    instance: Instance,
    candidate_start: DecodedCandidate,
    fallback_start: DecodedCandidate,
    *,
    state_id: str,
    target_set_id: str,
    fallback_target_set_id: str,
    incumbent_makespan: float,
    continuation_seed: int,
    seed_namespace: int,
    horizon: int,
    config: ALNSConfig,
) -> PairedContinuationAdvantage:
    """Evaluate candidate and fallback with the same continuation RNG stream."""
    common = {
        "state_id": state_id,
        "continuation_seed": continuation_seed,
        "seed_namespace": seed_namespace,
        "iterations": horizon,
        "config": config,
    }
    candidate = continue_frozen_alns_from_candidate(instance, candidate_start, **common)
    fallback = continue_frozen_alns_from_candidate(instance, fallback_start, **common)
    if candidate.derived_seed != fallback.derived_seed:
        raise RuntimeError("candidate/fallback continuation streams are not paired")
    return PairedContinuationAdvantage(
        state_id=state_id,
        target_set_id=target_set_id,
        fallback_target_set_id=fallback_target_set_id,
        continuation_seed=int(continuation_seed),
        horizon=int(horizon),
        incumbent_makespan=float(incumbent_makespan),
        candidate_best_makespan=float(candidate.best_makespan),
        fallback_best_makespan=float(fallback.best_makespan),
        advantage=fallback_relative_advantage(
            fallback.best_makespan, candidate.best_makespan, incumbent_makespan
        ),
        candidate=candidate,
        fallback=fallback,
    )


def aggregate_paired_advantages(
    rows: Sequence[PairedContinuationAdvantage],
) -> dict[str, float | int | str | bool]:
    """Aggregate the preregistered common seeds for one state/candidate/horizon."""
    if not rows:
        raise ValueError("at least one paired continuation row is required")
    keys = {
        (row.state_id, row.target_set_id, row.fallback_target_set_id, row.horizon)
        for row in rows
    }
    if len(keys) != 1 or len({row.continuation_seed for row in rows}) != len(rows):
        raise ValueError("paired rows must be one candidate with distinct seeds")
    first = rows[0]
    values = [row.advantage for row in rows]
    return {
        "state_id": first.state_id,
        "target_set_id": first.target_set_id,
        "fallback_target_set_id": first.fallback_target_set_id,
        "horizon": first.horizon,
        "continuation_seed_count": len(rows),
        "continuation_advantage_mean": statistics.fmean(values),
        "continuation_advantage_std": statistics.pstdev(values),
        "beats_fallback": statistics.fmean(value > 0 for value in values),
    }


def select_shortest_adequate_horizon(
    metrics: Mapping[int, Mapping[str, Any]],
    *,
    spearman_min: float = 0.70,
    ndcg1_min: float = 0.80,
    top1_agreement_min: float = 0.60,
) -> int:
    """Apply the frozen H4/H8 agreement rule; H12 is the fallback."""
    if 12 not in metrics:
        raise ValueError("H=12 reference metrics are required")
    for horizon in (4, 8):
        row = metrics.get(horizon)
        if row is None:
            continue
        scale_values = row.get("mean_spearman_by_scale", {})
        if (
            float(row["median_within_state_spearman"]) >= spearman_min
            and float(row["mean_ndcg_at_1"]) >= ndcg1_min
            and float(row["top1_agreement"]) >= top1_agreement_min
            and set(scale_values) == {"S", "M", "L"}
            and all(float(value) >= 0.0 for value in scale_values.values())
        ):
            return horizon
    return 12


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _overlap_fraction(target: Sequence[str], reference: Sequence[str]) -> float:
    target_set = set(target)
    return 0.0 if not target_set else len(target_set & set(reference)) / len(target_set)


def critical_and_bottleneck_operations(
    instance: Instance,
    current: DecodedCandidate,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Return the preregistered deterministic critical and bottleneck regions."""
    features = schedule_features(instance, current.schedule)
    critical = tuple(sorted(
        operation for operation, row in features.items()
        if bool(row["is_on_processing_critical_path"])
    ))
    proxy = bottleneck_proxy(current.schedule)
    scalar_by_proxy = {
        "ISLAND_PROCESSING_LOAD": "island_relative_load",
        "RECONFIGURATION": "local_reconfiguration_contribution",
        "W_LOGISTICS": "W_waiting_or_delay_contribution",
        "F_LOGISTICS": "F_waiting_or_delay_contribution",
        "CROSS_RESOURCE_SYNCHRONIZATION": "synchronization_wait_contribution",
    }
    if proxy == "PRECEDENCE_SEQUENCE":
        bottleneck = critical
    elif proxy in scalar_by_proxy:
        name = scalar_by_proxy[proxy]
        maximum = max(float(row[name]) for row in features.values())
        bottleneck = tuple(sorted(
            operation for operation, row in features.items()
            if maximum > 0 and float(row[name]) == maximum
        ))
        if not bottleneck and proxy == "CROSS_RESOURCE_SYNCHRONIZATION":
            bottleneck = tuple(sorted(
                operation for operation, row in features.items()
                if bool(row["is_on_processing_critical_path"])
                or bool(row["is_on_resource_critical_chain"])
            ))
    else:
        bottleneck = tuple(sorted(
            operation for operation, row in features.items()
            if bool(row["is_on_processing_critical_path"])
            or bool(row["is_on_resource_critical_chain"])
        ))
    return critical, bottleneck, proxy


def build_candidate_source_features(
    generated: ArmGenerationResult,
    *,
    state_id: str,
    operation_count: int,
    fallback_target_set_id: str,
    frozen_scores: Mapping[str, float],
    critical_operations: Sequence[str] = (),
    bottleneck_operations: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build outcome-blind full-bank context rows for the CAUR utility model."""
    if generated.requested_arm_count != 24:
        raise ValueError("Phase 6J requires the frozen 24-rule proposal generator")
    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    by_id = {arm.target_set_id: arm for arm in generated.arms}
    if set(by_id) != set(frozen_scores):
        raise ValueError("frozen scores must cover the complete deduplicated bank")
    if fallback_target_set_id not in by_id:
        raise ValueError("fallback must be a member of the full candidate bank")

    ranked = sorted(generated.arms, key=lambda arm: (-frozen_scores[arm.target_set_id], arm.target_set_id))
    score_rank = {arm.target_set_id: rank for rank, arm in enumerate(ranked, 1)}
    diversity_score = {
        arm.target_set_id: statistics.fmean(
            1.0 - _jaccard(arm.destroyed_operations, other.destroyed_operations)
            for other in generated.arms
            if other.target_set_id != arm.target_set_id
        )
        if len(generated.arms) > 1 else 0.0
        for arm in generated.arms
    }
    diverse = sorted(generated.arms, key=lambda arm: (-diversity_score[arm.target_set_id], arm.target_set_id))
    diversity_rank = {arm.target_set_id: rank for rank, arm in enumerate(diverse, 1)}
    fallback = by_id[fallback_target_set_id]
    best = ranked[0]
    denominator = max(1, len(generated.arms) - 1)

    rows: list[dict[str, Any]] = []
    for arm in generated.arms:
        target = arm.destroyed_operations
        rows.append({
            "state_id": state_id,
            "target_set_id": arm.target_set_id,
            "label_scope": FULL_BANK_SCOPE,
            "is_reduced_top8_audit": False,
            "primary_origin_rule": arm.origin_rules[0],
            "origin_destroy_operator": arm.origin_destroy_operator,
            "origin_family": arm.arm_family,
            "origin_rule_count": len(arm.origin_rules),
            "origin_family_count": len(arm.origin_families),
            "destroy_target_cardinality": len(target),
            "destroy_target_fraction": len(target) / operation_count,
            "fallback_overlap_fraction": _overlap_fraction(target, fallback.destroyed_operations),
            "fallback_jaccard": _jaccard(target, fallback.destroyed_operations),
            "critical_overlap_fraction": _overlap_fraction(target, critical_operations),
            "bottleneck_overlap_fraction": _overlap_fraction(target, bottleneck_operations),
            "best_frozen_score_jaccard": _jaccard(target, best.destroyed_operations),
            "normalized_frozen_score_rank": (score_rank[arm.target_set_id] - 1) / denominator,
            "normalized_diversity_rank": (diversity_rank[arm.target_set_id] - 1) / denominator,
            "is_fallback": arm.target_set_id == fallback_target_set_id,
        })
    return rows


def validate_full_bank_feature_rows(
    generated: ArmGenerationResult,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject truncated, duplicated, mislabeled, or incomplete candidate groups."""
    ids = [str(row.get("target_set_id")) for row in rows]
    checks = (
        generated.requested_arm_count == 24,
        generated.unique_arm_count + generated.duplicate_arm_count == 24,
        len(rows) == generated.unique_arm_count,
        len(ids) == len(set(ids)),
        set(ids) == {arm.target_set_id for arm in generated.arms},
        all(row.get("label_scope") == FULL_BANK_SCOPE for row in rows),
        all(row.get("is_reduced_top8_audit") is False for row in rows),
        all(set(CANDIDATE_INPUT_COLUMNS) <= set(row) for row in rows),
        not (set(CANDIDATE_INPUT_COLUMNS) & set(LABEL_COLUMNS)),
    )
    if not all(checks):
        raise ValueError("incomplete or invalid true-full-bank candidate group")


def validate_grouped_label_records(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate one collected state-level list without consuming label values as inputs."""
    if not rows:
        raise ValueError("grouped label list is empty")
    state_ids = {str(row.get("state_id")) for row in rows}
    target_ids = [str(row.get("target_set_id")) for row in rows]
    expected = int(rows[0].get("full_bank_unique_count", -1))
    if (
        len(state_ids) != 1
        or len(target_ids) != len(set(target_ids))
        or len(rows) != expected
        or any(row.get("label_scope") != FULL_BANK_SCOPE for row in rows)
        or any(row.get("is_reduced_top8_audit") is not False for row in rows)
        or any(not set(CANDIDATE_INPUT_COLUMNS) <= set(row) for row in rows)
        or any(not set(LABEL_COLUMNS) <= set(row) for row in rows)
    ):
        raise ValueError("invalid Phase 6J grouped full-bank labels")


def choose_caur_action(
    rows: Sequence[Mapping[str, Any]],
    *,
    fallback_target_set_id: str,
    p_min: float,
    lcb_lambda: float,
    delta_min: float,
    immediate_harm_floor: float,
) -> GateDecision:
    """Apply the deterministic selection-aware LCB intervention gate."""
    if not rows:
        raise ValueError("candidate rows are required")
    ids = {str(row["target_set_id"]) for row in rows}
    if fallback_target_set_id not in ids:
        raise ValueError("fallback is absent from candidate rows")
    winner = min(
        rows,
        key=lambda row: (
            -float(row["continuation_advantage_mean"]),
            str(row["target_set_id"]),
        ),
    )
    lcb = float(winner["continuation_advantage_mean"]) - lcb_lambda * float(
        winner["continuation_advantage_std"]
    )
    conditions = (
        ("PROBABILITY", float(winner["beats_fallback_probability"]) >= p_min),
        ("LCB", lcb > delta_min),
        ("SUPPORT", bool(winner["supported"])),
        ("IMMEDIATE_HARM", float(winner["immediate_utility_prediction"]) >= immediate_harm_floor),
    )
    failed = next((name for name, passed in conditions if not passed), None)
    intervened = failed is None and str(winner["target_set_id"]) != fallback_target_set_id
    return GateDecision(
        selected_target_set_id=(str(winner["target_set_id"]) if intervened else fallback_target_set_id),
        neural_target_set_id=str(winner["target_set_id"]),
        fallback_target_set_id=fallback_target_set_id,
        intervened=intervened,
        reason="INTERVENE" if intervened else (failed or "FALLBACK_ALREADY_BEST"),
        lcb=lcb,
    )


def grouped_oof_fold(scale: str, cf_level: str) -> int:
    """Return the frozen structural-cell fold for R12 cross-fitting."""
    folds = {
        ("S", "CF1"): 0, ("M", "CF2"): 0, ("L", "CF3"): 0,
        ("S", "CF2"): 1, ("M", "CF3"): 1, ("L", "CF1"): 1,
        ("S", "CF3"): 2, ("M", "CF1"): 2, ("L", "CF2"): 2,
    }
    try:
        return folds[(scale, cf_level)]
    except KeyError as error:
        raise ValueError(f"unknown Phase 6J structural cell: {scale}_{cf_level}") from error
