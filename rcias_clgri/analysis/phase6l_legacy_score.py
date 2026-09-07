"""Score-independent source features for Phase 6L.

This module is intentionally separate from the frozen Phase 6J feature path.
It consumes an already generated 24-rule bank and never accepts a policy,
historical model output, or continuation outcome.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from rcias_clgri.analysis.phase6j_caur import FULL_BANK_SCOPE
from rcias_clgri.search.phase6c import ArmGenerationResult, Phase6CTargetArm


CATEGORICAL_COLUMNS = (
    "primary_origin_rule",
    "origin_destroy_operator",
    "origin_family",
)
NUMERIC_COLUMNS = (
    "origin_rule_count",
    "origin_family_count",
    "destroy_target_cardinality",
    "destroy_target_fraction",
    "fallback_overlap_fraction",
    "fallback_jaccard",
    "critical_overlap_fraction",
    "bottleneck_overlap_fraction",
    "normalized_diversity_rank",
    "is_fallback",
)
ONLINE_INPUT_COLUMNS = (*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS)
REMOVED_LEGACY_INPUT_COLUMNS = (
    "best_frozen_score_jaccard",
    "normalized_frozen_score_rank",
)
FORBIDDEN_OUTCOME_COLUMNS = (
    "continuation_advantage_mean",
    "continuation_advantage_std",
    "beats_fallback",
    "immediate_utility",
    "candidate_decoded_makespan",
    "continuation_best_makespan",
    "fallback_continuation_best_makespan",
)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 0.0 if not union else len(left_set & right_set) / len(union)


def _overlap_fraction(target: Sequence[str], reference: Sequence[str]) -> float:
    target_set = set(target)
    return 0.0 if not target_set else len(target_set & set(reference)) / len(target_set)


def select_score_free_fallback(generated: ArmGenerationResult) -> Phase6CTargetArm:
    """Return the deduplicated canonical ``operator_related`` target.

    The Phase 6C generator constructs this target before any neural scoring.
    Requiring exactly one matching deduplicated arm makes the replacement fail
    closed if proposal provenance or deduplication semantics ever drift.
    """
    matches = [
        arm for arm in generated.arms if "operator_related" in arm.origin_rules
    ]
    if len(matches) != 1:
        raise ValueError(
            "Phase 6L requires exactly one deduplicated operator_related fallback"
        )
    fallback = matches[0]
    if fallback.destroyed_operations != generated.canonical_related_target:
        raise ValueError("operator_related provenance does not match the canonical target")
    return fallback


def build_score_free_candidate_source_features(
    generated: ArmGenerationResult,
    *,
    state_id: str,
    operation_count: int,
    critical_operations: Sequence[str] = (),
    bottleneck_operations: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build deterministic full-bank Phase 6L online features."""
    if generated.requested_arm_count != 24:
        raise ValueError("Phase 6L requires the frozen 24-rule proposal generator")
    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    fallback = select_score_free_fallback(generated)
    diversity_score = {
        arm.target_set_id: statistics.fmean(
            1.0 - _jaccard(arm.destroyed_operations, other.destroyed_operations)
            for other in generated.arms
            if other.target_set_id != arm.target_set_id
        )
        if len(generated.arms) > 1 else 0.0
        for arm in generated.arms
    }
    diverse = sorted(
        generated.arms,
        key=lambda arm: (-diversity_score[arm.target_set_id], arm.target_set_id),
    )
    diversity_rank = {arm.target_set_id: rank for rank, arm in enumerate(diverse, 1)}
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
            "fallback_overlap_fraction": _overlap_fraction(
                target, fallback.destroyed_operations
            ),
            "fallback_jaccard": _jaccard(target, fallback.destroyed_operations),
            "critical_overlap_fraction": _overlap_fraction(
                target, critical_operations
            ),
            "bottleneck_overlap_fraction": _overlap_fraction(
                target, bottleneck_operations
            ),
            "normalized_diversity_rank": (
                diversity_rank[arm.target_set_id] - 1
            ) / denominator,
            "is_fallback": arm.target_set_id == fallback.target_set_id,
        })
    validate_score_free_feature_rows(generated, rows)
    return rows


def validate_score_free_feature_rows(
    generated: ArmGenerationResult,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject incomplete banks, legacy-score inputs, and outcome leakage."""
    ids = [str(row.get("target_set_id")) for row in rows]
    keys = set().union(*(set(row) for row in rows)) if rows else set()
    checks = (
        generated.requested_arm_count == 24,
        generated.unique_arm_count + generated.duplicate_arm_count == 24,
        len(rows) == generated.unique_arm_count,
        len(ids) == len(set(ids)),
        ids == [arm.target_set_id for arm in generated.arms],
        all(row.get("label_scope") == FULL_BANK_SCOPE for row in rows),
        all(row.get("is_reduced_top8_audit") is False for row in rows),
        all(set(ONLINE_INPUT_COLUMNS) <= set(row) for row in rows),
        sum(bool(row.get("is_fallback")) for row in rows) == 1,
        not (set(REMOVED_LEGACY_INPUT_COLUMNS) & keys),
        not (set(FORBIDDEN_OUTCOME_COLUMNS) & keys),
        not any("frozen_score" in key for key in keys),
    )
    if not all(checks):
        raise ValueError("invalid Phase 6L score-free full-bank feature rows")
