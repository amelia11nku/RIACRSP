from dataclasses import replace
import inspect

import pytest

from rcias_clgri.analysis.phase6l_legacy_score import (
    FORBIDDEN_OUTCOME_COLUMNS,
    ONLINE_INPUT_COLUMNS,
    REMOVED_LEGACY_INPUT_COLUMNS,
    build_score_free_candidate_source_features,
    select_score_free_fallback,
)
from rcias_clgri.search.phase6c import ArmGenerationResult, ArmProposal, Phase6CTargetArm
from scripts.audit_phase6l_dependencies import dependency_entries


def generated_bank():
    arms = (
        Phase6CTargetArm("a", "ORIGINAL_OPERATOR", "related", ("operator_related",),
                         ("ORIGINAL_OPERATOR",), ("o1", "o2")),
        Phase6CTargetArm("b", "MATCHED_RANDOM", "random", ("matched_random_1",),
                         ("MATCHED_RANDOM",), ("o2", "o3")),
    )
    proposals = tuple(
        ArmProposal(f"rule_{index}", "MATCHED_RANDOM", "random", ("o2", "o3"))
        for index in range(22)
    )
    proposals = (
        ArmProposal("operator_related", "ORIGINAL_OPERATOR", "related", ("o1", "o2")),
        *proposals,
        ArmProposal("last", "MATCHED_RANDOM", "random", ("o2", "o3")),
    )
    return ArmGenerationResult(arms, proposals, ("o1", "o2"), 24, 2, 22)


def test_score_free_features_preserve_bank_order_and_exclude_legacy_and_labels():
    generated = generated_bank()
    rows = build_score_free_candidate_source_features(
        generated,
        state_id="state",
        operation_count=3,
        critical_operations=("o1",),
        bottleneck_operations=("o3",),
    )
    assert [row["target_set_id"] for row in rows] == ["a", "b"]
    assert [row["is_fallback"] for row in rows] == [True, False]
    assert all(set(ONLINE_INPUT_COLUMNS) <= set(row) for row in rows)
    keys = set().union(*(set(row) for row in rows))
    assert not keys.intersection(REMOVED_LEGACY_INPUT_COLUMNS)
    assert not keys.intersection(FORBIDDEN_OUTCOME_COLUMNS)


def test_score_free_fallback_fails_closed_on_provenance_drift():
    generated = generated_bank()
    assert select_score_free_fallback(generated).target_set_id == "a"
    with pytest.raises(ValueError, match="exactly one"):
        select_score_free_fallback(replace(generated, arms=generated.arms[1:]))
    duplicate = replace(generated.arms[1], origin_rules=("operator_related",))
    with pytest.raises(ValueError, match="exactly one"):
        select_score_free_fallback(replace(generated, arms=(*generated.arms, duplicate)))


def test_dependency_map_covers_known_live_interfaces():
    interfaces = {row["interface"] for row in dependency_entries()}
    required = {
        "historical scorer forward",
        "frozen_raw_score and calibrated historical outputs",
        "24-rule proposal generation",
        "target-set deduplication and identity",
        "candidate order and target_set_id tie breaks",
        "ALNS_RELATED_FALLBACK role",
        "fallback_overlap_fraction and fallback_jaccard",
        "best_frozen_score_jaccard",
        "normalized_frozen_score_rank",
        "normalized_diversity_rank",
        "continuation advantage and beats-fallback labels",
        "feature normalization and categorical vocabulary",
        "support mask",
        "OOF calibrator and decision thresholds",
        "repair, decoder and feasibility semantics",
    }
    assert required <= interfaces
    assert {row["classification"] for row in dependency_entries()} <= {
        "REMOVE", "REPLACE", "OFFLINE_TEACHER_ONLY", "KEEP"
    }


def test_online_feature_builder_cannot_receive_a_historical_model_or_scores():
    parameters = inspect.signature(build_score_free_candidate_source_features).parameters
    assert "policy" not in parameters
    assert "model" not in parameters
    assert "frozen_scores" not in parameters
