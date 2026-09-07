import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/phase6m_selective_confidence_v1.json").read_text())


def test_phase6m_primary_and_locked_boundaries_are_explicit():
    value = config()
    assert value["primary_family"] == "M1_SCORE_FREE_SELECTIVE_RISK"
    assert value["boundaries"]["historical_score_online_forward_calls"] == 0
    assert value["boundaries"]["candidate_generator_rules"] == 24
    assert value["boundaries"]["r13"].startswith("LOCKED")
    assert value["boundaries"]["r14"].startswith("LOCKED")
    assert value["boundaries"]["teacher_ablation"] is None


def test_selector_features_exclude_outcomes_and_historical_scores():
    value = config()
    serialized = json.dumps(value["selector_features"]).lower()
    for forbidden in (
        "continuation outcomes as input", "repair outcomes", "decoded makespan",
        "historical frozen scores", "r13", "r14",
    ):
        assert forbidden in serialized
    assert value["selector"]["target_rows"].startswith("two archived CRN")


def test_support_fix_separates_binary_and_keeps_semantic_fail_closed():
    support = config()["support"]
    assert support["excluded_from_continuous_distance"] == ["is_fallback"]
    assert support["fine_rule_oov"].startswith("allowed")
    assert "unknown origin_family" in support["hard_fail"]
    assert "unknown origin_destroy_operator" in support["hard_fail"]
    assert support["hard_support_boundary"] == 12.0


def test_final_scientific_gate_is_not_relaxed():
    gate = config()["gate"]
    assert gate["p_min_grid"] == [0.55, 0.65, 0.75]
    assert gate["lcb_lambda_grid"] == [0.5, 1.0]
    assert gate["delta_min_grid"] == [0.0, 0.0025, 0.005]
    assert gate["immediate_harm_floor"] == -0.005
    assert gate["minimum_direct_interventions_per_scale"] == 20
