import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/phase6o_neural_shortlist_v1"


def test_phase6o_training_completion_covers_all_nested_oof_runs():
    audit = json.loads(
        (PHASE / "training/completion_integrity_audit.json").read_text()
    )
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert len(audit["runs"]) == 9
    assert audit["oof_prediction_rows"] == 20441 * 3
    assert audit["ensemble_rows"] == 20441
    assert audit["states"] == 864
    assert audit["instances"] == 18
    assert len(audit["selected_epochs"]) == 9
    assert all(1 <= epoch <= 50 for epoch in audit["selected_epochs"])
    assert all(
        run["trainable_checkpoint_parameters"] == 667838
        for run in audit["runs"]
    )
    assert audit["historical_score_calls"] == 0
    assert audit["r13_accessed"] is False
    assert audit["r14_accessed"] is False


def test_phase6o_oof_gate_fails_exact_frozen_top_utility_requirements():
    quality = json.loads((PHASE / "quality/oof_quality_gate.json").read_text())
    assert quality["status"] == "FAIL"
    assert quality["decision"] == "MODEL_REVISION_TOP_UTILITY"
    assert quality["failed_hard_checks"] == [
        "common_raw_selected_lift_at_least_phase6l",
        "common_top1_regret_at_most_phase6l",
    ]
    assert quality["checks"]["phase6l_frozen_reference_exact"] is True
    assert quality["checks"]["expanded_raw_selected_lift_positive"] is True
    assert quality["checks"]["expanded_grouped_lcb_positive"] is True
    assert quality["checks"]["all_scale_selected_lift_nonnegative"] is True
    assert quality["checks"]["candidate_origin_diversity"] is True


def test_phase6o_oof_gate_is_fail_closed_before_solver_and_runtime():
    quality = json.loads((PHASE / "quality/oof_quality_gate.json").read_text())
    common = quality["metrics"]["common_original_288"]
    expanded = quality["metrics"]["expanded_864"]
    paired = quality["metrics"]["paired_improvement_same_relabel_truth"]
    assert common["selected_lift"] < quality["thresholds"][
        "common_raw_selected_lift_min"
    ]
    assert common["selection_regret"] > quality["thresholds"][
        "common_top1_regret_max"
    ]
    assert expanded["selected_lift"] > 0
    assert expanded["selected_lift_lcb"] > 0
    assert paired["mean_improvement"] < 0
    assert paired["instance_grouped_lcb"] < 0 < paired["instance_grouped_ucb"]
    assert quality["direct_decision_stage"] == "NOT_RUN_OOF_GATE_FAILED"
    assert quality["development_solver_pilot"] == "NOT_RUN_OOF_GATE_FAILED"
    assert quality["formal_runtime"] == "NOT_RUN_OOF_GATE_FAILED"
    assert quality["formal_r12_solver"] == "NOT_RUN_OOF_GATE_FAILED"
    assert quality["historical_score_calls"] == 0
    assert quality["gurobi_run"] is False
    assert quality["r13_accessed"] is False
    assert quality["r14_accessed"] is False
