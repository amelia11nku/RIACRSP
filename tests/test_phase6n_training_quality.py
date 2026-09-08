import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1"


def test_phase6n_n4_completion_audit_covers_all_nested_oof_runs():
    audit = json.loads(
        (PHASE / "training/completion_integrity_audit.json").read_text()
    )
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert len(audit["runs"]) == 9
    assert audit["oof_prediction_rows"] == 20441 * 3
    assert audit["inner_validation_prediction_rows"] == 20441 * 3
    assert audit["ensemble_rows"] == 20441
    assert audit["states"] == 864
    assert audit["historical_score_calls"] == 0
    assert audit["r13_accessed"] is False
    assert audit["r14_accessed"] is False


def test_phase6n_n5_representation_failure_is_single_gate_and_fail_closed():
    quality = json.loads((PHASE / "quality/raw_representation_gate.json").read_text())
    assert quality["status"] == "FAIL"
    assert quality["decision"] == "MODEL_REVISION_REPRESENTATION"
    assert quality["failed_hard_checks"] == [
        "paired_phase6l_lift_improvement_lcb_positive"
    ]
    assert quality["checks"]["phase6l_reference_exact"] is True
    assert quality["checks"]["common_spearman_improvement_at_least_0_02"] is True
    assert quality["checks"]["at_least_two_diagnostics_improve"] is True
    assert quality["checks"]["paired_phase6l_lift_improvement_lcb_positive"] is False
    assert quality["metrics"]["paired_improvement"]["mean_improvement"] < 0
    assert quality["metrics"]["paired_improvement"]["instance_grouped_lcb"] < 0
    assert quality["calibration_stage"] == "NOT_RUN_STOPPED_ON_N5_REPRESENTATION"
    assert quality["decision_stage"] == "NOT_RUN_STOPPED_ON_N5_REPRESENTATION"
    assert quality["runtime_stage"] == "NOT_RUN_STOPPED_ON_N5_REPRESENTATION"
    assert quality["solver_stage"] == "NOT_RUN_STOPPED_ON_N5_REPRESENTATION"
    assert quality["historical_score_calls"] == 0
    assert quality["r13_accessed"] is False
    assert quality["r14_accessed"] is False


def test_phase6n_quality_distinguishes_frozen_and_n5_bootstrap_lcbs():
    quality = json.loads((PHASE / "quality/raw_representation_gate.json").read_text())
    frozen = quality["metrics"]["phase6l_frozen_reference_lcb"]
    recomputed = quality["metrics"]["phase6l_common_reference"][
        "selected_lift_lcb"
    ]
    assert frozen == 0.0028662500281748144
    assert recomputed == 0.002935666985164265
    assert frozen > 0 and recomputed > 0
    assert frozen != recomputed
