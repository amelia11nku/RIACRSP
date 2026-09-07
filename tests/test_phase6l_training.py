import json
from pathlib import Path


TRAINING = Path("outputs/phase6l_legacy_score_decoupling_v1/training")


def test_phase6l_l3_completion_audit_and_gate_boundary():
    audit = json.loads((TRAINING / "completion_integrity_audit.json").read_text())
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert len(audit["runs"]) == 9
    assert audit["eligible_gate_count"] == 0
    assert audit["selected_gate"] is None
    assert audit["r13_accessed"] is False
    assert audit["r14_accessed"] is False


def test_phase6l_l4_quality_failure_is_fail_closed():
    quality = json.loads(
        Path("outputs/phase6l_legacy_score_decoupling_v1/quality/development_quality.json").read_text()
    )
    assert quality["decision"] == "MODEL_REVISION_QUALITY"
    assert all(quality["raw_essential_checks"].values())
    assert quality["eligible_gate_count"] == 0
    assert quality["selected_gate"] is None
    assert quality["failed_checks"] == [
        "retained_gate_with_scale_coverage",
        "positive_gated_lift_lcb",
        "nonnegative_gated_scale_lift",
    ]
    assert quality["runtime_stage"] == "NOT_RUN_STOPPED_ON_L4_QUALITY"
    assert quality["r12_qualification"] == "NOT_RUN_STOPPED_ON_L4_QUALITY"
    assert quality["r13_locked"] is True
    assert quality["r14_locked"] is True
