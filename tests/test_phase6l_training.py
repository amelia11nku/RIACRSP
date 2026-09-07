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
