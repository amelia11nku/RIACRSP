import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "outputs/phase6m_selective_confidence_v1/training"


def test_phase6m_m3_completion_is_independently_audited():
    audit = json.loads((TRAINING / "completion_integrity_audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["decision"] == "READY_FOR_M4"
    assert all(audit["checks"].values())
    assert len(audit["inner_ranker_runs"]) == 18
    assert len(audit["selector_runs"]) == 9
    assert audit["aggregate"]["prediction_rows"] == 20427
    assert audit["aggregate"]["ensemble_candidates"] == 6809
    assert audit["aggregate"]["states"] == 288
    assert audit["historical_score_online_forward_calls"] == 0
    assert audit["r13_accessed"] is False
    assert audit["r14_accessed"] is False


def test_phase6m_m3_preserves_ranker_and_improves_support_boundary():
    audit = json.loads((TRAINING / "completion_integrity_audit.json").read_text())
    aggregate = audit["aggregate"]
    assert aggregate["ranker_candidate_identity_preserved"] is True
    assert aggregate["ranker_winner_identity_preserved"] is True
    assert aggregate["hard_support_rate"] > 0.98
    assert aggregate["selected_winner_support_rate"] == 1.0
