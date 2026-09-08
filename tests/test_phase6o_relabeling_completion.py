from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling"


def test_phase6o_relabeling_completion_audit_is_full_and_locked():
    import json

    audit = json.loads((OUT / "completion_audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["states"] == 864
    assert audit["full_bank_candidates"] == 20441
    assert audit["targeted_candidates"] == audit["targeted_five_seed_candidates"] == 4786
    assert audit["additional_seed_rows"] == 14358
    assert audit["combined_seed_rows"] == 55240
    assert audit["base_implementation_states"] == 288
    assert audit["amended_implementation_states"] == 576
    assert audit["reanchored_original_states"] == 10
    assert audit["historical_score_calls"] == 0
    assert audit["r13_accessed"] is False and audit["r14_accessed"] is False


def test_phase6o_compact_additional_labels_match_frozen_union():
    additional = pd.read_parquet(OUT / "additional_seed_labels.parquet")
    union = pd.read_csv(
        ROOT
        / "outputs/phase6o_neural_shortlist_v1/preregistration/targeted_relabel_candidate_union.csv"
    )
    assert len(additional) == 14358
    assert set(zip(additional.state_id, additional.target_set_id)) == set(
        zip(union.state_id, union.target_set_id)
    )
    assert additional.groupby(["state_id", "target_set_id"]).continuation_seed.nunique().eq(3).all()
