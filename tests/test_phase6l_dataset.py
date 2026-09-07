import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from rcias_clgri.ni.phase6l_score_free_model import ScoreFreeCandidateContinuationHeads


DATA = Path("outputs/phase6l_legacy_score_decoupling_v1/data")


def test_score_free_dataset_manifest_and_fallback_labels():
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["derivation"]["states"] == 288
    assert manifest["derivation"]["candidates"] == 6809
    assert manifest["derivation"]["raw_seed_rows"] == 13618
    assert manifest["derivation"]["reanchored_states"] == 10
    assert manifest["derivation"]["unchanged_state_label_max_abs_delta"] == 0.0
    assert manifest["online_historical_score_forward_calls"] == 0
    frame = pd.read_parquet(DATA / "r12_score_free_grouped_labels.parquet")
    assert frame.groupby("state_id").is_fallback.sum().eq(1).all()
    assert frame.loc[frame.is_fallback, "continuation_advantage_mean"].abs().max() == 0.0


def test_score_free_dataset_excludes_historical_score_inputs():
    frame = pd.read_parquet(DATA / "r12_score_free_grouped_labels.parquet")
    assert "best_frozen_score_jaccard" not in frame
    assert "normalized_frozen_score_rank" not in frame
    assert "frozen_raw_score" not in frame
    schema = json.loads((DATA / "feature_schema.json").read_text())
    assert schema["historical_score_forward_calls"] == 0
    assert len(schema["numeric_columns"]) == 10


def test_score_free_head_requires_ten_numeric_features():
    heads = ScoreFreeCandidateContinuationHeads((25, 8, 6), dropout=0.0)
    actions = torch.randn(2, 128)
    state = torch.zeros(2, dtype=torch.long)
    fallback = torch.tensor([0])
    categorical = torch.zeros((2, 3), dtype=torch.long)
    result = heads(actions, state, fallback, categorical, torch.zeros((2, 10)))
    assert all(value.shape == (2,) for value in result)
    with pytest.raises(ValueError, match="ten score-free"):
        heads(actions, state, fallback, categorical, torch.zeros((2, 12)))
