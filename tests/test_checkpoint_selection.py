from __future__ import annotations

import pytest

from scripts.train_ppo import _legacy_validation_score, _weighted_validation_score


def _validation() -> dict[str, object]:
    return {
        "records": [
            {"level": "S", "normalized_makespan": 1.0},
            {"level": "S", "normalized_makespan": 3.0},
            {"level": "M", "normalized_makespan": 4.0},
            {"level": "L", "normalized_makespan": 8.0},
        ]
    }


def test_checkpoint_score_renormalizes_weights_over_active_levels():
    weights = {"S": 0.2, "M": 0.3, "L": 0.5}
    score = _weighted_validation_score(_validation(), ("S", "M"), weights)
    assert score == pytest.approx((0.2 * 2.0 + 0.3 * 4.0) / 0.5)


def test_checkpoint_score_includes_large_level_after_activation():
    weights = {"S": 0.2, "M": 0.3, "L": 0.5}
    score = _weighted_validation_score(_validation(), ("S", "M", "L"), weights)
    assert score == pytest.approx(0.2 * 2.0 + 0.3 * 4.0 + 0.5 * 8.0)


def test_checkpoint_score_rejects_missing_or_zero_active_weights():
    with pytest.raises(ValueError, match="missing"):
        _weighted_validation_score(_validation(), ("S", "M"), {"S": 1.0})
    with pytest.raises(ValueError, match="positive"):
        _weighted_validation_score(
            _validation(), ("S", "M"), {"S": 0.0, "M": 0.0}
        )
    with pytest.raises(ValueError, match="no records"):
        _weighted_validation_score(_validation(), ("S", "X"), {"S": 0.5, "X": 0.5})


def test_phase3_legacy_score_remains_record_weighted_and_excludes_large_level():
    assert _legacy_validation_score(_validation(), ("S", "M")) == pytest.approx(8.0 / 3.0)
    assert _legacy_validation_score(_validation(), ("S", "M", "L")) == pytest.approx(8.0 / 3.0)
