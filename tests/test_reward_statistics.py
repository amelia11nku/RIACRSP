from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.train_ppo import _reward_statistics


def test_reward_statistics_reports_zero_fraction_and_population_std():
    episodes = [SimpleNamespace(transitions=[
        SimpleNamespace(reward=0.0),
        SimpleNamespace(reward=-1.0),
        SimpleNamespace(reward=0.0),
    ])]
    result = _reward_statistics(episodes)
    assert result["count"] == 3.0
    assert result["zero_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["mean"] == pytest.approx(-1.0 / 3.0)
    assert result["std"] == pytest.approx(2.0**0.5 / 3.0)


def test_reward_statistics_rejects_empty_rollout():
    with pytest.raises(ValueError, match="at least one transition"):
        _reward_statistics([])
