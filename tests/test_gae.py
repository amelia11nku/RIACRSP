from __future__ import annotations

import torch

from rcias_clgri.learning.buffer import generalized_advantage_estimate, normalize_advantages


def test_gae_terminal():
    result = generalized_advantage_estimate(
        torch.tensor([-0.2]), torch.tensor([0.4]), torch.tensor([True]),
        gamma=1.0, gae_lambda=0.95,
    )
    assert torch.allclose(result.advantages, torch.tensor([-0.6]))
    assert torch.allclose(result.returns, torch.tensor([-0.2]))


def test_gae_known_sequence():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.25, 1.0])
    result = generalized_advantage_estimate(
        rewards, values, torch.tensor([False, False, True]),
        gamma=1.0, gae_lambda=1.0,
    )
    assert torch.allclose(result.returns, torch.tensor([6.0, 5.0, 3.0]))
    normalized = normalize_advantages(result.advantages)
    assert abs(float(normalized.mean())) < 1e-6
    assert torch.allclose(normalized.std(unbiased=False), torch.ones(()), atol=1e-6)
