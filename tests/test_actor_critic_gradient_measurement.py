import pytest
import torch

from scripts.analyze_gradient_interference import gradient_cosine


def test_actor_critic_gradient_cosine_measurement():
    actor = (torch.tensor([1.0, 0.0]), torch.tensor([1.0]))
    critic = (torch.tensor([-1.0, 0.0]), torch.tensor([-1.0]))
    assert gradient_cosine(actor, critic) == pytest.approx(-1.0)
