from __future__ import annotations

import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.buffer import RolloutBuffer
from rcias_clgri.learning.ppo import clipped_ppo_loss
from rcias_clgri.learning.rollout import collect_episode
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def test_ppo_ratio_uses_joint_logprob_and_is_one_when_unchanged():
    old = torch.tensor([-1.2, -0.7, -2.1])
    output = clipped_ppo_loss(
        new_joint_log_prob=old.clone(),
        old_joint_log_prob=old,
        advantages=torch.tensor([-1.0, 0.5, 1.5]),
        values=torch.tensor([0.1, 0.2, 0.3]),
        returns=torch.tensor([0.0, 0.1, 0.4]),
        entropy=torch.tensor([0.5, 0.4, 0.6]),
    )
    assert torch.equal(output.ratio, torch.ones(3))
    assert output.approx_kl.item() == 0.0
    assert output.clip_fraction.item() == 0.0
    assert torch.isfinite(output.total_loss)


def test_ppo_update_changes_parameters_with_finite_metrics(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    torch.manual_seed(51)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    )
    buffer = RolloutBuffer()
    for episode_seed in (71, 72):
        torch.manual_seed(episode_seed)
        episode = collect_episode(
            model, tensorizer, automotive_instance,
            device="cpu", deterministic=False, store_transitions=True,
        )
        for transition in episode.transitions:
            buffer.add(transition)
    buffer.compute_advantages()
    trainer = PPOTrainer(
        model, tensorizer,
        PPOConfig(update_epochs=1, minibatch_size=12, rollout_transitions=12, target_kl=10.0),
        device="cpu",
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = trainer.update(buffer, seed=81)
    assert any(
        not torch.equal(left, right.detach())
        for left, right in zip(before, model.parameters())
    )
    for key in ("total_loss", "policy_loss", "value_loss", "approx_kl", "gradient_norm_before"):
        assert torch.isfinite(torch.tensor(float(metrics[key])))
