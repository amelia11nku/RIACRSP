from __future__ import annotations

import copy
import random

import pytest
import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.teacher_anchor import active_stage_mean, freeze_teacher, teacher_stage_kl
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer, critic_hidden
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel
from rcias_clgri.training.curriculum import MixedScaleCurriculum


def _models(instance):
    state = build_graph_state(instance, InsertionDecoder(instance).empty_schedule())
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    model = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    return model, graph


def test_teacher_policy_frozen(automotive_instance):
    model, _ = _models(automotive_instance)
    freeze_teacher(model)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_teacher_kl_zero_then_positive_after_perturbation(automotive_instance):
    current, graph = _models(automotive_instance)
    teacher = freeze_teacher(copy.deepcopy(current))
    hidden = current.encode(graph)
    action = current.policy.sample_action(graph, hidden, deterministic=True).action
    initial = teacher_stage_kl(current, teacher, graph, hidden, action)
    assert sum(float(value.detach()) for value in initial.values()) == pytest.approx(0.0, abs=1e-7)
    with torch.no_grad():
        current.policy.operation_scorer[-1].weight.add_(0.1 * torch.randn_like(current.policy.operation_scorer[-1].weight))
    changed = teacher_stage_kl(current, teacher, graph, current.encode(graph), action)
    assert sum(float(value.detach()) for value in changed.values()) > 0.0


def test_normalized_entropy_averages_only_active_stages():
    values = {name: torch.tensor(value) for name, value in {"operation": .2, "island": .4, "w": .9, "f": .8}.items()}
    result = active_stage_mean(values, {"operation": 2, "island": 3, "w": 1, "f": 1})
    assert float(result) == pytest.approx(0.3)


def test_mixed_scale_sampling_covers_every_window():
    curriculum = MixedScaleCurriculum({"S": .25, "M": .35, "L": .4}, window_size=8)
    rng = random.Random(1)
    for _ in range(4):
        window = [curriculum.sample_level(rng) for _ in range(8)]
        assert set(window) == {"S", "M", "L"}


def test_critic_stop_gradient_detaches_encoder_hidden():
    source = {"O": torch.randn(2, 3, requires_grad=True)}
    detached = critic_hidden(source, stop_gradient=True)
    assert not detached["O"].requires_grad
    assert critic_hidden(source, stop_gradient=False)["O"] is source["O"]


def test_minimum_episode_rollout(automotive_instance):
    model, graph = _models(automotive_instance)
    tensorizer = GraphTensorizer.from_schema(GraphTensorizer(build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )).to_schema())
    trainer = PPOTrainer(
        model, tensorizer,
        PPOConfig(rollout_transitions=1, minimum_complete_episodes=2), device="cpu",
    )

    class Factory:
        def sample(self, seed, level):
            return automotive_instance

    class Curriculum:
        def sample_level(self, rng):
            return "S"

    buffer, episodes, _ = trainer.collect(
        Factory(), Curriculum(), seed_rng=random.Random(2), forbidden_seeds=set()
    )
    assert len(episodes) >= 2
    assert len(buffer) >= 2 * automotive_instance.num_operations
