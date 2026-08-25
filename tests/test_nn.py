from __future__ import annotations

import pytest
import torch

from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def _model_and_tensor(instance, schedule):
    graph = build_graph_state(instance, schedule)
    tensorizer = GraphTensorizer(graph)
    tensor = tensorizer.tensorize(graph)
    torch.manual_seed(17)
    model = RCIASNeuralModel(
        tensorizer,
        ModelConfig(embedding_dim=32, heads=4, layers=2, dropout=0.0),
    )
    return model, tensor


def test_rt_hgt_forward_backward_and_typed_shapes(automotive_instance):
    schedule = InsertionDecoder(automotive_instance).empty_schedule()
    model, tensor = _model_and_tensor(automotive_instance, schedule)
    action = Action("o11", "M1", "W1", "F1")
    losses = model.action_losses(tensor, action)
    assert set(losses) == {"operation", "island", "w", "f", "total"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters() if parameter.grad is not None
    ) > 0.0
    assert model(tensor).shape == torch.Size([])


def test_autoregressive_policy_uses_hard_masks_and_same_island_none(automotive_instance):
    decoder = InsertionDecoder(automotive_instance)
    schedule = decoder.empty_schedule()
    decoder.commit_action(
        schedule, decoder.probe_action(schedule, Action("o11", "M3", "W1", "F1"))
    )
    model, tensor = _model_and_tensor(automotive_instance, schedule)
    hidden = model.encode(tensor)
    operations = model.policy.operation_distribution(tensor, hidden)
    assert "o12" in operations.candidate_ids
    assert "o11" not in operations.candidate_ids
    islands = model.policy.island_distribution(tensor, hidden, "o13")
    assert set(islands.candidate_ids) == {"M1", "M3"}
    w_agvs = model.policy.w_distribution(tensor, hidden, "o13", "M3")
    assert w_agvs.candidate_ids == (None,)
    with pytest.raises(ValueError, match="violates the hard mask"):
        model.policy.imitation_loss(operations, "o11")


def test_policy_has_no_flat_cartesian_action_head(automotive_instance):
    schedule = InsertionDecoder(automotive_instance).empty_schedule()
    model, tensor = _model_and_tensor(automotive_instance, schedule)
    assert not hasattr(model.policy, "joint_action_head")
    hidden = model.encode(tensor)
    op = model.policy.operation_distribution(tensor, hidden).argmax()
    island = model.policy.island_distribution(tensor, hidden, op).argmax()
    w_agv = model.policy.w_distribution(tensor, hidden, op, island).argmax()
    f_agv = model.policy.f_distribution(tensor, hidden, op, island, w_agv).argmax()
    assert Action(op, island, w_agv, f_agv)
