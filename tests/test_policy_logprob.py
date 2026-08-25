from __future__ import annotations

import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def test_joint_logprob_is_stage_sum_and_unchanged_ratio_is_one(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    torch.manual_seed(7)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    ).eval()
    hidden = model.encode(graph)
    chosen = model.policy.sample_action(graph, hidden, deterministic=False)
    reevaluated = model.policy.evaluate_action(graph, hidden, chosen.action)
    assert torch.allclose(
        chosen.joint_log_prob,
        torch.stack(tuple(chosen.stage_log_probs.values())).sum(),
    )
    assert torch.allclose(chosen.joint_log_prob, reevaluated.joint_log_prob)
    assert torch.allclose(
        torch.exp(reevaluated.joint_log_prob - chosen.joint_log_prob),
        torch.ones(()),
    )


def test_illegal_candidate_probability_is_exact_zero(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    ).eval()
    distribution = model.policy.operation_distribution(graph, model.encode(graph))
    assert distribution.probability("o12").item() == 0.0
