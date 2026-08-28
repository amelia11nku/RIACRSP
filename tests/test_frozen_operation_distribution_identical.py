import copy

import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, OperationAnchoredModel, RCIASNeuralModel


def _model_and_graph(instance):
    state = build_graph_state(instance, InsertionDecoder(instance).empty_schedule())
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    model = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    return model, graph


def test_frozen_operation_distribution_identical_after_downstream_change(automotive_instance):
    bc, graph = _model_and_graph(automotive_instance)
    anchored = OperationAnchoredModel(copy.deepcopy(bc), copy.deepcopy(bc))
    before = anchored.frozen_operation.distribution(graph).probabilities().clone()
    with torch.no_grad():
        for parameter in anchored.downstream.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    after = anchored.frozen_operation.distribution(graph).probabilities()
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_frozen_operation_parameters_have_no_grad(automotive_instance):
    bc, graph = _model_and_graph(automotive_instance)
    anchored = OperationAnchoredModel(copy.deepcopy(bc), copy.deepcopy(bc))
    hidden = anchored.encode(graph)
    evaluation = anchored.sample_action_from_hidden(graph, hidden, deterministic=True)
    (-evaluation.joint_log_prob + anchored.value(hidden)).backward()
    assert all(not parameter.requires_grad for parameter in anchored.frozen_operation.parameters())
    assert all(parameter.grad is None for parameter in anchored.frozen_operation.parameters())
    assert any(parameter.grad is not None for parameter in anchored.downstream.parameters())

