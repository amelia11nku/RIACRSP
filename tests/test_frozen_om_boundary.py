import copy

import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, OperationAnchoredModel, RCIASNeuralModel


def test_frozen_om_distribution_and_wf_objective(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    bc = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    model = OperationAnchoredModel(
        copy.deepcopy(bc), copy.deepcopy(bc), frozen_prefix_stages=2
    )
    hidden = model.encode(graph)
    evaluation = model.sample_action_from_hidden(graph, hidden, deterministic=True)
    frozen = model.frozen_operation.prefix_distributions(graph, evaluation.action)
    assert evaluation.action.operation_id == frozen["operation"].argmax()
    assert evaluation.action.island_id == frozen["island"].argmax()
    torch.testing.assert_close(
        evaluation.joint_log_prob,
        evaluation.stage_log_probs["w"] + evaluation.stage_log_probs["f"],
    )
    torch.testing.assert_close(
        evaluation.joint_entropy,
        evaluation.stage_entropies["w"] + evaluation.stage_entropies["f"],
    )
    assert model.trainable_stages == ("w", "f")

