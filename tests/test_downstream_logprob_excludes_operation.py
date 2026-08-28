import copy

import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, OperationAnchoredModel, RCIASNeuralModel


def _evaluation(instance):
    state = build_graph_state(instance, InsertionDecoder(instance).empty_schedule())
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    bc = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    anchored = OperationAnchoredModel(copy.deepcopy(bc), copy.deepcopy(bc))
    hidden = anchored.encode(graph)
    return anchored.sample_action_from_hidden(graph, hidden, deterministic=True)


def test_downstream_logprob_excludes_operation(automotive_instance):
    evaluation = _evaluation(automotive_instance)
    expected = sum(evaluation.stage_log_probs[stage] for stage in ("island", "w", "f"))
    torch.testing.assert_close(evaluation.joint_log_prob, expected)


def test_downstream_entropy_excludes_operation(automotive_instance):
    evaluation = _evaluation(automotive_instance)
    expected = sum(evaluation.stage_entropies[stage] for stage in ("island", "w", "f"))
    torch.testing.assert_close(evaluation.joint_entropy, expected)


def test_downstream_ppo_ratio_uses_only_mwf_logprob(automotive_instance):
    evaluation = _evaluation(automotive_instance)
    shifted_operation_logprob = evaluation.stage_log_probs["operation"] - 5.0
    old_downstream = evaluation.joint_log_prob.detach() - 0.2
    ratio = torch.exp(evaluation.joint_log_prob - old_downstream)
    ratio_with_irrelevant_operation = torch.exp(
        evaluation.joint_log_prob + 0.0 * shifted_operation_logprob - old_downstream
    )
    torch.testing.assert_close(ratio, ratio_with_irrelevant_operation)
    torch.testing.assert_close(ratio, torch.exp(ratio.new_tensor(0.2)))

