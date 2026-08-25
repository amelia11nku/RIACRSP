from __future__ import annotations

import torch

from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def _partial_model(instance):
    decoder = InsertionDecoder(instance)
    schedule = decoder.empty_schedule()
    decoder.commit_action(
        schedule, decoder.probe_action(schedule, Action("o11", "M3", "W1", "F1"))
    )
    state = build_graph_state(instance, schedule)
    tensorizer = GraphTensorizer(state)
    graph = tensorizer.tensorize(state)
    torch.manual_seed(31)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    ).eval()
    return model, graph


def test_sampling_and_deterministic_single_stage(automotive_instance):
    model, graph = _partial_model(automotive_instance)
    hidden = model.encode(graph)
    greedy = model.greedy_action(graph)
    sampled = model.policy.sample_action(graph, hidden, deterministic=True)
    assert sampled.action == greedy
    same_island = Action("o13", "M3", None, "F1")
    evaluation = model.policy.evaluate_action(graph, hidden, same_island)
    assert evaluation.stage_candidate_counts["w"] == 1
    assert evaluation.stage_log_probs["w"].item() == 0.0
    assert evaluation.stage_entropies["w"].item() == 0.0
    assert torch.isfinite(evaluation.joint_log_prob)


def test_stochastic_action_is_always_hard_masked(automotive_instance):
    model, graph = _partial_model(automotive_instance)
    hidden = model.encode(graph)
    torch.manual_seed(99)
    for _ in range(20):
        result = model.policy.sample_action(graph, hidden, deterministic=False)
        action = result.action
        assert action.operation_id in graph.candidates.ready_operations
        assert graph.candidates.island_masks[action.operation_id][action.island_id]
        assert action.w_agv_id in graph.candidates.w_masks[(action.operation_id, action.island_id)]
        assert action.f_agv_id in graph.candidates.f_masks[(action.operation_id, action.island_id)]
