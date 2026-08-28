import copy

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.rollout import collect_episode
from rcias_clgri.learning.stagewise import collect_hybrid_episode
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def test_all_phase5b_hybrid_boundaries_are_sequentially_valid(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    model = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    reference = collect_episode(
        model,
        tensorizer,
        automotive_instance,
        device="cpu",
        deterministic=True,
        store_transitions=False,
    )
    for prefix_stages in (1, 2, 3):
        hybrid = collect_hybrid_episode(
            model,
            copy.deepcopy(model),
            tensorizer,
            automotive_instance,
            prefix_stages=prefix_stages,
            device="cpu",
        )
        assert hybrid.feasible
        assert hybrid.actions == reference.actions
        assert hybrid.makespan == reference.makespan

