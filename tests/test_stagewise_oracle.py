import copy

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.rollout import collect_episode
from rcias_clgri.learning.stagewise import collect_hybrid_episode
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def test_stagewise_oracle_is_sequential_and_matches_identical_policy(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    model = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    reference = collect_episode(
        model, tensorizer, automotive_instance, device="cpu", deterministic=True,
        store_transitions=False,
    )
    for prefix in (1, 2):
        hybrid = collect_hybrid_episode(
            model, copy.deepcopy(model), tensorizer, automotive_instance,
            prefix_stages=prefix, device="cpu",
        )
        assert hybrid.feasible
        assert hybrid.actions == reference.actions
        assert hybrid.makespan == reference.makespan
