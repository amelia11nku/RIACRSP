from __future__ import annotations

import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.evaluator import load_checkpoint, save_checkpoint
from rcias_clgri.learning.rollout import collect_episode
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel


def test_checkpoint_reproducibility(tmp_path, automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    torch.manual_seed(123)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, tensorizer, metadata={"seed": 123})
    loaded, restored_tensorizer, metadata = load_checkpoint(path, device="cpu")
    first = collect_episode(
        loaded, restored_tensorizer, automotive_instance,
        device="cpu", deterministic=True, store_transitions=False,
    )
    second = collect_episode(
        loaded, restored_tensorizer, automotive_instance,
        device="cpu", deterministic=True, store_transitions=False,
    )
    assert metadata["seed"] == 123
    assert first.actions == second.actions
    assert first.makespan == second.makespan


def test_checkpoint_loads_torch_version_metadata(tmp_path, automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    )
    path = tmp_path / "torch_version.pt"
    save_checkpoint(path, model, tensorizer, metadata={"torch": torch.__version__})
    _, _, metadata = load_checkpoint(path, device="cpu")
    assert str(metadata["torch"]) == str(torch.__version__)
