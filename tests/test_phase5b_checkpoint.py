import copy

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.evaluator import (
    load_operation_anchored_checkpoint,
    save_checkpoint,
    save_operation_anchored_checkpoint,
)
from rcias_clgri.nn import GraphTensorizer, ModelConfig, OperationAnchoredModel, RCIASNeuralModel


def test_operation_anchored_checkpoint_round_trip(tmp_path, automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(state)
    bc = RCIASNeuralModel(tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1))
    bc_path = tmp_path / "bc.pt"
    save_checkpoint(bc_path, bc, tensorizer, metadata={"kind": "bc"})
    anchored = OperationAnchoredModel(copy.deepcopy(bc), copy.deepcopy(bc))
    anchored_path = tmp_path / "anchored.pt"
    save_operation_anchored_checkpoint(
        anchored_path,
        anchored,
        tensorizer,
        frozen_operation_checkpoint=str(bc_path),
        metadata={"kind": "phase5b"},
    )
    restored, restored_tensorizer, metadata = load_operation_anchored_checkpoint(
        anchored_path, device="cpu"
    )
    assert metadata == {"kind": "phase5b"}
    assert restored_tensorizer.to_schema() == tensorizer.to_schema()
    assert all(not parameter.requires_grad for parameter in restored.frozen_operation.parameters())

