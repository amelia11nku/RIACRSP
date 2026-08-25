from __future__ import annotations

import torch

from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import BatchGraphTensor, GraphTensorizer, ModelConfig, RCIASNeuralModel


def test_single_vs_batch_encoder_equivalence(automotive_instance):
    decoder = InsertionDecoder(automotive_instance)
    schedule = decoder.empty_schedule()
    first_state = build_graph_state(automotive_instance, schedule)
    decoder.commit_action(
        schedule, decoder.probe_action(schedule, Action("o11", "M1", "W1", "F1"))
    )
    second_state = build_graph_state(automotive_instance, schedule)
    tensorizer = GraphTensorizer(first_state)
    graphs = [tensorizer.tensorize(first_state), tensorizer.tensorize(second_state)]
    batch = BatchGraphTensor.from_graphs(graphs)
    assert all(len(batch.graph_ptr[kind]) == 3 for kind in batch.graph_ptr)
    assert int(batch.candidate_to_graph["operation"].max()) == 1
    torch.manual_seed(19)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=32, heads=4, layers=2, dropout=0.0)
    ).eval()
    with torch.no_grad():
        singles = [model.encode(graph) for graph in graphs]
        splits = model.encode_batch(batch)
    for single, split in zip(singles, splits):
        for kind in single:
            assert torch.allclose(single[kind], split[kind], atol=2e-6, rtol=2e-6)
