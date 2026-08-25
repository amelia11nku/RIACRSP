from __future__ import annotations

import torch

from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.learning.buffer import RolloutBuffer, RolloutTransition
from rcias_clgri.nn import GraphTensorizer


def test_rollout_buffer_keeps_cpu_observations_and_computes_advantages(automotive_instance):
    state = build_graph_state(
        automotive_instance, InsertionDecoder(automotive_instance).empty_schedule()
    )
    graph = GraphTensorizer(state).tensorize(state)
    action = Action("o11", "M1", "W1", "F1")
    buffer = RolloutBuffer()
    buffer.add(RolloutTransition(
        graph, action, -1.0, {"operation": -1.0, "island": 0.0, "w": 0.0, "f": 0.0},
        0.2, -0.1, True, automotive_instance.instance_id, 0,
    ))
    batch = buffer.compute_advantages()
    assert len(buffer) == 1
    assert torch.isfinite(batch.returns).all()
    assert buffer.advantages.item() == 0.0
    buffer.clear()
    assert len(buffer) == 0
