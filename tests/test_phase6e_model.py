from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.encoder import NIModelConfig, STATIC_CANONICAL_RELATIONS
from rcias_clgri.ni.losses import NILossConfig, phase6e_loss
from rcias_clgri.ni.scorer import CSGTargetSetScorer
from rcias_clgri.ni.tensorize import CSGTensorizer, NIEdgeTensor
from rcias_clgri.search.common import candidate_from_actions, decode_candidate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_batch():
    instance = load_instance(ROOT / "instances/tiny/tiny_01.json")
    candidate = candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    decoded = decode_candidate(instance, candidate)
    graph = build_csg_from_schedule(
        instance,
        decoded.schedule,
        state_id="phase6e-model-test",
        search_progress=0.4,
        search_stage="40-60%",
    )
    operations = tuple(graph.operation_to_node)
    rows = []
    for index, selected in enumerate((operations[:2], operations[1:4], operations[-3:])):
        utility = (0.2, 0.05, -0.1)[index]
        rows.append({
            "state_id": graph.state_id,
            "target_set_id": f"target-{index}",
            "destroyed_operation_ids": json.dumps(selected),
            "mean_relative_improvement": utility,
            "rank_within_state": index + 1,
            "rank_percentile": (1.0, 0.5, 0.0)[index],
            "regret_to_best": 0.2 - utility,
            "top1": index == 0,
            "top3": True,
            "arm_family": "ORIGINAL_OPERATOR",
            "origin_destroy_operator": ("related", "critical", "random")[index],
        })
    tensorizer = CSGTensorizer()
    sample = NIStateSample(
        tensorizer.tensorize(graph),
        tensorize_action_records(graph, rows),
        {"scale": "S"},
    )
    return tensorizer, batch_state_samples([sample])


def test_full_csg_model_forward_backward_and_single_score(model_batch):
    tensorizer, batch = model_batch
    torch.manual_seed(6101)
    model = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(hidden_dim=32, layers=1, heads=4, dropout=0.0),
    )
    output = model(batch)
    assert output.scores.shape == (batch.action_count,)
    assert output.graph_embeddings.shape == (batch.state_count, 32)
    assert output.action_embeddings.shape == (batch.action_count, 32)
    losses = phase6e_loss(output.scores, batch, NILossConfig())
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["pair_count"].item() > 0
    edge_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if ".edge_key." in name and parameter.grad is not None
    ]
    assert edge_gradients
    assert any(torch.count_nonzero(gradient).item() for gradient in edge_gradients)


def test_edge_features_are_consumed_and_no_edge_ablation_is_explicit(model_batch):
    tensorizer, batch = model_batch
    torch.manual_seed(6102)
    model = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(hidden_dim=32, layers=1, heads=4, dropout=0.0),
    ).eval()
    zero_edges = {
        key: NIEdgeTensor(edge.spec, edge.index, torch.zeros_like(edge.features))
        for key, edge in batch.edges.items()
    }
    with torch.no_grad():
        regular = model(batch).scores
        zeroed = model(replace(batch, edges=zero_edges)).scores
    assert not torch.allclose(regular, zeroed)

    no_edge = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(
            hidden_dim=32, layers=1, heads=4, dropout=0.0, use_edge_features=False
        ),
    )
    assert len(no_edge.state_encoder.layers[0].edge_key) == 0
    assert no_edge(batch).scores.shape == regular.shape


def test_static_and_flat_ablations_have_the_declared_topology(model_batch):
    tensorizer, batch = model_batch
    static = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(
            hidden_dim=32, layers=1, heads=4, dropout=0.0, relation_mode="STATIC_CSG"
        ),
    )
    assert {
        tensorizer_relation.canonical_key
        for tensorizer_relation in tensorizer.relation_specs
        if tensorizer_relation.key in static.state_encoder.relation_keys
    } == STATIC_CANONICAL_RELATIONS
    assert len(static.state_encoder.relation_keys) == 2 * len(STATIC_CANONICAL_RELATIONS)
    assert static(batch).scores.shape == (batch.action_count,)

    flat = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(
            hidden_dim=32, layers=1, heads=4, dropout=0.0, message_passing=False
        ),
    )
    assert len(flat.state_encoder.layers) == 0
    assert flat(batch).scores.shape == (batch.action_count,)


def test_origin_identity_is_not_a_primary_model_input(model_batch):
    tensorizer, batch = model_batch
    torch.manual_seed(6103)
    model = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(hidden_dim=32, layers=1, heads=4, dropout=0.0),
    ).eval()
    changed_metadata = replace(
        batch,
        origin_destroy_operator=tuple("leak" for _ in batch.origin_destroy_operator),
        arm_family=tuple("leak" for _ in batch.arm_family),
    )
    with torch.no_grad():
        torch.testing.assert_close(model(batch).scores, model(changed_metadata).scores)


def test_pairwise_loss_has_correct_utility_direction(model_batch):
    _, batch = model_batch
    aligned = phase6e_loss(10 * batch.utility, batch, NILossConfig())
    reversed_scores = phase6e_loss(-10 * batch.utility, batch, NILossConfig())
    assert aligned["rank_loss"] < reversed_scores["rank_loss"]
