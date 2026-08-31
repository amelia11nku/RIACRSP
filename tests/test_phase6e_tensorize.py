from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.csg.schema import EDGE_TYPE_ORDER, NODE_TYPE_ORDER, CSGState
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.cache import load_shard_cache, write_shard_cache
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.tensorize import (
    BOTTLENECK_CATEGORIES,
    SEARCH_STAGES,
    CSGTensorizer,
)
from rcias_clgri.search.common import candidate_from_actions, decode_candidate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_graph() -> CSGState:
    instance = load_instance(ROOT / "instances/tiny/tiny_01.json")
    candidate = candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    decoded = decode_candidate(instance, candidate)
    return build_csg_from_schedule(
        instance,
        decoded.schedule,
        state_id="phase6e-tensor-test",
        search_progress=0.4,
        search_stage="40-60%",
    )


def test_tensorizer_preserves_frozen_feature_order_and_metadata(tiny_graph):
    tensorizer = CSGTensorizer(include_reverse=True)
    tensor = tensorizer.tensorize(tiny_graph)

    assert tuple(tensor.node_features) == NODE_TYPE_ORDER
    assert tensor.tensor_schema_hash == tensorizer.tensor_schema_hash
    assert tensor.graph_hash == tiny_graph.graph_hash
    assert tensor.operation_to_node == tiny_graph.operation_to_node
    for node_type in NODE_TYPE_ORDER:
        names = tensorizer.node_feature_names[node_type]
        expected = torch.tensor(
            [[float(node.features[name]) for name in names] for node in tiny_graph.nodes[node_type]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(tensor.node_features[node_type], expected)

    assert tensor.graph_numeric.tolist() == pytest.approx(
        [tiny_graph.graph_features["current_makespan"], 0.4]
    )
    expected_categories = [
        float(value == "40-60%") for value in SEARCH_STAGES
    ] + [
        float(value == tiny_graph.graph_categories["bottleneck_proxy"])
        for value in BOTTLENECK_CATEGORIES
    ]
    assert tensor.graph_categorical.tolist() == expected_categories


def test_canonical_and_mechanical_reverse_edges_are_exact(tiny_graph):
    tensorizer = CSGTensorizer(include_reverse=True)
    tensor = tensorizer.tensorize(tiny_graph)
    assert len(tensor.edges) == 2 * len(EDGE_TYPE_ORDER)

    for edge_key in EDGE_TYPE_ORDER:
        canonical = tensor.edges[edge_key]
        reverse = tensor.edges[f"REV__{edge_key}"]
        assert not canonical.spec.derived_reverse
        assert reverse.spec.derived_reverse
        assert reverse.spec.canonical_key == edge_key
        torch.testing.assert_close(reverse.index, canonical.index.flip(0))
        torch.testing.assert_close(reverse.features, canonical.features)
        assert canonical.features.shape[0] == canonical.index.shape[1]
        assert canonical.features.shape[1] == len(canonical.spec.edge_feature_names)


def _identifier_permutation(graph: CSGState) -> CSGState:
    names = {
        node.key: f"opaque-{node_type}-{index}"
        for node_type in NODE_TYPE_ORDER
        for index, node in enumerate(graph.nodes[node_type])
    }
    nodes = {
        node_type: tuple(replace(node, key=names[node.key]) for node in graph.nodes[node_type])
        for node_type in NODE_TYPE_ORDER
    }
    edges = {
        edge_key: tuple(
            replace(
                edge,
                source_key=names[edge.source_key],
                target_key=names[edge.target_key],
            )
            for edge in graph.edges[edge_key]
        )
        for edge_key in EDGE_TYPE_ORDER
    }
    operation_to_node = {
        names[operation]: index for operation, index in graph.operation_to_node.items()
    }
    return replace(
        graph,
        state_id="opaque-state",
        instance_id="opaque-instance",
        nodes=nodes,
        edges=edges,
        operation_to_node=operation_to_node,
        graph_hash="opaque-lookup-hash",
    )


def test_identifier_values_never_enter_predictive_tensors(tiny_graph):
    tensorizer = CSGTensorizer(include_reverse=True)
    original = tensorizer.tensorize(tiny_graph)
    renamed = tensorizer.tensorize(_identifier_permutation(tiny_graph))

    for node_type in NODE_TYPE_ORDER:
        torch.testing.assert_close(
            original.node_features[node_type], renamed.node_features[node_type]
        )
    for relation in original.edges:
        torch.testing.assert_close(original.edges[relation].index, renamed.edges[relation].index)
        torch.testing.assert_close(
            original.edges[relation].features, renamed.edges[relation].features
        )
    torch.testing.assert_close(original.graph_numeric, renamed.graph_numeric)
    torch.testing.assert_close(original.graph_categorical, renamed.graph_categorical)
    schema_text = str(tensorizer.schema_record()).lower()
    assert "candidate" not in schema_text
    assert "counterfactual" not in schema_text
    assert "relative_improvement" not in schema_text


def test_no_reverse_mode_and_empty_relations_have_stable_shapes(tiny_graph):
    tensorizer = CSGTensorizer(include_reverse=False)
    tensor = tensorizer.tensorize(tiny_graph)
    assert tuple(tensor.edges) == EDGE_TYPE_ORDER
    for edge in tensor.edges.values():
        assert edge.index.shape == (2, edge.features.shape[0])
        assert edge.features.ndim == 2
    assert tensor.tensor_bytes() > 0
    assert tensor.to("cpu").tensor_bytes() == tensor.tensor_bytes()
    assert CSGTensorizer(include_reverse=False).tensor_schema_hash == tensor.tensor_schema_hash


def test_unknown_graph_category_is_rejected(tiny_graph):
    tensorizer = CSGTensorizer()
    invalid = replace(
        tiny_graph,
        graph_categories={**tiny_graph.graph_categories, "search_stage": "future-stage"},
    )
    with pytest.raises(ValueError, match="unknown search_stage"):
        tensorizer.tensorize(invalid)


def _action_row(graph: CSGState, target_set_id: str, operations, utility: float, rank: int):
    return {
        "state_id": graph.state_id,
        "target_set_id": target_set_id,
        "destroyed_operation_ids": __import__("json").dumps(operations),
        "mean_relative_improvement": utility,
        "rank_within_state": rank,
        "rank_percentile": 1.0 if rank == 1 else 0.0,
        "regret_to_best": 0.0 if rank == 1 else 0.1,
        "top1": rank == 1,
        "top3": True,
        "arm_family": "ORIGINAL_OPERATOR",
        "origin_destroy_operator": "related" if rank == 1 else "random",
        "origin_rules": "[]",
        "origin_families": '["ORIGINAL_OPERATOR"]',
    }


def test_action_projection_keeps_all_candidates_and_labels_separate(tiny_graph):
    operations = tuple(tiny_graph.operation_to_node)
    rows = [
        _action_row(tiny_graph, "best", operations[2::-1], 0.2, 1),
        _action_row(tiny_graph, "other", operations[-2:], -0.1, 2),
    ]
    actions = tensorize_action_records(tiny_graph, rows)
    assert actions.target_set_ids == ("best", "other")
    assert actions.action_count == 2
    assert actions.action_ptr.tolist() == [0, 3, 5]
    assert actions.target_operation_indices[:3].tolist() == [0, 1, 2]
    assert actions.target_action_index.tolist() == [0, 0, 0, 1, 1]
    assert actions.utility.tolist() == pytest.approx([0.2, -0.1])
    assert actions.positive.tolist() == [1.0, 0.0]
    assert actions.origin_destroy_operator == ("related", "random")


def test_state_batch_reuses_each_graph_once_and_offsets_actions(tiny_graph):
    operations = tuple(tiny_graph.operation_to_node)
    rows = [
        _action_row(tiny_graph, "a", operations[:2], 0.2, 1),
        _action_row(tiny_graph, "b", operations[-2:], -0.1, 2),
    ]
    tensorizer = CSGTensorizer()
    first = NIStateSample(
        tensorizer.tensorize(tiny_graph),
        tensorize_action_records(tiny_graph, rows),
        {"scale": "S"},
    )
    second = replace(
        first,
        graph=replace(first.graph, state_id="second", instance_id="second-instance"),
    )
    batch = batch_state_samples([first, second])

    assert batch.state_count == 2
    assert batch.action_count == 4
    assert batch.action_ptr.tolist() == [0, 2, 4]
    assert batch.action_to_state.tolist() == [0, 0, 1, 1]
    assert batch.node_features["OP"].shape[0] == 2 * len(operations)
    assert batch.node_features["OP"].shape[0] != batch.action_count * len(operations)
    first_memberships = first.actions.membership_count
    assert batch.target_operation_indices[first_memberships:].min().item() >= len(operations)
    assert batch.target_action_index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    for edge in batch.edges.values():
        assert edge.index.shape[1] == edge.features.shape[0]
    assert batch.to("cpu").tensor_bytes() == batch.tensor_bytes()


def test_action_projection_rejects_cross_state_and_duplicates(tiny_graph):
    operations = tuple(tiny_graph.operation_to_node)
    row = _action_row(tiny_graph, "duplicate", operations[:2], 0.1, 1)
    with pytest.raises(ValueError, match="duplicate target_set_id"):
        tensorize_action_records(tiny_graph, [row, row])
    wrong = {**row, "state_id": "wrong"}
    with pytest.raises(ValueError, match="different state"):
        tensorize_action_records(tiny_graph, [wrong])


def test_versioned_shard_cache_round_trip_and_hash_guards(tiny_graph, tmp_path):
    operations = tuple(tiny_graph.operation_to_node)
    actions = tensorize_action_records(
        tiny_graph,
        [_action_row(tiny_graph, "cached", operations[:2], 0.1, 1)],
    )
    sample = NIStateSample(CSGTensorizer().tensorize(tiny_graph), actions, {"scale": "S"})
    path = tmp_path / "tiny.pt"
    record = write_shard_cache(
        path,
        [sample],
        instance_id=tiny_graph.instance_id,
        training_split="TRAIN",
        source_shard_sha256="source-hash",
    )
    loaded, metadata = load_shard_cache(
        path,
        expected_tensor_schema_hash=sample.graph.tensor_schema_hash,
        expected_source_shard_sha256="source-hash",
    )
    assert record["status"] == "COMPLETE"
    assert metadata["state_count"] == 1
    assert loaded[0].graph.graph_hash == sample.graph.graph_hash
    torch.testing.assert_close(loaded[0].actions.utility, actions.utility)
    with pytest.raises(ValueError, match="source shard mismatch"):
        load_shard_cache(path, expected_source_shard_sha256="wrong")
