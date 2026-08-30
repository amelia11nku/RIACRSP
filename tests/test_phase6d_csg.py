from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcias_clgri.csg import (
    build_csg,
    build_csg_from_schedule,
    csg_neighborhood_dot,
    load_schema,
    project_target_set,
    validate_csg,
)
from rcias_clgri.csg.serialize import calculate_graph_hash, canonical_json
from rcias_clgri.csg.temporal import temporal_features
from rcias_clgri.csg.validate import equivalent_under_node_mapping
from rcias_clgri.data.loader import load_instance, load_instance_dict
from rcias_clgri.data.phase6c import (
    candidate_sha256,
    candidate_to_json,
    reconstruct_state_from_instance,
)
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import Candidate, candidate_from_actions, decode_candidate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_csg():
    instance = load_instance(ROOT / "instances/tiny/tiny_01.json")
    candidate = candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    decoded = decode_candidate(instance, candidate)
    graph = build_csg_from_schedule(
        instance,
        decoded.schedule,
        state_id="tiny-state",
        search_progress=0.4,
        search_stage="40-60%",
    )
    return instance, candidate, decoded, graph


def test_csg_schema_is_complete_and_machine_readable():
    schema = load_schema()
    assert schema["version"] == "CSG-1.0"
    assert len(schema["node_types"]) == 8
    assert len(schema["edge_types"]) == 20
    assert {edge["class"] for edge in schema["edge_types"]} <= set(schema["edge_classes"])
    assert schema["action_projection"]["base_graph_shared_within_state"]


def test_deterministic_build_hash_and_canonical_serialization(tiny_csg):
    instance, _, decoded, first = tiny_csg
    second = build_csg_from_schedule(
        instance,
        decoded.schedule,
        state_id="tiny-state",
        search_progress=0.4,
        search_stage="40-60%",
    )
    assert first == second
    assert first.graph_hash == second.graph_hash == calculate_graph_hash(first)
    assert canonical_json(first) == canonical_json(second)


def test_phase6c_reconstruction_equivalence(tiny_csg):
    instance, candidate, decoded, expected = tiny_csg
    serialized = candidate_to_json(candidate)
    record = {
        "state_id": "tiny-state",
        "instance_id": instance.instance_id,
        "current_candidate": serialized,
        "candidate_sha256": candidate_sha256(serialized),
        "current_makespan": decoded.makespan,
        "search_progress": 0.4,
        "search_stage": "40-60%",
        "bottleneck_proxy": expected.graph_categories["bottleneck_proxy"],
    }
    reconstructed = reconstruct_state_from_instance(instance, record)
    actual = build_csg(reconstructed, record)
    assert actual.nodes == expected.nodes
    assert actual.edges == expected.edges
    assert actual.graph_hash == expected.graph_hash


def test_exact_relations_events_temporal_and_causal_semantics(tiny_csg):
    instance, _, decoded, graph = tiny_csg
    result = validate_csg(graph, instance, decoded.schedule)
    assert result.passed, result.violations
    assert result.checks["precedence_exact"]
    assert result.checks["eligibility_exact"]
    assert result.checks["support_exact"]
    assert result.checks["product_chain_exact"]
    assert result.checks["island_chain_exact"]
    assert result.checks["w_chain_exact"]
    assert result.checks["f_chain_exact"]
    assert result.checks["causal_subgraph_is_dag"]
    assert result.metrics["minimum_temporal_gap"] >= 0


def test_temporal_feature_keeps_signed_gap_without_clipping():
    features = temporal_features(10.0, 7.0, 20.0)
    assert features["temporal_gap"] == -3.0
    assert features["normalized_temporal_gap"] == -0.15
    assert features["binding_indicator"] == 0.0


def test_same_state_action_projection_is_graph_invariant_and_maps_operations(tiny_csg):
    _, _, _, graph = tiny_csg
    operations = tuple(graph.operation_to_node)
    first = project_target_set(graph, "a", operations[:2], {"arm_family": "TEST"})
    second = project_target_set(graph, "b", operations[-2:], {"arm_family": "TEST"})
    assert first.graph_hash == second.graph_hash == graph.graph_hash
    assert sum(first.target_mask) == sum(second.target_mask) == 2
    assert tuple(index for index, selected in enumerate(first.target_mask) if selected) == first.target_operation_node_indices
    with pytest.raises(KeyError, match="absent"):
        project_target_set(graph, "missing", ["not-an-operation"])
    with pytest.raises(ValueError, match="label fields"):
        project_target_set(graph, "leak", operations[:1], {"relative_improvement": 0.2})


def test_feature_normalization_forbidden_fields_and_debug_export(tiny_csg):
    _, _, decoded, graph = tiny_csg
    makespan = decoded.makespan
    for node in graph.nodes["OP"]:
        assert node.features["start_time_normalized"] == pytest.approx(node.features["start_time"] / max(makespan, 1))
        assert node.features["completion_time_normalized"] == pytest.approx(node.features["completion_time"] / max(makespan, 1))
        assert not any("counterfactual" in name or "improvement" in name for name in node.features)
    dot = csg_neighborhood_dot(graph, graph.nodes["OP"][0].key, hops=1)
    assert dot.startswith("digraph CSG")
    assert "gap=" in dot and "slack=" in dot


def _rename_json(value, names):
    if isinstance(value, dict):
        return {names.get(key, key): _rename_json(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_json(item, names) for item in value]
    if isinstance(value, str):
        return names.get(value, value)
    return value


def test_identifier_permutation_invariance():
    path = ROOT / "instances/tiny/tiny_01.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    original = load_instance_dict(raw)
    names = {original.instance_id: "renamed-tiny"}
    names.update({key: f"oP{index}" for index, key in enumerate(reversed(original.operations))})
    names.update({key: f"MP{index}" for index, key in enumerate(reversed(original.islands))})
    names.update({key: f"CP{index}" for index, key in enumerate(reversed(original.configurations))})
    names.update({key: f"WP{index}" for index, key in enumerate(reversed(original.agvs_w))})
    names.update({key: f"FP{index}" for index, key in enumerate(reversed(original.agvs_f))})
    names.update({key: f"JP{index}" for index, key in enumerate(reversed(original.products))})
    renamed = load_instance_dict(_rename_json(raw, names))

    candidate = candidate_from_actions(original, solve_dispatching(original, "H1").actions)
    renamed_candidate = Candidate(
        tuple(names[operation] for operation in candidate.operation_order),
        tuple(names[island] for island in candidate.island_assignment),
        tuple(names[resource] for resource in candidate.w_assignment),
        tuple(names[resource] for resource in candidate.f_assignment),
    )
    original_decoded = decode_candidate(original, candidate)
    renamed_decoded = decode_candidate(renamed, renamed_candidate)
    original_graph = build_csg_from_schedule(
        original, original_decoded.schedule, state_id="permutation", search_progress=0.2, search_stage="20-40%",
    )
    renamed_graph = build_csg_from_schedule(
        renamed, renamed_decoded.schedule, state_id="permutation", search_progress=0.2, search_stage="20-40%",
    )
    original_w = {
        task.operation_id: task.task_id
        for tasks in original_decoded.schedule.w_timelines.values()
        for task in tasks
    }
    renamed_w = {
        task.operation_id: task.task_id
        for tasks in renamed_decoded.schedule.w_timelines.values()
        for task in tasks
    }
    original_f = {
        task.operation_id: task.task_id
        for tasks in original_decoded.schedule.f_timelines.values()
        for task in tasks
    }
    renamed_f = {
        task.operation_id: task.task_id
        for tasks in renamed_decoded.schedule.f_timelines.values()
        for task in tasks
    }
    mapping = {
        "OP": {key: names[key] for key in original.operations},
        "ISLAND": {key: names[key] for key in original.islands},
        "CONFIG": {key: names[key] for key in original.configurations},
        "W_AGV": {key: names[key] for key in original.agvs_w},
        "F_AGV": {key: names[key] for key in original.agvs_f},
        "W_EVENT": {
            task_id: renamed_w[names[operation]] for operation, task_id in original_w.items()
        },
        "F_EVENT": {
            task_id: renamed_f[names[operation]] for operation, task_id in original_f.items()
        },
        "RECONF_EVENT": {
            f"R:{operation}": f"R:{names[operation]}"
            for operation in original.operations
            if f"R:{operation}" in {node.key for node in original_graph.nodes["RECONF_EVENT"]}
        },
    }
    assert validate_csg(renamed_graph, renamed, renamed_decoded.schedule).passed
    assert equivalent_under_node_mapping(original_graph, renamed_graph, mapping)


@pytest.mark.parametrize("scale", ["S", "M", "L"])
def test_small_medium_large_graph_construction(scale):
    candidates = sorted((ROOT / "instances/controlled/RCIAS-CB1-TRAIN/train").glob(f"*_{scale}_*.json"))
    instance = load_instance(candidates[0])
    candidate = candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    decoded = decode_candidate(instance, candidate)
    graph = build_csg_from_schedule(
        instance, decoded.schedule, state_id=f"scale-{scale}", search_progress=0.0, search_stage="0-20%",
    )
    assert validate_csg(graph, instance, decoded.schedule).passed
    assert len(graph.nodes["OP"]) == len(instance.operations)
