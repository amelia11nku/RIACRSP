from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

import torch

from rcias_clgri.data.loader import load_instance_dict
from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.env.objective import calculate_objective
from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel

ROOT = Path(__file__).resolve().parents[1]


def _all_feature_maps(graph):
    for nodes in graph.node_features.values():
        yield from nodes.values()
    yield from (edge.features for edge in graph.edges)
    yield from graph.operation_candidates.values()
    yield from graph.island_candidates.values()
    yield from graph.w_candidates.values()
    yield from graph.f_candidates.values()


def test_nonready_operations_have_static_features_and_no_cartesian_probes(automotive_instance):
    schedule = InsertionDecoder(automotive_instance).empty_schedule()
    graph = build_graph_state(automotive_instance, schedule)
    assert graph.operation_candidates["o12"]["dynamic_feature_valid"] == 0.0
    assert graph.operation_candidates["o12"]["is_actionable"] == 0.0
    assert all(key[0] != "o12" for key in graph.w_candidates)
    assert all(key[0] != "o12" for key in graph.f_candidates)
    assert all(
        graph.island_candidates[("o12", island)]["dynamic_feature_valid"] == 0.0
        for island in automotive_instance.operation_data["o12"].eligible_islands
    )
    ready_pairs = sum(
        len(automotive_instance.operation_data[op_id].eligible_islands)
        for op_id in graph.ready_operations
    )
    assert graph.probe_stats.machine_probes == ready_pairs
    assert graph.probe_stats.f_probes == ready_pairs * len(automotive_instance.agvs_f)
    assert graph.probe_stats.w_probes <= ready_pairs * len(automotive_instance.agvs_w)
    assert graph.probe_stats.to_dict()["total_probes"] <= ready_pairs * (
        len(automotive_instance.agvs_w) + len(automotive_instance.agvs_f) + 1
    )


def test_features_are_numeric_finite_normalized_and_dimensionally_stable(automotive_instance):
    decoder = InsertionDecoder(automotive_instance)
    empty = decoder.empty_schedule()
    first = build_graph_state(automotive_instance, empty)
    decoder.commit_action(empty, decoder.probe_action(empty, Action("o11", "M1", "W1", "F1")))
    second = build_graph_state(automotive_instance, empty)
    for feature_map in _all_feature_maps(first):
        assert all(isinstance(value, (int, float)) for value in feature_map.values())
        assert all(math.isfinite(value) for value in feature_map.values())
        assert all(abs(value) <= 1.5 for value in feature_map.values())
    for node_type in first.node_features:
        first_keys = {tuple(features) for features in first.node_features[node_type].values()}
        second_keys = {tuple(features) for features in second.node_features[node_type].values()}
        assert len(first_keys) == len(second_keys) == 1
        assert first_keys == second_keys
    island_dimensions = {tuple(features) for features in first.island_candidates.values()}
    assert len(island_dimensions) == 1


def _rename_instance(raw, island_map, config_map, w_map, f_map):
    renamed = copy.deepcopy(raw)
    sets = renamed["sets"]
    sets["islands"] = [island_map[item] for item in sets["islands"]]
    sets["nodes"] = [item if item == "WH" else island_map[item] for item in sets["nodes"]]
    sets["configurations"] = [config_map[item] for item in sets["configurations"]]
    sets["agvs_w"] = [w_map[item] for item in sets["agvs_w"]]
    sets["agvs_f"] = [f_map[item] for item in sets["agvs_f"]]
    renamed["islands"] = {
        island_map[island]: {
            **data,
            "supported_configurations": [config_map[item] for item in data["supported_configurations"]],
            "initial_configuration": config_map[data["initial_configuration"]],
        }
        for island, data in raw["islands"].items()
    }
    for op_id, data in renamed["operations"].items():
        original = raw["operations"][op_id]
        data["required_configuration"] = config_map[original["required_configuration"]]
        data["eligible_islands"] = [island_map[item] for item in original["eligible_islands"]]
        data["processing_time"] = {
            island_map[island]: value for island, value in original["processing_time"].items()
        }
    for kind in ("time", "cost"):
        renamed["reconfiguration"][kind] = {
            island_map[island]: {
                config_map[source]: {
                    config_map[target]: value for target, value in targets.items()
                }
                for source, targets in sources.items()
            }
            for island, sources in raw["reconfiguration"][kind].items()
        }
    node = lambda item: item if item == "WH" else island_map[item]
    renamed["logistics"]["distance"] = {
        node(source): {node(target): value for target, value in targets.items()}
        for source, targets in raw["logistics"]["distance"].items()
    }
    for kind in ("loaded_time", "empty_time"):
        renamed["logistics"]["W"][kind] = {
            w_map[vehicle]: {
                node(source): {node(target): value for target, value in targets.items()}
                for source, targets in origins.items()
            }
            for vehicle, origins in raw["logistics"]["W"][kind].items()
        }
    for kind in ("loaded_cost_per_distance", "empty_cost_per_distance"):
        renamed["logistics"]["W"][kind] = {
            w_map[vehicle]: value for vehicle, value in raw["logistics"]["W"][kind].items()
        }
    for kind in ("outbound_time", "return_time"):
        renamed["logistics"]["F"][kind] = {
            f_map[vehicle]: {island_map[island]: value for island, value in values.items()}
            for vehicle, values in raw["logistics"]["F"][kind].items()
        }
    for kind in ("outbound_cost_per_distance", "return_cost_per_distance"):
        renamed["logistics"]["F"][kind] = {
            f_map[vehicle]: value for vehicle, value in raw["logistics"]["F"][kind].items()
        }
    renamed["meta"]["instance_id"] = "tiny_01_relabelled"
    return renamed


def _edge_counter(graph, island_map, w_map, f_map):
    maps = {"O": {}, "J": {}, "M": island_map, "W": w_map, "F": f_map}
    return Counter(
        (
            edge.source_type,
            edge.relation,
            edge.target_type,
            maps[edge.source_type].get(edge.source_id, edge.source_id),
            maps[edge.target_type].get(edge.target_id, edge.target_id),
            tuple(sorted(edge.features.items())),
        )
        for edge in graph.edges
    )


def test_configuration_island_and_agv_relabeling_preserve_numeric_semantics():
    raw = json.loads((ROOT / "instances/tiny/tiny_01.json").read_text(encoding="utf-8"))
    island_map = {old: new for old, new in zip(raw["sets"]["islands"], ("M3", "M1", "M2"))}
    config_map = {
        old: new for old, new in zip(raw["sets"]["configurations"], ("C8", "C4", "C2", "C1"))
    }
    w_map = {"W1": "W9"}
    f_map = {"F1": "F9"}
    original = load_instance_dict(raw)
    renamed = load_instance_dict(_rename_instance(raw, island_map, config_map, w_map, f_map))
    original_graph = build_graph_state(original, InsertionDecoder(original).empty_schedule())
    renamed_graph = build_graph_state(renamed, InsertionDecoder(renamed).empty_schedule())
    for node_type, identifier_map in {"O": {}, "J": {}, "M": island_map, "W": w_map, "F": f_map}.items():
        for identifier, features in original_graph.node_features[node_type].items():
            assert features == renamed_graph.node_features[node_type][identifier_map.get(identifier, identifier)]
    assert _edge_counter(original_graph, island_map, w_map, f_map) == _edge_counter(
        renamed_graph, {}, {}, {}
    )
    assert {
        island_map[island] for island, allowed in original_graph.island_masks["o11"].items() if allowed
    } == {
        island for island, allowed in renamed_graph.island_masks["o11"].items() if allowed
    }
    tensorizer = GraphTensorizer(original_graph)
    torch.manual_seed(41)
    model = RCIASNeuralModel(
        tensorizer, ModelConfig(embedding_dim=16, heads=4, layers=1)
    ).eval()
    with torch.no_grad():
        original_action = model.greedy_action(tensorizer.tensorize(original_graph))
        renamed_action = model.greedy_action(tensorizer.tensorize(renamed_graph))
    assert renamed_action == Action(
        original_action.operation_id,
        island_map[original_action.island_id],
        None if original_action.w_agv_id is None else w_map[original_action.w_agv_id],
        f_map[original_action.f_agv_id],
    )

    exact = solve_tiny_exact(original, time_limit_seconds=30.0)
    original_decoder = InsertionDecoder(original)
    renamed_decoder = InsertionDecoder(renamed)
    original_schedule = original_decoder.empty_schedule()
    renamed_schedule = renamed_decoder.empty_schedule()
    for action in exact.actions:
        original_decoder.commit_action(original_schedule, original_decoder.probe_action(original_schedule, action))
        mapped = Action(
            action.operation_id,
            island_map[action.island_id],
            None if action.w_agv_id is None else w_map[action.w_agv_id],
            f_map[action.f_agv_id],
        )
        renamed_decoder.commit_action(renamed_schedule, renamed_decoder.probe_action(renamed_schedule, mapped))
    assert calculate_objective(original, original_schedule).to_dict() == calculate_objective(
        renamed, renamed_schedule
    ).to_dict()
