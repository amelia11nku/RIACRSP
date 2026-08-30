"""Structural, semantic, and determinism checks for CSG-1.0 graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping, Sequence

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.data.instance import Instance
from rcias_clgri.env.schedule import Schedule

from .schema import EDGE_SPECS, EDGE_TYPE_ORDER, NODE_TYPE_ORDER, CSGState, node_feature_names
from .serialize import calculate_graph_hash
from .temporal import TEMPORAL_FEATURE_NAMES


CAUSAL_CLASSES = frozenset({
    "REALIZED_RESOURCE_ORDER", "TEMPORAL_CAUSAL", "SYNCHRONIZATION",
})
FORBIDDEN_FEATURE_NAMES = frozenset({
    "counterfactual_makespan", "future_improvement", "rank_within_state",
    "regret_to_best", "relative_improvement", "repair_outcome",
})


@dataclass(frozen=True)
class CSGValidationResult:
    """Machine-readable result of one complete validation pass."""

    checks: Mapping[str, bool]
    metrics: Mapping[str, int | float]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.violations


def _pairs(graph: CSGState, edge_type: str) -> set[tuple[str, str]]:
    return {(edge.source_key, edge.target_key) for edge in graph.edges[edge_type]}


def _chain_pairs(chains: Mapping[str, Sequence[Any]], *, task_ids: bool) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for chain in chains.values():
        for left, right in zip(chain, chain[1:]):
            result.add(
                (left.task_id, right.task_id) if task_ids else (str(left), str(right))
            )
    return result


def _exact(
    checks: dict[str, bool],
    violations: list[str],
    name: str,
    actual: set[tuple[str, str]],
    expected: set[tuple[str, str]],
) -> None:
    checks[name] = actual == expected
    if actual != expected:
        violations.append(
            f"{name}: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )


def _causal_dag(graph: CSGState) -> tuple[bool, int]:
    typed_nodes = {
        (node_type, node.key)
        for node_type, nodes in graph.nodes.items()
        for node in nodes
    }
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    indegree = {node: 0 for node in typed_nodes}
    for edges in graph.edges.values():
        for edge in edges:
            if edge.edge_class not in CAUSAL_CLASSES:
                continue
            source = (edge.source_type, edge.source_key)
            target = (edge.target_type, edge.target_key)
            if target not in adjacency[source]:
                adjacency[source].add(target)
                indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    depth = {node: 0 for node in indegree}
    visited = 0
    maximum_depth = 0
    while queue:
        node = queue.popleft()
        visited += 1
        maximum_depth = max(maximum_depth, depth[node])
        for successor in adjacency[node]:
            depth[successor] = max(depth[successor], depth[node] + 1)
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    return visited == len(typed_nodes), maximum_depth


def validate_csg(
    graph: CSGState,
    instance: Instance,
    schedule: Schedule,
    *,
    raise_on_error: bool = False,
) -> CSGValidationResult:
    """Validate graph structure and meaning against its exact source schedule."""

    checks: dict[str, bool] = {}
    violations: list[str] = []
    reconfig_operations = {
        operation_id
        for operation_id, record in schedule.operation_schedules.items()
        if record.reconfiguration_end - record.reconfiguration_start > 1e-9
    }
    expected_counts = {
        "OP": len(instance.operations),
        "ISLAND": len(instance.islands),
        "CONFIG": len(instance.configurations),
        "W_AGV": len(instance.agvs_w),
        "F_AGV": len(instance.agvs_f),
        "W_EVENT": sum(len(tasks) for tasks in schedule.w_timelines.values()),
        "F_EVENT": sum(len(tasks) for tasks in schedule.f_timelines.values()),
        "RECONF_EVENT": len(reconfig_operations),
    }
    actual_counts = {node_type: len(graph.nodes[node_type]) for node_type in NODE_TYPE_ORDER}
    checks["node_type_coverage"] = actual_counts == expected_counts
    if not checks["node_type_coverage"]:
        violations.append(f"node_type_coverage: actual={actual_counts}, expected={expected_counts}")

    node_ids = [
        (node_type, node.key)
        for node_type, nodes in graph.nodes.items()
        for node in nodes
    ]
    checks["unique_typed_node_keys"] = len(node_ids) == len(set(node_ids))
    if not checks["unique_typed_node_keys"]:
        violations.append("unique_typed_node_keys: duplicate typed node key")

    feature_schema_ok = True
    leakage_ok = True
    for node_type, nodes in graph.nodes.items():
        expected = set(node_feature_names(node_type))
        for node in nodes:
            actual = set(node.features)
            if actual != expected:
                feature_schema_ok = False
                violations.append(
                    f"node_feature_schema[{node_type}:{node.key}]: "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )
            if actual & FORBIDDEN_FEATURE_NAMES:
                leakage_ok = False
                violations.append(
                    f"no_future_or_label_features[{node_type}:{node.key}]: "
                    f"{sorted(actual & FORBIDDEN_FEATURE_NAMES)}"
                )
    checks["node_feature_schema"] = feature_schema_ok
    checks["no_future_or_label_features"] = leakage_ok

    current_features = schedule_features(instance, schedule)
    operation_feature_semantics_ok = True
    for node in graph.nodes["OP"]:
        record = schedule.operation_schedules[node.key]
        current = current_features[node.key]
        expected = {
            "processing_time": float(record.processing_time),
            "start_time": float(record.start_time),
            "completion_time": float(record.completion_time),
            "operation_slack": float(current["operation_slack"]),
            "criticality_proxy": float(current["criticality_score"]),
            "w_delay": float(current["W_waiting_or_delay_contribution"]),
            "f_delay": float(current["F_waiting_or_delay_contribution"]),
            "synchronization_wait": float(current["synchronization_wait_contribution"]),
            "local_reconfiguration": float(current["local_reconfiguration_contribution"]),
            "island_relative_load": float(current["island_relative_load"]),
            "has_w_event": float(record.w_task_id is not None),
            "has_f_event": 1.0,
        }
        operation_feature_semantics_ok &= all(
            isclose(float(node.features[name]), value, abs_tol=1e-9)
            for name, value in expected.items()
        )
        operation_feature_semantics_ok &= all(
            isclose(
                float(node.features[normalized]),
                float(node.features[raw]) / max(float(graph.graph_features["current_makespan"]), 1.0),
                abs_tol=1e-9,
            )
            for raw, normalized in (
                ("start_time", "start_time_normalized"),
                ("completion_time", "completion_time_normalized"),
                ("operation_slack", "operation_slack_normalized"),
                ("w_delay", "w_delay_normalized"),
                ("f_delay", "f_delay_normalized"),
                ("synchronization_wait", "synchronization_wait_normalized"),
            )
        )
    checks["operation_feature_semantics"] = operation_feature_semantics_ok
    if not operation_feature_semantics_ok:
        violations.append("operation_feature_semantics: frozen Phase6C/current-schedule feature mismatch")

    known_nodes = set(node_ids)
    edge_schema_ok = True
    endpoint_ok = True
    temporal_schema_ok = True
    minimum_gap = float("inf")
    maximum_gap = float("-inf")
    temporal_semantics_ok = True
    for edge_type, edges in graph.edges.items():
        spec = EDGE_SPECS[edge_type]
        expected_edge_features = set(TEMPORAL_FEATURE_NAMES) if spec["temporal"] else set(spec.get("features", ()))
        for edge in edges:
            edge_schema_ok &= (
                edge.edge_key == edge_type
                and edge.source_type == spec["source"]
                and edge.target_type == spec["target"]
                and edge.edge_class == spec["class"]
                and edge.information_layer == spec["layer"]
            )
            endpoint_ok &= (
                (edge.source_type, edge.source_key) in known_nodes
                and (edge.target_type, edge.target_key) in known_nodes
            )
            temporal_schema_ok &= set(edge.features) == expected_edge_features
            if spec["temporal"] and set(edge.features) == set(TEMPORAL_FEATURE_NAMES):
                source_end = float(edge.features["source_end_time"])
                target_start = float(edge.features["target_start_time"])
                gap = float(edge.features["temporal_gap"])
                binding = float(edge.features["binding_indicator"])
                minimum_gap = min(minimum_gap, gap)
                maximum_gap = max(maximum_gap, gap)
                temporal_semantics_ok &= (
                    isclose(gap, target_start - source_end, abs_tol=1e-9)
                    and gap >= -1e-9
                    and binding in {0.0, 1.0}
                    and isclose(binding, float(abs(gap) <= 1e-9), abs_tol=1e-9)
                    and isclose(
                        float(edge.features["normalized_temporal_gap"]),
                        gap / max(float(graph.graph_features["current_makespan"]), 1.0),
                        abs_tol=1e-9,
                    )
                )
    checks["edge_schema"] = edge_schema_ok
    checks["edge_endpoints_exist"] = endpoint_ok
    checks["edge_feature_schema"] = temporal_schema_ok
    checks["temporal_semantics"] = temporal_semantics_ok
    if not edge_schema_ok:
        violations.append("edge_schema: relation type/class/layer mismatch")
    if not endpoint_ok:
        violations.append("edge_endpoints_exist: absent typed endpoint")
    if not temporal_schema_ok:
        violations.append("edge_feature_schema: relation features differ from schema")
    if not temporal_semantics_ok:
        violations.append("temporal_semantics: negative or inconsistent temporal gap")

    _exact(checks, violations, "precedence_exact", _pairs(graph, "OP__PRECEDES__OP"), {
        pair for product in instance.products for pair in instance.product_data[product].precedence
    })
    _exact(checks, violations, "eligibility_exact", _pairs(graph, "OP__ELIGIBLE_ON__ISLAND"), {
        (operation, island)
        for operation in instance.operations
        for island in instance.operation_data[operation].eligible_islands
    })
    eligibility_feature_semantics_ok = all(
        isclose(
            float(edge.features["processing_time"]),
            float(instance.processing_time[(edge.source_key, edge.target_key)]),
            abs_tol=1e-9,
        )
        and isclose(
            float(edge.features["processing_time_normalized"]),
            float(edge.features["processing_time"]) / float(graph.normalization["mean_processing_time"]),
            abs_tol=1e-9,
        )
        for edge in graph.edges["OP__ELIGIBLE_ON__ISLAND"]
    )
    checks["eligibility_feature_semantics"] = eligibility_feature_semantics_ok
    if not eligibility_feature_semantics_ok:
        violations.append("eligibility_feature_semantics: processing-time feature mismatch")
    _exact(checks, violations, "assignment_exact", _pairs(graph, "OP__ASSIGNED_TO__ISLAND"), {
        (operation, record.island_id) for operation, record in schedule.operation_schedules.items()
    })
    _exact(checks, violations, "requirement_exact", _pairs(graph, "OP__REQUIRES__CONFIG"), {
        (operation, instance.operation_data[operation].required_config)
        for operation in instance.operations
    })
    _exact(checks, violations, "support_exact", _pairs(graph, "ISLAND__SUPPORTS__CONFIG"), {
        (island, config)
        for island in instance.islands
        for config in instance.island_data[island].supported_configs
    })
    _exact(checks, violations, "current_configuration_exact", _pairs(graph, "ISLAND__CURRENT_CONFIG__CONFIG"), {
        (
            island,
            instance.island_data[island].initial_config
            if not schedule.island_timelines[island]
            else schedule.operation_schedules[schedule.island_timelines[island][-1]].config_id,
        )
        for island in instance.islands
    })
    _exact(checks, violations, "product_chain_exact", _pairs(graph, "OP__PRODUCT_NEXT__OP"), _chain_pairs(schedule.product_sequences, task_ids=False))
    _exact(checks, violations, "island_chain_exact", _pairs(graph, "OP__ISLAND_NEXT__OP"), _chain_pairs(schedule.island_timelines, task_ids=False))
    _exact(checks, violations, "w_chain_exact", _pairs(graph, "W_EVENT__W_NEXT__W_EVENT"), _chain_pairs(schedule.w_timelines, task_ids=True))
    _exact(checks, violations, "f_chain_exact", _pairs(graph, "F_EVENT__F_NEXT__F_EVENT"), _chain_pairs(schedule.f_timelines, task_ids=True))

    w_tasks = [task for tasks in schedule.w_timelines.values() for task in tasks]
    f_tasks = [task for tasks in schedule.f_timelines.values() for task in tasks]
    _exact(checks, violations, "w_synchronization_exact", _pairs(graph, "W_EVENT__ENABLES__OP"), {
        (task.task_id, task.operation_id) for task in w_tasks
    })
    _exact(checks, violations, "f_synchronization_exact", _pairs(graph, "F_EVENT__ENABLES__OP"), {
        (task.task_id, task.operation_id) for task in f_tasks
    })
    _exact(checks, violations, "w_execution_exact", _pairs(graph, "W_EVENT__EXECUTED_BY__W_AGV"), {
        (task.task_id, task.vehicle_id) for task in w_tasks
    })
    _exact(checks, violations, "f_execution_exact", _pairs(graph, "F_EVENT__EXECUTED_BY__F_AGV"), {
        (task.task_id, task.vehicle_id) for task in f_tasks
    })
    _exact(checks, violations, "w_release_exact", _pairs(graph, "OP__RELEASES_WORKPIECE_TO__W_EVENT"), {
        (task.predecessor_op, task.task_id) for task in w_tasks if task.predecessor_op is not None
    })
    expected_reconfig_nodes = {f"R:{operation}" for operation in reconfig_operations}
    _exact(checks, violations, "reconfiguration_synchronization_exact", _pairs(graph, "RECONF_EVENT__ENABLES__OP"), {
        (f"R:{operation}", operation) for operation in reconfig_operations
    })
    _exact(checks, violations, "reconfiguration_location_exact", _pairs(graph, "RECONF_EVENT__OCCURS_ON__ISLAND"), {
        (f"R:{operation}", schedule.operation_schedules[operation].island_id)
        for operation in reconfig_operations
    })
    reconfiguration_sources: dict[str, str] = {}
    reconfiguration_targets: dict[str, str] = {}
    reconfiguration_triggers: set[tuple[str, str]] = set()
    for island, sequence in schedule.island_timelines.items():
        previous_operation = None
        previous_config = instance.island_data[island].initial_config
        for operation in sequence:
            record = schedule.operation_schedules[operation]
            if operation in reconfig_operations:
                event = f"R:{operation}"
                reconfiguration_sources[event] = previous_config
                reconfiguration_targets[event] = record.config_id
                if previous_operation is not None:
                    reconfiguration_triggers.add((previous_operation, event))
            previous_operation = operation
            previous_config = record.config_id
    _exact(checks, violations, "reconfiguration_trigger_exact", _pairs(graph, "OP__TRIGGERS_RECONF__RECONF_EVENT"), reconfiguration_triggers)
    _exact(checks, violations, "reconfiguration_source_config_exact", _pairs(graph, "RECONF_EVENT__FROM_CONFIG__CONFIG"), set(reconfiguration_sources.items()))
    _exact(checks, violations, "reconfiguration_target_config_exact", _pairs(graph, "RECONF_EVENT__TO_CONFIG__CONFIG"), set(reconfiguration_targets.items()))
    checks["reconfiguration_nodes_exact"] = {
        node.key for node in graph.nodes["RECONF_EVENT"]
    } == expected_reconfig_nodes
    if not checks["reconfiguration_nodes_exact"]:
        violations.append("reconfiguration_nodes_exact: event keys differ from positive-duration transitions")

    dag_ok, causal_depth = _causal_dag(graph)
    checks["causal_subgraph_is_dag"] = dag_ok
    if not dag_ok:
        violations.append("causal_subgraph_is_dag: cycle detected")
    checks["operation_nodes_exact"] = {node.key for node in graph.nodes["OP"]} == set(instance.operations)
    checks["w_event_nodes_exact"] = {
        node.key for node in graph.nodes["W_EVENT"]
    } == {task.task_id for task in w_tasks}
    checks["f_event_nodes_exact"] = {
        node.key for node in graph.nodes["F_EVENT"]
    } == {task.task_id for task in f_tasks}
    w_task_by_id = {task.task_id: task for task in w_tasks}
    f_task_by_id = {task.task_id: task for task in f_tasks}
    checks["w_event_feature_semantics"] = all(
        isclose(float(node.features["start_time"]), float(w_task_by_id[node.key].empty_start), abs_tol=1e-9)
        and isclose(float(node.features["end_time"]), float(w_task_by_id[node.key].arrival_time), abs_tol=1e-9)
        and float(node.features["warehouse_origin"]) == float(w_task_by_id[node.key].pickup == "WH")
        and float(node.features["first_product_transport"]) == float(w_task_by_id[node.key].predecessor_op is None)
        for node in graph.nodes["W_EVENT"]
    )
    checks["f_event_feature_semantics"] = all(
        isclose(float(node.features["start_time"]), float(f_task_by_id[node.key].departure_wh), abs_tol=1e-9)
        and isclose(float(node.features["arrival_time"]), float(f_task_by_id[node.key].arrival_island), abs_tol=1e-9)
        and isclose(float(node.features["end_time"]), float(f_task_by_id[node.key].return_wh), abs_tol=1e-9)
        for node in graph.nodes["F_EVENT"]
    )
    checks["all_relation_buckets_present"] = set(graph.edges) == set(EDGE_TYPE_ORDER)
    checks["canonical_hash"] = bool(graph.graph_hash) and graph.graph_hash == calculate_graph_hash(graph)
    if not checks["canonical_hash"]:
        violations.append("canonical_hash: stored and calculated hashes differ")
    checks["graph_state_schema"] = (
        set(graph.graph_features) == {"current_makespan", "search_progress"}
        and set(graph.graph_categories) == {"search_stage", "bottleneck_proxy"}
    )
    if not checks["graph_state_schema"]:
        violations.append("graph_state_schema: graph-level fields differ from CSG-1.0")

    metrics: dict[str, int | float] = {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "causal_depth": causal_depth,
        "minimum_temporal_gap": 0.0 if minimum_gap == float("inf") else minimum_gap,
        "maximum_temporal_gap": 0.0 if maximum_gap == float("-inf") else maximum_gap,
    }
    metrics.update({f"nodes_{name.lower()}": count for name, count in actual_counts.items()})
    metrics.update({f"edges_{name.lower()}": len(graph.edges[name]) for name in EDGE_TYPE_ORDER})
    result = CSGValidationResult(dict(sorted(checks.items())), metrics, tuple(violations))
    if raise_on_error and not result.passed:
        raise ValueError("CSG validation failed:\n" + "\n".join(result.violations))
    return result


def equivalent_under_node_mapping(
    left: CSGState,
    right: CSGState,
    mapping: Mapping[str, Mapping[str, str]],
) -> bool:
    """Compare two graphs after an explicit typed-node renaming."""

    def nodes(graph: CSGState, apply: bool) -> set[tuple[Any, ...]]:
        return {
            (
                node_type,
                mapping.get(node_type, {}).get(node.key, node.key) if apply else node.key,
                tuple(sorted(node.features.items())),
            )
            for node_type, records in graph.nodes.items()
            for node in records
        }

    def edges(graph: CSGState, apply: bool) -> set[tuple[Any, ...]]:
        def renamed(node_type: str, key: str) -> str:
            return mapping.get(node_type, {}).get(key, key) if apply else key

        return {
            (
                edge.edge_key,
                renamed(edge.source_type, edge.source_key),
                renamed(edge.target_type, edge.target_key),
                tuple(sorted(edge.features.items())),
            )
            for records in graph.edges.values()
            for edge in records
        }

    return (
        left.schema_version == right.schema_version
        and left.graph_features == right.graph_features
        and left.graph_categories == right.graph_categories
        and left.diagnostic_metadata == right.diagnostic_metadata
        and left.normalization == right.normalization
        and nodes(left, True) == nodes(right, False)
        and edges(left, True) == edges(right, False)
    )
