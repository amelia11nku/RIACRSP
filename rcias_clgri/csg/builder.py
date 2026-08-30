"""Deterministic construction of a complete-schedule CSG-1.0 graph."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

from rcias_clgri.analysis.phase6a import bottleneck_proxy as infer_bottleneck_proxy
from rcias_clgri.data.instance import Instance
from rcias_clgri.data.phase6c import ReconstructedState, reconstruct_state
from rcias_clgri.env.schedule import Schedule

from .features import ExtractedFeatures, extract_node_features
from .schema import (
    CSGEdge, CSGNode, CSGState, EDGE_SPECS, EDGE_TYPE_ORDER, NODE_TYPE_ORDER,
    edge_key, node_feature_names,
)
from .serialize import attach_graph_hash
from .temporal import temporal_features


def _edge(
    source_type: str,
    relation: str,
    target_type: str,
    source_key: str,
    target_key: str,
    features: Mapping[str, float] | None = None,
) -> CSGEdge:
    key = edge_key(source_type, relation, target_type)
    spec = EDGE_SPECS[key]
    return CSGEdge(
        edge_key=key,
        source_type=source_type,
        relation=relation,
        target_type=target_type,
        source_key=source_key,
        target_key=target_key,
        edge_class=spec["class"],
        information_layer=spec["layer"],
        features=dict(features or {}),
    )


def _add(edges: dict[str, list[CSGEdge]], record: CSGEdge) -> None:
    edges[record.edge_key].append(record)


def _validate_feature_schema(extracted: ExtractedFeatures) -> None:
    for node_type in NODE_TYPE_ORDER:
        expected = set(node_feature_names(node_type))
        for key, features in extracted.nodes[node_type].items():
            actual = set(features)
            if actual != expected:
                raise ValueError(
                    f"{node_type}:{key} feature schema mismatch; "
                    f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
                )


def build_csg_from_schedule(
    instance: Instance,
    schedule: Schedule,
    *,
    state_id: str,
    search_progress: float,
    search_stage: str,
    natural_bottleneck_proxy: str | None = None,
    diagnostic_metadata: Mapping[str, object] | None = None,
) -> CSGState:
    """Build one action-independent graph from a complete feasible schedule."""
    extracted = extract_node_features(instance, schedule)
    _validate_feature_schema(extracted)
    norm = extracted.normalization
    makespan = max(record.completion_time for record in schedule.operation_schedules.values())
    nodes = {
        node_type: tuple(
            CSGNode(node_type, key, dict(features))
            for key, features in sorted(extracted.nodes[node_type].items())
        )
        for node_type in NODE_TYPE_ORDER
    }
    edges: dict[str, list[CSGEdge]] = defaultdict(list)

    for product in instance.products:
        for source, target in instance.product_data[product].precedence:
            _add(edges, _edge(
                "OP", "PRECEDES", "OP", source, target,
                temporal_features(
                    schedule.operation_schedules[source].completion_time,
                    schedule.operation_schedules[target].start_time,
                    makespan,
                ),
            ))
    for operation in instance.operations:
        operation_data = instance.operation_data[operation]
        assigned = schedule.operation_schedules[operation].island_id
        for island in operation_data.eligible_islands:
            processing = float(instance.processing_time[(operation, island)])
            _add(edges, _edge(
                "OP", "ELIGIBLE_ON", "ISLAND", operation, island,
                {
                    "processing_time": processing,
                    "processing_time_normalized": norm.processing(processing),
                },
            ))
        _add(edges, _edge("OP", "ASSIGNED_TO", "ISLAND", operation, assigned))
        _add(edges, _edge("OP", "REQUIRES", "CONFIG", operation, operation_data.required_config))
    for island in instance.islands:
        for config in instance.island_data[island].supported_configs:
            _add(edges, _edge("ISLAND", "SUPPORTS", "CONFIG", island, config))
        sequence = schedule.island_timelines[island]
        current_config = (
            instance.island_data[island].initial_config
            if not sequence else schedule.operation_schedules[sequence[-1]].config_id
        )
        _add(edges, _edge("ISLAND", "CURRENT_CONFIG", "CONFIG", island, current_config))

    for sequence in schedule.product_sequences.values():
        for source, target in zip(sequence, sequence[1:]):
            _add(edges, _edge(
                "OP", "PRODUCT_NEXT", "OP", source, target,
                temporal_features(
                    schedule.operation_schedules[source].completion_time,
                    schedule.operation_schedules[target].start_time,
                    makespan,
                ),
            ))
    for sequence in schedule.island_timelines.values():
        for source, target in zip(sequence, sequence[1:]):
            _add(edges, _edge(
                "OP", "ISLAND_NEXT", "OP", source, target,
                temporal_features(
                    schedule.operation_schedules[source].completion_time,
                    schedule.operation_schedules[target].reconfiguration_start,
                    makespan,
                ),
            ))

    for resource, tasks in schedule.w_timelines.items():
        for source, target in zip(tasks, tasks[1:]):
            _add(edges, _edge(
                "W_EVENT", "W_NEXT", "W_EVENT", source.task_id, target.task_id,
                temporal_features(source.arrival_time, target.empty_start, makespan),
            ))
        for task in tasks:
            _add(edges, _edge(
                "W_EVENT", "ENABLES", "OP", task.task_id, task.operation_id,
                temporal_features(
                    task.arrival_time,
                    schedule.operation_schedules[task.operation_id].start_time,
                    makespan,
                ),
            ))
            _add(edges, _edge("W_EVENT", "EXECUTED_BY", "W_AGV", task.task_id, resource))
            if task.predecessor_op is not None:
                _add(edges, _edge(
                    "OP", "RELEASES_WORKPIECE_TO", "W_EVENT",
                    task.predecessor_op, task.task_id,
                    temporal_features(
                        schedule.operation_schedules[task.predecessor_op].completion_time,
                        task.loaded_start,
                        makespan,
                    ),
                ))
    for resource, tasks in schedule.f_timelines.items():
        for source, target in zip(tasks, tasks[1:]):
            _add(edges, _edge(
                "F_EVENT", "F_NEXT", "F_EVENT", source.task_id, target.task_id,
                temporal_features(source.return_wh, target.departure_wh, makespan),
            ))
        for task in tasks:
            _add(edges, _edge(
                "F_EVENT", "ENABLES", "OP", task.task_id, task.operation_id,
                temporal_features(
                    task.arrival_island,
                    schedule.operation_schedules[task.operation_id].start_time,
                    makespan,
                ),
            ))
            _add(edges, _edge("F_EVENT", "EXECUTED_BY", "F_AGV", task.task_id, resource))

    for context in extracted.reconfiguration_contexts:
        _add(edges, _edge(
            "RECONF_EVENT", "ENABLES", "OP", context.key, context.operation_id,
            temporal_features(
                context.end_time,
                schedule.operation_schedules[context.operation_id].start_time,
                makespan,
            ),
        ))
        _add(edges, _edge("RECONF_EVENT", "OCCURS_ON", "ISLAND", context.key, context.island_id))
        _add(edges, _edge("RECONF_EVENT", "FROM_CONFIG", "CONFIG", context.key, context.source_config))
        _add(edges, _edge("RECONF_EVENT", "TO_CONFIG", "CONFIG", context.key, context.target_config))
        if context.previous_operation is not None:
            _add(edges, _edge(
                "OP", "TRIGGERS_RECONF", "RECONF_EVENT",
                context.previous_operation, context.key,
                temporal_features(
                    schedule.operation_schedules[context.previous_operation].completion_time,
                    context.start_time,
                    makespan,
                ),
            ))

    ordered_edges = {
        key: tuple(sorted(edges.get(key, ()), key=lambda item: (item.source_key, item.target_key)))
        for key in EDGE_TYPE_ORDER
    }
    operation_to_node = {
        node.key: index for index, node in enumerate(nodes["OP"])
    }
    metadata = {
        key: str(value) for key, value in (diagnostic_metadata or {}).items()
        if value is not None
    }
    graph = CSGState(
        schema_version="CSG-1.0",
        state_id=state_id,
        instance_id=instance.instance_id,
        nodes=nodes,
        edges=ordered_edges,
        graph_features={
            "current_makespan": float(makespan),
            "search_progress": float(search_progress),
        },
        graph_categories={
            "search_stage": str(search_stage),
            "bottleneck_proxy": str(
                natural_bottleneck_proxy or infer_bottleneck_proxy(schedule)
            ),
        },
        diagnostic_metadata=metadata,
        normalization=norm.to_dict(),
        operation_to_node=operation_to_node,
        graph_hash="",
    )
    return attach_graph_hash(graph)


def build_csg(
    reconstructed: ReconstructedState,
    state_record: Mapping[str, object],
) -> CSGState:
    """Build from the authoritative Phase 6C reconstruction result."""
    if abs(reconstructed.current_makespan - float(state_record["current_makespan"])) > 1e-9:
        raise ValueError("reconstructed state and state record disagree")
    diagnostics = {
        key: state_record.get(key) for key in ("scale", "CF_level", "RI_level", "TI_level")
    }
    return build_csg_from_schedule(
        reconstructed.instance,
        reconstructed.decoded.schedule,
        state_id=str(state_record["state_id"]),
        search_progress=reconstructed.search_progress,
        search_stage=reconstructed.search_stage,
        natural_bottleneck_proxy=str(state_record.get("bottleneck_proxy") or ""),
        diagnostic_metadata=diagnostics,
    )


def build_csg_from_record(
    state_record: Mapping[str, object],
    train_root: Path,
) -> CSGState:
    """Reconstruct through Phase 6C and build without future trajectory access."""
    reconstructed = reconstruct_state(state_record, train_root)
    return build_csg(reconstructed, state_record)
