"""Action-independent projection of Phase 6C target sets onto OP nodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from .schema import CSGState


FORBIDDEN_ACTION_FIELDS = {
    "counterfactual_makespan", "relative_improvement", "mean_relative_improvement",
    "rank_within_state", "regret_to_best", "repair_seed", "improvement_probability",
}


@dataclass(frozen=True)
class CSGActionView:
    state_id: str
    target_set_id: str
    graph_hash: str
    target_operation_node_indices: tuple[int, ...]
    target_mask: tuple[bool, ...]
    origin_rule_metadata: Mapping[str, object]


def project_target_set(
    graph: CSGState,
    target_set_id: str,
    target_operations: Sequence[str],
    origin_rule_metadata: Mapping[str, object] | None = None,
) -> CSGActionView:
    unique_operations = set(target_operations)
    if len(unique_operations) != len(target_operations):
        raise ValueError("target set contains duplicate operations")
    missing = sorted(operation for operation in unique_operations if operation not in graph.operation_to_node)
    if missing:
        raise KeyError(f"target operations absent from CSG: {missing}")
    operations = tuple(sorted(unique_operations, key=graph.operation_to_node.__getitem__))
    metadata = dict(origin_rule_metadata or {})
    forbidden = sorted(FORBIDDEN_ACTION_FIELDS.intersection(metadata))
    if forbidden:
        raise ValueError(f"action metadata contains label fields: {forbidden}")
    indices = tuple(graph.operation_to_node[operation] for operation in operations)
    selected = set(indices)
    return CSGActionView(
        state_id=graph.state_id,
        target_set_id=str(target_set_id),
        graph_hash=graph.graph_hash,
        target_operation_node_indices=indices,
        target_mask=tuple(index in selected for index in range(len(graph.nodes["OP"]))),
        origin_rule_metadata=metadata,
    )


def project_target_set_row(graph: CSGState, row: Mapping[str, object]) -> CSGActionView:
    if str(row["state_id"]) != graph.state_id:
        raise ValueError("target-set row belongs to a different state")
    operations = json.loads(str(row["destroyed_operation_ids"]))
    metadata = {
        key: row[key] for key in (
            "arm_family", "origin_destroy_operator", "origin_rules", "origin_families",
        ) if key in row
    }
    return project_target_set(graph, str(row["target_set_id"]), operations, metadata)
