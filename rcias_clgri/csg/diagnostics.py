"""Framework-neutral CSG-1.0 inspection helpers."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from .schema import CSGState


def graph_diagnostics(graph: CSGState) -> dict[str, Any]:
    """Return compact structural and synchronization statistics."""

    edge_counts = Counter({key: len(edges) for key, edges in graph.edges.items()})
    node_counts = {key: len(nodes) for key, nodes in graph.nodes.items()}
    binding = Counter()
    incoming_sync = Counter()
    gaps: list[float] = []
    for records in graph.edges.values():
        for edge in records:
            if "temporal_gap" in edge.features:
                gaps.append(float(edge.features["temporal_gap"]))
                if float(edge.features["binding_indicator"]) == 1.0:
                    binding[edge.edge_key] += 1
            if edge.relation == "ENABLES" and edge.target_type == "OP":
                incoming_sync[edge.target_key] += 1
    operation_count = max(node_counts["OP"], 1)
    return {
        "graph_hash": graph.graph_hash,
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "binding_edges_by_relation": dict(sorted(binding.items())),
        "operations_with_w_sync_ratio": edge_counts["W_EVENT__ENABLES__OP"] / operation_count,
        "operations_with_reconfiguration_ratio": edge_counts["RECONF_EVENT__ENABLES__OP"] / operation_count,
        "maximum_synchronization_fan_in": max(incoming_sync.values(), default=0),
        "minimum_temporal_gap": min(gaps, default=0.0),
        "maximum_temporal_gap": max(gaps, default=0.0),
    }


def csg_neighborhood_dot(graph: CSGState, operation_id: str, hops: int = 1) -> str:
    """Render a bounded Graphviz neighborhood for semantic debugging."""

    if hops < 0:
        raise ValueError("hops must be non-negative")
    center = ("OP", operation_id)
    nodes = {
        (node_type, node.key): node
        for node_type, records in graph.nodes.items()
        for node in records
    }
    if center not in nodes:
        raise KeyError(f"unknown operation node: {operation_id}")
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for records in graph.edges.values():
        for edge in records:
            source = (edge.source_type, edge.source_key)
            target = (edge.target_type, edge.target_key)
            adjacency[source].add(target)
            adjacency[target].add(source)
    selected = {center}
    queue = deque([(center, 0)])
    while queue:
        current, distance = queue.popleft()
        if distance == hops:
            continue
        for neighbor in adjacency[current]:
            if neighbor not in selected:
                selected.add(neighbor)
                queue.append((neighbor, distance + 1))
    aliases = {key: f"n{index}" for index, key in enumerate(sorted(selected))}
    lines = ["digraph CSG {", "  rankdir=LR;"]
    for typed_key in sorted(selected):
        node = nodes[typed_key]
        label = f"{node.node_type}\\n{node.key}"
        if node.node_type == "OP":
            label += f"\\nstart={node.features['start_time']:.3f}"
            label += f"\\nslack={node.features['operation_slack']:.3f}"
        lines.append(f'  {aliases[typed_key]} [label="{label}"];')
    for records in graph.edges.values():
        for edge in records:
            source = (edge.source_type, edge.source_key)
            target = (edge.target_type, edge.target_key)
            if source not in selected or target not in selected:
                continue
            label = edge.relation
            if "temporal_gap" in edge.features:
                label += f"\\ngap={edge.features['temporal_gap']:.3f}"
            lines.append(f'  {aliases[source]} -> {aliases[target]} [label="{label}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_csg_neighborhood(
    graph: CSGState,
    operation_id: str,
    path: str | Path,
    *,
    hops: int = 1,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(csg_neighborhood_dot(graph, operation_id, hops), encoding="utf-8")
    return output
