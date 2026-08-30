"""Canonical serialization, hashing, and table export for CSG-1.0."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .schema import CSGState, EDGE_TYPE_ORDER, NODE_TYPE_ORDER


def canonical_payload(graph: CSGState) -> dict:
    return {
        "schema_version": graph.schema_version,
        "state_id": graph.state_id,
        "instance_id": graph.instance_id,
        "nodes": {
            node_type: [
                {"key": node.key, "features": dict(sorted(node.features.items()))}
                for node in sorted(graph.nodes.get(node_type, ()), key=lambda item: item.key)
            ]
            for node_type in NODE_TYPE_ORDER
        },
        "edges": {
            key: [
                {
                    "source_key": edge.source_key,
                    "target_key": edge.target_key,
                    "features": dict(sorted(edge.features.items())),
                }
                for edge in sorted(
                    graph.edges.get(key, ()),
                    key=lambda item: (item.source_key, item.target_key, tuple(sorted(item.features.items()))),
                )
            ]
            for key in EDGE_TYPE_ORDER
        },
        "graph_features": dict(sorted(graph.graph_features.items())),
        "graph_categories": dict(sorted(graph.graph_categories.items())),
        "diagnostic_metadata": dict(sorted(graph.diagnostic_metadata.items())),
        "normalization": dict(sorted(graph.normalization.items())),
        "operation_to_node": dict(sorted(graph.operation_to_node.items())),
    }


def canonical_json(graph: CSGState) -> str:
    return json.dumps(canonical_payload(graph), sort_keys=True, separators=(",", ":"), allow_nan=False)


def calculate_graph_hash(graph: CSGState) -> str:
    return hashlib.sha256(canonical_json(graph).encode()).hexdigest()


def attach_graph_hash(graph: CSGState) -> CSGState:
    return replace(graph, graph_hash=calculate_graph_hash(graph))


def export_csg_tables(graph: CSGState, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for node_type in NODE_TYPE_ORDER:
        nodes = graph.nodes.get(node_type, ())
        fields = sorted({name for node in nodes for name in node.features})
        with (output / f"nodes_{node_type}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["node_key", *fields])
            writer.writeheader()
            for node in sorted(nodes, key=lambda item: item.key):
                writer.writerow({"node_key": node.key, **node.features})
    for key in EDGE_TYPE_ORDER:
        edges = graph.edges.get(key, ())
        fields = sorted({name for edge in edges for name in edge.features})
        with (output / f"edges_{key}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_key", "target_key", *fields])
            writer.writeheader()
            for edge in sorted(edges, key=lambda item: (item.source_key, item.target_key)):
                writer.writerow({"source_key": edge.source_key, "target_key": edge.target_key, **edge.features})
    payload = canonical_payload(graph) | {"graph_hash": graph.graph_hash}
    (output / "graph.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
