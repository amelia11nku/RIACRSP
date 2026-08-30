"""CSG-1.0 data records and machine-readable schema access."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "configs/csg_v1_schema.json"


def load_schema(path: Path = SCHEMA_PATH) -> dict:
    schema = json.loads(path.read_text())
    if schema.get("version") != "CSG-1.0":
        raise ValueError("unsupported CSG schema version")
    node_types = schema.get("node_types", {})
    edge_types = schema.get("edge_types", [])
    if len(node_types) != 8 or len(edge_types) != 20:
        raise ValueError("CSG-1.0 node/edge schema is incomplete")
    keys = [edge["key"] for edge in edge_types]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate CSG edge key")
    allowed_classes = set(schema["edge_classes"])
    if any(edge["class"] not in allowed_classes for edge in edge_types):
        raise ValueError("unknown CSG edge class")
    return schema


SCHEMA = load_schema()
NODE_TYPE_ORDER = tuple(SCHEMA["node_types"])
EDGE_TYPE_ORDER = tuple(edge["key"] for edge in SCHEMA["edge_types"])
EDGE_SPECS = {edge["key"]: edge for edge in SCHEMA["edge_types"]}


def edge_key(source_type: str, relation: str, target_type: str) -> str:
    return f"{source_type}__{relation}__{target_type}"


def node_feature_names(node_type: str) -> tuple[str, ...]:
    return tuple(SCHEMA["node_types"][node_type]["features"])


def _validate_numeric(features: Mapping[str, float], context: str) -> None:
    for name, value in features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context}.{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{context}.{name} is not finite")


@dataclass(frozen=True)
class CSGNode:
    node_type: str
    key: str
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPE_ORDER:
            raise ValueError(f"unknown CSG node type: {self.node_type}")
        _validate_numeric(self.features, f"{self.node_type}:{self.key}")


@dataclass(frozen=True)
class CSGEdge:
    edge_key: str
    source_type: str
    relation: str
    target_type: str
    source_key: str
    target_key: str
    edge_class: str
    information_layer: str
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.edge_key not in EDGE_SPECS:
            raise ValueError(f"unknown CSG edge type: {self.edge_key}")
        spec = EDGE_SPECS[self.edge_key]
        expected = (spec["source"], spec["relation"], spec["target"], spec["class"], spec["layer"])
        actual = (
            self.source_type, self.relation, self.target_type,
            self.edge_class, self.information_layer,
        )
        if actual != expected:
            raise ValueError(f"edge metadata mismatch for {self.edge_key}")
        _validate_numeric(self.features, self.edge_key)


@dataclass(frozen=True)
class CSGState:
    schema_version: str
    state_id: str
    instance_id: str
    nodes: Mapping[str, tuple[CSGNode, ...]]
    edges: Mapping[str, tuple[CSGEdge, ...]]
    graph_features: Mapping[str, float]
    graph_categories: Mapping[str, str]
    diagnostic_metadata: Mapping[str, str]
    normalization: Mapping[str, float]
    operation_to_node: Mapping[str, int]
    graph_hash: str

    @property
    def node_count(self) -> int:
        return sum(len(nodes) for nodes in self.nodes.values())

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.edges.values())

    def node(self, node_type: str, key: str) -> CSGNode:
        for node in self.nodes[node_type]:
            if node.key == key:
                return node
        raise KeyError((node_type, key))
