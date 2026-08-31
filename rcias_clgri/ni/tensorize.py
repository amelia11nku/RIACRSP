"""Production native-PyTorch tensorization of frozen CSG-1.0 graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Sequence

import torch

from rcias_clgri.csg.schema import (
    EDGE_SPECS,
    EDGE_TYPE_ORDER,
    NODE_TYPE_ORDER,
    SCHEMA,
    CSGState,
    node_feature_names,
)
from rcias_clgri.csg.temporal import TEMPORAL_FEATURE_NAMES


SEARCH_STAGES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
BOTTLENECK_CATEGORIES = (
    "PRECEDENCE_SEQUENCE",
    "ISLAND_PROCESSING_LOAD",
    "RECONFIGURATION",
    "W_LOGISTICS",
    "F_LOGISTICS",
    "CROSS_RESOURCE_SYNCHRONIZATION",
    "MIXED_OR_UNCERTAIN",
)
GRAPH_NUMERIC_NAMES = ("current_makespan", "search_progress")
GRAPH_CATEGORICAL_NAMES = tuple(
    [f"search_stage={value}" for value in SEARCH_STAGES]
    + [f"bottleneck_proxy={value}" for value in BOTTLENECK_CATEGORIES]
)


@dataclass(frozen=True)
class NIRelationSpec:
    key: str
    source_type: str
    relation: str
    target_type: str
    edge_feature_names: tuple[str, ...]
    edge_class: str
    information_layer: str
    derived_reverse: bool
    canonical_key: str


@dataclass(frozen=True)
class NIEdgeTensor:
    spec: NIRelationSpec
    index: torch.Tensor
    features: torch.Tensor

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "NIEdgeTensor":
        return NIEdgeTensor(
            self.spec,
            self.index.to(device, non_blocking=non_blocking),
            self.features.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class NITensorGraph:
    """One action-independent CSG tensor graph; identifiers remain lookup-only."""

    state_id: str
    instance_id: str
    graph_hash: str
    tensor_schema_hash: str
    node_features: Mapping[str, torch.Tensor]
    node_keys: Mapping[str, tuple[str, ...]]
    node_index: Mapping[str, Mapping[str, int]]
    edges: Mapping[str, NIEdgeTensor]
    graph_numeric: torch.Tensor
    graph_categorical: torch.Tensor
    operation_to_node: Mapping[str, int]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "NITensorGraph":
        return replace(
            self,
            node_features={
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.node_features.items()
            },
            edges={
                key: edge.to(device, non_blocking=non_blocking)
                for key, edge in self.edges.items()
            },
            graph_numeric=self.graph_numeric.to(device, non_blocking=non_blocking),
            graph_categorical=self.graph_categorical.to(device, non_blocking=non_blocking),
        )

    def tensor_bytes(self) -> int:
        tensors = [
            *self.node_features.values(),
            *(tensor for edge in self.edges.values() for tensor in (edge.index, edge.features)),
            self.graph_numeric,
            self.graph_categorical,
        ]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _canonical_relation_specs(include_reverse: bool) -> tuple[NIRelationSpec, ...]:
    canonical = []
    reverse = []
    for key in EDGE_TYPE_ORDER:
        raw = EDGE_SPECS[key]
        features = (
            tuple(TEMPORAL_FEATURE_NAMES)
            if raw["temporal"]
            else tuple(raw.get("features", ()))
        )
        canonical.append(NIRelationSpec(
            key=key,
            source_type=raw["source"],
            relation=raw["relation"],
            target_type=raw["target"],
            edge_feature_names=features,
            edge_class=raw["class"],
            information_layer=raw["layer"],
            derived_reverse=False,
            canonical_key=key,
        ))
        if include_reverse:
            reverse.append(NIRelationSpec(
                key=f"REV__{key}",
                source_type=raw["target"],
                relation=f"REV_{raw['relation']}",
                target_type=raw["source"],
                edge_feature_names=features,
                edge_class="DERIVED_REVERSE",
                information_layer=raw["layer"],
                derived_reverse=True,
                canonical_key=key,
            ))
    return tuple([*canonical, *reverse])


class CSGTensorizer:
    """Frozen-schema tensorizer with optional mechanical reverse relations."""

    def __init__(self, *, include_reverse: bool = True, dtype: torch.dtype = torch.float32) -> None:
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("CSG tensorization supports float32 or float64")
        self.include_reverse = include_reverse
        self.dtype = dtype
        self.node_feature_names = {
            node_type: node_feature_names(node_type) for node_type in NODE_TYPE_ORDER
        }
        self.relation_specs = _canonical_relation_specs(include_reverse)
        self.graph_numeric_names = GRAPH_NUMERIC_NAMES
        self.graph_categorical_names = GRAPH_CATEGORICAL_NAMES
        self.tensor_schema_hash = self._schema_hash()

    @property
    def node_input_dims(self) -> dict[str, int]:
        return {key: len(names) for key, names in self.node_feature_names.items()}

    @property
    def graph_input_dim(self) -> int:
        return len(self.graph_numeric_names) + len(self.graph_categorical_names)

    def _schema_payload(self) -> dict[str, object]:
        return {
            "schema": "phase6e-csg-tensor-schema-v1",
            "csg_schema_version": SCHEMA["version"],
            "include_reverse": self.include_reverse,
            "dtype": str(self.dtype),
            "node_feature_names": {
                key: list(value) for key, value in self.node_feature_names.items()
            },
            "relations": [
                {
                    "key": spec.key,
                    "source_type": spec.source_type,
                    "relation": spec.relation,
                    "target_type": spec.target_type,
                    "edge_feature_names": list(spec.edge_feature_names),
                    "edge_class": spec.edge_class,
                    "information_layer": spec.information_layer,
                    "derived_reverse": spec.derived_reverse,
                    "canonical_key": spec.canonical_key,
                }
                for spec in self.relation_specs
            ],
            "graph_numeric_names": list(self.graph_numeric_names),
            "graph_categorical_names": list(self.graph_categorical_names),
            "identifier_policy": "lookup-only; no identifier value tensor",
        }

    def _schema_hash(self) -> str:
        encoded = json.dumps(
            self._schema_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def schema_record(self) -> dict[str, object]:
        return {**self._schema_payload(), "tensor_schema_hash": self.tensor_schema_hash}

    @staticmethod
    def _feature_matrix(
        records: Sequence,
        names: tuple[str, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not records:
            return torch.empty((0, len(names)), dtype=dtype)
        values = [[float(record.features[name]) for name in names] for record in records]
        return torch.tensor(values, dtype=dtype)

    @staticmethod
    def _one_hot(value: str, vocabulary: tuple[str, ...], field: str) -> list[float]:
        if value not in vocabulary:
            raise ValueError(f"unknown {field}: {value}")
        return [float(candidate == value) for candidate in vocabulary]

    def tensorize(self, graph: CSGState) -> NITensorGraph:
        if graph.schema_version != "CSG-1.0":
            raise ValueError(f"expected CSG-1.0, got {graph.schema_version}")
        node_keys = {
            node_type: tuple(node.key for node in graph.nodes[node_type])
            for node_type in NODE_TYPE_ORDER
        }
        node_index = {
            node_type: {key: index for index, key in enumerate(keys)}
            for node_type, keys in node_keys.items()
        }
        node_features = {
            node_type: self._feature_matrix(
                graph.nodes[node_type], self.node_feature_names[node_type], self.dtype
            )
            for node_type in NODE_TYPE_ORDER
        }
        edges: dict[str, NIEdgeTensor] = {}
        for spec in self.relation_specs:
            records = graph.edges[spec.canonical_key]
            if records:
                if spec.derived_reverse:
                    sources = [node_index[spec.source_type][edge.target_key] for edge in records]
                    targets = [node_index[spec.target_type][edge.source_key] for edge in records]
                else:
                    sources = [node_index[spec.source_type][edge.source_key] for edge in records]
                    targets = [node_index[spec.target_type][edge.target_key] for edge in records]
                index = torch.tensor([sources, targets], dtype=torch.long)
                features = self._feature_matrix(records, spec.edge_feature_names, self.dtype)
            else:
                index = torch.empty((2, 0), dtype=torch.long)
                features = torch.empty((0, len(spec.edge_feature_names)), dtype=self.dtype)
            edges[spec.key] = NIEdgeTensor(spec, index, features)

        graph_numeric = torch.tensor(
            [float(graph.graph_features[name]) for name in GRAPH_NUMERIC_NAMES],
            dtype=self.dtype,
        )
        graph_categorical = torch.tensor(
            [
                *self._one_hot(
                    graph.graph_categories["search_stage"], SEARCH_STAGES, "search_stage"
                ),
                *self._one_hot(
                    graph.graph_categories["bottleneck_proxy"],
                    BOTTLENECK_CATEGORIES,
                    "bottleneck_proxy",
                ),
            ],
            dtype=self.dtype,
        )
        if graph.operation_to_node != node_index["OP"]:
            raise ValueError("CSG operation mapping disagrees with canonical OP tensor order")
        return NITensorGraph(
            state_id=graph.state_id,
            instance_id=graph.instance_id,
            graph_hash=graph.graph_hash,
            tensor_schema_hash=self.tensor_schema_hash,
            node_features=node_features,
            node_keys=node_keys,
            node_index=node_index,
            edges=edges,
            graph_numeric=graph_numeric,
            graph_categorical=graph_categorical,
            operation_to_node=dict(graph.operation_to_node),
        )
