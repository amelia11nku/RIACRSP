"""Convert pure-Python graph states into typed PyTorch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from rcias_clgri.graph.builder import GraphState

NODE_TYPES = ("O", "J", "M", "W", "F")
RELATION_TYPES = (
    ("J", "contains", "O"), ("O", "belongs_to", "J"),
    ("O", "precedence", "O"), ("O", "precedence_rev", "O"),
    ("O", "eligible_on", "M"), ("M", "can_process", "O"),
    ("M", "spatial", "M"),
    ("W", "reachable_to", "M"), ("M", "reachable_by", "W"),
    ("F", "deliver_to", "M"), ("M", "served_by", "F"),
    ("O", "actual_product_prev", "O"), ("O", "machine_prev", "O"),
)


def relation_key(source_type: str, relation: str, target_type: str) -> str:
    return f"{source_type}__{relation}__{target_type}"


@dataclass(frozen=True)
class EdgeTensor:
    source_type: str
    relation: str
    target_type: str
    index: torch.Tensor
    features: torch.Tensor


@dataclass(frozen=True)
class TensorizedCandidates:
    operation_features: Mapping[str, torch.Tensor]
    island_features: Mapping[tuple[str, str], torch.Tensor]
    w_features: Mapping[tuple[str, str, str | None], torch.Tensor]
    f_features: Mapping[tuple[str, str, str], torch.Tensor]
    ready_operations: tuple[str, ...]
    island_masks: Mapping[str, Mapping[str, bool]]
    w_masks: Mapping[tuple[str, str], tuple[str | None, ...]]
    f_masks: Mapping[tuple[str, str], tuple[str, ...]]


@dataclass(frozen=True)
class GraphTensor:
    node_features: Mapping[str, torch.Tensor]
    node_ids: Mapping[str, tuple[str, ...]]
    node_index: Mapping[str, Mapping[str, int]]
    edges: Mapping[str, EdgeTensor]
    candidates: TensorizedCandidates

    def to(self, device: torch.device | str) -> "GraphTensor":
        return GraphTensor(
            node_features={kind: value.to(device) for kind, value in self.node_features.items()},
            node_ids=self.node_ids,
            node_index=self.node_index,
            edges={
                key: EdgeTensor(
                    edge.source_type, edge.relation, edge.target_type,
                    edge.index.to(device), edge.features.to(device),
                )
                for key, edge in self.edges.items()
            },
            candidates=TensorizedCandidates(
                operation_features={key: value.to(device) for key, value in self.candidates.operation_features.items()},
                island_features={key: value.to(device) for key, value in self.candidates.island_features.items()},
                w_features={key: value.to(device) for key, value in self.candidates.w_features.items()},
                f_features={key: value.to(device) for key, value in self.candidates.f_features.items()},
                ready_operations=self.candidates.ready_operations,
                island_masks=self.candidates.island_masks,
                w_masks=self.candidates.w_masks,
                f_masks=self.candidates.f_masks,
            ),
        )


@dataclass(frozen=True)
class BatchGraphTensor:
    """Disjoint union of heterogeneous graphs for one encoder minibatch."""

    node_features: Mapping[str, torch.Tensor]
    node_ids: Mapping[str, tuple[str, ...]]
    node_index: Mapping[str, Mapping[str, int]]
    edges: Mapping[str, EdgeTensor]
    graphs: tuple[GraphTensor, ...]
    graph_ptr: Mapping[str, torch.Tensor]
    node_batch_index: Mapping[str, torch.Tensor]
    candidate_to_graph: Mapping[str, torch.Tensor]

    @classmethod
    def from_graphs(cls, graphs: Sequence[GraphTensor]) -> "BatchGraphTensor":
        items = tuple(graphs)
        if not items:
            raise ValueError("cannot batch an empty graph sequence")
        edge_keys = tuple(items[0].edges)
        if any(tuple(graph.edges) != edge_keys for graph in items[1:]):
            raise ValueError("all graphs must share one relation schema")
        node_features: dict[str, torch.Tensor] = {}
        node_ids: dict[str, tuple[str, ...]] = {}
        node_index: dict[str, dict[str, int]] = {}
        graph_ptr: dict[str, torch.Tensor] = {}
        node_batch_index: dict[str, torch.Tensor] = {}
        offsets_by_type: dict[str, list[int]] = {}
        for kind in NODE_TYPES:
            counts = [int(graph.node_features[kind].shape[0]) for graph in items]
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            offsets_by_type[kind] = offsets
            node_features[kind] = torch.cat(
                [graph.node_features[kind] for graph in items], dim=0
            )
            ids = tuple(
                f"g{graph_index}::{identifier}"
                for graph_index, graph in enumerate(items)
                for identifier in graph.node_ids[kind]
            )
            node_ids[kind] = ids
            node_index[kind] = {identifier: index for index, identifier in enumerate(ids)}
            graph_ptr[kind] = torch.tensor(offsets, dtype=torch.long)
            node_batch_index[kind] = torch.repeat_interleave(
                torch.arange(len(items), dtype=torch.long),
                torch.tensor(counts, dtype=torch.long),
            )
        edges: dict[str, EdgeTensor] = {}
        for key in edge_keys:
            exemplar = items[0].edges[key]
            indices: list[torch.Tensor] = []
            features: list[torch.Tensor] = []
            for graph_index, graph in enumerate(items):
                edge = graph.edges[key]
                shifted = edge.index.clone()
                shifted[0] += offsets_by_type[edge.source_type][graph_index]
                shifted[1] += offsets_by_type[edge.target_type][graph_index]
                indices.append(shifted)
                features.append(edge.features)
            edges[key] = EdgeTensor(
                exemplar.source_type,
                exemplar.relation,
                exemplar.target_type,
                torch.cat(indices, dim=1),
                torch.cat(features, dim=0),
            )

        def candidate_mapping(name: str) -> torch.Tensor:
            counts = []
            for graph in items:
                candidate = graph.candidates
                values = {
                    "operation": candidate.operation_features,
                    "island": candidate.island_features,
                    "w": candidate.w_features,
                    "f": candidate.f_features,
                }[name]
                counts.append(len(values))
            return torch.repeat_interleave(
                torch.arange(len(items), dtype=torch.long),
                torch.tensor(counts, dtype=torch.long),
            )

        return cls(
            node_features=node_features,
            node_ids=node_ids,
            node_index=node_index,
            edges=edges,
            graphs=items,
            graph_ptr=graph_ptr,
            node_batch_index=node_batch_index,
            candidate_to_graph={
                name: candidate_mapping(name) for name in ("operation", "island", "w", "f")
            },
        )

    def to(self, device: torch.device | str) -> "BatchGraphTensor":
        return BatchGraphTensor(
            node_features={kind: value.to(device) for kind, value in self.node_features.items()},
            node_ids=self.node_ids,
            node_index=self.node_index,
            edges={
                key: EdgeTensor(
                    edge.source_type, edge.relation, edge.target_type,
                    edge.index.to(device), edge.features.to(device),
                )
                for key, edge in self.edges.items()
            },
            graphs=tuple(graph.to(device) for graph in self.graphs),
            graph_ptr={kind: value.to(device) for kind, value in self.graph_ptr.items()},
            node_batch_index={
                kind: value.to(device) for kind, value in self.node_batch_index.items()
            },
            candidate_to_graph={
                name: value.to(device) for name, value in self.candidate_to_graph.items()
            },
        )

    def split_hidden(
        self, hidden: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], ...]:
        return tuple({
            kind: hidden[kind][
                int(self.graph_ptr[kind][index].item()):
                int(self.graph_ptr[kind][index + 1].item())
            ]
            for kind in NODE_TYPES
        } for index in range(len(self.graphs)))


class GraphTensorizer:
    """State-compatible tensor schema learned from one valid graph state."""

    def __init__(self, graph: GraphState) -> None:
        self.node_feature_names = {
            node_type: tuple(next(iter(graph.node_features[node_type].values())))
            for node_type in NODE_TYPES
        }
        self.relation_feature_names: dict[str, tuple[str, ...]] = {}
        for source_type, relation, target_type in RELATION_TYPES:
            key = relation_key(source_type, relation, target_type)
            names: tuple[str, ...] = ()
            for edge in graph.edges:
                if (edge.source_type, edge.relation, edge.target_type) == (
                    source_type, relation, target_type,
                ):
                    names = tuple(edge.features)
                    break
            self.relation_feature_names[key] = names or ("__constant__",)
        self.candidate_feature_names = {
            "operation": tuple(next(iter(graph.operation_candidates.values()))),
            "island": tuple(next(iter(graph.island_candidates.values()))),
            "w": tuple(next(iter(graph.w_candidates.values()))),
            "f": tuple(next(iter(graph.f_candidates.values()))),
        }

    @property
    def node_input_dims(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.node_feature_names.items()}

    @property
    def relation_specs(self) -> tuple[tuple[str, str, str, str, int], ...]:
        return tuple(
            (
                relation_key(source, relation, target), source, relation, target,
                len(self.relation_feature_names[relation_key(source, relation, target)]),
            )
            for source, relation, target in RELATION_TYPES
        )

    @property
    def candidate_input_dims(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.candidate_feature_names.items()}

    def to_schema(self) -> dict[str, object]:
        return {
            "node_feature_names": {
                key: list(value) for key, value in self.node_feature_names.items()
            },
            "relation_feature_names": {
                key: list(value) for key, value in self.relation_feature_names.items()
            },
            "candidate_feature_names": {
                key: list(value) for key, value in self.candidate_feature_names.items()
            },
        }

    @classmethod
    def from_schema(cls, schema: Mapping[str, object]) -> "GraphTensorizer":
        tensorizer = cls.__new__(cls)
        tensorizer.node_feature_names = {
            key: tuple(value) for key, value in schema["node_feature_names"].items()
        }
        tensorizer.relation_feature_names = {
            key: tuple(value) for key, value in schema["relation_feature_names"].items()
        }
        tensorizer.candidate_feature_names = {
            key: tuple(value) for key, value in schema["candidate_feature_names"].items()
        }
        return tensorizer

    @staticmethod
    def _vector(features: Mapping[str, float], names: tuple[str, ...]) -> list[float]:
        if names == ("__constant__",):
            if features:
                raise ValueError(f"unexpected relation features: {tuple(features)}")
            return [0.0]
        if tuple(features) != names:
            raise ValueError(f"feature schema changed: expected {names}, got {tuple(features)}")
        return [float(features[name]) for name in names]

    def tensorize(self, graph: GraphState) -> GraphTensor:
        node_ids = {
            node_type: tuple(graph.node_features[node_type]) for node_type in NODE_TYPES
        }
        node_index = {
            node_type: {identifier: index for index, identifier in enumerate(ids)}
            for node_type, ids in node_ids.items()
        }
        node_features = {
            node_type: torch.tensor([
                self._vector(graph.node_features[node_type][identifier], self.node_feature_names[node_type])
                for identifier in node_ids[node_type]
            ], dtype=torch.float32)
            for node_type in NODE_TYPES
        }
        grouped: dict[str, list] = {key: [] for key, *_ in self.relation_specs}
        for edge in graph.edges:
            grouped[relation_key(edge.source_type, edge.relation, edge.target_type)].append(edge)
        edges: dict[str, EdgeTensor] = {}
        for key, source, relation, target, feature_dim in self.relation_specs:
            records = grouped[key]
            if records:
                index = torch.tensor([
                    [node_index[source][edge.source_id] for edge in records],
                    [node_index[target][edge.target_id] for edge in records],
                ], dtype=torch.long)
                features = torch.tensor([
                    self._vector(edge.features, self.relation_feature_names[key]) for edge in records
                ], dtype=torch.float32)
            else:
                index = torch.empty((2, 0), dtype=torch.long)
                features = torch.empty((0, feature_dim), dtype=torch.float32)
            edges[key] = EdgeTensor(source, relation, target, index, features)

        def candidate_vectors(items, names):
            return {key: torch.tensor(self._vector(value, names), dtype=torch.float32) for key, value in items.items()}

        candidates = TensorizedCandidates(
            operation_features=candidate_vectors(
                graph.operation_candidates, self.candidate_feature_names["operation"]
            ),
            island_features=candidate_vectors(
                graph.island_candidates, self.candidate_feature_names["island"]
            ),
            w_features=candidate_vectors(graph.w_candidates, self.candidate_feature_names["w"]),
            f_features=candidate_vectors(graph.f_candidates, self.candidate_feature_names["f"]),
            ready_operations=graph.ready_operations,
            island_masks=graph.island_masks,
            w_masks=graph.w_masks,
            f_masks=graph.f_masks,
        )
        return GraphTensor(node_features, node_ids, node_index, edges, candidates)

    def batch(self, graphs: Sequence[GraphTensor]) -> BatchGraphTensor:
        return BatchGraphTensor.from_graphs(graphs)
