"""Compact relation-aware heterogeneous encoder for frozen CSG-1.0 tensors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn

from rcias_clgri.csg.schema import NODE_TYPE_ORDER

from .batching import NIBatch
from .tensorize import CSGTensorizer, NIRelationSpec


STATIC_CANONICAL_RELATIONS = {
    "OP__PRECEDES__OP",
    "OP__ELIGIBLE_ON__ISLAND",
    "OP__ASSIGNED_TO__ISLAND",
    "OP__REQUIRES__CONFIG",
    "ISLAND__SUPPORTS__CONFIG",
    "ISLAND__CURRENT_CONFIG__CONFIG",
}


@dataclass(frozen=True)
class NIModelConfig:
    hidden_dim: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    use_edge_features: bool = True
    relation_mode: str = "FULL_CSG"
    message_passing: bool = True
    utility_head: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.relation_mode not in {"FULL_CSG", "STATIC_CSG"}:
            raise ValueError("relation_mode must be FULL_CSG or STATIC_CSG")


def segment_mean(
    values: torch.Tensor,
    segments: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    output = values.new_zeros((segment_count, values.shape[-1]))
    output.index_add_(0, segments, values)
    counts = torch.bincount(segments, minlength=segment_count).to(values.dtype).unsqueeze(-1)
    return output / counts.clamp_min(1)


def segment_softmax(scores: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    if scores.shape[0] == 0:
        return torch.empty_like(scores)
    segment_count = int(segments.max().item()) + 1
    expanded = segments.view(-1, *([1] * (scores.ndim - 1))).expand_as(scores)
    maxima = torch.full(
        (segment_count, *scores.shape[1:]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    numerator = torch.exp(scores - maxima[segments])
    denominator = torch.zeros_like(maxima)
    denominator.index_add_(0, segments, numerator)
    return numerator / denominator[segments].clamp_min(torch.finfo(scores.dtype).tiny)


class CSGRelationLayer(nn.Module):
    def __init__(
        self,
        relation_specs: Sequence[NIRelationSpec],
        config: NIModelConfig,
    ) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.dim = dim
        self.heads = config.heads
        self.head_dim = dim // config.heads
        self.use_edge_features = config.use_edge_features
        self.query = nn.ModuleDict({
            node_type: nn.Linear(dim, dim, bias=False) for node_type in NODE_TYPE_ORDER
        })
        self.key = nn.ModuleDict({
            node_type: nn.Linear(dim, dim, bias=False) for node_type in NODE_TYPE_ORDER
        })
        self.value = nn.ModuleDict({
            node_type: nn.Linear(dim, dim, bias=False) for node_type in NODE_TYPE_ORDER
        })
        self.relation_key = nn.ModuleDict({
            spec.key: nn.Linear(dim, dim, bias=False) for spec in relation_specs
        })
        self.relation_value = nn.ModuleDict({
            spec.key: nn.Linear(dim, dim, bias=False) for spec in relation_specs
        })
        self.edge_key = nn.ModuleDict({
            spec.key: nn.Linear(len(spec.edge_feature_names), dim, bias=False)
            for spec in relation_specs if config.use_edge_features and spec.edge_feature_names
        })
        self.edge_value = nn.ModuleDict({
            spec.key: nn.Linear(len(spec.edge_feature_names), dim, bias=False)
            for spec in relation_specs if config.use_edge_features and spec.edge_feature_names
        })
        self.output = nn.ModuleDict({
            node_type: nn.Linear(dim, dim, bias=False) for node_type in NODE_TYPE_ORDER
        })
        self.attention_norm = nn.ModuleDict({
            node_type: nn.LayerNorm(dim) for node_type in NODE_TYPE_ORDER
        })
        self.feedforward = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(config.dropout),
                nn.Linear(2 * dim, dim),
            )
            for node_type in NODE_TYPE_ORDER
        })
        self.feedforward_norm = nn.ModuleDict({
            node_type: nn.LayerNorm(dim) for node_type in NODE_TYPE_ORDER
        })
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: NIBatch,
    ) -> dict[str, torch.Tensor]:
        aggregates = {
            node_type: torch.zeros_like(hidden[node_type]) for node_type in NODE_TYPE_ORDER
        }
        relation_counts = {
            node_type: hidden[node_type].new_zeros((hidden[node_type].shape[0], 1))
            for node_type in NODE_TYPE_ORDER
        }
        for relation_key, edge in batch.edges.items():
            if edge.index.shape[1] == 0:
                continue
            source_index, target_index = edge.index
            source = hidden[edge.spec.source_type][source_index]
            target = hidden[edge.spec.target_type][target_index]
            query = self.query[edge.spec.target_type](target)
            key = self.relation_key[relation_key](self.key[edge.spec.source_type](source))
            value = self.relation_value[relation_key](
                self.value[edge.spec.source_type](source)
            )
            if (
                self.use_edge_features
                and edge.spec.edge_feature_names
                and edge.features.shape[1] > 0
            ):
                key = key + self.edge_key[relation_key](edge.features)
                value = value + self.edge_value[relation_key](edge.features)
            query = query.view(-1, self.heads, self.head_dim)
            key = key.view(-1, self.heads, self.head_dim)
            value = value.view(-1, self.heads, self.head_dim)
            scores = (query * key).sum(dim=-1) / math.sqrt(self.head_dim)
            weights = segment_softmax(scores, target_index)
            messages = (weights.unsqueeze(-1) * value).reshape(-1, self.dim)
            aggregates[edge.spec.target_type].index_add_(0, target_index, messages)
            relation_counts[edge.spec.target_type][torch.unique(target_index)] += 1

        output = {}
        for node_type in NODE_TYPE_ORDER:
            aggregate = aggregates[node_type] / relation_counts[node_type].clamp_min(1)
            attended = self.attention_norm[node_type](
                hidden[node_type] + self.dropout(self.output[node_type](aggregate))
            )
            output[node_type] = self.feedforward_norm[node_type](
                attended + self.dropout(self.feedforward[node_type](attended))
            )
        return output


class CSGStateEncoder(nn.Module):
    def __init__(self, tensorizer: CSGTensorizer, config: NIModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.input_projection = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(tensorizer.node_input_dims[node_type], dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            for node_type in NODE_TYPE_ORDER
        })
        relation_specs = tuple(
            spec for spec in tensorizer.relation_specs
            if config.relation_mode == "FULL_CSG"
            or spec.canonical_key in STATIC_CANONICAL_RELATIONS
        )
        self.relation_keys = frozenset(spec.key for spec in relation_specs)
        if config.message_passing:
            self.layers = nn.ModuleList([
                CSGRelationLayer(relation_specs, config) for _ in range(config.layers)
            ])
            self.flat_layers = nn.ModuleList()
        else:
            self.layers = nn.ModuleList()
            self.flat_layers = nn.ModuleList([
                nn.ModuleDict({
                    node_type: nn.Sequential(
                        nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(config.dropout),
                        nn.Linear(2 * dim, dim), nn.Dropout(config.dropout), nn.LayerNorm(dim),
                    )
                    for node_type in NODE_TYPE_ORDER
                })
                for _ in range(config.layers)
            ])
        graph_input_dim = len(NODE_TYPE_ORDER) * dim + tensorizer.graph_input_dim
        self.graph_projection = nn.Sequential(
            nn.Linear(graph_input_dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, batch: NIBatch) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hidden = {
            node_type: self.input_projection[node_type](batch.node_features[node_type])
            for node_type in NODE_TYPE_ORDER
        }
        if self.config.message_passing:
            filtered_batch = replace_batch_edges(batch, self.relation_keys)
            for layer in self.layers:
                hidden = layer(hidden, filtered_batch)
        else:
            for layer in self.flat_layers:
                hidden = {
                    node_type: hidden[node_type] + layer[node_type](hidden[node_type])
                    for node_type in NODE_TYPE_ORDER
                }
        pooled = [
            segment_mean(
                hidden[node_type], batch.node_batch_index[node_type], batch.state_count
            )
            for node_type in NODE_TYPE_ORDER
        ]
        graph_numeric = torch.stack(
            (
                torch.log1p(batch.graph_numeric[:, 0].clamp_min(0)),
                batch.graph_numeric[:, 1],
            ),
            dim=1,
        )
        graph_embedding = self.graph_projection(torch.cat(
            [*pooled, graph_numeric, batch.graph_categorical], dim=1
        ))
        return hidden, graph_embedding


def replace_batch_edges(batch: NIBatch, relation_keys: frozenset[str]) -> NIBatch:
    if len(relation_keys) == len(batch.edges):
        return batch
    from dataclasses import replace
    return replace(
        batch,
        edges={key: edge for key, edge in batch.edges.items() if key in relation_keys},
    )
