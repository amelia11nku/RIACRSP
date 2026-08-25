"""Relation-aware temporal heterogeneous graph transformer."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from .config import ModelConfig
from .tensorizer import BatchGraphTensor, GraphTensor, NODE_TYPES


class RTHGTLayer(nn.Module):
    """One typed attention layer with relation messages and edge-time bias."""

    def __init__(self, relation_specs, config: ModelConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        self.dim = dim
        self.heads = config.heads
        self.head_dim = dim // config.heads
        self.query = nn.ModuleDict({kind: nn.Linear(dim, dim, bias=False) for kind in NODE_TYPES})
        self.key = nn.ModuleDict({kind: nn.Linear(dim, dim, bias=False) for kind in NODE_TYPES})
        self.value = nn.ModuleDict({kind: nn.Linear(dim, dim, bias=False) for kind in NODE_TYPES})
        self.relation_key = nn.ModuleDict()
        self.relation_message = nn.ModuleDict()
        self.edge_bias = nn.ModuleDict()
        self.edge_gate = nn.ModuleDict()
        for key, _source, _relation, _target, edge_dim in relation_specs:
            self.relation_key[key] = nn.Linear(dim, dim, bias=False)
            self.relation_message[key] = nn.Linear(dim, dim, bias=False)
            self.edge_bias[key] = nn.Linear(edge_dim, config.heads, bias=False)
            self.edge_gate[key] = nn.Linear(edge_dim, config.heads, bias=True)
        self.output = nn.ModuleDict({kind: nn.Linear(dim, dim, bias=False) for kind in NODE_TYPES})
        self.norm_attention = nn.ModuleDict({kind: nn.LayerNorm(dim) for kind in NODE_TYPES})
        hidden = dim * config.feedforward_multiplier
        self.feedforward = nn.ModuleDict({
            kind: nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            for kind in NODE_TYPES
        })
        self.norm_feedforward = nn.ModuleDict({kind: nn.LayerNorm(dim) for kind in NODE_TYPES})
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if scores.shape[0] == 0:
            return torch.empty_like(scores)
        if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
            raise ValueError("targets must contain one segment index per score row")
        segment_count = int(targets.max().item()) + 1
        expanded = targets.view(-1, *([1] * (scores.ndim - 1))).expand_as(scores)
        segment_shape = (segment_count, *scores.shape[1:])
        maxima = torch.full(
            segment_shape, -torch.inf, dtype=scores.dtype, device=scores.device
        )
        maxima.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
        shifted = scores - maxima[targets]
        numerators = torch.exp(shifted)
        denominators = torch.zeros(
            segment_shape, dtype=scores.dtype, device=scores.device
        )
        denominators.index_add_(0, targets, numerators)
        return numerators / denominators[targets].clamp_min(torch.finfo(scores.dtype).tiny)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        graph: GraphTensor | BatchGraphTensor,
    ) -> dict[str, torch.Tensor]:
        aggregates = {kind: torch.zeros_like(hidden[kind]) for kind in NODE_TYPES}
        relation_counts = {
            kind: torch.zeros((hidden[kind].shape[0], 1), device=hidden[kind].device)
            for kind in NODE_TYPES
        }
        for key, edge in graph.edges.items():
            if edge.index.shape[1] == 0:
                continue
            source_index, target_index = edge.index[0], edge.index[1]
            q = self.query[edge.target_type](hidden[edge.target_type])[target_index]
            k = self.relation_key[key](self.key[edge.source_type](hidden[edge.source_type])[source_index])
            v = self.relation_message[key](self.value[edge.source_type](hidden[edge.source_type])[source_index])
            q = q.view(-1, self.heads, self.head_dim)
            k = k.view(-1, self.heads, self.head_dim)
            v = v.view(-1, self.heads, self.head_dim)
            scores = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = scores + self.edge_bias[key](edge.features)
            weights = self._segment_softmax(scores, target_index)
            gates = torch.sigmoid(self.edge_gate[key](edge.features))
            messages = (weights * gates).unsqueeze(-1) * v
            aggregates[edge.target_type].index_add_(
                0, target_index, messages.reshape(-1, self.dim)
            )
            touched = torch.unique(target_index)
            relation_counts[edge.target_type][touched] += 1.0
        updated: dict[str, torch.Tensor] = {}
        for kind in NODE_TYPES:
            aggregate = aggregates[kind] / relation_counts[kind].clamp_min(1.0)
            attention = self.norm_attention[kind](
                hidden[kind] + self.dropout(self.output[kind](aggregate))
            )
            updated[kind] = self.norm_feedforward[kind](
                attention + self.dropout(self.feedforward[kind](attention))
            )
        return updated


class RTHGTEncoder(nn.Module):
    """Type projections followed by stacked relation-aware temporal layers."""

    def __init__(self, node_input_dims, relation_specs, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.ModuleDict({
            kind: nn.Linear(node_input_dims[kind], config.embedding_dim) for kind in NODE_TYPES
        })
        self.input_norm = nn.ModuleDict({
            kind: nn.LayerNorm(config.embedding_dim) for kind in NODE_TYPES
        })
        self.layers = nn.ModuleList([
            RTHGTLayer(relation_specs, config) for _ in range(config.layers)
        ])

    def forward(
        self, graph: GraphTensor | BatchGraphTensor,
    ) -> dict[str, torch.Tensor]:
        hidden = {
            kind: self.input_norm[kind](self.input_projection[kind](graph.node_features[kind]))
            for kind in NODE_TYPES
        }
        for layer in self.layers:
            hidden = layer(hidden, graph)
        return hidden
