"""Relation- and type-aware multi-head attention for the NGAS CSG."""
from __future__ import annotations

import math
import torch
from torch import nn

from rcias_clgri.csg.schema import EDGE_TYPE_ORDER, NODE_TYPE_ORDER
from rcias_ngas.csg.revised_features import EDGE_FEATURE_DIM, NODE_DIM


def segment_softmax(scores: torch.Tensor, segments: torch.Tensor,
                    segment_count: int) -> torch.Tensor:
    if not len(scores):
        return torch.empty_like(scores)
    expanded = segments[:, None].expand_as(scores)
    maxima = torch.full(
        (segment_count, scores.shape[1]), -torch.inf,
        device=scores.device, dtype=scores.dtype)
    maxima.scatter_reduce_(0, expanded, scores, reduce='amax', include_self=True)
    numerator = torch.exp(scores - maxima[segments])
    denominator = torch.zeros_like(maxima).index_add_(0, segments, numerator)
    return numerator / denominator[segments].clamp_min(torch.finfo(scores.dtype).tiny)


def _typed_transform(hidden: torch.Tensor, types: torch.Tensor,
                     modules: nn.ModuleList) -> torch.Tensor:
    output_dim = modules[0](hidden[:0]).shape[-1]
    output = hidden.new_zeros((len(hidden), output_dim))
    for index, module in enumerate(modules):
        mask = types == index
        if bool(mask.any()):
            output[mask] = module(hidden[mask])
    return output


class RTHGTLayer(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError('RT-HGT hidden dimension must be divisible by heads')
        self.hidden, self.heads, self.head_dim = hidden, heads, hidden // heads
        type_count = len(NODE_TYPE_ORDER)
        relation_count = 2 * len(EDGE_TYPE_ORDER)
        self.query = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(type_count))
        self.key = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(type_count))
        self.value = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(type_count))
        self.relation_key = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(relation_count))
        self.relation_value = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(relation_count))
        self.edge_key = nn.ModuleList(
            nn.Linear(EDGE_FEATURE_DIM, hidden, bias=False) for _ in range(relation_count))
        self.edge_value = nn.ModuleList(
            nn.Linear(EDGE_FEATURE_DIM, hidden, bias=False) for _ in range(relation_count))
        self.relation_prior = nn.Parameter(torch.ones(relation_count, heads))
        self.output = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(type_count))
        self.attention_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(type_count))
        self.feedforward = nn.ModuleList(nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
            for _ in range(type_count))
        self.feedforward_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(type_count))

    def forward(self, hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        source, target = batch['edge_index']
        query = _typed_transform(hidden, batch['types'], self.query)[target]
        key = _typed_transform(hidden, batch['types'], self.key)[source]
        value = _typed_transform(hidden, batch['types'], self.value)[source]
        related_key, related_value = torch.zeros_like(key), torch.zeros_like(value)
        for relation_index in range(len(self.relation_key)):
            mask = batch['relations'] == relation_index
            if not bool(mask.any()):
                continue
            related_key[mask] = (
                self.relation_key[relation_index](key[mask])
                + self.edge_key[relation_index](batch['edge_features'][mask]))
            related_value[mask] = (
                self.relation_value[relation_index](value[mask])
                + self.edge_value[relation_index](batch['edge_features'][mask]))
        query = query.view(-1, self.heads, self.head_dim)
        related_key = related_key.view(-1, self.heads, self.head_dim)
        related_value = related_value.view(-1, self.heads, self.head_dim)
        scores = (query * related_key).sum(-1) / math.sqrt(self.head_dim)
        scores = scores * self.relation_prior[batch['relations']]
        weights = segment_softmax(scores, target, len(hidden))
        messages = (weights.unsqueeze(-1) * related_value).reshape(-1, self.hidden)
        aggregate = torch.zeros_like(hidden).index_add_(0, target, messages)
        output = hidden.clone()
        for type_index in range(len(NODE_TYPE_ORDER)):
            mask = batch['types'] == type_index
            if not bool(mask.any()):
                continue
            attended = self.attention_norm[type_index](
                hidden[mask] + self.output[type_index](aggregate[mask]))
            output[mask] = self.feedforward_norm[type_index](
                attended + self.feedforward[type_index](attended))
        return output


class RTHGTEncoder(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.ModuleList(nn.Sequential(
            nn.Linear(NODE_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
            for _ in NODE_TYPE_ORDER)
        self.layers = nn.ModuleList(RTHGTLayer(hidden, heads) for _ in range(layers))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = _typed_transform(batch['x'], batch['types'], self.input)
        for layer in self.layers:
            hidden = layer(hidden, batch)
        return hidden
