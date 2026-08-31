"""Permutation-invariant target-set encoder over CSG operation embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn

from .batching import NIBatch
from .encoder import segment_mean, segment_softmax


def segment_max(
    values: torch.Tensor,
    segments: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    output = torch.full(
        (segment_count, values.shape[-1]),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    expanded = segments.unsqueeze(-1).expand_as(values)
    output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


class TargetSetEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        operation_hidden: torch.Tensor,
        graph_embedding: torch.Tensor,
        batch: NIBatch,
    ) -> torch.Tensor:
        selected = operation_hidden[batch.target_operation_indices]
        action_index = batch.target_action_index
        mean = segment_mean(selected, action_index, batch.action_count)
        maximum = segment_max(selected, action_index, batch.action_count)
        query = self.attention_query(graph_embedding[batch.action_to_state])
        keys = self.attention_key(selected)
        logits = (
            keys * query[action_index]
        ).sum(dim=-1) / math.sqrt(self.hidden_dim)
        weights = segment_softmax(logits, action_index)
        attention = selected.new_zeros((batch.action_count, self.hidden_dim))
        attention.index_add_(0, action_index, weights.unsqueeze(-1) * selected)
        target_sizes = torch.bincount(action_index, minlength=batch.action_count).to(
            selected.dtype
        )
        op_counts = (batch.node_ptr["OP"][1:] - batch.node_ptr["OP"][:-1]).to(
            selected.dtype
        )
        normalized_size = (
            target_sizes / op_counts[batch.action_to_state].clamp_min(1)
        ).unsqueeze(-1)
        return self.projection(torch.cat(
            [mean, maximum, attention, graph_embedding[batch.action_to_state], normalized_size],
            dim=-1,
        ))
