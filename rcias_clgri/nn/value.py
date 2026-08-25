"""Graph-level value baseline for future policy optimization."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .tensorizer import NODE_TYPES


class ValueHead(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(NODE_TYPES) * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, hidden: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pooled = torch.cat([hidden[kind].mean(dim=0) for kind in NODE_TYPES])
        return self.network(pooled).squeeze(-1)
