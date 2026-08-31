"""Primary offline CSG target-set improvement scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .action_encoder import TargetSetEncoder
from .batching import NIBatch
from .encoder import CSGStateEncoder, NIModelConfig
from .tensorize import CSGTensorizer


@dataclass(frozen=True)
class NIModelOutput:
    scores: torch.Tensor
    graph_embeddings: torch.Tensor
    action_embeddings: torch.Tensor
    node_embeddings: Mapping[str, torch.Tensor]
    utility_predictions: torch.Tensor | None = None


class CSGTargetSetScorer(nn.Module):
    """Scores actual target membership without origin-operator identity inputs."""

    def __init__(self, tensorizer: CSGTensorizer, config: NIModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tensor_schema_hash = tensorizer.tensor_schema_hash
        self.state_encoder = CSGStateEncoder(tensorizer, config)
        self.action_encoder = TargetSetEncoder(config.hidden_dim, config.dropout)
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.utility_head = (
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
            if config.utility_head else None
        )

    def forward(self, batch: NIBatch) -> NIModelOutput:
        if batch.tensor_schema_hash != self.tensor_schema_hash:
            raise ValueError("model/batch tensor schema mismatch")
        node_embeddings, graph_embeddings = self.state_encoder(batch)
        action_embeddings = self.action_encoder(
            node_embeddings["OP"], graph_embeddings, batch
        )
        scores = self.score_head(action_embeddings).squeeze(-1)
        utility_predictions = (
            self.utility_head(action_embeddings).squeeze(-1)
            if self.utility_head is not None else None
        )
        return NIModelOutput(
            scores,
            graph_embeddings,
            action_embeddings,
            node_embeddings,
            utility_predictions,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
