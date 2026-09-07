"""Phase 6L score-free continuation model over the frozen CSG encoder."""

from __future__ import annotations

import torch
from torch import nn

from rcias_clgri.ni.batching import NIBatch
from rcias_clgri.ni.phase6j_caur_model import CAUROutput
from rcias_clgri.ni.scorer import CSGTargetSetScorer


class ScoreFreeCandidateContinuationHeads(nn.Module):
    """Fuse action embeddings with ten deterministic source features."""

    def __init__(
        self,
        categorical_sizes: tuple[int, int, int],
        *,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(size, 4) for size in categorical_sizes
        ])
        self.context_projection = nn.Sequential(
            nn.Linear(22, 32),
            nn.GELU(),
        )
        self.candidate_context_projection = nn.Sequential(
            nn.Linear(3 * embedding_dim + 32, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(embedding_dim, 24),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(24, 1),
            )

        self.continuation_advantage_head = head()
        self.beats_fallback_head = head()
        self.immediate_utility_head = head()

    def forward(
        self,
        action_embeddings: torch.Tensor,
        action_to_state: torch.Tensor,
        fallback_action_indices: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if categorical.ndim != 2 or categorical.shape[1] != 3:
            raise ValueError("Phase 6L requires three categorical feature columns")
        if numeric.ndim != 2 or numeric.shape[1] != 10:
            raise ValueError("Phase 6L requires ten score-free numeric columns")
        if len(action_embeddings) != len(categorical) or len(numeric) != len(categorical):
            raise ValueError("Phase 6L action and source-feature rows are misaligned")
        fallback = action_embeddings[fallback_action_indices][action_to_state]
        categories = torch.cat([
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.categorical_embeddings)
        ], dim=1)
        context = self.context_projection(torch.cat([categories, numeric], dim=1))
        fused = self.candidate_context_projection(torch.cat([
            action_embeddings,
            fallback,
            action_embeddings - fallback,
            context,
        ], dim=1))
        return (
            self.continuation_advantage_head(fused).squeeze(-1),
            self.beats_fallback_head(fused).squeeze(-1),
            self.immediate_utility_head(fused).squeeze(-1),
        )


class ScoreFreeCAURModel(nn.Module):
    """Single preregistered Phase 6L family with a frozen base encoder."""

    family = "L1_SCORE_FREE_CONT_FROZEN"

    def __init__(
        self,
        base: CSGTargetSetScorer,
        categorical_sizes: tuple[int, int, int],
        *,
        family: str,
    ) -> None:
        super().__init__()
        if family != self.family:
            raise ValueError(f"unsupported Phase 6L family: {family}")
        base.score_head = nn.Identity()
        base.utility_head = None
        self.base = base
        self.heads = ScoreFreeCandidateContinuationHeads(
            categorical_sizes,
            embedding_dim=base.config.hidden_dim,
            dropout=base.config.dropout,
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ScoreFreeCAURModel":
        super().train(mode)
        self.base.eval()
        self.heads.train(mode)
        return self

    def forward(
        self,
        batch: NIBatch,
        *,
        fallback_action_indices: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> CAUROutput:
        with torch.no_grad():
            node, graph = self.base.state_encoder(batch)
            action = self.base.action_encoder(node["OP"], graph, batch)
        advantage, beats, immediate = self.heads(
            action,
            batch.action_to_state,
            fallback_action_indices,
            categorical,
            numeric,
        )
        return CAUROutput(advantage, beats, immediate, action)

    def parameter_counts(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        return total, trainable
