"""Small continuation-aware heads over the frozen Phase 6F CSG encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rcias_clgri.ni.batching import NIBatch
from rcias_clgri.ni.scorer import CSGTargetSetScorer


@dataclass(frozen=True)
class CAUROutput:
    advantage: torch.Tensor
    beats_fallback_logit: torch.Tensor
    immediate_utility: torch.Tensor
    action_embeddings: torch.Tensor


class CandidateContinuationHeads(nn.Module):
    """Fuse candidate, fallback, and outcome-blind source context."""

    def __init__(
        self,
        categorical_sizes: tuple[int, int, int],
        *,
        numeric_dim: int = 12,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(size, 4) for size in categorical_sizes
        ])
        self.context_projection = nn.Sequential(
            nn.Linear(numeric_dim + 12, 32),
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
            raise ValueError("CAUR requires three categorical feature columns")
        if numeric.ndim != 2 or numeric.shape[1] != 12:
            raise ValueError("CAUR requires twelve numeric feature columns")
        if len(action_embeddings) != len(categorical) or len(numeric) != len(categorical):
            raise ValueError("CAUR action and candidate-context rows are misaligned")
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


class CAURModel(nn.Module):
    """J1/J2 model with an auditable frozen-base trainability boundary."""

    def __init__(
        self,
        base: CSGTargetSetScorer,
        categorical_sizes: tuple[int, int, int],
        *,
        family: str,
    ) -> None:
        super().__init__()
        if family not in {"J1_CONT_FROZEN", "J2_CONT_LASTBLOCK"}:
            raise ValueError(f"unsupported CAUR family: {family}")
        self.family = family
        base.score_head = nn.Identity()
        base.utility_head = None
        self.base = base
        self.heads = CandidateContinuationHeads(
            categorical_sizes,
            embedding_dim=base.config.hidden_dim,
            dropout=base.config.dropout,
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        if family == "J2_CONT_LASTBLOCK":
            for parameter in self.base.state_encoder.layers[-1].parameters():
                parameter.requires_grad_(True)
            for parameter in self.base.action_encoder.projection.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "CAURModel":
        super().train(mode)
        self.base.eval()
        if mode and self.family == "J2_CONT_LASTBLOCK":
            self.base.state_encoder.layers[-1].train()
            self.base.action_encoder.projection.train()
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
        if self.family == "J1_CONT_FROZEN":
            with torch.no_grad():
                node, graph = self.base.state_encoder(batch)
                action = self.base.action_encoder(node["OP"], graph, batch)
        else:
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


def _within_state_standardize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean()
    return centered / values.std(unbiased=False).clamp_min(1e-6)


def caur_grouped_state_loss(
    advantage_prediction: torch.Tensor,
    beats_fallback_logit: torch.Tensor,
    immediate_prediction: torch.Tensor,
    advantage_target: torch.Tensor,
    beats_fallback_target: torch.Tensor,
    immediate_target: torch.Tensor,
    action_ptr: torch.Tensor,
    *,
    gap_scale: float,
    immediate_delta: float,
    pairwise_weight: float = 1.0,
    listnet_weight: float = 0.75,
    advantage_huber_weight: float = 0.5,
    beats_bce_weight: float = 0.25,
    immediate_huber_weight: float = 0.10,
    gap_weight_clip: tuple[float, float] = (0.25, 4.0),
) -> dict[str, torch.Tensor]:
    """State-balanced five-part CAUR objective on complete candidate lists."""
    tensors = (
        advantage_prediction,
        beats_fallback_logit,
        immediate_prediction,
        advantage_target,
        beats_fallback_target,
        immediate_target,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("CAUR loss inputs must be one-dimensional")
    if len({len(tensor) for tensor in tensors}) != 1:
        raise ValueError("CAUR loss inputs must have identical lengths")
    if gap_scale <= 0 or immediate_delta <= 0:
        raise ValueError("CAUR loss scales must be positive")
    if action_ptr.ndim != 1 or action_ptr[0] != 0 or action_ptr[-1] != len(tensors[0]):
        raise ValueError("invalid CAUR state action pointers")

    terms: dict[str, list[torch.Tensor]] = {
        "pairwise_loss": [],
        "listnet_loss": [],
        "advantage_huber_loss": [],
        "beats_fallback_bce_loss": [],
        "immediate_huber_loss": [],
    }
    pair_count = 0
    for start, stop in zip(action_ptr[:-1].tolist(), action_ptr[1:].tolist()):
        predicted = advantage_prediction[start:stop]
        target = advantage_target[start:stop]
        predicted_z = _within_state_standardize(predicted)
        target_z = _within_state_standardize(target)
        gaps = target.unsqueeze(1) - target.unsqueeze(0)
        better = gaps > 1e-12
        pair_count += int(better.sum().item())
        if bool(better.any()):
            predicted_gaps = predicted_z.unsqueeze(1) - predicted_z.unsqueeze(0)
            weights = (gaps.abs() / gap_scale).clamp(*gap_weight_clip)
            terms["pairwise_loss"].append(
                (weights[better] * F.softplus(-predicted_gaps[better])).mean()
            )
        else:
            terms["pairwise_loss"].append(predicted.sum() * 0.0)
        terms["listnet_loss"].append(-(
            torch.softmax(target_z, dim=0) * torch.log_softmax(predicted_z, dim=0)
        ).sum())
        terms["advantage_huber_loss"].append(F.huber_loss(
            predicted, target, delta=gap_scale
        ))
        terms["beats_fallback_bce_loss"].append(F.binary_cross_entropy_with_logits(
            beats_fallback_logit[start:stop],
            beats_fallback_target[start:stop],
        ))
        terms["immediate_huber_loss"].append(F.huber_loss(
            immediate_prediction[start:stop],
            immediate_target[start:stop],
            delta=immediate_delta,
        ))
    means = {name: torch.stack(values).mean() for name, values in terms.items()}
    total = (
        pairwise_weight * means["pairwise_loss"]
        + listnet_weight * means["listnet_loss"]
        + advantage_huber_weight * means["advantage_huber_loss"]
        + beats_bce_weight * means["beats_fallback_bce_loss"]
        + immediate_huber_weight * means["immediate_huber_loss"]
    )
    return {"loss": total, **means, "pair_count": torch.as_tensor(pair_count)}
