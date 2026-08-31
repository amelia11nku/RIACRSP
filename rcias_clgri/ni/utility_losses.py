"""Pre-registered utility-aware Phase 6F loss families."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .batching import NIBatch


OBJECTIVES = {
    "O1_PHASE6E_REFERENCE",
    "O2_REGRET_WEIGHTED_RANKING",
    "O3_UTILITY_AWARE_MULTITASK",
}


@dataclass(frozen=True)
class Phase6FLossConfig:
    objective: str
    rank_weight: float = 1.0
    classification_weight: float = 0.25
    utility_weight: float = 0.0
    positive_weight: float = 1.0
    rank_gap_scale: float = 1.0
    rank_weight_min: float = 0.25
    rank_weight_max: float = 4.0
    utility_clip: float = 1.0
    huber_delta: float = 0.25
    distillation_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown Phase 6F objective: {self.objective}")
        if self.rank_weight < 0 or self.classification_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.utility_weight < 0 or self.distillation_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.rank_gap_scale <= 0 or self.utility_clip <= 0 or self.huber_delta <= 0:
            raise ValueError("utility scaling constants must be positive")
        if not 0 < self.rank_weight_min <= self.rank_weight_max:
            raise ValueError("invalid regret-ranking weight bounds")
        if self.objective == "O3_UTILITY_AWARE_MULTITASK" and self.utility_weight <= 0:
            raise ValueError("O3 requires a positive utility loss weight")


def within_state_standardize(scores: torch.Tensor, action_ptr: torch.Tensor) -> torch.Tensor:
    """Center and scale action scores independently inside every state."""
    result = torch.empty_like(scores)
    for start, stop in zip(action_ptr[:-1].tolist(), action_ptr[1:].tolist()):
        local = scores[start:stop]
        result[start:stop] = (local - local.mean()) / local.std(unbiased=False).clamp_min(1e-6)
    return result


def phase6f_loss(
    scores: torch.Tensor,
    utility_predictions: torch.Tensor | None,
    batch: NIBatch,
    config: Phase6FLossConfig,
    *,
    teacher_scores: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    better, worse = batch.rank_better_index, batch.rank_worse_index
    zero = scores.sum() * 0
    if better.numel():
        pair_losses = F.softplus(-(scores[better] - scores[worse]))
        if config.objective == "O1_PHASE6E_REFERENCE":
            ranking = pair_losses.mean()
        else:
            gaps = (batch.utility[better] - batch.utility[worse]).abs()
            pair_weights = (gaps / config.rank_gap_scale).clamp(
                config.rank_weight_min, config.rank_weight_max
            )
            ranking = (pair_weights * pair_losses).mean()
    else:
        ranking = zero

    positive_weight = torch.tensor(
        config.positive_weight, dtype=scores.dtype, device=scores.device
    )
    classification = F.binary_cross_entropy_with_logits(
        scores, batch.positive.to(scores.dtype), pos_weight=positive_weight
    )

    if config.utility_weight:
        if utility_predictions is None:
            raise ValueError("utility-aware objective requires a utility prediction head")
        target = batch.utility.clamp(-config.utility_clip, config.utility_clip)
        target = target / config.utility_clip
        utility = F.huber_loss(
            utility_predictions, target.to(utility_predictions.dtype), delta=config.huber_delta
        )
    else:
        utility = zero

    if config.distillation_weight:
        if teacher_scores is None:
            raise ValueError("positive distillation weight requires teacher scores")
        student = within_state_standardize(scores, batch.action_ptr)
        teacher = within_state_standardize(teacher_scores.detach(), batch.action_ptr)
        distillation = F.mse_loss(student, teacher)
    else:
        distillation = zero

    total = (
        config.rank_weight * ranking
        + config.classification_weight * classification
        + config.utility_weight * utility
        + config.distillation_weight * distillation
    )
    return {
        "loss": total,
        "rank_loss": ranking,
        "classification_loss": classification,
        "utility_loss": utility,
        "distillation_loss": distillation,
        "pair_count": torch.tensor(better.numel(), device=scores.device),
    }
