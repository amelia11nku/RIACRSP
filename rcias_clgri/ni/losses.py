"""State-conditioned ranking/classification losses for Phase 6E."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .batching import NIBatch


@dataclass(frozen=True)
class NILossConfig:
    rank_weight: float = 1.0
    classification_weight: float = 0.25
    positive_weight: float = 1.0


def phase6e_loss(
    scores: torch.Tensor,
    batch: NIBatch,
    config: NILossConfig,
) -> dict[str, torch.Tensor]:
    better, worse = batch.rank_better_index, batch.rank_worse_index
    if better.numel():
        ranking = F.softplus(-(scores[better] - scores[worse])).mean()
    else:
        ranking = scores.sum() * 0
    positive_weight = torch.tensor(
        config.positive_weight, dtype=scores.dtype, device=scores.device
    )
    classification = F.binary_cross_entropy_with_logits(
        scores, batch.positive.to(scores.dtype), pos_weight=positive_weight
    )
    total = config.rank_weight * ranking + config.classification_weight * classification
    return {
        "loss": total,
        "rank_loss": ranking,
        "classification_loss": classification,
        "pair_count": torch.tensor(better.numel(), device=scores.device),
    }
