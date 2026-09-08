"""Frozen-encoder, top-utility-aligned candidate critic for Phase 6O."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .phase6n_candidate_conditioned import CandidateConditionedCSGModel


FAMILY = "O1_TOP_UTILITY_FROZEN_ENCODER"


class TopUtilityCSGModel(CandidateConditionedCSGModel):
    """Phase 6N representation with the complete Phase 6F encoder frozen."""

    family = FAMILY

    def __init__(self, base, categorical_sizes: tuple[int, int, int]) -> None:
        super().__init__(
            base, categorical_sizes, family="N1_CANDIDATE_CONDITIONED_CSG"
        )
        for parameter in self.state_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "TopUtilityCSGModel":
        nn.Module.train(self, mode)
        self.state_encoder.eval()
        self.pooler.train(mode)
        self.fallback_fusion.train(mode)
        self.continuation_advantage_head.train(mode)
        self.beats_fallback_head.train(mode)
        return self


def top_utility_loss(
    advantage_prediction: torch.Tensor,
    beats_fallback_logit: torch.Tensor,
    advantage_target: torch.Tensor,
    beats_fallback_target: torch.Tensor,
    action_ptr: torch.Tensor,
    state_weights: torch.Tensor,
    *,
    noise_margin: float,
    policy_temperature: float,
    utility_scale: float,
    regret_scale: float,
    huber_delta: float,
    top_set_weight: float = 1.0,
    regret_weight: float = 1.0,
    expected_utility_weight: float = 1.0,
    advantage_huber_weight: float = 0.1,
    beats_bce_weight: float = 0.1,
    pair_weight_clip: tuple[float, float] = (0.25, 4.0),
) -> dict[str, torch.Tensor]:
    """Compute the preregistered whole-state noise-aware top-selection loss."""
    vectors = (
        advantage_prediction,
        beats_fallback_logit,
        advantage_target,
        beats_fallback_target,
    )
    if any(value.ndim != 1 for value in vectors):
        raise ValueError("Phase 6O loss inputs must be vectors")
    if len({len(value) for value in vectors}) != 1:
        raise ValueError("Phase 6O loss vectors must be aligned")
    if action_ptr.ndim != 1 or int(action_ptr[0]) != 0 or int(action_ptr[-1]) != len(vectors[0]):
        raise ValueError("invalid Phase 6O action pointers")
    state_count = len(action_ptr) - 1
    if state_weights.shape != (state_count,) or not bool(torch.isfinite(state_weights).all()):
        raise ValueError("invalid Phase 6O state weights")
    if bool((state_weights <= 0).any()):
        raise ValueError("Phase 6O state weights must be positive")
    scales = (noise_margin, policy_temperature, utility_scale, regret_scale, huber_delta)
    if any(not 0 < float(value) for value in scales):
        raise ValueError("Phase 6O objective scales must be positive")

    terms: dict[str, list[torch.Tensor]] = {
        "top_set_cross_entropy": [],
        "regret_logistic": [],
        "soft_expected_utility": [],
        "advantage_huber": [],
        "beats_fallback_bce": [],
    }
    pair_count = near_count = 0
    for start, stop in zip(action_ptr[:-1].tolist(), action_ptr[1:].tolist()):
        prediction = advantage_prediction[start:stop]
        target = advantage_target[start:stop]
        beats_logit = beats_fallback_logit[start:stop]
        beats_target = beats_fallback_target[start:stop]
        near = target >= target.max() - noise_margin
        rest = ~near
        near_count += int(near.sum())
        terms["top_set_cross_entropy"].append(-torch.log_softmax(prediction, dim=0)[near].mean())

        if bool(rest.any()):
            gaps = target[near, None] - target[None, rest]
            keep = gaps > noise_margin
            pair_count += int(keep.sum())
            if bool(keep.any()):
                predicted_gaps = prediction[near, None] - prediction[None, rest]
                pair_weights = (gaps / regret_scale).clamp(*pair_weight_clip)
                logistic = (pair_weights[keep] * F.softplus(-predicted_gaps[keep])).mean()
            else:
                logistic = prediction.sum() * 0.0
        else:
            logistic = prediction.sum() * 0.0
        terms["regret_logistic"].append(logistic)
        policy = torch.softmax(prediction / policy_temperature, dim=0)
        terms["soft_expected_utility"].append(-(policy * target).sum() / utility_scale)
        terms["advantage_huber"].append(F.huber_loss(
            prediction, target, delta=huber_delta
        ))
        terms["beats_fallback_bce"].append(F.binary_cross_entropy_with_logits(
            beats_logit, beats_target
        ))

    normalized_weights = state_weights / state_weights.mean()
    means = {
        name: (torch.stack(values) * normalized_weights).mean()
        for name, values in terms.items()
    }
    total = (
        top_set_weight * means["top_set_cross_entropy"]
        + regret_weight * means["regret_logistic"]
        + expected_utility_weight * means["soft_expected_utility"]
        + advantage_huber_weight * means["advantage_huber"]
        + beats_bce_weight * means["beats_fallback_bce"]
    )
    return {
        "loss": total,
        **means,
        "pair_count": torch.as_tensor(pair_count, device=total.device),
        "near_count": torch.as_tensor(near_count, device=total.device),
    }
