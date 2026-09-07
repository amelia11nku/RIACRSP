"""Score-free selective-risk features, support, model, and loss for Phase 6M."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from rcias_clgri.analysis.phase6l_legacy_score import NUMERIC_COLUMNS as PHASE6L_NUMERIC


FAMILY = "M1_SCORE_FREE_SELECTIVE_RISK"
SUPPORT_NUMERIC_COLUMNS = tuple(column for column in PHASE6L_NUMERIC if column != "is_fallback")
STRUCTURAL_NUMERIC_COLUMNS = (
    *SUPPORT_NUMERIC_COLUMNS,
    "is_fallback",
    "target_progress",
    "search_progress",
)
CATEGORICAL_COLUMNS = (
    "primary_origin_rule",
    "origin_destroy_operator",
    "origin_family",
    "scale",
    "bottleneck_proxy",
)
RANKER_FEATURE_COLUMNS = (
    "ranker_advantage_mean",
    "ranker_advantage_std",
    "ranker_beats_fallback_logit_mean",
    "ranker_beats_fallback_logit_std",
    "ranker_beats_fallback_probability_mean",
    "ranker_beats_fallback_probability_std",
    "ranker_immediate_utility_mean",
    "ranker_immediate_utility_std",
    "candidate_fallback_advantage_margin",
    "candidate_best_other_advantage_margin",
    "seed_top1_vote_fraction",
    "seed_rank_mean_normalized",
    "seed_rank_std_normalized",
)
BANK_FEATURE_COLUMNS = (
    "candidate_bank_size_div_24",
    "hard_supported_candidate_fraction",
    "candidate_score_std",
    "candidate_score_spread",
    "candidate_score_normalized_entropy",
    "fraction_candidates_predicted_above_fallback",
)
SUPPORT_FEATURE_COLUMNS = (
    "fine_rule_oov",
    "maximum_absolute_robust_z",
    "root_mean_square_robust_z",
    "fraction_numeric_abs_z_above_4",
    "fraction_numeric_outside_training_range",
    "continuous_support_score",
)
SELECTOR_NUMERIC_COLUMNS = (
    *RANKER_FEATURE_COLUMNS,
    *BANK_FEATURE_COLUMNS,
    *SUPPORT_FEATURE_COLUMNS,
    *STRUCTURAL_NUMERIC_COLUMNS,
)
FORBIDDEN_ONLINE_COLUMNS = (
    "continuation_advantage",
    "continuation_advantage_mean",
    "continuation_advantage_std",
    "beats_fallback",
    "continuation_best_makespan",
    "fallback_continuation_best_makespan",
    "candidate_continuation_best_makespan",
    "candidate_decoded_makespan",
    "repair_trial_makespans",
    "frozen_raw_score",
    "frozen_calibrated_probability",
    "best_frozen_score_jaccard",
    "normalized_frozen_score_rank",
)


@dataclass(frozen=True)
class SupportTransform:
    vocabularies: dict[str, tuple[str, ...]]
    medians: dict[str, float]
    scales: dict[str, float]
    minima: dict[str, float]
    maxima: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "vocabularies": {key: list(value) for key, value in self.vocabularies.items()},
            "medians": self.medians,
            "scales": self.scales,
            "minima": self.minima,
            "maxima": self.maxima,
            "continuous_numeric_columns": list(SUPPORT_NUMERIC_COLUMNS),
            "binary_excluded_from_distance": ["is_fallback"],
            "hard_support_boundary": 12.0,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SupportTransform":
        return cls(
            {key: tuple(items) for key, items in value["vocabularies"].items()},
            {key: float(item) for key, item in value["medians"].items()},
            {key: float(item) for key, item in value["scales"].items()},
            {key: float(item) for key, item in value["minima"].items()},
            {key: float(item) for key, item in value["maxima"].items()},
        )


@dataclass(frozen=True)
class SelectorTransform:
    vocabularies: dict[str, tuple[str, ...]]
    medians: dict[str, float]
    scales: dict[str, float]

    @property
    def feature_names(self) -> tuple[str, ...]:
        numeric = tuple(f"numeric:{column}" for column in SELECTOR_NUMERIC_COLUMNS)
        categorical = tuple(
            f"category:{column}:{value}"
            for column in CATEGORICAL_COLUMNS
            for value in ("<UNK>", *self.vocabularies[column])
        )
        return (*numeric, *categorical)

    def to_dict(self) -> dict:
        return {
            "vocabularies": {key: list(value) for key, value in self.vocabularies.items()},
            "medians": self.medians,
            "scales": self.scales,
            "numeric_columns": list(SELECTOR_NUMERIC_COLUMNS),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "numeric_clip": [-10.0, 10.0],
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SelectorTransform":
        return cls(
            {key: tuple(items) for key, items in value["vocabularies"].items()},
            {key: float(item) for key, item in value["medians"].items()},
            {key: float(item) for key, item in value["scales"].items()},
        )


def _required(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = set(columns) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"invalid {context}: missing={sorted(missing)}, empty={frame.empty}")


def fit_support_transform(frame: pd.DataFrame) -> SupportTransform:
    _required(
        frame,
        (*SUPPORT_NUMERIC_COLUMNS, "primary_origin_rule", "origin_destroy_operator", "origin_family"),
        "Phase 6M support fit",
    )
    vocabularies = {
        column: tuple(sorted(frame[column].astype(str).unique()))
        for column in ("primary_origin_rule", "origin_destroy_operator", "origin_family")
    }
    medians, scales, minima, maxima = {}, {}, {}, {}
    for column in SUPPORT_NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite Phase 6M support fit column: {column}")
        median = float(np.median(values))
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        mad = float(np.median(np.abs(values - median)))
        span = float(values.max() - values.min())
        medians[column] = median
        scales[column] = max(iqr, 1.4826 * mad, 0.05 * span, 0.001)
        minima[column] = float(values.min())
        maxima[column] = float(values.max())
    return SupportTransform(vocabularies, medians, scales, minima, maxima)


def support_features(frame: pd.DataFrame, transform: SupportTransform) -> pd.DataFrame:
    _required(
        frame,
        (*SUPPORT_NUMERIC_COLUMNS, "primary_origin_rule", "origin_destroy_operator", "origin_family"),
        "Phase 6M support transform",
    )
    robust = np.column_stack([
        (frame[column].to_numpy(dtype=float) - transform.medians[column])
        / transform.scales[column]
        for column in SUPPORT_NUMERIC_COLUMNS
    ])
    finite = np.isfinite(robust).all(axis=1)
    absolute = np.abs(robust)
    maximum = np.where(finite, absolute.max(axis=1), np.inf)
    rms = np.where(finite, np.sqrt(np.mean(robust ** 2, axis=1)), np.inf)
    above_four = np.where(finite, np.mean(absolute > 4.0, axis=1), 1.0)
    outside = np.column_stack([
        (frame[column].to_numpy(dtype=float) < transform.minima[column])
        | (frame[column].to_numpy(dtype=float) > transform.maxima[column])
        for column in SUPPORT_NUMERIC_COLUMNS
    ]).mean(axis=1)
    rule_known = frame.primary_origin_rule.astype(str).isin(transform.vocabularies["primary_origin_rule"])
    operator_known = frame.origin_destroy_operator.astype(str).isin(
        transform.vocabularies["origin_destroy_operator"]
    )
    family_known = frame.origin_family.astype(str).isin(transform.vocabularies["origin_family"])
    hard_supported = finite & operator_known & family_known & (maximum <= 12.0)
    continuous = np.exp(-np.maximum(maximum - 2.0, 0.0) / 4.0 - outside)
    return pd.DataFrame({
        "fine_rule_oov": (~rule_known).to_numpy(dtype=float),
        "high_level_operator_known": operator_known.to_numpy(dtype=bool),
        "high_level_family_known": family_known.to_numpy(dtype=bool),
        "maximum_absolute_robust_z": maximum,
        "root_mean_square_robust_z": rms,
        "fraction_numeric_abs_z_above_4": above_four,
        "fraction_numeric_outside_training_range": outside,
        "continuous_support_score": continuous,
        "hard_supported": hard_supported,
    }, index=frame.index)


def _normalized_entropy(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    scale = float(np.std(values, ddof=0))
    standardized = np.zeros_like(values) if scale == 0.0 else (values - values.mean()) / scale
    shifted = standardized - standardized.max()
    probability = np.exp(shifted) / np.exp(shifted).sum()
    return float(-(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum() / math.log(len(values)))


def build_selector_feature_frame(
    ranker_predictions: pd.DataFrame,
    support_transform: SupportTransform,
) -> pd.DataFrame:
    required = (
        "state_id", "target_set_id", "training_seed", "is_fallback",
        "predicted_continuation_advantage", "predicted_beats_fallback_logit",
        "predicted_beats_fallback_probability_raw", "predicted_immediate_utility",
        *CATEGORICAL_COLUMNS, *STRUCTURAL_NUMERIC_COLUMNS,
    )
    _required(ranker_predictions, required, "Phase 6M ranker predictions")
    keys = ["state_id", "target_set_id"]
    seeds = tuple(sorted(int(value) for value in ranker_predictions.training_seed.unique()))
    if len(seeds) != 3 or ranker_predictions.duplicated([*keys, "training_seed"]).any():
        raise ValueError("Phase 6M requires exactly three unique ranker predictions per candidate")
    counts = ranker_predictions.groupby(keys).training_seed.nunique()
    if not counts.eq(3).all():
        raise ValueError("incomplete Phase 6M ranker ensemble")
    first = ranker_predictions[ranker_predictions.training_seed.eq(seeds[0])].copy()
    first = first.sort_values(keys, kind="stable").reset_index(drop=True)
    allowed_metadata = [
        *keys, "instance_id", "oof_fold", "CF_level", "search_stage",
        *CATEGORICAL_COLUMNS, *STRUCTURAL_NUMERIC_COLUMNS, "is_fallback",
        "fallback_target_set_id", "requested_bank_count", "full_bank_unique_count",
    ]
    allowed_metadata = list(dict.fromkeys(column for column in allowed_metadata if column in first))
    result = first[allowed_metadata].copy()
    if result.duplicated(keys).any() or not result.requested_bank_count.eq(24).all():
        raise ValueError("Phase 6M candidate identity or 24-rule bank boundary failed")
    bank_sizes = result.groupby("state_id").size()
    declared_sizes = result.groupby("state_id").full_bank_unique_count.first()
    if not bank_sizes.eq(declared_sizes).all():
        raise ValueError("Phase 6M full deduplicated bank was replaced or truncated")
    fallback_counts = result.groupby("state_id").is_fallback.sum()
    if not fallback_counts.eq(1).all():
        raise ValueError("Phase 6M canonical fallback is missing or nonunique")
    fallback_rows = result[result.is_fallback.astype(bool)]
    if not fallback_rows.target_set_id.astype(str).eq(
        fallback_rows.fallback_target_set_id.astype(str)
    ).all():
        raise ValueError("Phase 6M canonical fallback identity changed")
    aggregate = ranker_predictions.groupby(keys, sort=True).agg(
        ranker_advantage_mean=("predicted_continuation_advantage", "mean"),
        ranker_advantage_std=("predicted_continuation_advantage", lambda x: float(np.std(x, ddof=0))),
        ranker_beats_fallback_logit_mean=("predicted_beats_fallback_logit", "mean"),
        ranker_beats_fallback_logit_std=("predicted_beats_fallback_logit", lambda x: float(np.std(x, ddof=0))),
        ranker_beats_fallback_probability_mean=("predicted_beats_fallback_probability_raw", "mean"),
        ranker_beats_fallback_probability_std=("predicted_beats_fallback_probability_raw", lambda x: float(np.std(x, ddof=0))),
        ranker_immediate_utility_mean=("predicted_immediate_utility", "mean"),
        ranker_immediate_utility_std=("predicted_immediate_utility", lambda x: float(np.std(x, ddof=0))),
    ).reset_index()
    result = result.merge(aggregate, on=keys, validate="one_to_one")
    support = support_features(result, support_transform)
    result = pd.concat([result.reset_index(drop=True), support.reset_index(drop=True)], axis=1)
    state_rows = []
    for state_id, group in result.groupby("state_id", sort=True):
        positions = group.index.to_numpy()
        mean = group.ranker_advantage_mean.to_numpy(dtype=float)
        fallback_positions = np.flatnonzero(group.is_fallback.to_numpy(dtype=bool))
        if len(fallback_positions) != 1:
            raise ValueError(f"Phase 6M requires one canonical fallback: {state_id}")
        fallback_score = mean[int(fallback_positions[0])]
        seed_group = ranker_predictions[ranker_predictions.state_id.eq(state_id)]
        seed_scores = []
        seed_ranks = []
        seed_winners = []
        target_ids = group.target_set_id.astype(str).to_numpy()
        for seed in seeds:
            current = seed_group[seed_group.training_seed.eq(seed)].set_index("target_set_id").loc[target_ids]
            scores = current.predicted_continuation_advantage.to_numpy(dtype=float)
            seed_scores.append(scores)
            order = np.lexsort((target_ids, -scores))
            ranks = np.empty(len(group), dtype=float)
            ranks[order] = np.arange(len(group), dtype=float) / max(len(group) - 1, 1)
            seed_ranks.append(ranks)
            seed_winners.append(int(order[0]))
        rank_matrix = np.vstack(seed_ranks)
        votes = np.zeros(len(group), dtype=float)
        for index in seed_winners:
            votes[index] += 1.0 / len(seeds)
        best_order = np.lexsort((target_ids, -mean))
        best_index = int(best_order[0])
        second_index = int(best_order[1])
        best_other = np.full(len(group), mean[best_index], dtype=float)
        best_other[best_index] = mean[second_index]
        bank_support = float(group.hard_supported.mean())
        bank_std = float(np.std(mean, ddof=0))
        bank_spread = float(mean.max() - mean.min())
        bank_entropy = _normalized_entropy(mean)
        fraction_above = float(np.mean(mean > fallback_score))
        for local, position in enumerate(positions):
            state_rows.append({
                "index": position,
                "candidate_fallback_advantage_margin": mean[local] - fallback_score,
                "candidate_best_other_advantage_margin": mean[local] - best_other[local],
                "seed_top1_vote_fraction": votes[local],
                "seed_rank_mean_normalized": float(rank_matrix[:, local].mean()),
                "seed_rank_std_normalized": float(rank_matrix[:, local].std(ddof=0)),
                "candidate_bank_size_div_24": len(group) / 24.0,
                "hard_supported_candidate_fraction": bank_support,
                "candidate_score_std": bank_std,
                "candidate_score_spread": bank_spread,
                "candidate_score_normalized_entropy": bank_entropy,
                "fraction_candidates_predicted_above_fallback": fraction_above,
            })
    state_features = pd.DataFrame(state_rows).set_index("index").sort_index()
    result = pd.concat([result, state_features], axis=1)
    result = result.sort_values(keys, kind="stable").reset_index(drop=True)
    _required(result, (*SELECTOR_NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS), "selector features")
    if set(FORBIDDEN_ONLINE_COLUMNS) & set(result):
        raise ValueError("outcome or historical-score leakage in Phase 6M selector features")
    if not np.isfinite(result[list(SELECTOR_NUMERIC_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite Phase 6M selector feature")
    return result


def fit_selector_transform(frame: pd.DataFrame) -> SelectorTransform:
    _required(frame, (*SELECTOR_NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS), "selector transform fit")
    vocabularies = {
        column: tuple(sorted(frame[column].astype(str).unique()))
        for column in CATEGORICAL_COLUMNS
    }
    medians, scales = {}, {}
    for column in SELECTOR_NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite selector fit column: {column}")
        medians[column] = float(np.median(values))
        scales[column] = max(
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 0.001
        )
    return SelectorTransform(vocabularies, medians, scales)


def transform_selector_features(frame: pd.DataFrame, transform: SelectorTransform) -> np.ndarray:
    _required(frame, (*SELECTOR_NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS), "selector transform")
    numeric = np.column_stack([
        np.clip(
            (frame[column].to_numpy(dtype=float) - transform.medians[column])
            / transform.scales[column],
            -10.0, 10.0,
        ) for column in SELECTOR_NUMERIC_COLUMNS
    ])
    categorical_parts = []
    for column in CATEGORICAL_COLUMNS:
        vocabulary = transform.vocabularies[column]
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        indices = frame[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=int)
        one_hot = np.zeros((len(frame), len(vocabulary) + 1), dtype=float)
        one_hot[np.arange(len(frame)), indices] = 1.0
        categorical_parts.append(one_hot)
    output = np.column_stack([numeric, *categorical_parts]).astype(np.float32)
    if output.shape[1] != len(transform.feature_names) or not np.isfinite(output).all():
        raise ValueError("invalid transformed Phase 6M selector features")
    return output


@dataclass(frozen=True)
class SelectiveRiskOutput:
    continuation_mean: torch.Tensor
    continuation_scale: torch.Tensor
    seed_positive_logit: torch.Tensor


class SelectiveRiskMLP(nn.Module):
    def __init__(self, input_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.GELU(),
        )
        self.output = nn.Linear(16, 3)

    def forward(self, features: torch.Tensor) -> SelectiveRiskOutput:
        raw = self.output(self.backbone(features))
        return SelectiveRiskOutput(
            raw[:, 0], F.softplus(raw[:, 1]) + 1e-4, raw[:, 2]
        )


def initialize_selector(input_dim: int, seed: int, *, dropout: float = 0.1) -> SelectiveRiskMLP:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SelectiveRiskMLP(input_dim, dropout=dropout)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def selective_risk_loss(
    output: SelectiveRiskOutput,
    seed_outcomes: torch.Tensor,
    action_ptr: torch.Tensor,
    *,
    gaussian_weight: float = 1.0,
    bce_weight: float = 0.5,
    huber_weight: float = 0.25,
    huber_delta: float = 0.02,
) -> dict[str, torch.Tensor]:
    if seed_outcomes.ndim != 2 or seed_outcomes.shape[1] != 2:
        raise ValueError("Phase 6M requires exactly two CRN outcomes per candidate")
    if len(output.continuation_mean) != len(seed_outcomes):
        raise ValueError("selector output and outcome rows are misaligned")
    mean = output.continuation_mean[:, None]
    scale = output.continuation_scale[:, None]
    gaussian = (0.5 * ((seed_outcomes - mean) / scale) ** 2 + torch.log(scale)).mean(dim=1)
    positive = seed_outcomes.gt(0).to(dtype=output.seed_positive_logit.dtype)
    bce = F.binary_cross_entropy_with_logits(
        output.seed_positive_logit[:, None].expand_as(positive), positive, reduction="none"
    ).mean(dim=1)
    huber = F.huber_loss(
        output.continuation_mean, seed_outcomes.mean(dim=1), delta=huber_delta, reduction="none"
    )
    combined = gaussian_weight * gaussian + bce_weight * bce + huber_weight * huber
    state_losses = []
    for start, end in zip(action_ptr[:-1], action_ptr[1:]):
        state_losses.append(combined[int(start):int(end)].mean())
    loss = torch.stack(state_losses).mean()
    return {
        "loss": loss,
        "gaussian_nll": torch.stack([
            gaussian[int(start):int(end)].mean() for start, end in zip(action_ptr[:-1], action_ptr[1:])
        ]).mean(),
        "seed_positive_bce": torch.stack([
            bce[int(start):int(end)].mean() for start, end in zip(action_ptr[:-1], action_ptr[1:])
        ]).mean(),
        "candidate_mean_huber": torch.stack([
            huber[int(start):int(end)].mean() for start, end in zip(action_ptr[:-1], action_ptr[1:])
        ]).mean(),
    }
