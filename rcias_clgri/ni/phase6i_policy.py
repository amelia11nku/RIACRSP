"""Frozen policy helpers for Phase 6I-MR calibration and selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from rcias_clgri.ni.calibration import (
    FrozenCalibrator,
    fit_probability_calibrator,
    fit_utility_calibrator,
)


ACTION_KEY = ("state_id", "target_set_id")


@dataclass(frozen=True)
class CrossFittedCalibration:
    """A deployment calibrator plus leakage-safe R09 fitted predictions."""

    calibrator: FrozenCalibrator
    predictions: np.ndarray


def ensemble_oof_predictions(
    frame: pd.DataFrame,
    *,
    value_column: str,
    score_column: str,
    expected_training_seeds: Iterable[int],
) -> pd.DataFrame:
    """Average seed-specific grouped-OOF outputs after strict row validation."""
    required = {
        *ACTION_KEY,
        "training_seed",
        "held_fold",
        "oof_fold",
        "decoded_immediate_utility",
        "positive_label",
        "fallback_target_set_id",
        "fallback_decoded_utility",
        "scale",
        value_column,
        score_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")

    seeds = tuple(sorted(int(seed) for seed in expected_training_seeds))
    observed = tuple(sorted(int(seed) for seed in frame.training_seed.unique()))
    if observed != seeds:
        raise ValueError(f"training seed mismatch: expected {seeds}, observed {observed}")
    if frame.duplicated([*ACTION_KEY, "training_seed"]).any():
        raise ValueError("duplicate action/training-seed OOF rows")
    counts = frame.groupby(list(ACTION_KEY), sort=False).training_seed.nunique()
    if not counts.eq(len(seeds)).all():
        raise ValueError("every OOF action must have one prediction from every seed")
    if not (frame.held_fold.astype(str) == frame.oof_fold.astype(str)).all():
        raise ValueError("held fold does not match the action's grouped OOF fold")

    invariant_columns = [
        "decoded_immediate_utility",
        "positive_label",
        "fallback_target_set_id",
        "fallback_decoded_utility",
        "scale",
        "oof_fold",
    ]
    for column in invariant_columns:
        if frame.groupby(list(ACTION_KEY), sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"seed-specific rows disagree on invariant column {column}")

    first_seed = frame[frame.training_seed == seeds[0]].copy()
    first_seed = first_seed.sort_values(list(ACTION_KEY), kind="stable").reset_index(drop=True)
    means = (
        frame.groupby(list(ACTION_KEY), sort=True)[[value_column, score_column]]
        .mean()
        .rename(columns={
            value_column: "ensemble_raw_value",
            score_column: "ensemble_raw_score",
        })
        .reset_index()
    )
    result = first_seed.drop(columns=["training_seed", value_column, score_column])
    result = result.merge(means, on=list(ACTION_KEY), validate="one_to_one")
    if not np.isfinite(
        result[["ensemble_raw_value", "ensemble_raw_score"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("non-finite ensemble OOF prediction")
    return result.sort_values(list(ACTION_KEY), kind="stable").reset_index(drop=True)


def _fold_values(folds: pd.Series) -> tuple[str, ...]:
    values = tuple(sorted(str(value) for value in folds.unique()))
    if len(values) < 2:
        raise ValueError("cross-fitted calibration requires at least two folds")
    return values


def cross_fit_utility_calibration(
    predictor: pd.Series | np.ndarray,
    outcome: pd.Series | np.ndarray,
    folds: pd.Series | np.ndarray,
) -> CrossFittedCalibration:
    values = np.asarray(predictor, dtype=float)
    targets = np.asarray(outcome, dtype=float)
    fold_array = np.asarray(folds).astype(str)
    if values.ndim != 1 or targets.shape != values.shape or fold_array.shape != values.shape:
        raise ValueError("utility calibration inputs must be aligned vectors")
    predictions = np.empty_like(values)
    for fold in _fold_values(pd.Series(fold_array)):
        held = fold_array == fold
        calibrator = fit_utility_calibrator(values[~held], targets[~held])
        predictions[held] = calibrator.predict(values[held])
    return CrossFittedCalibration(
        calibrator=fit_utility_calibrator(values, targets),
        predictions=predictions,
    )


def cross_fit_probability_calibration(
    scores: pd.Series | np.ndarray,
    positive: pd.Series | np.ndarray,
    folds: pd.Series | np.ndarray,
) -> CrossFittedCalibration:
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(positive, dtype=int)
    fold_array = np.asarray(folds).astype(str)
    if values.ndim != 1 or labels.shape != values.shape or fold_array.shape != values.shape:
        raise ValueError("probability calibration inputs must be aligned vectors")
    predictions = np.empty_like(values)
    for fold in _fold_values(pd.Series(fold_array)):
        held = fold_array == fold
        calibrator = fit_probability_calibrator(values[~held], labels[~held], "PLATT")
        predictions[held] = calibrator.predict(values[held])
    return CrossFittedCalibration(
        calibrator=fit_probability_calibrator(values, labels, "PLATT"),
        predictions=predictions,
    )


def fit_support_bounds(
    frame: pd.DataFrame, columns: Iterable[str]
) -> dict[str, dict[str, float]]:
    """Freeze literal R09 finite ranges; no protected-split quantiles are used."""
    bounds: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in frame:
            raise ValueError(f"missing support feature: {column}")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite support feature: {column}")
        bounds[column] = {"minimum": float(values.min()), "maximum": float(values.max())}
    return bounds


def support_mask(
    frame: pd.DataFrame, bounds: dict[str, dict[str, float]]
) -> np.ndarray:
    supported = np.ones(len(frame), dtype=bool)
    for column, limits in bounds.items():
        if column not in frame:
            return np.zeros(len(frame), dtype=bool)
        values = frame[column].to_numpy(dtype=float)
        supported &= np.isfinite(values)
        supported &= values >= float(limits["minimum"])
        supported &= values <= float(limits["maximum"])
    return supported


def select_immediate_actions(
    frame: pd.DataFrame,
    *,
    probability_threshold: float,
    utility_threshold: float,
) -> pd.DataFrame:
    """Apply the frozen filter/rank/threshold/fallback rule once per state."""
    required = {
        *ACTION_KEY,
        "scale",
        "calibrated_probability",
        "calibrated_utility",
        "ensemble_raw_value",
        "ensemble_raw_score",
        "supported",
        "decoded_immediate_utility",
        "fallback_target_set_id",
        "fallback_decoded_utility",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"policy frame is missing columns: {sorted(missing)}")
    if frame.duplicated(list(ACTION_KEY)).any():
        raise ValueError("policy frame contains duplicate actions")

    fallback_counts = frame.groupby("state_id").agg(
        fallback_id_count=("fallback_target_set_id", "nunique"),
        fallback_value_count=("fallback_decoded_utility", "nunique"),
    )
    if not (
        fallback_counts.fallback_id_count.eq(1).all()
        and fallback_counts.fallback_value_count.eq(1).all()
    ):
        raise ValueError("inconsistent fallback within a state")
    states = (
        frame.groupby("state_id", sort=True)
        .agg(
            instance_id=("instance_id", "first"),
            scale=("scale", "first"),
            fallback_target_set_id=("fallback_target_set_id", "first"),
            fallback_realized_utility=("fallback_decoded_utility", "first"),
            forced_action_count=("target_set_id", "size"),
            best_forced_utility=("decoded_immediate_utility", "max"),
        )
        .reset_index()
    )
    eligible = frame[
        frame.supported.astype(bool)
        & frame.calibrated_probability.ge(float(probability_threshold))
    ].sort_values(
        ["state_id", "ensemble_raw_value", "ensemble_raw_score", "target_set_id"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    chosen = eligible.drop_duplicates("state_id", keep="first")[[
        "state_id", "target_set_id", "calibrated_utility",
        "decoded_immediate_utility", "supported",
    ]].rename(columns={
        "target_set_id": "candidate_target_set_id",
        "calibrated_utility": "candidate_calibrated_utility",
        "decoded_immediate_utility": "candidate_realized_utility",
        "supported": "candidate_supported",
    })
    result = states.merge(chosen, on="state_id", how="left", validate="one_to_one")
    result["intervened"] = (
        result.candidate_target_set_id.notna()
        & result.candidate_calibrated_utility.ge(float(utility_threshold))
    )
    result["selected_target_set_id"] = np.where(
        result.intervened,
        result.candidate_target_set_id,
        result.fallback_target_set_id,
    )
    result["selected_calibrated_utility"] = result.candidate_calibrated_utility.where(
        result.intervened
    )
    result["selected_realized_utility"] = result.candidate_realized_utility.where(
        result.intervened, result.fallback_realized_utility
    )
    result["selected_lift_over_fallback"] = (
        result.selected_realized_utility - result.fallback_realized_utility
    )
    result["selected_sign_error"] = (
        result.intervened
        & result.candidate_calibrated_utility.gt(0.0)
        & result.candidate_realized_utility.le(0.0)
    )
    result["selected_supported"] = result.candidate_supported.where(
        result.intervened, True
    ).astype(bool)
    result["best_forced_lift_over_fallback"] = (
        result.best_forced_utility - result.fallback_realized_utility
    )
    return result.drop(columns=[
        "candidate_target_set_id", "candidate_calibrated_utility",
        "candidate_realized_utility", "candidate_supported",
    ])


def state_ranking_metrics(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "ensemble_raw_value",
    target_column: str = "decoded_immediate_utility",
) -> pd.DataFrame:
    """Compute frozen within-state metrics with state as the first aggregation."""
    required = {
        "state_id", "instance_id", "scale", "CF_level", "search_stage",
        "candidate_role", prediction_column, target_column,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"ranking frame is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for state_id, group in frame.groupby("state_id", sort=True):
        truth = group[target_column].to_numpy(dtype=float)
        score = group[prediction_column].to_numpy(dtype=float)
        comparable = 0
        concordant = 0.0
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                truth_delta = truth[first] - truth[second]
                if truth_delta == 0:
                    continue
                comparable += 1
                score_delta = score[first] - score[second]
                concordant += (
                    1.0 if truth_delta * score_delta > 0
                    else 0.5 if score_delta == 0
                    else 0.0
                )
        predicted_order = np.lexsort((
            group.target_set_id.astype(str).to_numpy(),
            -score,
        ))
        chosen_index = int(predicted_order[0])
        truth_order = np.lexsort((
            group.target_set_id.astype(str).to_numpy(),
            -truth,
        ))
        rank = np.empty(len(group), dtype=int)
        rank[truth_order] = np.arange(len(group))
        relevance = len(group) - rank

        def ndcg_at(k: int) -> float:
            ideal = np.sort(relevance)[::-1][:k]
            selected = relevance[predicted_order[:k]]
            discount = 1.0 / np.log2(np.arange(2, k + 2))
            return float(np.sum(selected * discount) / np.sum(ideal * discount))

        fallback = group[group.candidate_role.eq("ALNS_RELATED_FALLBACK")]
        fallback_value = (
            float(fallback[target_column].iloc[0]) if len(fallback) else math.nan
        )
        spearman = spearmanr(truth, score).statistic
        kendall = kendalltau(truth, score).statistic
        rows.append({
            "state_id": state_id,
            "instance_id": str(group.instance_id.iloc[0]),
            "scale": str(group.scale.iloc[0]),
            "CF_level": str(group.CF_level.iloc[0]),
            "search_stage": str(group.search_stage.iloc[0]),
            "spearman": float(spearman) if np.isfinite(spearman) else math.nan,
            "kendall": float(kendall) if np.isfinite(kendall) else math.nan,
            "pairwise_accuracy": concordant / comparable if comparable else math.nan,
            "pair_inversion_rate": (
                1.0 - concordant / comparable if comparable else math.nan
            ),
            "ndcg_at_1": ndcg_at(1),
            "ndcg_at_2": ndcg_at(min(2, len(group))),
            "top1_agreement": float(chosen_index == int(truth_order[0])),
            "selected_value": float(truth[chosen_index]),
            "selected_regret": float(np.max(truth) - truth[chosen_index]),
            "fallback_value": fallback_value,
            "selected_lift_over_fallback": float(truth[chosen_index] - fallback_value),
            "selected_positive": float(truth[chosen_index] > 0),
            "selected_sign_error": float(
                (score[chosen_index] >= 0) != (truth[chosen_index] >= 0)
            ),
        })
    return pd.DataFrame(rows)


def summarize_ranking_metrics(state_metrics: pd.DataFrame) -> dict[str, float]:
    """Aggregate state metrics through equal-weight instance means."""
    if state_metrics.empty:
        raise ValueError("cannot summarize empty state metrics")
    numeric = [
        "spearman", "kendall", "pairwise_accuracy", "pair_inversion_rate",
        "ndcg_at_1", "ndcg_at_2", "top1_agreement", "selected_value",
        "selected_regret", "selected_lift_over_fallback", "selected_positive",
        "selected_sign_error",
    ]
    instance = state_metrics.groupby(
        ["instance_id", "scale"], as_index=False
    )[numeric].mean()
    result = {
        f"overall_per_instance_{column}": float(
            instance.groupby("instance_id")[column].mean().mean()
        )
        for column in numeric
    }
    for scale in ("S", "M", "L"):
        group = instance[instance.scale == scale]
        for column in numeric:
            result[f"scale_{scale}_per_instance_{column}"] = (
                float(group[column].mean()) if len(group) else math.nan
            )
    return result
