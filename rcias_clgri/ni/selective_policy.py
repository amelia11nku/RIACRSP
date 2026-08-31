"""Validation-only selective neural intervention policy for Phase 6F."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InterventionThresholds:
    confidence: float
    predicted_utility: float
    decision_margin: float


def _neural_choices(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["state_id", "score", "target_set_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    chosen = ordered.groupby("state_id", sort=False).head(1).copy()
    second = ordered.groupby("state_id", sort=False).nth(1)["calibrated_probability"]
    chosen = chosen.set_index("state_id")
    chosen["decision_margin"] = (
        chosen["calibrated_probability"] - second.reindex(chosen.index).fillna(0.0)
    )
    return chosen.reset_index()


def _related_fallback(frame: pd.DataFrame) -> pd.DataFrame:
    if "origin_rules" not in frame:
        raise ValueError("canonical related fallback requires frozen origin_rules")
    related = frame[
        frame["origin_rules"].str.contains(
            '"operator_related"', regex=False, na=False
        )
    ].copy()
    counts = related.groupby("state_id").size()
    if len(counts) != frame["state_id"].nunique() or not counts.eq(1).all():
        raise ValueError("canonical related fallback must exist exactly once per state")
    return related.set_index("state_id")


def evaluate_thresholds(
    frame: pd.DataFrame,
    thresholds: InterventionThresholds,
) -> dict[str, float]:
    required = {
        "state_id", "target_set_id", "score", "calibrated_probability",
        "calibrated_utility", "mean_relative_improvement", "regret_to_best",
        "arm_family", "origin_destroy_operator",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"selective-policy frame missing columns: {sorted(missing)}")
    neural = _neural_choices(frame).set_index("state_id")
    fallback = _related_fallback(frame).reindex(neural.index)
    intervene = (
        neural["calibrated_probability"].ge(thresholds.confidence)
        & neural["calibrated_utility"].ge(thresholds.predicted_utility)
        & neural["decision_margin"].ge(thresholds.decision_margin)
    )
    neural_utility = neural["mean_relative_improvement"].astype(float)
    fallback_utility = fallback["mean_relative_improvement"].astype(float)
    hybrid_utility = neural_utility.where(intervene, fallback_utility)
    neural_regret = neural["regret_to_best"].astype(float)
    fallback_regret = fallback["regret_to_best"].astype(float)
    hybrid_regret = neural_regret.where(intervene, fallback_regret)
    intervention_utility = neural_utility[intervene]
    return {
        "coverage": float(intervene.mean()),
        "intervention_state_count": int(intervene.sum()),
        "mean_intervention_utility": (
            float(intervention_utility.mean()) if intervene.any() else np.nan
        ),
        "intervention_positive_fraction": (
            float(intervention_utility.gt(0).mean()) if intervene.any() else np.nan
        ),
        "incremental_utility_vs_fallback": float(
            (hybrid_utility - fallback_utility).mean()
        ),
        "hybrid_selected_utility": float(hybrid_utility.mean()),
        "hybrid_selected_positive_fraction": float(hybrid_utility.gt(0).mean()),
        "hybrid_mean_regret": float(hybrid_regret.mean()),
        "fallback_selected_utility": float(fallback_utility.mean()),
        "fallback_mean_regret": float(fallback_regret.mean()),
        "fallback_rate": float(1.0 - intervene.mean()),
    }


def threshold_study(
    frame: pd.DataFrame,
    *,
    minimum_coverage: float = 0.2,
) -> tuple[pd.DataFrame, InterventionThresholds]:
    neural = _neural_choices(frame)
    quantiles = (0.0, 0.25, 0.5, 0.75)
    confidence = sorted(set(float(value) for value in np.quantile(
        neural["calibrated_probability"], quantiles
    )))
    utility = sorted(set(float(value) for value in np.quantile(
        neural["calibrated_utility"], quantiles
    )))
    margin = sorted(set(float(value) for value in np.quantile(
        neural["decision_margin"], quantiles
    )))
    rows = []
    for values in itertools.product(confidence, utility, margin):
        thresholds = InterventionThresholds(*values)
        rows.append({
            "confidence_threshold": thresholds.confidence,
            "utility_threshold": thresholds.predicted_utility,
            "margin_threshold": thresholds.decision_margin,
            **evaluate_thresholds(frame, thresholds),
        })
    result = pd.DataFrame(rows)
    eligible = result[result["coverage"] >= minimum_coverage].copy()
    if eligible.empty:
        raise ValueError("no selective policy meets the frozen minimum coverage")
    selected = eligible.sort_values(
        ["hybrid_selected_utility", "hybrid_mean_regret", "coverage",
         "confidence_threshold", "utility_threshold", "margin_threshold"],
        ascending=[False, True, False, True, True, True],
        kind="stable",
    ).iloc[0]
    thresholds = InterventionThresholds(
        float(selected["confidence_threshold"]),
        float(selected["utility_threshold"]),
        float(selected["margin_threshold"]),
    )
    result["selected"] = (
        result["confidence_threshold"].eq(thresholds.confidence)
        & result["utility_threshold"].eq(thresholds.predicted_utility)
        & result["margin_threshold"].eq(thresholds.decision_margin)
    )
    return result, thresholds
