"""Classification, within-state ranking, and selected-action metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


SCORE_COLUMN = "score"


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if array.size and np.isfinite(array).any() else math.nan


def evaluate_action_scores(frame: pd.DataFrame, *, score: str = SCORE_COLUMN) -> dict[str, float]:
    required = {
        "state_id", "mean_relative_improvement", "rank_percentile", "regret_to_best",
        "top1", score,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"scored action frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("cannot evaluate an empty scored action frame")
    positive = frame["mean_relative_improvement"].gt(0).astype(int)
    if positive.nunique() == 2:
        roc_auc = float(roc_auc_score(positive, frame[score]))
        pr_auc = float(average_precision_score(positive, frame[score]))
    else:
        roc_auc = math.nan
        pr_auc = math.nan

    pairwise = []
    correlations = []
    ndcg = []
    top1_hits = []
    top3_hits = []
    selected_utility = []
    selected_positive = []
    selected_top3 = []
    selected_regret = []
    selected_rank_percentile = []
    oracle_utility = []
    for _, group in frame.groupby("state_id", sort=False):
        scores = group[score].to_numpy(dtype=float)
        utility = group["mean_relative_improvement"].to_numpy(dtype=float)
        concordant = 0.0
        comparable = 0
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                truth_delta = utility[first] - utility[second]
                if truth_delta == 0:
                    continue
                score_delta = scores[first] - scores[second]
                comparable += 1
                concordant += 1.0 if truth_delta * score_delta > 0 else 0.5 if score_delta == 0 else 0.0
        if comparable:
            pairwise.append(concordant / comparable)
        correlation = spearmanr(utility, scores).statistic
        if math.isfinite(float(correlation)):
            correlations.append(float(correlation))
        relevance = group["rank_percentile"].to_numpy(dtype=float).reshape(1, -1)
        if np.ptp(relevance) > 0:
            ndcg.append(float(ndcg_score(relevance, scores.reshape(1, -1))))
        order = np.argsort(-scores, kind="stable")
        chosen = group.iloc[int(order[0])]
        top1_hits.append(float(bool(chosen["top1"])))
        top3_hits.append(float(group.iloc[order[:3]]["top1"].astype(bool).any()))
        selected_utility.append(float(chosen["mean_relative_improvement"]))
        selected_positive.append(float(chosen["mean_relative_improvement"] > 0))
        selected_top3.append(float(bool(chosen.get("top3", False))))
        selected_regret.append(float(chosen["regret_to_best"]))
        selected_rank_percentile.append(float(chosen["rank_percentile"]))
        oracle_utility.append(float(group["mean_relative_improvement"].max()))
    return {
        "action_count": int(len(frame)),
        "state_count": int(frame["state_id"].nunique()),
        "positive_fraction": float(positive.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pairwise_accuracy": _safe_mean(pairwise),
        "within_state_spearman": _safe_mean(correlations),
        "ndcg": _safe_mean(ndcg),
        "top1_accuracy": _safe_mean(top1_hits),
        "top3_recall": _safe_mean(top3_hits),
        "mean_selected_utility": _safe_mean(selected_utility),
        "median_selected_utility": float(np.median(selected_utility)),
        "selected_positive_fraction": _safe_mean(selected_positive),
        "selected_top3_fraction": _safe_mean(selected_top3),
        "mean_selected_regret": _safe_mean(selected_regret),
        "mean_selected_rank_percentile": _safe_mean(selected_rank_percentile),
        "mean_oracle_utility": _safe_mean(oracle_utility),
    }


def selected_action_table(
    frame: pd.DataFrame,
    *,
    model_name: str,
    score: str = SCORE_COLUMN,
) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("state_id", sort=False):
        chosen = group.sort_values(score, ascending=False, kind="stable").iloc[0].to_dict()
        chosen["model"] = model_name
        chosen["selected_positive"] = bool(chosen["mean_relative_improvement"] > 0)
        rows.append(chosen)
    return pd.DataFrame(rows)
