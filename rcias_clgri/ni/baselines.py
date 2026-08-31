"""Frozen non-graph and heuristic baselines for Phase 6E evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TABULAR_CATEGORICAL = (
    "scale", "CF_level", "RI_level", "TI_level", "search_stage",
    "bottleneck_proxy", "arm_family",
)
TABULAR_NUMERIC = (
    "current_makespan", "destroy_count", "target_mean_criticality",
    "target_max_criticality", "target_mean_slack", "target_min_slack",
    "target_mean_W_delay", "target_mean_F_delay", "target_mean_island_load",
    "target_mean_reconfiguration", "target_mean_eligible_islands",
    "target_mean_sync_delay", "target_critical_fraction",
    "target_resource_critical_fraction", "target_product_diversity",
    "target_island_diversity",
)
ORIGINAL_OPERATORS = (
    "random", "critical", "overloaded_island", "high_reconfiguration",
    "w_bottleneck", "f_bottleneck", "related",
)


def random_selection_expectation(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact B0 expectation over every candidate, without Monte Carlo."""
    rows = []
    for state_id, group in frame.groupby("state_id", sort=False):
        rows.append({
            "state_id": state_id,
            "model": "B0_RANDOM_EXPECTATION",
            "candidate_count": len(group),
            "mean_selected_utility": float(group["mean_relative_improvement"].mean()),
            "selected_positive_fraction": float(
                group["mean_relative_improvement"].gt(0).mean()
            ),
            "mean_selected_regret": float(group["regret_to_best"].mean()),
        })
    return pd.DataFrame(rows)


def fixed_original_selection(frame: pd.DataFrame, operator: str) -> pd.DataFrame:
    if operator not in ORIGINAL_OPERATORS:
        raise ValueError(f"unknown original operator: {operator}")
    if "origin_rules" not in frame:
        raise ValueError("fixed original selection requires frozen origin_rules")
    original = frame[
        frame["origin_rules"].str.contains(
            f'"operator_{operator}"', regex=False, na=False
        )
    ].copy()
    counts = original.groupby("state_id").size()
    if (counts > 1).any():
        raise ValueError(f"multiple original {operator} arms found in one state")
    original["model"] = f"FIXED_ORIGINAL_{operator.upper()}"
    original["selected_positive"] = original["mean_relative_improvement"].gt(0)
    return original


def choose_best_fixed_original(
    validation: pd.DataFrame,
    operators: tuple[str, ...] = ORIGINAL_OPERATORS,
) -> tuple[str, pd.DataFrame]:
    rows = []
    expected_states = validation["state_id"].nunique()
    for operator in operators:
        selected = fixed_original_selection(validation, operator)
        if selected["state_id"].nunique() != expected_states:
            raise ValueError(f"original {operator} does not cover every validation state")
        rows.append({
            "origin_destroy_operator": operator,
            "state_count": selected["state_id"].nunique(),
            "mean_utility": float(selected["mean_relative_improvement"].mean()),
            "positive_fraction": float(selected["mean_relative_improvement"].gt(0).mean()),
            "mean_regret": float(selected["regret_to_best"].mean()),
        })
    summary = pd.DataFrame(rows).sort_values(
        ["mean_utility", "positive_fraction", "origin_destroy_operator"],
        ascending=[False, False, True],
        kind="stable",
    )
    if summary.empty:
        raise ValueError("validation data contains no original operator arms")
    return str(summary.iloc[0]["origin_destroy_operator"]), summary


def merge_tabular_features(
    aggregates: pd.DataFrame,
    target_features: pd.DataFrame,
) -> pd.DataFrame:
    return aggregates.merge(
        target_features,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )


@dataclass
class Phase6CTabularDiagnostic:
    """B3 reproduction of the strongest legal Phase 6C set diagnostic."""

    sample_limit: int = 250_000
    random_state: int = 670001

    def __post_init__(self) -> None:
        preprocessing = ColumnTransformer([
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(TABULAR_CATEGORICAL),
            ),
            (
                "numeric",
                StandardScaler(),
                list(TABULAR_NUMERIC),
            ),
        ])
        self.pipeline = Pipeline([
            ("features", preprocessing),
            (
                "logistic",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    random_state=self.random_state,
                ),
            ),
        ])
        self.fitted = False

    @property
    def feature_columns(self) -> list[str]:
        return [*TABULAR_CATEGORICAL, *TABULAR_NUMERIC]

    def fit(self, train: pd.DataFrame) -> "Phase6CTabularDiagnostic":
        if not train["training_split"].eq("TRAIN").all():
            raise ValueError("B3 may be fitted only on TRAIN")
        sample = train if len(train) <= self.sample_limit else train.sample(
            n=self.sample_limit, random_state=self.random_state
        )
        digest = hashlib.sha256()
        for row in sample[["state_id", "target_set_id"]].itertuples(index=False):
            digest.update(f"{row.state_id}\t{row.target_set_id}\n".encode())
        self.training_sample_count = len(sample)
        self.training_sample_sha256 = digest.hexdigest()
        self.training_label = "positive_under_2_of_3"
        if "positive_under_2_of_3" not in sample:
            raise ValueError(
                "B3 requires the frozen Phase 6C positive_under_2_of_3 label"
            )
        label = sample["positive_under_2_of_3"].astype(int)
        if label.nunique() != 2:
            raise ValueError("B3 training sample must contain both classes")
        self.pipeline.fit(sample[self.feature_columns], label)
        self.fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("B3 must be fitted before scoring")
        return self.pipeline.predict_proba(frame[self.feature_columns])[:, 1]
