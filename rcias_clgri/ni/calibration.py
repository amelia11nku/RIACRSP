"""Leakage-safe calibration helpers for Phase 6F validation scores."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class FrozenCalibrator:
    method: str
    parameters: dict[str, object]

    def predict(self, scores) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if self.method == "SIGMOID_IDENTITY":
            logits = np.clip(values, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-logits))
        if self.method == "PLATT":
            coefficient = float(self.parameters["coefficient"])
            intercept = float(self.parameters["intercept"])
            logits = np.clip(coefficient * values + intercept, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-logits))
        if self.method == "ISOTONIC":
            return np.interp(
                values,
                np.asarray(self.parameters["x_thresholds"], dtype=float),
                np.asarray(self.parameters["y_thresholds"], dtype=float),
            )
        if self.method == "BETA":
            raw_probability = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
            raw_probability = np.clip(raw_probability, 1e-12, 1.0 - 1e-12)
            transformed = (
                float(self.parameters["log_probability_coefficient"])
                * np.log(raw_probability)
                + float(self.parameters["log_one_minus_probability_coefficient"])
                * np.log1p(-raw_probability)
                + float(self.parameters["intercept"])
            )
            return 1.0 / (1.0 + np.exp(-np.clip(transformed, -40.0, 40.0)))
        raise ValueError(f"unknown calibration method: {self.method}")

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "parameters": self.parameters}


def fit_probability_calibrator(scores, positive, method: str) -> FrozenCalibrator:
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(positive, dtype=int)
    if values.ndim != 1 or labels.shape != values.shape or np.unique(labels).size != 2:
        raise ValueError("probability calibration requires aligned scores and two classes")
    if method == "SIGMOID_IDENTITY":
        return FrozenCalibrator(method, {})
    if method == "PLATT":
        model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
        model.fit(values.reshape(-1, 1), labels)
        return FrozenCalibrator(method, {
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        })
    if method == "ISOTONIC":
        model = IsotonicRegression(out_of_bounds="clip").fit(values, labels)
        return FrozenCalibrator(method, {
            "x_thresholds": model.X_thresholds_.astype(float).tolist(),
            "y_thresholds": model.y_thresholds_.astype(float).tolist(),
        })
    if method == "BETA":
        raw_probability = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
        raw_probability = np.clip(raw_probability, 1e-12, 1.0 - 1e-12)
        transformed = np.column_stack((
            np.log(raw_probability),
            np.log1p(-raw_probability),
        ))
        model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
        model.fit(transformed, labels)
        return FrozenCalibrator(method, {
            "log_probability_coefficient": float(model.coef_[0, 0]),
            "log_one_minus_probability_coefficient": float(model.coef_[0, 1]),
            "intercept": float(model.intercept_[0]),
        })
    raise ValueError(f"unsupported probability calibration method: {method}")


def fit_utility_calibrator(predictor, utility) -> FrozenCalibrator:
    values = np.asarray(predictor, dtype=float)
    outcomes = np.asarray(utility, dtype=float)
    if values.ndim != 1 or outcomes.shape != values.shape:
        raise ValueError("utility calibration requires aligned one-dimensional arrays")
    model = IsotonicRegression(out_of_bounds="clip").fit(values, outcomes)
    return FrozenCalibrator("ISOTONIC", {
        "x_thresholds": model.X_thresholds_.astype(float).tolist(),
        "y_thresholds": model.y_thresholds_.astype(float).tolist(),
    })


def reliability_table(probability, positive, *, bins: int = 10) -> pd.DataFrame:
    prediction = np.asarray(probability, dtype=float)
    labels = np.asarray(positive, dtype=float)
    if prediction.shape != labels.shape or bins < 2:
        raise ValueError("invalid reliability inputs")
    indices = np.minimum((np.clip(prediction, 0, 1) * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = indices == index
        rows.append({
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": int(mask.sum()),
            "mean_confidence": float(prediction[mask].mean()) if mask.any() else math.nan,
            "positive_fraction": float(labels[mask].mean()) if mask.any() else math.nan,
        })
    return pd.DataFrame(rows)


def calibration_metrics(probability, positive, *, bins: int = 10) -> dict[str, float]:
    prediction = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(positive, dtype=float)
    if prediction.ndim != 1 or labels.shape != prediction.shape:
        raise ValueError("calibration metrics require aligned one-dimensional arrays")
    table = reliability_table(prediction, labels, bins=bins)
    nonempty = table[table["count"] > 0]
    ece = float((
        nonempty["count"]
        / max(int(nonempty["count"].sum()), 1)
        * (nonempty["mean_confidence"] - nonempty["positive_fraction"]).abs()
    ).sum())
    return {
        "brier_score": float(np.mean((prediction - labels) ** 2)),
        "expected_calibration_error": ece,
        "negative_log_likelihood": float(-np.mean(
            labels * np.log(prediction) + (1.0 - labels) * np.log1p(-prediction)
        )),
        "mean_predicted_probability": float(prediction.mean()),
        "realized_positive_fraction": float(labels.mean()),
    }
