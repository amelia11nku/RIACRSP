#!/usr/bin/env python3
"""Compare stratified Phase 6G live CSG states with Phase 6C TRAIN states."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/phase6g"
OUT = PHASE / "drift_audit"
REFERENCE = OUT / "phase6c_training_state_reference.parquet"
FEATURES = (
    "mean_slack_ratio",
    "mean_w_delay_ratio",
    "mean_f_delay_ratio",
    "mean_island_relative_load",
    "mean_local_reconfiguration_ratio",
    "search_progress",
)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_reference() -> pd.DataFrame:
    if REFERENCE.exists():
        return pd.read_parquet(REFERENCE)
    rows = []
    columns = [
        "state_id", "operation_id", "operation_slack",
        "W_waiting_or_delay_contribution", "F_waiting_or_delay_contribution",
        "island_relative_load", "local_reconfiguration_contribution",
    ]
    paths = sorted((ROOT / "outputs/phase6c/dataset/train").rglob("target_membership.parquet"))
    for index, path in enumerate(paths, 1):
        membership = pd.read_parquet(path, columns=columns).drop_duplicates(
            ["state_id", "operation_id"]
        )
        states = pd.read_parquet(
            path.with_name("states.parquet"),
            columns=["state_id", "scale", "CF_level", "current_makespan", "search_progress"],
        )
        grouped = membership.groupby("state_id").agg(
            mean_operation_slack=("operation_slack", "mean"),
            mean_w_delay=("W_waiting_or_delay_contribution", "mean"),
            mean_f_delay=("F_waiting_or_delay_contribution", "mean"),
            mean_island_relative_load=("island_relative_load", "mean"),
            mean_local_reconfiguration=("local_reconfiguration_contribution", "mean"),
        ).reset_index().merge(states, on="state_id", validate="one_to_one")
        denominator = grouped.current_makespan.clip(lower=1.0)
        grouped["mean_slack_ratio"] = grouped.mean_operation_slack / denominator
        grouped["mean_w_delay_ratio"] = grouped.mean_w_delay / denominator
        grouped["mean_f_delay_ratio"] = grouped.mean_f_delay / denominator
        grouped["mean_local_reconfiguration_ratio"] = grouped.mean_local_reconfiguration / denominator
        rows.append(grouped[["state_id", "scale", "CF_level", *FEATURES]])
        if index % 25 == 0 or index == len(paths):
            print(f"PHASE6G_DRIFT_REFERENCE {index}/{len(paths)}", flush=True)
    reference = pd.concat(rows, ignore_index=True)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    temporary = REFERENCE.with_name(REFERENCE.name + f".tmp.{os.getpid()}")
    reference.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(REFERENCE)
    return reference


def psi(reference: np.ndarray, live: np.ndarray) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        low = min(reference.min(), live.min())
        high = max(reference.max(), live.max())
        if low == high:
            return 0.0
        edges = np.linspace(low, high, 11)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    live_counts = np.histogram(live, bins=edges)[0].astype(float)
    ref_fraction = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    live_fraction = np.clip(live_counts / max(live_counts.sum(), 1), 1e-6, None)
    return float(np.sum((live_fraction - ref_fraction) * np.log(live_fraction / ref_fraction)))


def severity(std_shift: float, psi_value: float) -> str:
    if std_shift >= 1.0 or psi_value >= 0.25:
        return "HIGH"
    if std_shift >= 0.5 or psi_value >= 0.10:
        return "MODERATE"
    return "LOW"


def compare(reference: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", "ALL", reference, live)]
    for scale in ("S", "M", "L"):
        groups.append((
            "scale", scale,
            reference[reference.scale == scale], live[live.scale == scale],
        ))
    for group_type, group_value, ref, observed in groups:
        for feature in FEATURES:
            ref_values = ref[feature].dropna().to_numpy(float)
            live_values = observed[feature].dropna().to_numpy(float)
            ref_std = float(np.std(ref_values, ddof=0))
            std_shift = abs(float(np.mean(live_values) - np.mean(ref_values))) / max(ref_std, 1e-12)
            psi_value = psi(ref_values, live_values)
            quantile_shift = float(np.mean(np.abs(
                np.quantile(live_values, [.1, .5, .9]) - np.quantile(ref_values, [.1, .5, .9])
            )) / max(ref_std, 1e-12))
            rows.append({
                "group_type": group_type,
                "group_value": group_value,
                "feature": feature,
                "training_state_count": len(ref_values),
                "live_state_count": len(live_values),
                "training_mean": float(np.mean(ref_values)),
                "live_mean": float(np.mean(live_values)),
                "training_std": ref_std,
                "standardized_mean_shift": std_shift,
                "normalized_quantile_shift": quantile_shift,
                "psi": psi_value,
                "severity": severity(std_shift, psi_value),
            })
    return pd.DataFrame(rows)


def main() -> None:
    progress = json.loads((OUT / "progress.json").read_text())
    if progress.get("status") != "COMPLETE":
        raise RuntimeError("drift feature-capture audit is not complete")
    paths = sorted((PHASE / "live_logs/drift_audit").rglob("*.parquet"))
    if len(paths) != 9:
        raise RuntimeError(f"expected 9 drift logs, found {len(paths)}")
    live = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    if live[list(FEATURES)].isna().any().any():
        raise RuntimeError("live drift features contain missing values")
    reference = build_reference()
    summary = compare(reference, live)
    atomic_csv(summary, PHASE / "statistics/live_drift_summary.csv")
    overall = summary[summary.group_type == "overall"]
    order = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    overall_severity = max(overall.severity, key=order.__getitem__)
    atomic_json({
        "schema": "phase6g-live-state-drift-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_capture_protocol": "9 DEV-HOLDOUT instances, seed 670301, R100, full 2N budget; excluded from primary performance statistics",
        "training_reference": "Phase6C TRAIN state-operation rows deduplicated to state-level means",
        "time_feature_normalization": "raw state feature divided by current makespan",
        "training_state_count": len(reference),
        "live_iteration_state_count": len(live),
        "overall_classification": overall_severity,
        "high_features": overall.loc[overall.severity == "HIGH", "feature"].tolist(),
        "moderate_features": overall.loc[overall.severity == "MODERATE", "feature"].tolist(),
    }, PHASE / "audit/live_state_drift_audit.json")

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = [
        {"LOW": "#70AD47", "MODERATE": "#FFC000", "HIGH": "#C00000"}[value]
        for value in overall.severity
    ]
    labels = [value.replace("mean_", "").replace("_ratio", "").replace("_", " ") for value in overall.feature]
    ax.bar(labels, overall.psi, color=colors)
    ax.axhline(.1, color="#FFC000", linestyle="--", linewidth=1)
    ax.axhline(.25, color="#C00000", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=35)
    ax.set(ylabel="Population stability index")
    ax.set_title("Phase6C TRAIN to Phase6G live state shift")
    fig.tight_layout()
    figures = PHASE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "live_state_distribution_shift.png", dpi=180)
    fig.savefig(figures / "live_state_distribution_shift.pdf")
    plt.close(fig)
    print(f"PHASE6G_DRIFT_ANALYSIS_COMPLETE classification={overall_severity}")


if __name__ == "__main__":
    main()
