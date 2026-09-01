#!/usr/bin/env python3
"""Fit Phase 6H post-hoc calibration and preregistered CAL-FIT gate candidates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.calibration import (  # noqa: E402
    FrozenCalibrator,
    calibration_metrics,
    fit_probability_calibrator,
    fit_utility_calibrator,
    reliability_table,
)


CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
COLLECTION = ROOT / "outputs/phase6h_calibration/collection"
OUT = ROOT / "outputs/phase6h_calibration/calibration"
CANDIDATES = OUT / "candidate_policies"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def load_fit_rows(config: dict) -> pd.DataFrame:
    integrity = load_json(COLLECTION / "collection_integrity.json")
    if integrity.get("status") != "PASS" or integrity.get("cal_holdout_opened") is not False:
        raise RuntimeError("complete CAL-FIT-only collection is required before fitting")
    summary = pd.read_csv(COLLECTION / "collection_run_summary.csv")
    expected = 9 * len(config["seeds"]["CAL_FIT_COLLECTION"])
    if len(summary) != expected or set(summary.instance_relative_path.str.split("/").str[0]) != {"cal_fit"}:
        raise RuntimeError("collection summary is not the frozen CAL-FIT task set")
    frames = []
    for row in summary.itertuples(index=False):
        path = COLLECTION / "live_logs" / row.instance_id / f"seed_{row.seed}.parquet"
        if digest(path) != row.live_log_sha256:
            raise RuntimeError(f"CAL-FIT live-log hash mismatch: {path}")
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "instance_id", "scale", "CF_level", "raw_score", "raw_utility",
        "realized_positive", "realized_immediate_utility", "search_progress",
    }
    if not required <= set(frame) or not (frame.calibration_split == "CAL_FIT").all():
        raise RuntimeError("CAL-FIT calibration rows are incomplete or contaminated")
    return frame


def fold_map(frame: pd.DataFrame) -> dict[str, int]:
    scales = {name: index for index, name in enumerate(("S", "M", "L"))}
    cfs = {name: index for index, name in enumerate(("CF1", "CF2", "CF3"))}
    groups = frame[["instance_id", "scale", "CF_level"]].drop_duplicates()
    return {
        row.instance_id: (scales[row.scale] + cfs[row.CF_level]) % 3
        for row in groups.itertuples(index=False)
    }


def fit_probability_cv(
    frame: pd.DataFrame, config: dict, phase6f: dict
) -> tuple[FrozenCalibrator, str, pd.DataFrame, pd.DataFrame]:
    methods = list(config["probability_calibration"]["candidate_methods"])
    folds = fold_map(frame)
    labels = frame.realized_positive.astype(int).to_numpy()
    scores = frame.raw_score.to_numpy(dtype=float)
    records = []
    predictions = []
    for method_index, method in enumerate(methods):
        oof = np.empty(len(frame), dtype=float)
        for fold in range(3):
            test = frame.instance_id.map(folds).to_numpy() == fold
            train = ~test
            calibrator = (
                FrozenCalibrator(**phase6f["probability_calibrator"])
                if method == "PHASE6G"
                else fit_probability_calibrator(scores[train], labels[train], method)
            )
            oof[test] = calibrator.predict(scores[test])
            metrics = calibration_metrics(
                oof[test], labels[test],
                bins=int(config["probability_calibration"]["reliability_bins"]),
            )
            records.append({"method": method, "fold": fold, **metrics})
        aggregate = calibration_metrics(
            oof, labels,
            bins=int(config["probability_calibration"]["reliability_bins"]),
        )
        predictions.append(pd.DataFrame({
            "row_index": np.arange(len(frame)),
            "method": method,
            "oof_probability": oof,
            "realized_positive": labels,
        }))
        records.append({
            "method": method,
            "fold": "OOF_ALL",
            **aggregate,
            "simplicity_order": method_index,
        })
    metrics = pd.DataFrame(records)
    aggregate = metrics[metrics.fold == "OOF_ALL"].copy()
    selected = aggregate.sort_values([
        "negative_log_likelihood", "brier_score",
        "expected_calibration_error", "simplicity_order",
    ]).iloc[0]
    method = str(selected.method)
    calibrator = (
        FrozenCalibrator(**phase6f["probability_calibrator"])
        if method == "PHASE6G"
        else fit_probability_calibrator(scores, labels, method)
    )
    return calibrator, method, metrics, pd.concat(predictions, ignore_index=True)


def utility_metrics(prediction: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    correlation = spearmanr(prediction, outcome).statistic
    return {
        "mae": float(np.mean(np.abs(prediction - outcome))),
        "rmse": float(np.sqrt(np.mean((prediction - outcome) ** 2))),
        "spearman_r": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def fit_utility_cv(
    frame: pd.DataFrame, config: dict, phase6f: dict
) -> tuple[FrozenCalibrator, str, pd.DataFrame]:
    folds = fold_map(frame)
    predictors = frame.raw_utility.to_numpy(dtype=float)
    outcomes = frame.realized_immediate_utility.to_numpy(dtype=float)
    methods = list(config["utility_calibration"]["candidate_methods"])
    records = []
    oof_by_method: dict[str, np.ndarray] = {}
    for method in methods:
        oof = np.empty(len(frame), dtype=float)
        for fold in range(3):
            test = frame.instance_id.map(folds).to_numpy() == fold
            train = ~test
            calibrator = (
                FrozenCalibrator(**phase6f["utility_calibrator"])
                if method == "PHASE6G"
                else fit_utility_calibrator(predictors[train], outcomes[train])
            )
            oof[test] = calibrator.predict(predictors[test])
            records.append({
                "method": method,
                "fold": fold,
                **utility_metrics(oof[test], outcomes[test]),
            })
        oof_by_method[method] = oof
        records.append({
            "method": method,
            "fold": "OOF_ALL",
            **utility_metrics(oof, outcomes),
        })
    metrics = pd.DataFrame(records)
    aggregate = metrics[metrics.fold == "OOF_ALL"].copy()
    phase6g_rank = float(aggregate.loc[aggregate.method == "PHASE6G", "spearman_r"].iloc[0])
    minimum_rank = phase6g_rank - float(
        config["utility_calibration"]["maximum_spearman_degradation"]
    )
    eligible = aggregate[aggregate.spearman_r >= minimum_rank]
    selected = eligible.sort_values(["mae", "rmse", "method"]).iloc[0]
    method = str(selected.method)
    calibrator = (
        FrozenCalibrator(**phase6f["utility_calibrator"])
        if method == "PHASE6G"
        else fit_utility_calibrator(predictors, outcomes)
    )
    return calibrator, method, metrics


def choose_threshold(
    frame: pd.DataFrame,
    probability: np.ndarray,
    utility: np.ndarray,
    config: dict,
    *,
    use_utility: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    study = config["intervention_gate_study"]
    rows = []
    utility_grid = study["utility_threshold_grid"] if use_utility else [-1e9]
    outcome = frame.realized_immediate_utility.to_numpy(dtype=float)
    positive = frame.realized_positive.to_numpy(dtype=bool)
    for probability_threshold in study["probability_threshold_grid"]:
        for utility_threshold in utility_grid:
            selected = (probability >= probability_threshold) & (utility >= utility_threshold)
            count = int(selected.sum())
            coverage = float(selected.mean())
            eligible = (
                count >= int(study["minimum_selected_samples"])
                and float(study["minimum_coverage"]) <= coverage
                <= float(study["maximum_coverage"])
            )
            rows.append({
                "probability_threshold": float(probability_threshold),
                "utility_threshold": float(utility_threshold),
                "selected_count": count,
                "coverage": coverage,
                "mean_realized_immediate_utility": (
                    float(outcome[selected].mean()) if count else np.nan
                ),
                "realized_positive_fraction": (
                    float(positive[selected].mean()) if count else np.nan
                ),
                "eligible": eligible,
            })
    table = pd.DataFrame(rows)
    eligible = table[table.eligible]
    if eligible.empty:
        raise RuntimeError("no preregistered threshold satisfies coverage/sample constraints")
    selected = eligible.sort_values([
        "mean_realized_immediate_utility", "realized_positive_fraction",
        "coverage", "probability_threshold", "utility_threshold",
    ], ascending=[False, False, False, False, False]).iloc[0]
    return {
        "confidence": float(selected.probability_threshold),
        "predicted_utility": float(selected.utility_threshold),
        "decision_margin": -1e9,
    }, table


def write_policy(
    name: str,
    *,
    probability: FrozenCalibrator,
    utility: FrozenCalibrator,
    thresholds: dict,
    support_guard: dict,
    config: dict,
) -> dict:
    payload = {
        "schema": "phase6h-candidate-policy-v1",
        "status": "CAL_FIT_ONLY_CANDIDATE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_name": name,
        "checkpoint_sha256": config["frozen_phase6f"]["checkpoint_sha256"],
        "probability_calibrator": probability.to_dict(),
        "utility_calibrator": utility.to_dict(),
        "thresholds": thresholds,
        "support_guard": support_guard,
        "candidate_bank_repair_acceptance_frozen": True,
        "cal_holdout_opened": False,
    }
    path = CANDIDATES / f"{name}.json"
    atomic_json(payload, path)
    return {
        "policy_name": name,
        "relative_path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "probability_method": probability.method,
        "utility_method": utility.method,
        **thresholds,
        "support_guard_enabled": bool(support_guard["enabled"]),
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    phase6f = load_json(ROOT / config["frozen_phase6f"]["experiment_freeze"])
    frame = load_fit_rows(config)
    probability, probability_method, probability_cv, oof = fit_probability_cv(
        frame, config, phase6f
    )
    utility, utility_method, utility_cv = fit_utility_cv(frame, config, phase6f)
    probability_values = probability.predict(frame.raw_score.to_numpy(dtype=float))
    utility_values = utility.predict(frame.raw_utility.to_numpy(dtype=float))

    reliability_rows = []
    for method in config["probability_calibration"]["candidate_methods"]:
        method_oof = oof[oof.method == method]
        table = reliability_table(
            method_oof.oof_probability,
            method_oof.realized_positive,
            bins=int(config["probability_calibration"]["reliability_bins"]),
        )
        table.insert(0, "method", method)
        reliability_rows.append(table)
    atomic_csv(probability_cv, OUT / "probability_cross_validation.csv")
    atomic_csv(pd.concat(reliability_rows, ignore_index=True), OUT / "reliability_bins.csv")
    atomic_csv(utility_cv, OUT / "utility_cross_validation.csv")
    atomic_parquet(oof, OUT / "probability_oof_predictions.parquet")

    feature_names = config["support_guard"]["features"]
    bounds = {
        name: {
            "lower": float(frame[name].quantile(config["support_guard"]["lower_quantile"])),
            "upper": float(frame[name].quantile(config["support_guard"]["upper_quantile"])),
        }
        for name in feature_names
    }
    support_disabled = {"enabled": False, "bounds": {}}
    support_enabled = {
        "enabled": True,
        "bounds": bounds,
        "maximum_out_of_range_features": int(
            config["support_guard"]["maximum_out_of_range_features"]
        ),
    }
    probability_thresholds, probability_grid = choose_threshold(
        frame, probability_values, utility_values, config, use_utility=False
    )
    combined_thresholds, combined_grid = choose_threshold(
        frame, probability_values, utility_values, config, use_utility=True
    )
    probability_grid.insert(0, "gate", "CALIBRATED_PROBABILITY")
    combined_grid.insert(0, "gate", "CALIBRATED_PROBABILITY_UTILITY")
    atomic_csv(
        pd.concat((probability_grid, combined_grid), ignore_index=True),
        OUT / "threshold_grid.csv",
    )

    candidate_rows = []
    candidate_rows.append(write_policy(
        "PHASE6G_REFERENCE",
        probability=FrozenCalibrator(**phase6f["probability_calibrator"]),
        utility=FrozenCalibrator(**phase6f["utility_calibrator"]),
        thresholds=dict(phase6f["selective_intervention_thresholds"]),
        support_guard=support_disabled,
        config=config,
    ))
    candidate_rows.append(write_policy(
        "CALIBRATED_PROBABILITY",
        probability=probability,
        utility=utility,
        thresholds=probability_thresholds,
        support_guard=support_disabled,
        config=config,
    ))
    candidate_rows.append(write_policy(
        "CALIBRATED_PROBABILITY_UTILITY",
        probability=probability,
        utility=utility,
        thresholds=combined_thresholds,
        support_guard=support_disabled,
        config=config,
    ))
    candidate_rows.append(write_policy(
        "CALIBRATED_PROBABILITY_UTILITY_SUPPORT",
        probability=probability,
        utility=utility,
        thresholds=combined_thresholds,
        support_guard=support_enabled,
        config=config,
    ))
    candidates = pd.DataFrame(candidate_rows)
    atomic_csv(candidates, OUT / "candidate_policy_manifest.csv")

    frame = frame.copy()
    frame["phase6h_probability"] = probability_values
    frame["phase6h_utility"] = utility_values
    progress_labels = pd.cut(
        frame.search_progress,
        bins=[0, .2, .4, .6, .8, 1.0],
        include_lowest=True,
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    frame["search_stage"] = progress_labels.astype(str)
    subgroup_rows = []
    for grouping in ("scale", "CF_level", "search_stage"):
        for group, part in frame.groupby(grouping, observed=True):
            metrics = calibration_metrics(
                part.phase6h_probability,
                part.realized_positive,
                bins=int(config["probability_calibration"]["reliability_bins"]),
            )
            subgroup_rows.append({
                "grouping": grouping,
                "group": group,
                "count": len(part),
                **metrics,
                **{f"utility_{key}": value for key, value in utility_metrics(
                    part.phase6h_utility.to_numpy(dtype=float),
                    part.realized_immediate_utility.to_numpy(dtype=float),
                ).items()},
            })
    atomic_csv(pd.DataFrame(subgroup_rows), OUT / "calibration_by_subgroup.csv")

    artifact = {
        "schema": "phase6h-calibration-fit-artifact-v1",
        "status": "FITTED_ON_CAL_FIT_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase6h_config_sha256": digest(CONFIG_PATH),
        "checkpoint_sha256": config["frozen_phase6f"]["checkpoint_sha256"],
        "cal_fit_state_count": len(frame),
        "cal_fit_instance_count": frame.instance_id.nunique(),
        "probability_method": probability_method,
        "probability_calibrator": probability.to_dict(),
        "utility_method": utility_method,
        "utility_calibrator": utility.to_dict(),
        "support_guard": support_enabled,
        "candidate_policy_hashes": {
            row["policy_name"]: row["sha256"] for row in candidate_rows
        },
        "cal_holdout_opened": False,
    }
    artifact_path = OUT / "calibration_fit_artifact.json"
    atomic_json(artifact, artifact_path)
    atomic_json({
        "schema": "phase6h-calibration-fit-integrity-v1",
        "status": "PASS",
        "artifact_sha256": digest(artifact_path),
        "candidate_count": len(candidate_rows),
        "candidate_manifest_sha256": digest(OUT / "candidate_policy_manifest.csv"),
        "all_data_cal_fit": bool((frame.calibration_split == "CAL_FIT").all()),
        "cal_holdout_opened": False,
    }, OUT / "fit_integrity.json")
    print(
        f"PHASE6H_CALIBRATION_FIT_COMPLETE states={len(frame)} "
        f"probability={probability_method} utility={utility_method}",
        flush=True,
    )


if __name__ == "__main__":
    main()
