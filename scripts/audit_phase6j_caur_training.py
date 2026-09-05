#!/usr/bin/env python3
"""Reproduce the completed regular-family R12 evidence before activating J3."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_phase6j_caur as regular  # noqa: E402

OUTPUT = regular.OUT / "completion_integrity_audit.json"


def audit_regular_training() -> dict:
    protocol = regular.validate_protocol()
    protocol_sha = regular.digest(regular.PROTOCOL_PATH)
    progress = regular.load_json(regular.OUT / "progress.json")
    worker = regular.load_json(regular.OUT / "worker_status.json")
    if not (
        progress["status"] == "COMPLETE_J1_J2"
        and progress["completed_runs"] == progress["expected_runs"] == 18
        and worker["status"] == "COMPLETE" and worker["exit_code"] == 0
    ):
        raise RuntimeError("regular-family R12 training is not complete")
    artifacts = {}

    def remember(path):
        artifacts[str(path.relative_to(ROOT))] = regular.digest(path)

    source = pd.read_parquet(regular.SOURCE_PATH)
    source["oof_fold"] = [
        regular.grouped_oof_fold(str(scale), str(cf))
        for scale, cf in zip(source.scale, source.CF_level)
    ]
    key = ["state_id", "target_set_id"]
    lookup = source.set_index(key).sort_index()
    frames, run_rows = [], []
    for family in regular.FAMILIES:
        for seed in protocol["training"]["seeds"]:
            for fold in range(3):
                paths = regular.run_paths(family, int(seed), fold)
                if not regular.valid_run(paths, protocol_sha):
                    raise RuntimeError(f"invalid completed fold: {paths[2]}")
                record = regular.load_json(paths[2])
                frame = pd.read_parquet(paths[1])
                checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
                expected = lookup[lookup.oof_fold.eq(fold)]
                actual = frame.set_index(key).sort_index()
                pd.testing.assert_index_equal(expected.index, actual.index)
                for column in source.columns.difference(key):
                    pd.testing.assert_series_equal(expected[column], actual[column])
                if not (
                    frame.held_fold.eq(fold).all()
                    and frame.model_family.eq(family).all()
                    and frame.training_seed.eq(seed).all()
                    and frame.state_id.nunique() == 96
                    and frame.instance_id.nunique() == 6
                ):
                    raise RuntimeError("OOF run identity or grouped coverage failed")
                values = frame.filter(regex="^predicted_").to_numpy(float)
                if not np.isfinite(values).all():
                    raise RuntimeError("non-finite OOF prediction")
                if not all(torch.isfinite(tensor).all() for tensor in
                           checkpoint["trainable_model_state"].values()):
                    raise RuntimeError("non-finite checkpoint")
                if any(checkpoint[name] != value for name, value in {
                    "training_protocol_sha256": protocol_sha,
                    "model_family": family, "training_seed": seed, "held_fold": fold,
                    "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"],
                }.items()):
                    raise RuntimeError("checkpoint identity does not match run")
                history = record["history"]
                if {history["inner_training_fold"], history["inner_validation_fold"], fold} != {0, 1, 2}:
                    raise RuntimeError("nested epoch-selection folds overlap")
                if record["best_epoch"] != len(history["outer_final_fit"]):
                    raise RuntimeError("outer fit did not use the selected epoch")
                transform = regular.fit_feature_transform(source[source.oof_fold.ne(fold)])
                if checkpoint["feature_transform"] != transform.to_dict():
                    raise RuntimeError("normalization does not match training folds")
                frames.append(frame)
                run_rows.append({
                    "family": family, "seed": seed, "held_fold": fold,
                    "best_epoch": record["best_epoch"],
                    "runtime_seconds": record["runtime_seconds"],
                })
                for path in paths:
                    remember(path)
    combined = pd.concat(frames, ignore_index=True)
    aggregate = regular.OUT / "oof_predictions.parquet"
    sort_key = ["model_family", "training_seed", *key]
    pd.testing.assert_frame_equal(
        combined.sort_values(sort_key).reset_index(drop=True),
        pd.read_parquet(aggregate).sort_values(sort_key).reset_index(drop=True),
    )
    remember(aggregate)
    summaries = {}
    for family in regular.FAMILIES:
        result = regular.ensemble_family(combined, family, protocol)
        summary_path = regular.OUT / f"{family}_oof_summary.json"
        saved = regular.load_json(summary_path)
        for field in ("metrics", "calibration", "selected_gate"):
            if saved[field] != result[field]:
                raise RuntimeError(f"cannot reproduce {family} {field}")
        for name, field in (("ensemble_oof", "ensemble"),
                            ("state_metrics", "states"),
                            ("selected_winners", "selected_winners")):
            path = regular.OUT / f"{family}_{name}.parquet"
            pd.testing.assert_frame_equal(pd.read_parquet(path), result[field])
            remember(path)
        summaries[family] = saved
        remember(summary_path)
        for suffix in ("calibration_metrics.csv", "gate_grid.csv"):
            remember(regular.OUT / f"{family}_{suffix}")
    manifest_path = regular.CACHE / "tensor_manifest.csv"
    if regular.digest(manifest_path) != protocol["input_hashes"]["tensor_manifest"]:
        raise RuntimeError("tensor manifest changed")
    manifest = pd.read_csv(manifest_path)
    for row in manifest.itertuples(index=False):
        path = Path(row.cache_path)
        if path.stat().st_size != row.cache_bytes or regular.digest(path) != row.cache_sha256:
            raise RuntimeError(f"tensor shard bytes changed: {path}")
        remember(path)
    for path in (regular.PROTOCOL_PATH, regular.SOURCE_PATH, manifest_path,
                 regular.OUT / "progress.json", regular.OUT / "worker_status.json",
                 regular.OUT / "j3_activation_decision.json",
                 regular.OUT / "family_oof_metrics.csv",
                 regular.OUT / "three_seed_stability.csv"):
        remember(path)
    metrics = [summary["metrics"] for summary in summaries.values()]
    activation = (
        max(row["pairwise_accuracy"] for row in metrics) < 0.60
        or any(value < 0 for row in metrics for value in row["mean_spearman_by_scale"].values())
    )
    recorded = regular.load_json(regular.OUT / "j3_activation_decision.json")
    if recorded["j3_activated"] != activation:
        raise RuntimeError("J3 activation cannot be reproduced")
    return {
        "schema": "phase6j-caur-regular-training-integrity-v1", "status": "PASS",
        "runs": 18, "prediction_rows": len(combined), "states_per_family": 288,
        "actions_per_family": len(source), "training_protocol_sha256": protocol_sha,
        "checks": {
            "worker_exit_and_counts": True, "fold_files_and_hashes": True,
            "labels_features_and_fold_assignments_exact": True,
            "fold_normalization_reproduced": True, "checkpoint_values_finite": True,
            "nested_fold_separation": True, "aggregate_predictions_reproduced": True,
            "ensemble_metrics_calibration_and_gate_reproduced": True,
            "tensor_shard_bytes_exact": True, "j3_activation_reproduced": True,
            "r13_r14_locked": True,
        },
        "j3_required": activation, "runs_audited": run_rows,
        "artifact_sha256": artifacts, "r13_accessed": False, "r14_accessed": False,
        "scientific_stage_complete": False,
        "remaining_before_r13": ["conditional J3", "latency", "origin collapse audit",
                                  "R12 eligibility", "deployable family bundle"],
    }


if __name__ == "__main__":
    result = audit_regular_training()
    if OUTPUT.exists() and regular.load_json(OUTPUT) != result:
        raise RuntimeError("existing regular-family audit changed")
    regular.atomic_json(result, OUTPUT)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"artifact_sha256", "runs_audited"}}, indent=2))
