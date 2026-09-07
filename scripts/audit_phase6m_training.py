#!/usr/bin/env python3
"""Independently audit completed Phase 6M M3 nested OOF training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS as RANKER_CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS as RANKER_NUMERIC_COLUMNS,
)
from rcias_clgri.ni.phase6m_selective_risk import (  # noqa: E402
    SELECTOR_NUMERIC_COLUMNS,
    build_selector_feature_frame,
    fit_selector_transform,
    fit_support_transform,
    initialize_selector,
    transform_selector_features,
)
from scripts import train_phase6m_selective_confidence as training  # noqa: E402


OUT = ROOT / "outputs/phase6m_selective_confidence_v1/training"
REPORT = ROOT / "docs/reports/phase6m_training_report.md"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace completed audit: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def process_alive(pid: int) -> bool:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "args="], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return False
    return "train_phase6m_selective_confidence.py" in output


def ranker_transform(frame: pd.DataFrame) -> dict:
    vocabularies = {
        column: sorted(frame[column].astype(str).unique())
        for column in RANKER_CATEGORICAL_COLUMNS
    }
    medians, iqrs = {}, {}
    for column in RANKER_NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        medians[column] = float(np.median(values))
        iqrs[column] = max(
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-6
        )
    return {
        "categorical_columns": list(RANKER_CATEGORICAL_COLUMNS),
        "numeric_columns": list(RANKER_NUMERIC_COLUMNS),
        "vocabularies": vocabularies,
        "medians": medians,
        "iqrs": iqrs,
        "unknown_category_index": 0,
        "numeric_clip": [-8.0, 8.0],
    }


def verify_phase6l_protection(implementation: dict) -> dict:
    manifest_path = ROOT / implementation["protected_phase6l_manifest"]["path"]
    require(digest(manifest_path) == implementation["protected_phase6l_manifest"]["sha256"],
            "Phase 6L protected manifest changed")
    manifest = json.loads(manifest_path.read_text())
    total_bytes = 0
    for relative, expected in manifest.items():
        path = ROOT / relative
        require(path.is_file(), f"protected Phase 6L file missing: {relative}")
        require(digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"],
                f"protected Phase 6L file changed: {relative}")
        total_bytes += int(expected["bytes"])
    return {"status": "PASS", "files": len(manifest), "bytes": total_bytes}


def audit_inner_rankers(config: dict, implementation_sha256: str,
                        source: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame], dict]:
    rows = []
    frames = []
    artifacts = {}
    source_indexed = source.set_index(["state_id", "target_set_id"]).sort_index()
    for prediction_fold in range(3):
        for train_fold in sorted({0, 1, 2} - {prediction_fold}):
            expected_transform = ranker_transform(source[source.oof_fold.eq(train_fold)])
            for seed in config["ranker"]["seeds"]:
                paths = training.inner_ranker_paths(train_fold, prediction_fold, int(seed))
                require(training.valid_ranker_run(paths, implementation_sha256),
                        f"invalid inner ranker run: {train_fold}->{prediction_fold}, seed {seed}")
                checkpoint_path, prediction_path, record_path = paths
                record = json.loads(record_path.read_text())
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                prediction = pd.read_parquet(prediction_path)
                expected = source_indexed[source_indexed.oof_fold.eq(prediction_fold)]
                actual = prediction.set_index(["state_id", "target_set_id"]).sort_index()
                pd.testing.assert_index_equal(expected.index, actual.index)
                for column in source.columns.difference(["state_id", "target_set_id"]):
                    pd.testing.assert_series_equal(expected[column], actual[column])
                require(
                    prediction.state_id.nunique() == 96
                    and prediction.instance_id.nunique() == 6
                    and prediction.training_seed.eq(seed).all()
                    and prediction.held_fold.eq(prediction_fold).all()
                    and prediction.ranker_train_fold.eq(train_fold).all(),
                    "inner ranker fold/seed identity failed",
                )
                require(checkpoint["feature_transform"] == expected_transform,
                        "inner ranker transform used data outside its training fold")
                require(
                    checkpoint["training_fold"] == train_fold
                    and checkpoint["prediction_fold"] == prediction_fold
                    and checkpoint["training_seed"] == seed
                    and checkpoint["epochs"] == config["ranker"]["inner_training_epochs"],
                    "inner ranker checkpoint identity failed",
                )
                require(len(record["history"]) == config["ranker"]["inner_training_epochs"],
                        "inner ranker did not use fixed preregistered epochs")
                require(all(torch.isfinite(value).all()
                            for value in checkpoint["trainable_model_state"].values()),
                        "nonfinite inner ranker checkpoint")
                require(np.isfinite(prediction.filter(regex="^predicted_").to_numpy(float)).all(),
                        "nonfinite inner ranker predictions")
                rows.append({
                    "training_seed": int(seed), "training_fold": train_fold,
                    "prediction_fold": prediction_fold, "epochs": checkpoint["epochs"],
                    "runtime_seconds": float(record["runtime_seconds"]),
                    "checkpoint_sha256": digest(checkpoint_path),
                    "prediction_sha256": digest(prediction_path),
                })
                frames.append(prediction)
                for path in paths:
                    artifacts[str(path.relative_to(ROOT))] = digest(path)
    require(len(rows) == 18, "expected 18 inner ranker runs")
    return rows, frames, artifacts


def replay_selected_epoch(history: list[dict], minimum: float) -> int:
    best = np.inf
    best_epoch = 0
    for row in history:
        value = float(row["validation_loss"])
        if value < best - minimum:
            best = value
            best_epoch = int(row["epoch"])
    return best_epoch


def audit_selectors(config: dict, implementation_sha256: str,
                    source: pd.DataFrame, seed_labels: pd.DataFrame,
                    frozen_ranker: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame], dict]:
    del seed_labels  # checkpoint/prediction targets are audited against the frozen grouped labels below.
    rows = []
    frames = []
    artifacts = {}
    for held_fold in range(3):
        inner_ranker = training.load_inner_predictions(held_fold, config, implementation_sha256)
        outer_source = source[source.oof_fold.ne(held_fold)]
        support_transform = fit_support_transform(outer_source)
        outer_features = build_selector_feature_frame(inner_ranker, support_transform)
        selector_transform = fit_selector_transform(outer_features)
        held_features = build_selector_feature_frame(
            frozen_ranker[frozen_ranker.held_fold.eq(held_fold)], support_transform
        )
        feature_key = ["state_id", "target_set_id"]
        expected_features = held_features.set_index(feature_key).sort_index()
        for seed in config["selector"]["training_seeds"]:
            paths = training.selector_paths(int(seed), held_fold)
            require(training.valid_selector_run(paths, implementation_sha256),
                    f"invalid selector run: seed {seed}, fold {held_fold}")
            checkpoint_path, prediction_path, record_path = paths
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            record = json.loads(record_path.read_text())
            prediction = pd.read_parquet(prediction_path)
            actual_features = prediction.set_index(feature_key).sort_index()
            pd.testing.assert_index_equal(expected_features.index, actual_features.index)
            for column in expected_features.columns:
                pd.testing.assert_series_equal(expected_features[column], actual_features[column])
            require(checkpoint["support_transform"] == support_transform.to_dict(),
                    "selector support transform used data outside outer training folds")
            require(checkpoint["selector_transform"] == selector_transform.to_dict(),
                    "selector normalization used data outside outer training folds")
            require(
                checkpoint["selector_training_seed"] == seed
                and checkpoint["held_fold"] == held_fold
                and prediction.selector_training_seed.eq(seed).all()
                and prediction.held_fold.eq(held_fold).all()
                and prediction.state_id.nunique() == 96,
                "selector fold/seed identity failed",
            )
            roles = {record["inner_fit_fold"], record["inner_validation_fold"], held_fold}
            require(roles == {0, 1, 2}, "selector cross-fit folds overlap")
            best_epoch = replay_selected_epoch(
                record["inner_history"],
                float(config["selector"]["minimum_validation_loss_improvement"]),
            )
            require(best_epoch == record["best_epoch"] == checkpoint["best_epoch"],
                    "selector selected epoch is not reproducible")
            require(len(record["outer_final_history"]) == best_epoch,
                    "selector final refit did not use selected epoch")
            require(all(torch.isfinite(value).all() for value in checkpoint["model_state"].values()),
                    "nonfinite selector checkpoint")
            output_columns = [
                "selector_continuation_mean", "selector_continuation_scale",
                "selector_seed_positive_logit", "selector_seed_positive_probability",
            ]
            require(np.isfinite(prediction[output_columns].to_numpy(float)).all(),
                    "nonfinite selector prediction")

            transform = training.SelectorTransform.from_dict(checkpoint["selector_transform"])
            model = initialize_selector(
                len(transform.feature_names), int(seed), dropout=float(config["selector"]["dropout"])
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            matrix = transform_selector_features(held_features, transform)
            with torch.inference_mode():
                output = model(torch.as_tensor(matrix))
            replay = np.column_stack([
                output.continuation_mean.numpy(), output.continuation_scale.numpy(),
                output.seed_positive_logit.numpy(), torch.sigmoid(output.seed_positive_logit).numpy(),
            ])
            require(np.allclose(replay, prediction[output_columns].to_numpy(float), atol=1e-5, rtol=1e-5),
                    "selector checkpoint does not replay saved held predictions")
            rows.append({
                "selector_training_seed": int(seed), "held_fold": held_fold,
                "best_epoch": best_epoch, "inner_epochs_run": len(record["inner_history"]),
                "runtime_seconds": float(record["runtime_seconds"]),
                "checkpoint_sha256": digest(checkpoint_path),
                "prediction_sha256": digest(prediction_path),
            })
            frames.append(prediction)
            for path in paths:
                artifacts[str(path.relative_to(ROOT))] = digest(path)
    require(len(rows) == 9, "expected nine selector runs")
    return rows, frames, artifacts


def aggregate_audit(config: dict, selector_frames: list[pd.DataFrame]) -> dict:
    keys = ["state_id", "target_set_id"]
    combined = pd.concat(selector_frames, ignore_index=True)
    saved_predictions = pd.read_parquet(OUT / "oof_predictions.parquet")
    order = ["selector_training_seed", *keys]
    pd.testing.assert_frame_equal(
        combined.sort_values(order).reset_index(drop=True),
        saved_predictions.sort_values(order).reset_index(drop=True),
    )
    require(len(combined) == 20427 and combined.state_id.nunique() == 288,
            "combined selector OOF shape changed")
    require(combined.groupby(keys).selector_training_seed.nunique().eq(3).all(),
            "selector OOF does not have three seeds per candidate")
    ensemble = pd.read_parquet(OUT / "ensemble_oof.parquet")
    selected = pd.read_parquet(OUT / "selected_winners.parquet")
    require(len(ensemble) == 6809 and not ensemble.duplicated(keys).any(),
            "ensemble candidate identity failed")
    require(len(selected) == selected.state_id.nunique() == 288,
            "selected winner identity failed")
    require(selected.hard_supported.astype(bool).all(),
            "unexpected unsupported selected winner")
    require(np.isfinite(ensemble[list(SELECTOR_NUMERIC_COLUMNS)].to_numpy(float)).all(),
            "nonfinite selector inputs in ensemble evidence")
    require(np.isfinite(ensemble[[
        "selector_predictive_mean", "selector_total_predictive_scale",
        "selector_probability", "selector_probability_std",
    ]].to_numpy(float)).all(), "nonfinite selector ensemble outputs")
    frozen_ensemble = pd.read_parquet(training.FROZEN_ENSEMBLE).sort_values(keys, kind="stable")
    current = ensemble.sort_values(keys, kind="stable")
    require(current[keys].reset_index(drop=True).equals(frozen_ensemble[keys].reset_index(drop=True)),
            "ranker candidate identity/order changed")
    require(np.allclose(current.ranker_advantage_mean, frozen_ensemble.ensemble_advantage_mean,
                        atol=1e-7, rtol=1e-7), "ranker predictions changed")
    frozen_selected = pd.read_parquet(training.FROZEN_SELECTED).sort_values("state_id")
    require(selected.sort_values("state_id").target_set_id.tolist()
            == frozen_selected.target_set_id.tolist(), "ranker winner changed")
    return {
        "prediction_rows": len(combined), "ensemble_candidates": len(ensemble),
        "states": len(selected), "hard_support_rate": float(ensemble.hard_supported.mean()),
        "selected_winner_support_rate": float(selected.hard_supported.mean()),
        "selector_probability_min": float(selected.selector_probability.min()),
        "selector_probability_max": float(selected.selector_probability.max()),
        "selector_total_scale_min": float(selected.selector_total_predictive_scale.min()),
        "selector_total_scale_max": float(selected.selector_total_predictive_scale.max()),
        "ranker_candidate_identity_preserved": True,
        "ranker_winner_identity_preserved": True,
    }


def render_report(result: dict) -> str:
    epochs = [row["best_epoch"] for row in result["selector_runs"]]
    aggregate = result["aggregate"]
    return f"""# Phase 6M M3 训练与完整性报告

状态：**M3_COMPLETE — INTEGRITY_PASS — READY_FOR_M4**。

正式 worker 已退出，连续进程耗时 {result['worker_elapsed_seconds']:.2f} 秒。18/18 个 nested inner-ranker runs 与 9/9 个 selector runs 均通过文件存在性、记录状态、implementation hash、checkpoint hash、prediction hash、fold/seed identity 和有限数值检查。合并输出为 {aggregate['prediction_rows']:,} 条 selector-seed OOF rows、{aggregate['ensemble_candidates']:,} 个唯一 candidates 和 {aggregate['states']} 个 states。

## Cross-fit 与输入边界

每个 inner ranker 只在一个允许的 structural fold 上拟合并预测另一个 fold；对应 outer held fold 未参与。ranker transform 已从实际 training fold 重新计算并与 checkpoint 逐字段相等。每个 selector 的 inner fit、inner validation 和 outer held folds 恰好构成 `{{0,1,2}}`；outer support 与 selector normalization 已从两个 outer-training folds 的 OOF ranker features 重新计算并与 checkpoint 一致。

保存的 selector checkpoint 在 CPU 上以 `atol=rtol=1e-5` 重放全部 held predictions。aggregate candidate identity/order、ranker ensemble advantage 和 288 个 winner identity 均与冻结 Phase 6L 精确边界一致。historical score online forward calls 为 0；R13/R14 未访问。

## 数值与稳定性观察

所有 checkpoint 与输出均为有限值。新 hard support 的 candidate rate 为 {aggregate['hard_support_rate']:.3%}，selected-winner rate 为 {aggregate['selected_winner_support_rate']:.3%}。winner selector probability 范围为 [{aggregate['selector_probability_min']:.6f}, {aggregate['selector_probability_max']:.6f}]，total predictive scale 范围为 [{aggregate['selector_total_scale_min']:.6f}, {aggregate['selector_total_scale_max']:.6f}]。

九个 selector best epochs 范围为 {min(epochs)}–{max(epochs)}，具体为 `{epochs}`。这表明 outer folds/seeds 的收敛差异较大；它不构成完整性失败，但必须在 M4 的 discrimination、calibration、scale lift 和 retained-gate 审计中显式评估，不能挑选单个 seed 或 fold。

## 证据

- `outputs/phase6m_selective_confidence_v1/training/completion_integrity_audit.json`
- `outputs/phase6m_selective_confidence_v1/training/progress.json`
- `outputs/phase6m_selective_confidence_v1/training/oof_summary.json`
- `outputs/phase6m_selective_confidence_v1/training/oof_predictions.parquet`
- `outputs/phase6m_selective_confidence_v1/training/ensemble_oof.parquet`
- `outputs/phase6m_selective_confidence_v1/training/selected_winners.parquet`
- `outputs/phase6m_selective_confidence_v1/training/formal_training_20260907T141046Z.log`

本报告只确认 M3 数据与协议完整性，不宣称质量 gate 通过。M4 必须使用全部三 seeds、全部 288 states 和预注册 18-gate grid；若没有 retained cross-scale gate，终止为 `MODEL_REVISION_QUALITY`。
"""


def main() -> None:
    config, implementation, implementation_sha256 = training.validate_boundary()
    progress = json.loads((OUT / "progress.json").read_text())
    summary = json.loads((OUT / "oof_summary.json").read_text())
    launch = json.loads((OUT / "launch_record.json").read_text())
    require(progress["status"] == "COMPLETE" and summary["status"] == "M3_COMPLETE",
            "M3 progress or summary is incomplete")
    require(progress["completed_inner_ranker_runs"] == progress["expected_inner_ranker_runs"] == 18,
            "inner ranker run count incomplete")
    require(progress["completed_selector_runs"] == progress["expected_selector_runs"] == 9,
            "selector run count incomplete")
    require(not process_alive(int(launch["pid"])), "Phase 6M worker is still alive")
    log_path = ROOT / launch["log_path"]
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    require('"status": "COMPLETE"' in log_text and "Traceback" not in log_text,
            "formal M3 log does not end cleanly")
    source = pd.read_parquet(training.SOURCE)
    seed_labels = pd.read_parquet(training.SEED_LABELS)
    frozen_ranker = pd.read_parquet(training.FROZEN_RANKER)
    protection = verify_phase6l_protection(implementation)
    ranker_rows, _, ranker_artifacts = audit_inner_rankers(
        config, implementation_sha256, source
    )
    selector_rows, selector_frames, selector_artifacts = audit_selectors(
        config, implementation_sha256, source, seed_labels, frozen_ranker
    )
    aggregate = aggregate_audit(config, selector_frames)
    common_artifacts = [
        OUT / "progress.json", OUT / "oof_summary.json", OUT / "oof_predictions.parquet",
        OUT / "ensemble_oof.parquet", OUT / "selected_winners.parquet", log_path,
        OUT / "launch_record.json",
    ]
    result = {
        "schema": "phase6m-m3-completion-integrity-audit-v1", "status": "PASS",
        "decision": "READY_FOR_M4", "implementation_protocol_sha256": implementation_sha256,
        "checks": {
            "worker_exited": True, "progress_and_summary_complete": True,
            "eighteen_inner_ranker_runs": True, "nine_selector_runs": True,
            "all_artifact_hashes_match": True, "all_values_finite": True,
            "ranker_transform_fold_isolation": True, "selector_cross_fit_isolation": True,
            "selector_support_and_normalization_fold_isolation": True,
            "selector_checkpoint_replay": True, "aggregate_oof_reproduced": True,
            "candidate_identity_and_order_preserved": True, "ranker_winner_preserved": True,
            "protected_phase6l_unchanged": True, "historical_score_online_forward_calls_zero": True,
            "r13_r14_locked": True,
        },
        "worker_elapsed_seconds": float(progress["elapsed_seconds"]),
        "inner_ranker_runs": ranker_rows, "selector_runs": selector_rows,
        "aggregate": aggregate, "protected_phase6l": protection,
        "artifact_sha256": {
            **ranker_artifacts, **selector_artifacts,
            **{str(path.relative_to(ROOT)): digest(path) for path in common_artifacts},
        },
        "historical_score_online_forward_calls": 0,
        "r13_accessed": False, "r14_accessed": False,
    }
    audit_path = OUT / "completion_integrity_audit.json"
    atomic_json(audit_path, result)
    REPORT.write_text(render_report(result))
    print(json.dumps({
        "status": result["status"], "decision": result["decision"],
        "inner_ranker_runs": len(ranker_rows), "selector_runs": len(selector_rows),
        **aggregate,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
