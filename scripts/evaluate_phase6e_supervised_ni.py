#!/usr/bin/env python3
"""One-time frozen Phase 6E validation rehearsal and internal-holdout evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.baselines import (  # noqa: E402
    Phase6CTabularDiagnostic,
    fixed_original_selection,
)
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import file_sha256, load_shard_cache  # noqa: E402
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.metrics import evaluate_action_scores, selected_action_table  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.ni.trainer import NITrainingConfig  # noqa: E402


HOLDOUT_SPLIT = "TRAIN_INTERNAL_HOLDOUT"
VALIDATION_SPLIT = "TRAIN_VALIDATION"
EXPECTED_HOLDOUT_STATES = 20_000
EXPECTED_HOLDOUT_ACTIONS = 472_452
BASE_COLUMNS = [
    "instance_id", "training_split", "state_id", "scale", "CF_level",
    "RI_level", "TI_level", "search_stage", "bottleneck_proxy",
    "current_makespan", "target_set_id", "arm_family",
    "origin_destroy_operator", "origin_rules", "destroy_count", "mean_relative_improvement",
    "positive_under_2_of_3", "rank_within_state", "rank_percentile", "top1",
    "top3", "regret_to_best",
]
STRUCTURAL_DIMENSIONS = (
    "scale", "CF_level", "RI_level", "TI_level", "search_stage",
    "bottleneck_proxy",
)
SCORE_METHODS = (
    "FULL_CSG_SEED_660201", "FULL_CSG_SEED_660202", "FULL_CSG_SEED_660203",
    "FULL_CSG_ENSEMBLE", "FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES",
    "B3_PHASE6C_TABULAR",
)
SELECTED_METHODS = (
    "FULL_CSG_ENSEMBLE", "B0_RANDOM_EXPECTATION", "B1_FIXED_RELATED",
    "B2_BEST_FIXED_ORIGINAL", "B3_PHASE6C_TABULAR", "FLAT_SET",
    "STATIC_CSG", "NO_EDGE_FEATURES",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_guards(args: argparse.Namespace) -> dict[str, object]:
    final_audit = read_json(args.final_training_audit)
    ablation_audit = read_json(args.ablation_training_audit)
    baseline_freeze = read_json(args.baseline_freeze)
    if final_audit.get("status") != "PASS":
        raise ValueError("final training audit is not PASS")
    if ablation_audit.get("status") != "PASS":
        raise ValueError("ablation training audit is not PASS")
    if baseline_freeze.get("status") != "FROZEN_BEFORE_INTERNAL_HOLDOUT":
        raise ValueError("baseline freeze is not valid")
    plan_hash = file_sha256(args.evaluation_config)
    final_manifest_hash = file_sha256(args.final_checkpoint_manifest)
    ablation_manifest_hash = file_sha256(args.ablation_checkpoint_manifest)
    if final_audit.get("evaluation_plan_sha256") != plan_hash:
        raise ValueError("evaluation plan changed after final training audit")
    if ablation_audit.get("evaluation_plan_sha256") != plan_hash:
        raise ValueError("evaluation plan changed after ablation audit")
    if baseline_freeze.get("evaluation_plan_sha256") != plan_hash:
        raise ValueError("evaluation plan changed after baseline freeze")
    if final_audit.get("checkpoint_manifest_sha256") != final_manifest_hash:
        raise ValueError("final checkpoint manifest changed after audit")
    if ablation_audit.get("checkpoint_manifest_sha256") != ablation_manifest_hash:
        raise ValueError("ablation checkpoint manifest changed after audit")
    if baseline_freeze.get("ablation_training_audit_sha256") != file_sha256(
        args.ablation_training_audit
    ):
        raise ValueError("ablation audit changed after baseline freeze")
    if baseline_freeze.get("B3_model_sha256") != file_sha256(args.tabular_model):
        raise ValueError("frozen B3 model hash mismatch")
    return {
        "evaluation_plan_sha256": plan_hash,
        "final_training_audit_sha256": file_sha256(args.final_training_audit),
        "ablation_training_audit_sha256": file_sha256(args.ablation_training_audit),
        "baseline_freeze_sha256": file_sha256(args.baseline_freeze),
        "final_checkpoint_manifest_sha256": final_manifest_hash,
        "ablation_checkpoint_manifest_sha256": ablation_manifest_hash,
        "tabular_model_sha256": file_sha256(args.tabular_model),
    }


def checkpoint_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    final_manifest = read_json(args.final_checkpoint_manifest)
    ablation_manifest = read_json(args.ablation_checkpoint_manifest)
    if final_manifest.get("status") != "COMPLETE":
        raise ValueError("final checkpoint manifest is incomplete")
    if ablation_manifest.get("status") != "COMPLETE":
        raise ValueError("ablation checkpoint manifest is incomplete")
    specifications = []
    for record in final_manifest["checkpoints"]:
        specifications.append({
            "method": f"FULL_CSG_SEED_{int(record['seed'])}",
            **record,
        })
    for record in ablation_manifest["checkpoints"]:
        specifications.append({"method": str(record["variant"]), **record})
    expected = {
        "FULL_CSG_SEED_660201", "FULL_CSG_SEED_660202", "FULL_CSG_SEED_660203",
        "FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES",
    }
    if {str(spec["method"]) for spec in specifications} != expected:
        raise ValueError("checkpoint methods do not match frozen evaluation plan")
    for specification in specifications:
        checkpoint = Path(str(specification["checkpoint_path"]))
        if file_sha256(checkpoint) != specification["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    return specifications


def cache_records(
    manifest_path: Path,
    split: str,
    *,
    max_shards: int | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    records = manifest[
        manifest["status"].eq("COMPLETE") & manifest["training_split"].eq(split)
    ].sort_values("instance_id")
    if max_shards is not None:
        records = records.head(max_shards)
    if records.empty:
        raise ValueError(f"cache manifest has no complete {split} records")
    return records


def score_artifact_paths(output_root: Path, method: str) -> tuple[Path, Path]:
    stem = method.lower()
    return output_root / "scores" / f"{stem}.parquet", output_root / "scores" / f"{stem}.json"


def valid_score_artifact(
    output_root: Path,
    method: str,
    checkpoint_sha256: str,
    expected_actions: int,
    split: str,
) -> pd.DataFrame | None:
    score_path, record_path = score_artifact_paths(output_root, method)
    if not score_path.exists() or not record_path.exists():
        return None
    record = read_json(record_path)
    valid = (
        record.get("status") == "COMPLETE"
        and record.get("method") == method
        and record.get("checkpoint_sha256") == checkpoint_sha256
        and record.get("evaluation_split") == split
        and int(record.get("action_count", -1)) == expected_actions
        and int(record.get("score_bytes", -1)) == score_path.stat().st_size
        and record.get("score_sha256") == file_sha256(score_path)
    )
    if not valid:
        return None
    frame = pd.read_parquet(score_path)
    if len(frame) != expected_actions or frame[["state_id", "target_set_id"]].duplicated().any():
        return None
    return frame


def save_score_artifact(
    frame: pd.DataFrame,
    output_root: Path,
    *,
    method: str,
    checkpoint_sha256: str,
    split: str,
    profile: dict[str, object],
) -> None:
    score_path, record_path = score_artifact_paths(output_root, method)
    write_parquet(frame, score_path)
    record = {
        "schema": "phase6e-score-artifact-v1",
        "status": "COMPLETE",
        "method": method,
        "evaluation_split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "state_count": int(frame["state_id"].nunique()),
        "action_count": len(frame),
        "score_path": str(score_path.resolve()),
        "score_bytes": score_path.stat().st_size,
        "score_sha256": file_sha256(score_path),
        **profile,
    }
    write_json(record_path, record)


def score_checkpoint(
    records: pd.DataFrame,
    specification: dict[str, object],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    method = str(specification["method"])
    checkpoint_path = Path(str(specification["checkpoint_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = NIModelConfig(**checkpoint["model_config"])
    training_config = NITrainingConfig(**checkpoint["training_config"])
    tensorizer = CSGTensorizer()
    model = CSGTargetSetScorer(tensorizer, model_config)
    if checkpoint.get("tensor_schema_hash") != model.tensor_schema_hash:
        raise ValueError(f"tensor schema mismatch for {method}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    del checkpoint
    amp_enabled = training_config.mixed_precision and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    frames = []
    total_states = int(records["state_count"].sum())
    completed_states = 0
    started = time.perf_counter()
    with torch.no_grad():
        for completed_shards, record in enumerate(records.to_dict("records"), start=1):
            samples, _ = load_shard_cache(
                Path(str(record["cache_path"])),
                expected_tensor_schema_hash=model.tensor_schema_hash,
                expected_source_shard_sha256=str(record["source_shard_sha256"]),
            )
            scale = str(samples[0].structural_metadata["scale"])
            batch_size = training_config.batch_size(scale)
            for start in range(0, len(samples), batch_size):
                batch = batch_state_samples(samples[start:start + batch_size]).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    scores = model(batch).scores
                frames.append(pd.DataFrame({
                    "state_id": [
                        batch.state_ids[index]
                        for index in batch.action_to_state.detach().cpu().tolist()
                    ],
                    "target_set_id": batch.target_set_ids,
                    "score": scores.detach().float().cpu().numpy(),
                }))
                del batch, scores
            completed_states += len(samples)
            elapsed = time.perf_counter() - started
            rate = completed_states / max(elapsed, 1e-9)
            event = {
                "event": "evaluation_shard",
                "method": method,
                "completed_shards": completed_shards,
                "total_shards": len(records),
                "completed_states": completed_states,
                "total_states": total_states,
                "states_per_second": rate,
                "eta_seconds": (total_states - completed_states) / max(rate, 1e-9),
                "instance_id": record["instance_id"],
            }
            print(json.dumps(event), flush=True)
    elapsed = time.perf_counter() - started
    scored = pd.concat(frames, ignore_index=True)
    profile = {
        "runtime_seconds": elapsed,
        "states_per_second": total_states / max(elapsed, 1e-9),
        "gpu_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda" else 0.0
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored, profile


def load_aggregates(
    dataset_root: Path,
    directory: str,
    *,
    instance_ids: set[str] | None = None,
) -> pd.DataFrame:
    paths = sorted(dataset_root.glob(f"{directory}/*/target_set_aggregates.parquet"))
    if instance_ids is not None:
        paths = [path for path in paths if path.parent.name in instance_ids]
    if not paths:
        raise ValueError(f"no Phase 6C aggregates for {directory}")
    frame = pd.concat(
        [pd.read_parquet(path, columns=BASE_COLUMNS) for path in paths],
        ignore_index=True,
    )
    if frame[["state_id", "target_set_id"]].duplicated().any():
        raise ValueError(f"duplicate state/action keys in {directory}")
    return frame


def attach_scores(base: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    result = base.merge(
        scores[["state_id", "target_set_id", "score"]],
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(base):
        raise ValueError("score/base action coverage mismatch")
    return result


def selected_columns(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    columns = [
        "state_id", "scale", "CF_level", "RI_level", "TI_level",
        "search_stage", "bottleneck_proxy",
    ]
    return frame.assign(
        model=method,
        selected_utility=frame["mean_relative_improvement"].astype(float),
        selected_positive=frame["mean_relative_improvement"].gt(0).astype(float),
        selected_top3=frame["top3"].astype(float),
        selected_regret=frame["regret_to_best"].astype(float),
        selected_rank_percentile=frame["rank_percentile"].astype(float),
    )[[*columns, "model", "target_set_id", "selected_utility", "selected_positive",
       "selected_top3", "selected_regret", "selected_rank_percentile"]]


def random_expected_table(base: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "state_id", "scale", "CF_level", "RI_level", "TI_level",
        "search_stage", "bottleneck_proxy",
    ]
    result = base.assign(
        positive=base["mean_relative_improvement"].gt(0).astype(float),
        top3_float=base["top3"].astype(float),
    ).groupby(dimensions, sort=False, as_index=False).agg(
        selected_utility=("mean_relative_improvement", "mean"),
        selected_positive=("positive", "mean"),
        selected_top3=("top3_float", "mean"),
        selected_regret=("regret_to_best", "mean"),
        selected_rank_percentile=("rank_percentile", "mean"),
    )
    result["model"] = "B0_RANDOM_EXPECTATION"
    result["target_set_id"] = "<EXACT_EXPECTATION>"
    return result[[*dimensions, "model", "target_set_id", "selected_utility",
                   "selected_positive", "selected_top3", "selected_regret",
                   "selected_rank_percentile"]]


def summarize_selected(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("model", sort=False).agg(
        state_count=("state_id", "nunique"),
        mean_selected_utility=("selected_utility", "mean"),
        median_selected_utility=("selected_utility", "median"),
        selected_positive_fraction=("selected_positive", "mean"),
        selected_top3_fraction=("selected_top3", "mean"),
        mean_selected_regret=("selected_regret", "mean"),
        mean_selected_rank_percentile=("selected_rank_percentile", "mean"),
    ).reset_index()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def pairwise_statistics(
    selected: pd.DataFrame,
    *,
    expected_state_count: int = EXPECTED_HOLDOUT_STATES,
) -> pd.DataFrame:
    primary = selected[selected["model"].eq("FULL_CSG_ENSEMBLE")][
        ["state_id", "selected_utility"]
    ].rename(columns={"selected_utility": "primary_utility"})
    rows = []
    for comparator in SELECTED_METHODS:
        if comparator == "FULL_CSG_ENSEMBLE":
            continue
        other = selected[selected["model"].eq(comparator)][
            ["state_id", "selected_utility"]
        ].rename(columns={"selected_utility": "comparator_utility"})
        paired = primary.merge(other, on="state_id", validate="one_to_one")
        if len(paired) != expected_state_count:
            raise ValueError(f"paired state coverage mismatch for {comparator}")
        difference = paired["primary_utility"] - paired["comparator_utility"]
        tolerance = 1e-12
        if np.all(np.abs(difference) <= tolerance):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(
                difference,
                zero_method="wilcox",
                alternative="two-sided",
                method="approx",
            )
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append({
            "primary": "FULL_CSG_ENSEMBLE",
            "comparator": comparator,
            "state_count": len(paired),
            "mean_paired_utility_delta": float(difference.mean()),
            "median_paired_utility_delta": float(difference.median()),
            "win_count": int((difference > tolerance).sum()),
            "tie_count": int((difference.abs() <= tolerance).sum()),
            "loss_count": int((difference < -tolerance).sum()),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_value_bh_fdr"] = value
        row["significant_at_0_05"] = bool(value < 0.05)
    return pd.DataFrame(rows)


def structural_metrics(
    base: pd.DataFrame,
    score_frames: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regimes: list[tuple[str, str, pd.Series]] = [
        ("ALL", "ALL", pd.Series(True, index=base.index))
    ]
    for dimension in STRUCTURAL_DIMENSIONS:
        for value in sorted(base[dimension].astype(str).unique()):
            regimes.append((dimension, value, base[dimension].astype(str).eq(value)))
    for method, scores in score_frames.items():
        scored = attach_scores(base, scores)
        for dimension, value, mask in regimes:
            metrics = evaluate_action_scores(scored.loc[mask])
            rows.append({
                "model": method,
                "metric_family": "ACTION_SCORING",
                "regime_dimension": dimension,
                "regime_value": value,
                **metrics,
            })
    selected_regimes = [("ALL", "ALL", None)]
    for dimension in STRUCTURAL_DIMENSIONS:
        for value in sorted(selected[dimension].astype(str).unique()):
            selected_regimes.append((dimension, value, dimension))
    for method, method_frame in selected.groupby("model", sort=False):
        for dimension, value, column in selected_regimes:
            part = method_frame if column is None else method_frame[
                method_frame[column].astype(str).eq(value)
            ]
            summary = summarize_selected(part).iloc[0].to_dict()
            rows.append({
                "model": method,
                "metric_family": "SELECTED_ACTION",
                "regime_dimension": dimension,
                "regime_value": value,
                **summary,
            })
    return pd.DataFrame(rows)


def run_preflight(
    args: argparse.Namespace,
    specifications: list[dict[str, object]],
) -> None:
    records = cache_records(
        args.cache_manifest,
        VALIDATION_SPLIT,
        max_shards=args.preflight_validation_shards,
    )
    output_root = args.output_root / "preflight"
    reference_keys = None
    results = []
    score_frames: dict[str, pd.DataFrame] = {}
    for specification in specifications:
        scored, profile = score_checkpoint(
            records, specification, device=torch.device(args.device)
        )
        keys = scored[["state_id", "target_set_id"]]
        if reference_keys is None:
            reference_keys = keys
        elif not reference_keys.equals(keys):
            raise ValueError("preflight model score ordering/coverage mismatch")
        save_score_artifact(
            scored,
            output_root,
            method=str(specification["method"]),
            checkpoint_sha256=str(specification["checkpoint_sha256"]),
            split=VALIDATION_SPLIT,
            profile=profile,
        )
        score_frames[str(specification["method"])] = scored
        results.append({"method": specification["method"], **profile})
    seed_methods = [f"FULL_CSG_SEED_{seed}" for seed in (660201, 660202, 660203)]
    ensemble = reference_keys.copy()
    ensemble["score"] = np.mean(np.stack([
        score_frames[method]["score"].to_numpy(dtype=float) for method in seed_methods
    ]), axis=0)
    score_frames["FULL_CSG_ENSEMBLE"] = ensemble

    base = load_aggregates(
        args.dataset_root,
        "validation",
        instance_ids=set(records["instance_id"].astype(str)),
    )
    if len(base) != int(records["action_count"].sum()):
        raise ValueError("preflight source/cache action coverage mismatch")
    tabular = joblib.load(args.tabular_model)
    features = pd.read_parquet(args.target_features)
    modeled = base.merge(
        features, on=["state_id", "target_set_id"], how="inner", validate="one_to_one"
    )
    b3_scores = modeled[["state_id", "target_set_id"]].copy()
    b3_scores["score"] = tabular.score(modeled)
    score_frames["B3_PHASE6C_TABULAR"] = b3_scores
    selected_tables = []
    metric_rows = []
    for method in SCORE_METHODS:
        scored = attach_scores(base, score_frames[method])
        metric_rows.append({"model": method, **evaluate_action_scores(scored)})
        if method in SELECTED_METHODS:
            selected_tables.append(selected_columns(
                selected_action_table(scored, model_name=method), method
            ))
    selected_tables.append(selected_columns(
        fixed_original_selection(base, "related"), "B1_FIXED_RELATED"
    ))
    best_operator = str(read_json(args.baseline_freeze)["B2_best_fixed_original_operator"])
    selected_tables.append(selected_columns(
        fixed_original_selection(base, best_operator), "B2_BEST_FIXED_ORIGINAL"
    ))
    selected_tables.append(random_expected_table(base))
    selected = pd.concat(selected_tables, ignore_index=True)
    expected_states = int(records["state_count"].sum())
    statistics = pairwise_statistics(
        selected, expected_state_count=expected_states
    )
    regimes = structural_metrics(base, score_frames, selected)
    write_csv(pd.DataFrame(metric_rows), output_root / "preflight_metrics.csv")
    write_csv(summarize_selected(selected), output_root / "preflight_selected_utility.csv")
    write_csv(statistics, output_root / "preflight_pairwise_statistics.csv")
    write_csv(regimes, output_root / "preflight_structural_regimes.csv")
    result = {
        "schema": "phase6e-holdout-evaluation-preflight-v1",
        "status": "PASS",
        "evaluation_split": VALIDATION_SPLIT,
        "shard_count": len(records),
        "state_count": int(records["state_count"].sum()),
        "action_count": int(records["action_count"].sum()),
        "methods": results,
        "score_method_count": len(SCORE_METHODS),
        "selected_method_count": int(selected["model"].nunique()),
        "paired_comparison_count": len(statistics),
        "structural_metric_row_count": len(regimes),
        "internal_holdout_accessed": False,
    }
    write_json(output_root / "preflight_result.json", result)
    print(json.dumps({"event": "preflight_complete", **result}), flush=True)


def open_holdout_gate(
    args: argparse.Namespace,
    guards: dict[str, object],
) -> None:
    access_path = args.output_root / "holdout_access_record.json"
    expected = {
        "schema": "phase6e-internal-holdout-access-v1",
        "experiment_version": args.experiment_version,
        "evaluation_split": HOLDOUT_SPLIT,
        "expected_state_count": EXPECTED_HOLDOUT_STATES,
        "expected_action_count": EXPECTED_HOLDOUT_ACTIONS,
        **guards,
    }
    if access_path.exists():
        existing = read_json(access_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(f"holdout resume guard mismatch: {key}")
        if existing.get("status") not in {"IN_PROGRESS", "COMPLETE"}:
            raise ValueError("invalid holdout access status")
        print(json.dumps({"event": "holdout_gate_resume", **existing}), flush=True)
        return
    record = {
        **expected,
        "status": "IN_PROGRESS",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_selection_after_open": False,
    }
    write_json(access_path, record)
    print(json.dumps({"event": "holdout_gate_opened", **record}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiment-version", default="phase6e-holdout-v2")
    parser.add_argument("--preflight-validation-shards", type=int)
    parser.add_argument(
        "--evaluation-config", type=Path, default=ROOT / "configs/phase6e_evaluation.json"
    )
    parser.add_argument(
        "--cache-manifest", type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "outputs/phase6c/dataset"
    )
    parser.add_argument(
        "--target-features", type=Path,
        default=ROOT / "outputs/phase6c/diagnostics/target_set_preaction_features.parquet",
    )
    parser.add_argument(
        "--final-training-audit", type=Path,
        default=ROOT / "outputs/phase6e/audit/final_training_audit.json",
    )
    parser.add_argument(
        "--ablation-training-audit", type=Path,
        default=ROOT / "outputs/phase6e/audit/ablation_training_audit.json",
    )
    parser.add_argument(
        "--baseline-freeze", type=Path,
        default=ROOT / "outputs/phase6e/audit/baseline_freeze_v2.json",
    )
    parser.add_argument(
        "--tabular-model", type=Path,
        default=ROOT / "outputs/phase6e/baselines/v2/phase6c_tabular_diagnostic.joblib",
    )
    parser.add_argument(
        "--final-checkpoint-manifest", type=Path,
        default=ROOT / "outputs/phase6e/training/final_seeds/checkpoint_manifest.json",
    )
    parser.add_argument(
        "--ablation-checkpoint-manifest", type=Path,
        default=ROOT / "outputs/phase6e/ablations/training/checkpoint_manifest.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/evaluation_v2"
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    guards = validate_guards(args)
    specifications = checkpoint_specs(args)
    if args.preflight_validation_shards is not None:
        if args.preflight_validation_shards < 1:
            raise ValueError("--preflight-validation-shards must be positive")
        run_preflight(args, specifications)
        return

    records = cache_records(args.cache_manifest, HOLDOUT_SPLIT)
    if (
        int(records["state_count"].sum()) != EXPECTED_HOLDOUT_STATES
        or int(records["action_count"].sum()) != EXPECTED_HOLDOUT_ACTIONS
        or len(records) != 81
    ):
        raise ValueError("frozen holdout cache coverage mismatch")
    open_holdout_gate(args, guards)
    access_path = args.output_root / "holdout_access_record.json"
    if read_json(access_path).get("status") == "COMPLETE":
        print(json.dumps({"event": "holdout_evaluation_already_complete"}), flush=True)
        return

    score_frames: dict[str, pd.DataFrame] = {}
    runtime_rows = []
    for specification in specifications:
        method = str(specification["method"])
        checkpoint_hash = str(specification["checkpoint_sha256"])
        existing = valid_score_artifact(
            args.output_root, method, checkpoint_hash,
            EXPECTED_HOLDOUT_ACTIONS, HOLDOUT_SPLIT,
        )
        if existing is not None:
            score_frames[method] = existing
            record = read_json(score_artifact_paths(args.output_root, method)[1])
            runtime_rows.append({"method": method, **{
                key: record[key] for key in (
                    "runtime_seconds", "states_per_second", "gpu_peak_memory_mib"
                )
            }})
            print(json.dumps({"event": "score_artifact_resume", "method": method}), flush=True)
            continue
        scored, profile = score_checkpoint(
            records, specification, device=torch.device(args.device)
        )
        if len(scored) != EXPECTED_HOLDOUT_ACTIONS:
            raise ValueError(f"holdout action coverage mismatch for {method}")
        save_score_artifact(
            scored,
            args.output_root,
            method=method,
            checkpoint_sha256=checkpoint_hash,
            split=HOLDOUT_SPLIT,
            profile=profile,
        )
        score_frames[method] = scored
        runtime_rows.append({"method": method, **profile})

    seed_methods = [f"FULL_CSG_SEED_{seed}" for seed in (660201, 660202, 660203)]
    reference = score_frames[seed_methods[0]][["state_id", "target_set_id"]]
    ensemble_values = []
    for method in seed_methods:
        if not reference.equals(score_frames[method][["state_id", "target_set_id"]]):
            raise ValueError("full-seed score ordering mismatch")
        ensemble_values.append(score_frames[method]["score"].to_numpy(dtype=float))
    ensemble = reference.copy()
    ensemble["score"] = np.mean(np.stack(ensemble_values), axis=0)
    score_frames["FULL_CSG_ENSEMBLE"] = ensemble
    ensemble_path, _ = score_artifact_paths(args.output_root, "FULL_CSG_ENSEMBLE")
    write_parquet(ensemble, ensemble_path)

    print(json.dumps({"event": "load_holdout_labels_once"}), flush=True)
    base = load_aggregates(args.dataset_root, "internal_holdout")
    if len(base) != EXPECTED_HOLDOUT_ACTIONS or base["state_id"].nunique() != EXPECTED_HOLDOUT_STATES:
        raise ValueError("holdout source coverage mismatch")

    model = joblib.load(args.tabular_model)
    if not isinstance(model, Phase6CTabularDiagnostic) or not model.fitted:
        raise ValueError("frozen B3 model is invalid")
    features = pd.read_parquet(args.target_features)
    modeled = base.merge(
        features, on=["state_id", "target_set_id"], how="inner", validate="one_to_one"
    )
    del features
    if len(modeled) != len(base):
        raise ValueError("B3 holdout feature coverage mismatch")
    b3_scores = modeled[["state_id", "target_set_id"]].copy()
    b3_scores["score"] = model.score(modeled)
    score_frames["B3_PHASE6C_TABULAR"] = b3_scores
    write_parquet(
        b3_scores,
        score_artifact_paths(args.output_root, "B3_PHASE6C_TABULAR")[0],
    )

    metric_rows = []
    selected_tables = []
    for method in SCORE_METHODS:
        scored = attach_scores(base, score_frames[method])
        metrics = evaluate_action_scores(scored)
        robust_label = scored["positive_under_2_of_3"].astype(int)
        metrics["robust_label_roc_auc"] = float(roc_auc_score(robust_label, scored["score"]))
        metrics["robust_label_pr_auc"] = float(
            average_precision_score(robust_label, scored["score"])
        )
        metric_rows.append({
            "model": method,
            "evaluation_split": HOLDOUT_SPLIT,
            **metrics,
        })
        if method in SELECTED_METHODS:
            selected_tables.append(selected_columns(
                selected_action_table(scored, model_name=method), method
            ))
        print(json.dumps({"event": "method_metrics_complete", "method": method}), flush=True)

    baseline_freeze = read_json(args.baseline_freeze)
    related = fixed_original_selection(base, "related")
    selected_tables.append(selected_columns(related, "B1_FIXED_RELATED"))
    best_operator = str(baseline_freeze["B2_best_fixed_original_operator"])
    selected_tables.append(selected_columns(
        fixed_original_selection(base, best_operator), "B2_BEST_FIXED_ORIGINAL"
    ))
    selected_tables.append(random_expected_table(base))
    selected = pd.concat(selected_tables, ignore_index=True)
    if set(selected["model"]) != set(SELECTED_METHODS):
        raise ValueError("selected-action method coverage mismatch")
    if not selected.groupby("model")["state_id"].nunique().eq(EXPECTED_HOLDOUT_STATES).all():
        raise ValueError("selected-action state coverage mismatch")

    metrics_frame = pd.DataFrame(metric_rows)
    selected_summary = summarize_selected(selected)
    statistics = pairwise_statistics(selected)
    print(json.dumps({"event": "structural_metrics_start"}), flush=True)
    regimes = structural_metrics(base, score_frames, selected)
    ablation = metrics_frame[
        metrics_frame["model"].isin(
            ["FULL_CSG_ENSEMBLE", "FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES"]
        )
    ].merge(selected_summary, on="model", suffixes=("_ranking", "_selected"))

    write_csv(metrics_frame, args.output_root / "internal_holdout_metrics.csv")
    write_csv(selected_summary, args.output_root / "selected_action_utility.csv")
    write_parquet(selected, args.output_root / "state_selected_actions.parquet")
    write_csv(statistics, ROOT / "outputs/phase6e/statistics/pairwise_statistics.csv")
    write_csv(regimes, args.output_root / "structural_regime_metrics.csv")
    write_csv(ablation, ROOT / "outputs/phase6e/ablations/ablation_summary.csv")
    write_csv(pd.DataFrame(runtime_rows), args.output_root / "holdout_inference_runtime.csv")

    output_paths = [
        args.output_root / "internal_holdout_metrics.csv",
        args.output_root / "selected_action_utility.csv",
        args.output_root / "state_selected_actions.parquet",
        ROOT / "outputs/phase6e/statistics/pairwise_statistics.csv",
        args.output_root / "structural_regime_metrics.csv",
        ROOT / "outputs/phase6e/ablations/ablation_summary.csv",
        args.output_root / "holdout_inference_runtime.csv",
    ]
    completion = {
        "schema": "phase6e-internal-holdout-evaluation-v1",
        "status": "COMPLETE",
        "experiment_version": args.experiment_version,
        "evaluation_split": HOLDOUT_SPLIT,
        "state_count": EXPECTED_HOLDOUT_STATES,
        "action_count": EXPECTED_HOLDOUT_ACTIONS,
        "method_count": len(SELECTED_METHODS),
        "score_method_count": len(SCORE_METHODS),
        "B2_best_fixed_original_operator": best_operator,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in output_paths
        },
        **guards,
    }
    write_json(args.output_root / "evaluation_completion.json", completion)
    access = read_json(access_path)
    access.update({
        "status": "COMPLETE",
        "completed_at_utc": completion["completed_at_utc"],
        "evaluation_completion_sha256": file_sha256(
            args.output_root / "evaluation_completion.json"
        ),
        "checkpoint_selection_after_open": False,
    })
    write_json(access_path, access)
    print(json.dumps({"event": "holdout_evaluation_complete", **completion}), flush=True)


if __name__ == "__main__":
    main()
