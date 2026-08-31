#!/usr/bin/env python3
"""Evaluate frozen Phase 6E and Phase 6F policies on the same R06 states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from rcias_clgri.ni.calibration import (  # noqa: E402
    FrozenCalibrator,
    calibration_metrics,
    reliability_table,
)
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.metrics import (  # noqa: E402
    evaluate_action_scores,
    selected_action_table,
)
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.selective_policy import InterventionThresholds  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.ni.trainer import NITrainingConfig  # noqa: E402
from scripts.evaluate_phase6e_supervised_ni import (  # noqa: E402
    attach_scores,
    cache_records,
    load_aggregates,
    random_expected_table,
    selected_columns,
    summarize_selected,
    write_csv,
    write_json,
    write_parquet,
)


SPLIT = "REVISION_HOLDOUT"
EXPECTED_STATES = 8_100
EXPECTED_ACTIONS = 191_416
PRIMARY = "PHASE6F_REVISED_HYBRID_SEED_660301"
PAIRWISE_COMPARATORS = (
    "PHASE6E_FULL_CSG_ENSEMBLE",
    "PHASE6E_DEPLOYABLE_SINGLE_SEED_660201",
    "PHASE6C_TABULAR",
    "B1_FIXED_RELATED",
    "PHASE6E_FLAT_SET",
)
STRUCTURAL_DIMENSIONS = (
    "scale", "CF_level", "RI_level", "TI_level", "search_stage",
    "bottleneck_proxy",
)


def canonical_hash(payload: dict[str, object], excluded: str) -> str:
    body = {key: value for key, value in payload.items() if key != excluded}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_guards(args: argparse.Namespace) -> dict[str, str]:
    opened = json.loads(args.label_open_record.read_text(encoding="utf-8"))
    freeze = json.loads(args.experiment_freeze.read_text(encoding="utf-8"))
    cache_audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    feature_audit = json.loads(args.feature_audit.read_text(encoding="utf-8"))
    baseline_freeze = json.loads(args.phase6e_baseline_freeze.read_text(encoding="utf-8"))
    checks = {
        "r06_open_authorized": (
            opened.get("status") == "OPENED_FOR_FROZEN_EVALUATION"
            and opened.get("revision_holdout_labels_opened_after_model_freeze") is True
        ),
        "experiment_freeze_hash_valid": freeze.get("freeze_hash") == canonical_hash(
            freeze, "freeze_hash"
        ),
        "open_bound_to_experiment_freeze": opened.get("model_freeze_hash") == freeze.get(
            "freeze_hash"
        ),
        "selected_checkpoint_hash_valid": file_sha256(
            Path(str(freeze["selected_checkpoint_path"]))
        ) == freeze.get("selected_checkpoint_sha256"),
        "cache_audit_pass": cache_audit.get("status") == "PASS",
        "feature_audit_pass": feature_audit.get("status") == "PASS",
        "phase6e_baselines_frozen": baseline_freeze.get("status")
        == "FROZEN_BEFORE_INTERNAL_HOLDOUT",
        "phase6c_tabular_hash_valid": file_sha256(args.tabular_model)
        == baseline_freeze.get("B3_model_sha256"),
    }
    if not all(checks.values()):
        raise ValueError(f"R06 evaluation guard failed: {checks}")
    return {
        "experiment_freeze_sha256": file_sha256(args.experiment_freeze),
        "label_open_record_sha256": file_sha256(args.label_open_record),
        "cache_audit_sha256": file_sha256(args.cache_audit),
        "feature_audit_sha256": file_sha256(args.feature_audit),
        "phase6e_baseline_freeze_sha256": file_sha256(args.phase6e_baseline_freeze),
    }


def checkpoint_specs(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    old_manifest = json.loads(args.phase6e_manifest.read_text(encoding="utf-8"))
    ablations = json.loads(args.phase6e_ablation_manifest.read_text(encoding="utf-8"))
    revised_manifest = json.loads(args.phase6f_manifest.read_text(encoding="utf-8"))
    if any(item.get("status") != "COMPLETE" for item in (
        old_manifest, ablations, revised_manifest
    )):
        raise ValueError("checkpoint manifest is incomplete")
    old = [{
        **record,
        "method": f"PHASE6E_FULL_CSG_SEED_{int(record['seed'])}",
    } for record in old_manifest["checkpoints"]]
    flat_records = [
        record for record in ablations["checkpoints"] if record["variant"] == "FLAT_SET"
    ]
    if len(flat_records) != 1:
        raise ValueError("frozen Phase 6E FLAT_SET checkpoint is missing")
    old.append({**flat_records[0], "method": "PHASE6E_FLAT_SET"})
    revised = [{
        **record,
        "method": f"PHASE6F_REVISED_SEED_{int(record['seed'])}",
    } for record in revised_manifest["checkpoints"]]
    if {int(record["seed"]) for record in old[:3]} != {660201, 660202, 660203}:
        raise ValueError("Phase 6E seed set changed")
    if {int(record["seed"]) for record in revised} != {660301, 660302, 660303}:
        raise ValueError("Phase 6F seed set changed")
    if int(revised_manifest["selected_seed"]) != 660301:
        raise ValueError("deployment seed changed after R06 opening")
    for specification in [*old, *revised]:
        path = Path(str(specification["checkpoint_path"]))
        if file_sha256(path) != specification["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
    return old, revised


def score_paths(output_root: Path, method: str) -> tuple[Path, Path]:
    stem = method.lower()
    return output_root / "scores" / f"{stem}.parquet", output_root / "scores" / f"{stem}.json"


def load_score_artifact(
    output_root: Path,
    method: str,
    checkpoint_hash: str,
    expected_actions: int,
) -> pd.DataFrame | None:
    score_path, metadata_path = score_paths(output_root, method)
    if not score_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not (
        metadata.get("status") == "COMPLETE"
        and metadata.get("method") == method
        and metadata.get("checkpoint_sha256") == checkpoint_hash
        and int(metadata.get("action_count", -1)) == expected_actions
        and metadata.get("score_sha256") == file_sha256(score_path)
    ):
        return None
    frame = pd.read_parquet(score_path)
    if len(frame) != expected_actions or frame[["state_id", "target_set_id"]].duplicated().any():
        return None
    return frame


def save_score_artifact(
    frame: pd.DataFrame,
    output_root: Path,
    method: str,
    checkpoint_hash: str,
    profile: dict[str, float],
) -> None:
    score_path, metadata_path = score_paths(output_root, method)
    write_parquet(frame, score_path)
    write_json(metadata_path, {
        "schema": "phase6f-r06-score-artifact-v1",
        "status": "COMPLETE",
        "method": method,
        "evaluation_split": SPLIT,
        "checkpoint_sha256": checkpoint_hash,
        "state_count": int(frame["state_id"].nunique()),
        "action_count": len(frame),
        "score_sha256": file_sha256(score_path),
        **profile,
    })


def score_checkpoint(
    records: pd.DataFrame,
    specification: dict[str, object],
    device: torch.device,
    *,
    require_utility: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    method = str(specification["method"])
    checkpoint = torch.load(
        Path(str(specification["checkpoint_path"])),
        map_location="cpu",
        weights_only=False,
    )
    model = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    if checkpoint["tensor_schema_hash"] != model.tensor_schema_hash:
        raise ValueError(f"tensor schema mismatch for {method}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    training = NITrainingConfig(**checkpoint["training_config"])
    del checkpoint
    amp_enabled = training.mixed_precision and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    frames = []
    completed_states = 0
    total_states = int(records["state_count"].sum())
    started = time.perf_counter()
    with torch.no_grad():
        for shard_index, record in enumerate(records.to_dict("records"), start=1):
            samples, _ = load_shard_cache(
                ROOT / Path(str(record["cache_path"])),
                expected_tensor_schema_hash=model.tensor_schema_hash,
                expected_source_shard_sha256=str(record["source_shard_sha256"]),
            )
            batch_size = training.batch_size(str(samples[0].structural_metadata["scale"]))
            for start in range(0, len(samples), batch_size):
                batch = batch_state_samples(samples[start:start + batch_size]).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(batch)
                if require_utility and output.utility_predictions is None:
                    raise ValueError(f"utility head missing for {method}")
                action_state = batch.action_to_state.detach().cpu().tolist()
                values = {
                    "state_id": [batch.state_ids[index] for index in action_state],
                    "target_set_id": batch.target_set_ids,
                    "score": output.scores.detach().float().cpu().numpy(),
                }
                if require_utility:
                    values["utility_prediction"] = (
                        output.utility_predictions.detach().float().cpu().numpy()
                    )
                frames.append(pd.DataFrame(values))
                del batch, output
            completed_states += len(samples)
            elapsed = time.perf_counter() - started
            rate = completed_states / max(elapsed, 1e-9)
            print(json.dumps({
                "event": "r06_scoring_shard",
                "method": method,
                "completed_shards": shard_index,
                "total_shards": len(records),
                "completed_states": completed_states,
                "total_states": total_states,
                "states_per_second": rate,
                "eta_seconds": (total_states - completed_states) / max(rate, 1e-9),
            }), flush=True)
    elapsed = time.perf_counter() - started
    frame = pd.concat(frames, ignore_index=True)
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
    return frame, profile


def score_or_resume(
    records: pd.DataFrame,
    specification: dict[str, object],
    output_root: Path,
    device: torch.device,
    *,
    require_utility: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    method = str(specification["method"])
    checkpoint_hash = str(specification["checkpoint_sha256"])
    expected_actions = int(records["action_count"].sum())
    existing = load_score_artifact(
        output_root, method, checkpoint_hash, expected_actions
    )
    if existing is not None:
        metadata = json.loads(score_paths(output_root, method)[1].read_text(encoding="utf-8"))
        print(json.dumps({"event": "r06_score_resume", "method": method}), flush=True)
        return existing, {
            "method": method,
            "runtime_seconds": metadata["runtime_seconds"],
            "states_per_second": metadata["states_per_second"],
            "gpu_peak_memory_mib": metadata["gpu_peak_memory_mib"],
            "resumed": True,
        }
    frame, profile = score_checkpoint(
        records, specification, device, require_utility=require_utility
    )
    save_score_artifact(frame, output_root, method, checkpoint_hash, profile)
    return frame, {"method": method, **profile, "resumed": False}


def calibrated_revised_frame(
    base: pd.DataFrame,
    scores: pd.DataFrame,
    freeze: dict[str, object],
) -> pd.DataFrame:
    frame = base.merge(
        scores,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    probability = FrozenCalibrator(**freeze["probability_calibrator"])
    utility = FrozenCalibrator(**freeze["utility_calibrator"])
    frame["calibrated_probability"] = probability.predict(frame["score"])
    frame["calibrated_utility"] = utility.predict(frame["utility_prediction"])
    return frame


def selective_selection(
    frame: pd.DataFrame,
    freeze: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = InterventionThresholds(**freeze["selective_intervention_thresholds"])
    ordered = frame.sort_values(
        ["state_id", "score", "target_set_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    neural = ordered.groupby("state_id", sort=False).head(1).copy().set_index("state_id")
    second = ordered.groupby("state_id", sort=False).nth(1)["calibrated_probability"]
    neural["decision_margin"] = (
        neural["calibrated_probability"] - second.reindex(neural.index).fillna(0.0)
    )
    related = frame[
        frame["origin_rules"].str.contains('"operator_related"', regex=False, na=False)
    ].copy()
    counts = related.groupby("state_id").size()
    if len(counts) != len(neural) or not counts.eq(1).all():
        raise ValueError("R06 related fallback is not exactly one action per state")
    fallback = related.set_index("state_id").reindex(neural.index)
    intervene = (
        neural["calibrated_probability"].ge(thresholds.confidence)
        & neural["calibrated_utility"].ge(thresholds.predicted_utility)
        & neural["decision_margin"].ge(thresholds.decision_margin)
    )
    chosen = neural.where(intervene, fallback).reset_index()
    chosen["intervened"] = intervene.to_numpy(dtype=bool)
    chosen["model"] = PRIMARY
    selected = selected_columns(chosen, PRIMARY)
    selected["intervened"] = chosen["intervened"].to_numpy(dtype=bool)
    neural_utility = neural["mean_relative_improvement"].astype(float)
    fallback_utility = fallback["mean_relative_improvement"].astype(float)
    hybrid_utility = neural_utility.where(intervene, fallback_utility)
    hybrid_regret = neural["regret_to_best"].astype(float).where(
        intervene, fallback["regret_to_best"].astype(float)
    )
    policy = {
        "coverage": float(intervene.mean()),
        "intervention_state_count": int(intervene.sum()),
        "incremental_utility_vs_fallback": float(
            (hybrid_utility - fallback_utility).mean()
        ),
        "hybrid_selected_utility": float(hybrid_utility.mean()),
        "hybrid_selected_positive_fraction": float(hybrid_utility.gt(0).mean()),
        "hybrid_mean_regret": float(hybrid_regret.mean()),
        "fallback_selected_utility": float(fallback_utility.mean()),
        "fallback_mean_regret": float(fallback["regret_to_best"].mean()),
    }
    return selected, policy


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank in range(len(p_values), 0, -1):
        index = int(order[rank - 1])
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, p_values[int(index)] * (len(p_values) - rank))
        adjusted[int(index)] = min(running, 1.0)
    return adjusted.tolist()


def paired_statistics(selected: pd.DataFrame, expected_states: int) -> pd.DataFrame:
    primary = selected[selected["model"].eq(PRIMARY)][
        ["state_id", "selected_utility"]
    ].rename(columns={"selected_utility": "primary_utility"})
    rows = []
    for comparator in PAIRWISE_COMPARATORS:
        other = selected[selected["model"].eq(comparator)][
            ["state_id", "selected_utility"]
        ].rename(columns={"selected_utility": "comparator_utility"})
        paired = primary.merge(other, on="state_id", validate="one_to_one")
        if len(paired) != expected_states:
            raise ValueError(f"paired coverage mismatch for {comparator}")
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
            "primary": PRIMARY,
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
    p_values = [float(row["p_value"]) for row in rows]
    for row, bh, holm in zip(rows, bh_adjust(p_values), holm_adjust(p_values)):
        row["p_value_bh_fdr"] = bh
        row["p_value_holm"] = holm
        row["significant_bh_0_05"] = bool(bh < 0.05)
        row["significant_holm_0_05"] = bool(holm < 0.05)
    return pd.DataFrame(rows)


def structural_summary(
    base: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    regimes: list[tuple[str, str | None]] = [("ALL", None)]
    for dimension in STRUCTURAL_DIMENSIONS:
        regimes.extend((dimension, value) for value in sorted(
            selected[dimension].astype(str).unique()
        ))
    for method, method_frame in selected.groupby("model", sort=False):
        for dimension, value in regimes:
            part = method_frame if value is None else method_frame[
                method_frame[dimension].astype(str).eq(value)
            ]
            if part.empty:
                continue
            row = summarize_selected(part).iloc[0].to_dict()
            rows.append({
                "model": method,
                "regime_dimension": dimension,
                "regime_value": "ALL" if value is None else value,
                **row,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument(
        "--cache-manifest", type=Path,
        default=ROOT / "outputs/phase6f/tensorization/revision_holdout_cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=ROOT / "outputs/phase6f/revision_holdout/sealed_labels",
    )
    parser.add_argument(
        "--tabular-features", type=Path,
        default=ROOT / "outputs/phase6f/evaluation/r06_target_set_preaction_features.parquet",
    )
    parser.add_argument(
        "--tabular-model", type=Path,
        default=ROOT / "outputs/phase6e/baselines/v2/phase6c_tabular_diagnostic.joblib",
    )
    parser.add_argument(
        "--phase6e-manifest", type=Path,
        default=ROOT / "outputs/phase6e/training/final_seeds/checkpoint_manifest.json",
    )
    parser.add_argument(
        "--phase6e-ablation-manifest", type=Path,
        default=ROOT / "outputs/phase6e/ablations/training/checkpoint_manifest.json",
    )
    parser.add_argument(
        "--phase6f-manifest", type=Path,
        default=ROOT / "outputs/phase6f/training/final_seeds/checkpoint_manifest.json",
    )
    parser.add_argument(
        "--phase6e-baseline-freeze", type=Path,
        default=ROOT / "outputs/phase6e/audit/baseline_freeze_v2.json",
    )
    parser.add_argument(
        "--experiment-freeze", type=Path,
        default=ROOT / "outputs/phase6f/audit/experiment_freeze.json",
    )
    parser.add_argument(
        "--label-open-record", type=Path,
        default=ROOT / "outputs/phase6f/audit/revision_holdout_label_open.json",
    )
    parser.add_argument(
        "--cache-audit", type=Path,
        default=ROOT / "outputs/phase6f/audit/revision_holdout_cache_audit.json",
    )
    parser.add_argument(
        "--feature-audit", type=Path,
        default=ROOT / "outputs/phase6f/audit/r06_tabular_feature_audit.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/phase6f/evaluation",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    guards = validate_guards(args)
    old_specs, revised_specs = checkpoint_specs(args)
    records = cache_records(args.cache_manifest, SPLIT, max_shards=args.max_shards)
    expected_states = int(records["state_count"].sum())
    expected_actions = int(records["action_count"].sum())
    if args.max_shards is None and (
        len(records) != 81
        or expected_states != EXPECTED_STATES
        or expected_actions != EXPECTED_ACTIONS
    ):
        raise ValueError("full R06 cache coverage mismatch")
    device = torch.device(args.device)
    score_frames: dict[str, pd.DataFrame] = {}
    runtime_rows = []
    for specification in old_specs:
        frame, runtime = score_or_resume(
            records, specification, args.output_root, device, require_utility=False
        )
        score_frames[str(specification["method"])] = frame
        runtime_rows.append(runtime)
    for specification in revised_specs:
        frame, runtime = score_or_resume(
            records, specification, args.output_root, device, require_utility=True
        )
        score_frames[str(specification["method"])] = frame
        runtime_rows.append(runtime)

    old_seed_methods = [f"PHASE6E_FULL_CSG_SEED_{seed}" for seed in (660201, 660202, 660203)]
    reference = score_frames[old_seed_methods[0]][["state_id", "target_set_id"]]
    for method in old_seed_methods[1:]:
        if not reference.equals(score_frames[method][["state_id", "target_set_id"]]):
            raise ValueError("Phase 6E ensemble score ordering mismatch")
    ensemble = reference.copy()
    ensemble["score"] = np.mean(np.stack([
        score_frames[method]["score"].to_numpy(dtype=float)
        for method in old_seed_methods
    ]), axis=0)
    score_frames["PHASE6E_FULL_CSG_ENSEMBLE"] = ensemble

    instance_ids = set(records["instance_id"].astype(str))
    base = load_aggregates(
        args.dataset_root,
        "revision_holdout",
        instance_ids=instance_ids,
    )
    if len(base) != expected_actions or base["state_id"].nunique() != expected_states:
        raise ValueError("R06 label/cache coverage mismatch")
    tabular = joblib.load(args.tabular_model)
    if not isinstance(tabular, Phase6CTabularDiagnostic) or not tabular.fitted:
        raise ValueError("frozen Phase 6C tabular model is invalid")
    features = pd.read_parquet(args.tabular_features)
    modeled = base.merge(
        features,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(modeled) != len(base):
        raise ValueError("R06 tabular feature coverage mismatch")
    tabular_scores = modeled[["state_id", "target_set_id"]].copy()
    tabular_scores["score"] = tabular.score(modeled)
    score_frames["PHASE6C_TABULAR"] = tabular_scores

    freeze = json.loads(args.experiment_freeze.read_text(encoding="utf-8"))
    selected_tables = []
    metric_rows = []
    scored_methods = [
        *old_seed_methods,
        "PHASE6E_FULL_CSG_ENSEMBLE",
        "PHASE6E_FLAT_SET",
        "PHASE6C_TABULAR",
        *(f"PHASE6F_REVISED_SEED_{seed}" for seed in (660301, 660302, 660303)),
    ]
    for method in scored_methods:
        scored = attach_scores(base, score_frames[method])
        metrics = evaluate_action_scores(scored)
        robust = scored["positive_under_2_of_3"].astype(int)
        metrics["robust_label_roc_auc"] = float(roc_auc_score(robust, scored["score"]))
        metrics["robust_label_pr_auc"] = float(
            average_precision_score(robust, scored["score"])
        )
        metric_rows.append({"model": method, "evaluation_split": SPLIT, **metrics})
        selected_tables.append(selected_columns(
            selected_action_table(scored, model_name=method), method
        ))

    deploy_scores = score_frames["PHASE6F_REVISED_SEED_660301"]
    calibrated = calibrated_revised_frame(base, deploy_scores, freeze)
    hybrid, policy = selective_selection(calibrated, freeze)
    selected_tables.append(hybrid)
    probability_metrics = calibration_metrics(
        calibrated["calibrated_probability"],
        calibrated["positive_under_2_of_3"].astype(int),
    )
    reliability = reliability_table(
        calibrated["calibrated_probability"],
        calibrated["positive_under_2_of_3"].astype(int),
    )

    related = fixed_original_selection(base, "related")
    selected_tables.append(selected_columns(related, "B1_FIXED_RELATED"))
    selected_tables.append(random_expected_table(base))
    selected = pd.concat(selected_tables, ignore_index=True)
    deploy_single = selected[selected["model"].eq("PHASE6E_FULL_CSG_SEED_660201")].copy()
    deploy_single["model"] = "PHASE6E_DEPLOYABLE_SINGLE_SEED_660201"
    selected = pd.concat([selected, deploy_single], ignore_index=True)
    if not selected.groupby("model")["state_id"].nunique().eq(expected_states).all():
        raise ValueError("selected-policy state coverage mismatch")

    metrics = pd.DataFrame(metric_rows)
    selected_summary = summarize_selected(selected)
    statistics = paired_statistics(selected, expected_states)
    regimes = structural_summary(base, selected)
    calibration_summary = pd.DataFrame([{
        "model": PRIMARY,
        "evaluation_split": SPLIT,
        **probability_metrics,
        **policy,
    }])

    write_csv(metrics, args.output_root / "revision_holdout_metrics.csv")
    write_csv(selected_summary, args.output_root / "revision_holdout_selected_action_summary.csv")
    write_parquet(selected, args.output_root / "revision_holdout_state_selected_actions.parquet")
    write_csv(regimes, args.output_root / "revision_holdout_structural_summary.csv")
    write_csv(calibration_summary, args.output_root / "revision_holdout_calibration_summary.csv")
    write_csv(reliability, args.output_root / "revision_holdout_reliability.csv")
    write_csv(pd.DataFrame(runtime_rows), args.output_root / "revision_holdout_inference_runtime.csv")
    statistics_path = (
        args.output_root / "revision_holdout_pairwise_statistics.csv"
        if args.max_shards is not None
        else ROOT / "outputs/phase6f/statistics/revision_holdout_pairwise_statistics.csv"
    )
    write_csv(statistics, statistics_path)

    outputs = [
        args.output_root / "revision_holdout_metrics.csv",
        args.output_root / "revision_holdout_selected_action_summary.csv",
        args.output_root / "revision_holdout_state_selected_actions.parquet",
        args.output_root / "revision_holdout_structural_summary.csv",
        args.output_root / "revision_holdout_calibration_summary.csv",
        args.output_root / "revision_holdout_reliability.csv",
        args.output_root / "revision_holdout_inference_runtime.csv",
        statistics_path,
    ]
    completion = {
        "schema": "phase6f-r06-evaluation-v1",
        "status": "COMPLETE",
        "evaluation_split": SPLIT,
        "shard_count": len(records),
        "state_count": expected_states,
        "action_count": expected_actions,
        "deployment_model": PRIMARY,
        "deployment_checkpoint_selected_before_r06": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selective_policy": policy,
        "outputs": {
            str(path.resolve().relative_to(ROOT)): file_sha256(path)
            for path in outputs
        },
        **guards,
    }
    write_json(args.output_root / "revision_holdout_evaluation_completion.json", completion)
    print(json.dumps({"event": "phase6f_r06_evaluation_complete", **completion}), flush=True)


if __name__ == "__main__":
    main()
