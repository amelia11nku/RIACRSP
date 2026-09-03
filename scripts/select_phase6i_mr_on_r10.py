#!/usr/bin/env python3
"""Execute the frozen one-time, seed-robust Phase 6I-MR R10 selection rule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.calibration import FrozenCalibrator, calibration_metrics  # noqa: E402
from rcias_clgri.ni.phase6i_heads import build_phase6i_head  # noqa: E402
from rcias_clgri.ni.phase6i_policy import (  # noqa: E402
    select_immediate_actions,
    state_ranking_metrics,
    summarize_ranking_metrics,
    support_mask,
)
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)
from scripts.train_phase6i_mr_u3 import load_model as load_u3_base_model  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
TRAINING_PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
U3_PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/u3_training_protocol.json"
PRE_R10_FREEZE = ROOT / "outputs/phase6i_mr/pre_r10/pre_r10_freeze.json"
REGISTRY_PATH = ROOT / "outputs/phase6i_mr/pre_r10/candidate_registry.json"
AUTHORIZATION = ROOT / "outputs/phase6i_mr/frozen/r10_collection_authorization.json"
CACHE_ROOT = ROOT / "outputs/phase6i_mr/r10_selection/cache"
OUT = ROOT / "outputs/phase6i_mr/r10_selection"
BOOTSTRAP_SEED = 688102
BOOTSTRAP_REPLICATES = 10_000
LOW_COUNT_FORCED_ACTIONS = 120
ABSTENTION_UPPER_BOUND = 0.0025


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_inputs() -> tuple[dict, dict, dict, dict, pd.DataFrame, list]:
    config = load_json(CONFIG_PATH)
    training_protocol = load_json(TRAINING_PROTOCOL_PATH)
    u3_protocol = load_json(U3_PROTOCOL_PATH)
    freeze = load_json(PRE_R10_FREEZE)
    registry = load_json(REGISTRY_PATH)
    authorization = load_json(AUTHORIZATION)
    cache_integrity = load_json(CACHE_ROOT / "cache_integrity.json")
    if not all([
        freeze.get("status") == "PASS",
        authorization.get("pre_r10_freeze_sha256") == digest(PRE_R10_FREEZE),
        authorization.get("code_hashes", {}).get(
            "scripts/select_phase6i_mr_on_r10.py"
        ) == digest(Path(__file__)),
        cache_integrity.get("status") == "PASS",
        cache_integrity.get("pre_r10_freeze_sha256") == digest(PRE_R10_FREEZE),
        cache_integrity.get("r10_accessed") is True,
        cache_integrity.get("r11_accessed") is False,
        registry.get("r10_accessed") is False,
        registry.get("r11_accessed") is False,
    ]):
        raise RuntimeError("R10 selection inputs or frozen code hash are invalid")
    manifest = pd.read_csv(CACHE_ROOT / "cache_manifest.csv")
    embedding_frames = [
        pd.read_parquet(ROOT / path)
        for path in manifest.sort_values("instance_id").embedding_path
    ]
    embeddings = pd.concat(embedding_frames, ignore_index=True)
    if embeddings.duplicated(["state_id", "target_set_id"]).any():
        raise RuntimeError("duplicate R10 action cache rows")
    samples = []
    for path in manifest.sort_values("instance_id").tensor_cache_path:
        shard, _ = load_shard_cache(
            ROOT / path,
            expected_tensor_schema_hash=cache_integrity["tensor_schema_hash"],
            expected_source_shard_sha256=cache_integrity["source_sha256"],
        )
        samples.extend(shard)
    if len(samples) != 1620:
        raise RuntimeError("R10 tensor cache does not contain 1620 states")
    return config, training_protocol, u3_protocol, registry, embeddings, samples


def _load_head_models(candidate: dict, device: torch.device) -> list[torch.nn.Module]:
    models = []
    for artifact in candidate["artifacts"]:
        path = ROOT / artifact["checkpoint_path"]
        if digest(path) != artifact["checkpoint_sha256"]:
            raise RuntimeError(f"candidate checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_phase6i_head(candidate["model_family"], dropout=0.1)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(device)
        models.append(model)
    return models


def _load_u3_models(
    candidate: dict, protocol: dict, device: torch.device
) -> list[torch.nn.Module]:
    models = []
    for artifact in candidate["artifacts"]:
        path = ROOT / artifact["checkpoint_path"]
        if digest(path) != artifact["checkpoint_sha256"]:
            raise RuntimeError(f"U3 checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        seed = int(artifact["training_seed"])
        model = load_u3_base_model(seed, protocol, device)
        state = model.state_dict()
        state.update(checkpoint["trainable_model_state"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def _predict_heads(
    candidate: dict,
    frame: pd.DataFrame,
    context_columns: list[str],
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    models = _load_head_models(candidate, device)
    embedding_columns = sorted(c for c in frame if c.startswith("embedding_"))
    rows = []
    latencies = []
    for _, group in frame.groupby("state_id", sort=True):
        ordered = group.sort_values("target_set_id", kind="stable")
        embeddings = torch.tensor(
            ordered[embedding_columns].to_numpy(dtype=np.float32), device=device
        )
        context = torch.tensor(
            ordered[context_columns].to_numpy(dtype=np.float32), device=device
        )
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            values = [model(embeddings, context) for model in models]
        _synchronize(device)
        head_ms = (time.perf_counter() - started) * 1000.0
        base_ms = float(ordered.model_inference_and_scoring_ms.iloc[0])
        latencies.append(base_ms + head_ms)
        for model_index, artifact in enumerate(candidate["artifacts"]):
            output = values[model_index].detach().float().cpu().numpy()
            for target_set_id, value in zip(ordered.target_set_id, output):
                rows.append({
                    "state_id": str(ordered.state_id.iloc[0]),
                    "target_set_id": str(target_set_id),
                    "training_seed": int(artifact["training_seed"]),
                    "predicted_normalized_value": float(value),
                    "predicted_raw_score": float(
                        ordered.loc[
                            ordered.target_set_id == target_set_id,
                            "frozen_reference_score",
                        ].iloc[0]
                    ),
                })
    return pd.DataFrame(rows), np.asarray(latencies, dtype=float)


def _predict_u3(
    candidate: dict,
    samples: list,
    protocol: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    models = _load_u3_models(candidate, protocol, device)
    rows = []
    latencies = []
    for sample in sorted(samples, key=lambda item: item.graph.state_id):
        batch = batch_state_samples([sample]).to(device)
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = [model(batch) for model in models]
        _synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
        for model_index, artifact in enumerate(candidate["artifacts"]):
            utility = outputs[model_index].utility_predictions.detach().float().cpu().numpy()
            score = outputs[model_index].scores.detach().float().cpu().numpy()
            for target_set_id, value, raw_score in zip(
                batch.target_set_ids, utility, score
            ):
                rows.append({
                    "state_id": str(sample.graph.state_id),
                    "target_set_id": str(target_set_id),
                    "training_seed": int(artifact["training_seed"]),
                    "predicted_normalized_value": float(value),
                    "predicted_raw_score": float(raw_score),
                })
    return pd.DataFrame(rows), np.asarray(latencies, dtype=float)


def _ensemble_frame(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate: dict,
    support_bounds: dict,
) -> pd.DataFrame:
    if predictions.duplicated(["state_id", "target_set_id", "training_seed"]).any():
        raise RuntimeError("duplicate R10 seed prediction")
    means = (
        predictions.groupby(["state_id", "target_set_id"])[
            ["predicted_normalized_value", "predicted_raw_score"]
        ]
        .mean()
        .rename(columns={
            "predicted_normalized_value": "ensemble_raw_value",
            "predicted_raw_score": "ensemble_raw_score",
        })
        .reset_index()
    )
    result = base.merge(
        means, on=["state_id", "target_set_id"], validate="one_to_one"
    )
    calibration = load_json(ROOT / candidate["calibration_path"])
    probability = FrozenCalibrator(**calibration["probability_calibrator"])
    utility = FrozenCalibrator(**calibration["utility_calibrator"])
    result["calibrated_probability"] = probability.predict(result.ensemble_raw_score)
    result["calibrated_utility"] = utility.predict(result.ensemble_raw_value)
    result["supported"] = support_mask(result, support_bounds)
    return result


def _bootstrap_upper(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    values = frame.groupby("instance_id")[column].mean().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(values, (BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.975))


def _support_diagnostic(selected: pd.DataFrame) -> tuple[bool, list[dict[str, object]], float]:
    rows = []
    for scale in ("S", "M", "L"):
        group = selected[selected.scale == scale]
        interventions = group[group.intervened]
        abstained = group[~group.intervened]
        best_upper = _bootstrap_upper(abstained, "best_forced_utility")
        lift_upper = _bootstrap_upper(
            abstained, "best_forced_lift_over_fallback"
        )
        mean_best = float(abstained.best_forced_utility.mean()) if len(abstained) else math.nan
        mean_lift = (
            float(abstained.best_forced_lift_over_fallback.mean())
            if len(abstained) else math.nan
        )
        abstention_supported = bool(
            int(abstained.forced_action_count.sum()) >= LOW_COUNT_FORCED_ACTIONS
            and (
                (mean_best <= 0.0 and best_upper <= ABSTENTION_UPPER_BOUND)
                or (mean_lift <= 0.0 and lift_upper <= ABSTENTION_UPPER_BOUND)
            )
        )
        unsupported = (
            float((~interventions.selected_supported).mean()) if len(interventions) else 0.0
        )
        rows.append({
            "scale": scale,
            "state_count": len(group),
            "selected_action_count": len(interventions),
            "coverage": float(group.intervened.mean()),
            "unsupported_intervention_fraction": unsupported,
            "selected_mean_realized_utility": (
                float(interventions.selected_realized_utility.mean())
                if len(interventions) else math.nan
            ),
            "selected_mean_lift_over_fallback": (
                float(interventions.selected_lift_over_fallback.mean())
                if len(interventions) else math.nan
            ),
            "selected_sign_error_rate": (
                float(interventions.selected_sign_error.mean())
                if len(interventions) else math.nan
            ),
            "selected_calibration_bias": (
                float((
                    interventions.selected_calibrated_utility
                    - interventions.selected_realized_utility
                ).mean()) if len(interventions) else math.nan
            ),
            "abstained_forced_action_count": int(abstained.forced_action_count.sum()),
            "abstained_mean_best_forced_utility": mean_best,
            "abstained_mean_best_lift_over_fallback": mean_lift,
            "abstained_best_utility_grouped_bootstrap_upper_97_5": best_upper,
            "abstained_lift_grouped_bootstrap_upper_97_5": lift_upper,
            "abstention_supported": abstention_supported,
            "scale_support_eligible": bool(
                (len(interventions) >= 100 or abstention_supported) and unsupported <= 0.10
            ),
        })
    coverage = np.asarray([row["coverage"] for row in rows], dtype=float)
    ratio = float(coverage.max() / coverage.min()) if np.all(coverage > 0) else math.inf
    imbalance_ok = bool(ratio <= 8.0 or rows[int(np.argmin(coverage))]["abstention_supported"])
    return bool(all(row["scale_support_eligible"] for row in rows) and imbalance_ok), rows, ratio


def _seed_stability(
    predictions: pd.DataFrame, base: pd.DataFrame
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = []
    for seed, values in predictions.groupby("training_seed", sort=True):
        merged = base.merge(
            values[["state_id", "target_set_id", "predicted_normalized_value"]],
            on=["state_id", "target_set_id"],
            validate="one_to_one",
        )
        states = state_ranking_metrics(
            merged, prediction_column="predicted_normalized_value"
        )
        records.append({"training_seed": int(seed), **summarize_ranking_metrics(states)})
    frame = pd.DataFrame(records)
    overall = frame.overall_per_instance_spearman.to_numpy(dtype=float)
    scale_records = {}
    for scale in ("S", "M", "L"):
        values = frame[f"scale_{scale}_per_instance_spearman"].to_numpy(dtype=float)
        scale_records[scale] = {
            "mean": float(values.mean()),
            "worst": float(values.min()),
            "nonnegative_seed_count": int(np.sum(values >= 0.0)),
            "eligible": bool(
                values.mean() >= 0.0
                and np.sum(values >= 0.0) >= 2
                and values.min() >= -0.10
            ),
        }
    summary = {
        "mean_overall_spearman": float(overall.mean()),
        "worst_seed_overall_spearman": float(overall.min()),
        "seed_standard_deviation": float(overall.std(ddof=1)),
        "scales": scale_records,
    }
    summary["eligible"] = bool(
        summary["mean_overall_spearman"] > 0.0
        and summary["worst_seed_overall_spearman"] >= 0.0
        and all(record["eligible"] for record in scale_records.values())
    )
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R10 selection requested CUDA but CUDA is unavailable")
    config, training_protocol, u3_protocol, registry, base, samples = _load_inputs()
    support_bounds = load_json(
        ROOT / registry["support_bounds_path"]
    )["bounds"]
    phase6h = load_json(ROOT / registry["u0_comparator"]["phase6h_policy_path"])
    reference_probability = FrozenCalibrator(**phase6h["probability_calibrator"])
    reference_metrics = calibration_metrics(
        reference_probability.predict(base.frozen_reference_score), base.positive_label
    )
    grids = config["live_policy"]
    threshold_lookup = {
        f"P{pi:02d}_U{ui:02d}": (float(p), float(u))
        for pi, p in enumerate(grids["probability_threshold_grid"])
        for ui, u in enumerate(grids["immediate_utility_threshold_grid"])
    }
    long_predictions = []
    candidate_summaries = []
    threshold_summaries = []
    scale_diagnostics = []
    started = time.perf_counter()
    for candidate in registry["immediate_candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate["model_family"] == "U3":
            predictions, latencies = _predict_u3(
                candidate, samples, u3_protocol, device
            )
        else:
            predictions, latencies = _predict_heads(
                candidate,
                base,
                list(training_protocol["context_columns"]),
                device,
            )
        predictions["candidate_id"] = candidate_id
        long_predictions.append(predictions)
        ensemble = _ensemble_frame(base, predictions, candidate, support_bounds)
        ensemble["candidate_id"] = candidate_id
        prediction_path = OUT / "predictions" / f"{candidate_id}.parquet"
        atomic_parquet(ensemble, prediction_path)
        state = state_ranking_metrics(ensemble)
        ranking = summarize_ranking_metrics(state)
        stability, seed_records = _seed_stability(predictions, base)
        probability = calibration_metrics(
            ensemble.calibrated_probability, ensemble.positive_label
        )
        probability_eligible = bool(
            probability["expected_calibration_error"] <= 0.10
            and probability["brier_score"]
            <= 1.10 * reference_metrics["brier_score"]
            and probability["negative_log_likelihood"]
            <= 1.10 * reference_metrics["negative_log_likelihood"]
        )
        ranking_eligible = bool(
            ranking["overall_per_instance_spearman"] > 0.0
            and all(
                ranking[f"scale_{scale}_per_instance_spearman"] >= 0.0
                for scale in ("S", "M", "L")
            )
        )
        candidate_summaries.append({
            "candidate_id": candidate_id,
            "parameter_count": int(candidate["parameter_count"]),
            "p50_neural_decision_ms": float(np.quantile(latencies, 0.50)),
            "p90_neural_decision_ms": float(np.quantile(latencies, 0.90)),
            "ranking_eligible": ranking_eligible,
            "probability_eligible": probability_eligible,
            "seed_stability_eligible": stability["eligible"],
            "ranking": ranking,
            "seed_stability": stability,
            "seed_metrics": seed_records,
            "probability_metrics": probability,
            "prediction_path": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": digest(prediction_path),
        })
        for threshold_id in candidate["eligible_threshold_set_ids"]:
            probability_threshold, utility_threshold = threshold_lookup[threshold_id]
            selected = select_immediate_actions(
                ensemble,
                probability_threshold=probability_threshold,
                utility_threshold=utility_threshold,
            )
            support_eligible, scale_rows, coverage_ratio = _support_diagnostic(selected)
            for row in scale_rows:
                scale_diagnostics.append({
                    "candidate_id": candidate_id,
                    "threshold_set_id": threshold_id,
                    "probability_threshold": probability_threshold,
                    "utility_threshold": utility_threshold,
                    "cross_scale_coverage_ratio": coverage_ratio,
                    **row,
                })
            interventions = selected[selected.intervened]
            selected_regret = float((
                selected.best_forced_utility - selected.selected_realized_utility
            ).mean())
            major = [
                row for row in scale_rows if row["selected_action_count"] >= 100
            ]
            calibration_sign_eligible = bool(all(
                row["selected_sign_error_rate"] <= 0.50
                and abs(row["selected_calibration_bias"]) <= 0.05
                for row in major
            ))
            lift = float(selected.selected_lift_over_fallback.mean())
            summary = candidate_summaries[-1]
            eligible = bool(
                ranking_eligible
                and stability["eligible"]
                and probability_eligible
                and calibration_sign_eligible
                and support_eligible
                and lift > 0.0
            )
            threshold_summaries.append({
                "candidate_id": candidate_id,
                "threshold_set_id": threshold_id,
                "probability_threshold": probability_threshold,
                "utility_threshold": utility_threshold,
                "r10_eligible": eligible,
                "ranking_eligible": ranking_eligible,
                "seed_stability_eligible": stability["eligible"],
                "probability_eligible": probability_eligible,
                "calibration_sign_eligible": calibration_sign_eligible,
                "support_eligible": support_eligible,
                "intervention_count": len(interventions),
                "coverage": float(selected.intervened.mean()),
                "mean_selected_realized_utility_all_states": float(
                    selected.selected_realized_utility.mean()
                ),
                "mean_selected_lift_over_fallback": lift,
                "mean_selected_regret": selected_regret,
                "pair_inversion_rate": ranking[
                    "overall_per_instance_pair_inversion_rate"
                ],
                "ndcg_at_1": ranking["overall_per_instance_ndcg_at_1"],
                "ndcg_at_2": ranking["overall_per_instance_ndcg_at_2"],
                "worst_seed_spearman": stability["worst_seed_overall_spearman"],
                "mean_seed_spearman": stability["mean_overall_spearman"],
                "seed_standard_deviation": stability["seed_standard_deviation"],
                "p90_neural_decision_ms": summary["p90_neural_decision_ms"],
                "parameter_count": summary["parameter_count"],
            })
        print(json.dumps({
            "event": "r10_candidate_complete",
            "candidate_id": candidate_id,
            "elapsed_seconds": time.perf_counter() - started,
        }), flush=True)

    long_path = OUT / "r10_seed_predictions.parquet"
    candidate_path = OUT / "r10_candidate_metrics.json"
    threshold_path = OUT / "r10_threshold_metrics.csv"
    scale_path = OUT / "r10_scale_support_diagnostics.csv"
    atomic_parquet(pd.concat(long_predictions, ignore_index=True), long_path)
    atomic_json({"candidates": candidate_summaries}, candidate_path)
    threshold_frame = pd.DataFrame(threshold_summaries)
    atomic_csv(threshold_frame, threshold_path)
    atomic_csv(pd.DataFrame(scale_diagnostics), scale_path)
    eligible = threshold_frame[threshold_frame.r10_eligible].copy()
    if len(eligible):
        selected = eligible.sort_values(
            [
                "mean_selected_lift_over_fallback", "mean_selected_regret",
                "pair_inversion_rate", "ndcg_at_1", "ndcg_at_2",
                "worst_seed_spearman", "mean_seed_spearman",
                "seed_standard_deviation", "p90_neural_decision_ms",
                "parameter_count", "candidate_id", "threshold_set_id",
            ],
            ascending=[
                False, True, True, False, False, False, False, True, True,
                True, True, True,
            ],
            kind="stable",
        ).iloc[0]
        status = "SELECTED_IMMUTABLY_FOR_R10_TRANSLATION"
        selected_record = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in selected.to_dict().items()
        }
    else:
        status = "MODEL_REVISION_NO_R10_ELIGIBLE_CANDIDATE"
        selected_record = None
    decision = {
        "schema": "phase6i-mr-r10-selection-decision-v1.2",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": registry["r10_selection_rule"],
        "selection_tie_break_fields": [
            "lift desc", "regret asc", "inversion asc", "NDCG@1 desc",
            "NDCG@2 desc", "worst-seed desc", "mean-seed desc",
            "seed std asc", "p90 latency asc", "parameter count asc",
            "candidate ID asc", "threshold ID asc",
        ],
        "selected": selected_record,
        "eligible_candidate_threshold_count": len(eligible),
        "reference_probability_metrics": reference_metrics,
        "input_hashes": {
            "pre_r10_freeze": digest(PRE_R10_FREEZE),
            "candidate_registry": digest(REGISTRY_PATH),
            "cache_integrity": digest(CACHE_ROOT / "cache_integrity.json"),
        },
        "output_hashes": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [long_path, candidate_path, threshold_path, scale_path]
        },
        "runtime_seconds": time.perf_counter() - started,
        "r10_accessed": True,
        "r10_refit_performed": False,
        "r11_accessed": False,
    }
    atomic_json(decision, OUT / "selection_decision.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
