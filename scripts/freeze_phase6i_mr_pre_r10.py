#!/usr/bin/env python3
"""Fit R09-only calibrators and freeze the complete pre-R10 candidate registry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.calibration import (  # noqa: E402
    FrozenCalibrator,
    calibration_metrics,
)
from rcias_clgri.ni.phase6i_policy import (  # noqa: E402
    cross_fit_probability_calibration,
    cross_fit_utility_calibration,
    ensemble_oof_predictions,
    fit_support_bounds,
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


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
TRAINING_PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
U3_PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/u3_training_protocol.json"
PHASE6H_POLICY_PATH = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
MODEL_ROOT = ROOT / "outputs/phase6i_mr/model_training"
R09_COLLECTION = ROOT / "outputs/phase6i_mr/collection/r09"
OUT = ROOT / "outputs/phase6i_mr/pre_r10"
CALIBRATION_OUT = OUT / "calibration"

TRAINING_SEEDS = (686101, 686102, 686103)
BOOTSTRAP_SEED = 688101
BOOTSTRAP_REPLICATES = 10_000
R09_LOW_COUNT_FORCED_ACTIONS = 120
R09_ABSTENTION_UPPER_BOUND = 0.0025

IMMEDIATE_CANDIDATES = (
    ("U1_R09_ONLY", "U1", "R09_ONLY", "immediate_oof_predictions.parquet"),
    ("U1_MIXED_OLD_NEW", "U1", "MIXED_OLD_NEW", "immediate_oof_predictions.parquet"),
    ("U1_HARD_AGG_20_60_20", "U1", "HARD_AGG_20_60_20", "hard_agg_oof_predictions.parquet"),
    ("U2_R09_ONLY", "U2", "R09_ONLY", "immediate_oof_predictions.parquet"),
    ("U2_MIXED_OLD_NEW", "U2", "MIXED_OLD_NEW", "immediate_oof_predictions.parquet"),
    ("U2_HARD_AGG_20_60_20", "U2", "HARD_AGG_20_60_20", "hard_agg_oof_predictions.parquet"),
    ("U3_HARD_AGG_20_60_20", "U3", "HARD_AGG_20_60_20", "u3/u3_oof_predictions.parquet"),
)


def _finite_correlation(left, right) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else math.nan


def _bootstrap_upper(values: pd.DataFrame, column: str) -> float:
    if values.empty:
        return math.nan
    instance_values = (
        values.groupby("instance_id")[column].mean().to_numpy(dtype=float)
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(
        instance_values,
        size=(BOOTSTRAP_REPLICATES, len(instance_values)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.975))


def _full_artifact_paths(family: str, variant: str, seed: int) -> tuple[Path, Path]:
    if family == "U3":
        base = MODEL_ROOT / "u3/full" / f"seed_{seed}"
    else:
        base = MODEL_ROOT / "full_artifacts" / family / variant / f"seed_{seed}"
    return base.with_suffix(".pt"), base.with_suffix(".json")


def _validate_artifacts(
    family: str, variant: str, protocol_hash: str
) -> tuple[list[dict[str, object]], int]:
    records = []
    parameter_counts = set()
    state_hashes = set()
    for seed in TRAINING_SEEDS:
        checkpoint_path, record_path = _full_artifact_paths(family, variant, seed)
        record = load_json(record_path)
        checkpoint_hash = digest(checkpoint_path)
        protocol_field = (
            "u3_training_protocol_sha256" if family == "U3"
            else "training_protocol_sha256"
        )
        if not all([
            record.get("status") == "COMPLETE",
            int(record.get("training_seed")) == seed,
            record.get("checkpoint_sha256") == checkpoint_hash,
            record.get(protocol_field) == protocol_hash,
            record.get("r10_accessed") is False,
            record.get("r11_accessed") is False,
        ]):
            raise RuntimeError(f"invalid full artifact record: {record_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if family == "U3":
            tensors = checkpoint["trainable_model_state"]
            parameter_count = sum(value.numel() for value in tensors.values())
        else:
            tensors = checkpoint["model_state_dict"]
            parameter_count = sum(value.numel() for value in tensors.values())
        if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
            raise RuntimeError(f"non-finite model tensor: {checkpoint_path}")
        parameter_counts.add(parameter_count)
        state_hashes.add(checkpoint_hash)
        records.append({
            "training_seed": seed,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_hash,
            "record_path": str(record_path.relative_to(ROOT)),
            "record_sha256": digest(record_path),
        })
    if len(parameter_counts) != 1 or len(state_hashes) != len(TRAINING_SEEDS):
        raise RuntimeError(f"invalid three-seed artifact family: {family}/{variant}")
    return records, int(parameter_counts.pop())


def _seed_metrics(family: str, variant: str) -> list[dict[str, object]]:
    path = (
        MODEL_ROOT / "u3/u3_oof_metrics.parquet"
        if family == "U3"
        else MODEL_ROOT / (
            "hard_agg_oof_metrics.parquet"
            if variant.startswith("HARD") else "immediate_oof_metrics.parquet"
        )
    )
    frame = pd.read_parquet(path)
    selected = frame[(frame.model_family == family) & (frame.data_variant == variant)]
    if tuple(sorted(selected.training_seed.astype(int))) != TRAINING_SEEDS:
        raise RuntimeError(f"missing seed metrics: {family}/{variant}")
    columns = [
        "training_seed", "overall_per_instance_spearman",
        "scale_S_per_instance_spearman", "scale_M_per_instance_spearman",
        "scale_L_per_instance_spearman", "pairwise_accuracy", "ndcg_at_1",
        "top1_agreement", "selected_lift_over_fallback", "selected_sign_error",
    ]
    return selected[columns].sort_values("training_seed").to_dict("records")


def _seed_stability(records: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(records)
    overall = frame.overall_per_instance_spearman.to_numpy(dtype=float)
    scale_checks = {}
    for scale in ("S", "M", "L"):
        values = frame[f"scale_{scale}_per_instance_spearman"].to_numpy(dtype=float)
        scale_checks[scale] = {
            "mean": float(values.mean()),
            "worst": float(values.min()),
            "nonnegative_seed_count": int(np.sum(values >= 0.0)),
            "sign_stable": bool(np.sum(values >= 0.0) >= 2 and values.min() >= -0.10),
        }
    return {
        "mean_overall_spearman": float(overall.mean()),
        "worst_seed_overall_spearman": float(overall.min()),
        "seed_standard_deviation": float(overall.std(ddof=1)),
        "scales": scale_checks,
    }


def _calibration_payload(
    candidate_id: str,
    probability: FrozenCalibrator,
    utility: FrozenCalibrator,
    *,
    probability_origin: str,
    utility_target: str,
) -> dict[str, object]:
    return {
        "schema": "phase6i-mr-r09-calibration-v1.2",
        "status": "FROZEN_BEFORE_R10_ACCESS",
        "candidate_id": candidate_id,
        "probability_calibrator": probability.to_dict(),
        "probability_origin": probability_origin,
        "utility_calibrator": utility.to_dict(),
        "utility_target": utility_target,
        "ranking_value": "uncalibrated arithmetic mean of three seed-specific normalized outputs",
        "threshold_value": "cross-fitted calibrated output on R09; full-R09 calibrator for deployment",
        "r10_accessed": False,
        "r11_accessed": False,
    }


def _threshold_audit(
    candidate_id: str,
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict[str, object]] = []
    eligible_ids: list[str] = []
    minimum_direct = int(
        config["live_policy"]["r09_threshold_constraints"]
        ["minimum_selected_actions_per_scale_for_direct_estimation"]
    )
    maximum_unsupported = float(
        config["live_policy"]["r09_threshold_constraints"]
        ["maximum_unsupported_intervention_fraction"]
    )
    for probability_index, probability_threshold in enumerate(
        config["live_policy"]["probability_threshold_grid"]
    ):
        for utility_index, utility_threshold in enumerate(
            config["live_policy"]["immediate_utility_threshold_grid"]
        ):
            threshold_id = f"P{probability_index:02d}_U{utility_index:02d}"
            selected = select_immediate_actions(
                frame,
                probability_threshold=float(probability_threshold),
                utility_threshold=float(utility_threshold),
            )
            scale_records = []
            for scale in ("S", "M", "L"):
                group = selected[selected.scale == scale]
                interventions = group[group.intervened]
                abstained = group[~group.intervened]
                selected_count = len(interventions)
                unsupported_fraction = (
                    float((~interventions.selected_supported).mean())
                    if selected_count else 0.0
                )
                best_bootstrap_upper = _bootstrap_upper(
                    abstained, "best_forced_utility"
                )
                lift_bootstrap_upper = _bootstrap_upper(
                    abstained, "best_forced_lift_over_fallback"
                )
                mean_best = (
                    float(abstained.best_forced_utility.mean()) if len(abstained) else math.nan
                )
                mean_lift = (
                    float(abstained.best_forced_lift_over_fallback.mean())
                    if len(abstained) else math.nan
                )
                abstention_supported = bool(
                    int(abstained.forced_action_count.sum()) >= R09_LOW_COUNT_FORCED_ACTIONS
                    and (
                        (
                            mean_best <= 0.0
                            and best_bootstrap_upper <= R09_ABSTENTION_UPPER_BOUND
                        )
                        or (
                            mean_lift <= 0.0
                            and lift_bootstrap_upper <= R09_ABSTENTION_UPPER_BOUND
                        )
                    )
                )
                direct = selected_count >= minimum_direct
                scale_eligible = bool(
                    (direct or abstention_supported)
                    and unsupported_fraction <= maximum_unsupported
                )
                scale_record = {
                    "candidate_id": candidate_id,
                    "threshold_set_id": threshold_id,
                    "probability_threshold": float(probability_threshold),
                    "utility_threshold": float(utility_threshold),
                    "scale": scale,
                    "state_count": len(group),
                    "selected_action_count": selected_count,
                    "coverage": float(group.intervened.mean()),
                    "selected_mean_realized_utility": (
                        float(interventions.selected_realized_utility.mean())
                        if selected_count else math.nan
                    ),
                    "selected_mean_lift_over_fallback": (
                        float(interventions.selected_lift_over_fallback.mean())
                        if selected_count else math.nan
                    ),
                    "selected_sign_error_rate": (
                        float(interventions.selected_sign_error.mean())
                        if selected_count else math.nan
                    ),
                    "unsupported_intervention_fraction": unsupported_fraction,
                    "abstained_state_count": len(abstained),
                    "abstained_forced_action_count": int(abstained.forced_action_count.sum()),
                    "abstained_mean_best_forced_utility": mean_best,
                    "abstained_mean_best_lift_over_fallback": mean_lift,
                    "abstained_best_utility_grouped_bootstrap_upper_97_5": best_bootstrap_upper,
                    "abstained_lift_grouped_bootstrap_upper_97_5": lift_bootstrap_upper,
                    "direct_estimation": direct,
                    "abstention_supported": abstention_supported,
                    "scale_support_eligible": scale_eligible,
                }
                records.append(scale_record)
                scale_records.append(scale_record)
            coverages = np.asarray([row["coverage"] for row in scale_records], dtype=float)
            positive = coverages[coverages > 0]
            coverage_ratio = (
                float(positive.max() / positive.min())
                if len(positive) == 3 else math.inf
            )
            low_scale_index = int(np.argmin(coverages))
            imbalance_supported = bool(
                coverage_ratio <= 8.0
                or scale_records[low_scale_index]["abstention_supported"]
            )
            threshold_eligible = bool(
                all(row["scale_support_eligible"] for row in scale_records)
                and imbalance_supported
            )
            for row in scale_records:
                row["cross_scale_coverage_ratio"] = coverage_ratio
                row["coverage_imbalance_supported"] = imbalance_supported
                row["threshold_eligible_for_r10"] = threshold_eligible
                row["solver_no_collapse_condition"] = "PENDING_R10_TRANSLATION_IF_EXCEPTION_USED"
            if threshold_eligible:
                eligible_ids.append(threshold_id)
    return pd.DataFrame(records), eligible_ids


def _full_bank_audit() -> dict[str, object]:
    frame = pd.read_parquet(R09_COLLECTION / "full_bank_audit_table.parquet")
    rows = []
    for state_id, group in frame.groupby("state_id", sort=True):
        best = float(group.decoded_immediate_utility.max())
        broad = group[group.in_broad_four]
        top_eight = group[group.in_top_eight]
        true_best = group[np.isclose(group.decoded_immediate_utility, best)]
        rows.append({
            "state_id": state_id,
            "instance_id": str(group.instance_id.iloc[0]),
            "scale": str(group.scale.iloc[0]),
            "bank_size": len(group),
            "true_best_absent_from_broad_four": not bool(true_best.in_broad_four.any()),
            "true_best_absent_from_top_eight": not bool(true_best.in_top_eight.any()),
            "broad_four_true_top4_recall": float(
                group[group.full_bank_true_rank <= 4].in_broad_four.mean()
            ),
            "top_eight_true_top4_recall": float(
                group[group.full_bank_true_rank <= 4].in_top_eight.mean()
            ),
            "broad_four_regret_to_full_best": best - float(broad.decoded_immediate_utility.max()),
            "top_eight_regret_to_full_best": best - float(top_eight.decoded_immediate_utility.max()),
        })
    states = pd.DataFrame(rows)
    atomic_csv(states, OUT / "r09_full_bank_audit_states.csv")
    return {
        "schema": "phase6i-mr-r09-full-bank-audit-v1.2",
        "status": "DIAGNOSTIC_ONLY",
        "state_count": len(states),
        "mean_bank_size": float(states.bank_size.mean()),
        "true_best_absent_from_broad_four_rate": float(
            states.true_best_absent_from_broad_four.mean()
        ),
        "true_best_absent_from_top_eight_rate": float(
            states.true_best_absent_from_top_eight.mean()
        ),
        "mean_broad_four_true_top4_recall": float(states.broad_four_true_top4_recall.mean()),
        "mean_top_eight_true_top4_recall": float(states.top_eight_true_top4_recall.mean()),
        "mean_broad_four_regret_to_full_best": float(
            states.broad_four_regret_to_full_best.mean()
        ),
        "mean_top_eight_regret_to_full_best": float(
            states.top_eight_regret_to_full_best.mean()
        ),
        "candidate_bank_change_allowed": False,
        "r10_accessed": False,
        "r11_accessed": False,
    }


def _continuation_diagnostic(
    phase6h_probability: FrozenCalibrator,
    training_protocol_hash: str,
) -> dict[str, object]:
    predictions_path = MODEL_ROOT / "continuation_oof_predictions.parquet"
    source = pd.read_parquet(predictions_path)
    ensemble = ensemble_oof_predictions(
        source,
        value_column="predicted_normalized_value",
        score_column="frozen_reference_score",
        expected_training_seeds=TRAINING_SEEDS,
    )
    calibration = cross_fit_utility_calibration(
        ensemble.ensemble_raw_value,
        ensemble.continuation_value,
        ensemble.oof_fold,
    )
    ensemble["calibrated_continuation_value"] = calibration.predictions
    calibration_path = CALIBRATION_OUT / "U2H_CONTINUATION_DIAGNOSTIC.json"
    atomic_json(_calibration_payload(
        "U2H_CONTINUATION_DIAGNOSTIC",
        phase6h_probability,
        calibration.calibrator,
        probability_origin="REUSED_PHASE6H_PLATT_FROZEN_SCORE_HEAD",
        utility_target="fixed_horizon_continuation_value",
    ), calibration_path)
    artifacts = []
    parameter_counts = set()
    for seed in TRAINING_SEEDS:
        base = (
            MODEL_ROOT / "full_artifacts_continuation/U2_H_CONTINUATION/CONTINUATION_ONLY"
            / f"seed_{seed}"
        )
        checkpoint_path, record_path = base.with_suffix(".pt"), base.with_suffix(".json")
        record = load_json(record_path)
        if not all([
            record.get("status") == "COMPLETE",
            record.get("checkpoint_sha256") == digest(checkpoint_path),
            record.get("training_protocol_sha256") == training_protocol_hash,
            record.get("r10_accessed") is False,
            record.get("r11_accessed") is False,
        ]):
            raise RuntimeError(f"invalid continuation artifact: {record_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        parameter_counts.add(sum(value.numel() for value in checkpoint["model_state_dict"].values()))
        artifacts.append({
            "training_seed": seed,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": digest(checkpoint_path),
            "record_path": str(record_path.relative_to(ROOT)),
            "record_sha256": digest(record_path),
        })
    seed_records = pd.read_parquet(MODEL_ROOT / "continuation_oof_metrics.parquet")
    mean_overall = float(seed_records.overall_per_instance_spearman.mean())
    scale_means = {
        scale: float(seed_records[f"scale_{scale}_per_instance_spearman"].mean())
        for scale in ("S", "M", "L")
    }
    return {
        "candidate_id": "U2H_CONTINUATION_DIAGNOSTIC",
        "role": "ACTIVATED_AND_REPORTED_NOT_R10_SELECTABLE",
        "r10_selectable": False,
        "exclusion_reasons": [
            "R09 grouped-OOF continuation mean overall Spearman is not positive",
            "R09 grouped-OOF continuation has negative scale means",
            "the 27-state diagnostic cannot satisfy the frozen 100-selected-actions-per-scale direct-support rule",
        ],
        "mean_overall_spearman": mean_overall,
        "scale_mean_spearman": scale_means,
        "calibration_path": str(calibration_path.relative_to(ROOT)),
        "calibration_sha256": digest(calibration_path),
        "parameter_count": int(parameter_counts.pop()),
        "artifacts": artifacts,
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    training_protocol = load_json(TRAINING_PROTOCOL_PATH)
    u3_protocol = load_json(U3_PROTOCOL_PATH)
    phase6h_policy = load_json(PHASE6H_POLICY_PATH)
    training_protocol_hash = digest(TRAINING_PROTOCOL_PATH)
    u3_protocol_hash = digest(U3_PROTOCOL_PATH)
    if not all([
        training_protocol.get("r10_accessed") is False,
        training_protocol.get("r11_accessed") is False,
        u3_protocol.get("r10_accessed") is False,
        u3_protocol.get("r11_accessed") is False,
        load_json(R09_COLLECTION / "collection_integrity.json").get("status") == "PASS",
        not (ROOT / "outputs/phase6i_mr/collection/r10/access_ledger.json").exists(),
    ]):
        raise RuntimeError("pre-R10 access boundary is not clean")

    raw_context_columns = [
        name.replace("context__", "raw_context__", 1)
        for name in training_protocol["context_columns"]
    ]
    support_source = pd.read_parquet(MODEL_ROOT / "immediate_oof_predictions.parquet")
    support_source = support_source[
        (support_source.model_family == "U1")
        & (support_source.data_variant == "R09_ONLY")
        & (support_source.training_seed == TRAINING_SEEDS[0])
    ]
    support_bounds = fit_support_bounds(support_source, raw_context_columns)
    support_payload = {
        "schema": "phase6i-mr-r09-support-bounds-v1.2",
        "status": "FROZEN_BEFORE_R10_ACCESS",
        "method": "inclusive literal finite min/max for required R09 raw context features",
        "bounds": support_bounds,
        "r10_accessed": False,
        "r11_accessed": False,
    }
    support_path = OUT / "support_bounds.json"
    atomic_json(support_payload, support_path)

    phase6h_probability = FrozenCalibrator(**phase6h_policy["probability_calibrator"])
    candidate_records = []
    all_thresholds = []
    ranking_records = []
    for candidate_id, family, variant, prediction_name in IMMEDIATE_CANDIDATES:
        source_path = MODEL_ROOT / prediction_name
        source = pd.read_parquet(source_path)
        source = source[(source.model_family == family) & (source.data_variant == variant)]
        score_column = "adapted_score" if family == "U3" else "frozen_reference_score"
        ensemble = ensemble_oof_predictions(
            source,
            value_column="predicted_normalized_value",
            score_column=score_column,
            expected_training_seeds=TRAINING_SEEDS,
        )
        utility = cross_fit_utility_calibration(
            ensemble.ensemble_raw_value,
            ensemble.decoded_immediate_utility,
            ensemble.oof_fold,
        )
        if family == "U3":
            probability = cross_fit_probability_calibration(
                ensemble.ensemble_raw_score,
                ensemble.positive_label,
                ensemble.oof_fold,
            )
            probability_calibrator = probability.calibrator
            probability_values = probability.predictions
            probability_origin = "R09_GROUPED_CROSS_FITTED_PLATT_FOR_ADAPTED_SCORE"
        else:
            probability_calibrator = phase6h_probability
            probability_values = phase6h_probability.predict(ensemble.ensemble_raw_score)
            probability_origin = "REUSED_PHASE6H_PLATT_FROZEN_SCORE_HEAD"
        ensemble["calibrated_probability"] = probability_values
        ensemble["calibrated_utility"] = utility.predictions
        ensemble["supported"] = support_mask(ensemble, support_bounds)
        compact_columns = [
            "state_id", "instance_id", "scale", "CF_level", "search_stage",
            "candidate_role", "target_set_id", "decoded_immediate_utility",
            "positive_label", "fallback_target_set_id", "fallback_decoded_utility",
            "oof_fold", "ensemble_raw_value", "ensemble_raw_score",
            "calibrated_probability", "calibrated_utility", "supported",
            *raw_context_columns,
        ]
        prediction_path = OUT / "r09_oof" / f"{candidate_id}.parquet"
        atomic_parquet(ensemble[compact_columns], prediction_path)
        calibration_path = CALIBRATION_OUT / f"{candidate_id}.json"
        atomic_json(_calibration_payload(
            candidate_id,
            probability_calibrator,
            utility.calibrator,
            probability_origin=probability_origin,
            utility_target="decoded_immediate_utility",
        ), calibration_path)

        states = state_ranking_metrics(ensemble)
        ranking = summarize_ranking_metrics(states)
        ranking_records.append({"candidate_id": candidate_id, **ranking})
        threshold_table, eligible_thresholds = _threshold_audit(
            candidate_id, ensemble, config
        )
        all_thresholds.append(threshold_table)
        artifacts, parameter_count = _validate_artifacts(
            family,
            variant,
            u3_protocol_hash if family == "U3" else training_protocol_hash,
        )
        seed_metrics = _seed_metrics(family, variant)
        candidate_records.append({
            "candidate_id": candidate_id,
            "model_family": family,
            "data_variant": variant,
            "target": "immediate_utility",
            "r10_selectable": bool(eligible_thresholds),
            "ensemble_rule": "arithmetic mean of the three seed-specific normalized utility outputs",
            "score_rule": (
                "arithmetic mean of three adapted scores"
                if family == "U3" else "single shared frozen Phase6F score"
            ),
            "parameter_count": parameter_count,
            "artifacts": artifacts,
            "oof_source_path": str(source_path.relative_to(ROOT)),
            "oof_source_sha256": digest(source_path),
            "r09_ensemble_predictions_path": str(prediction_path.relative_to(ROOT)),
            "r09_ensemble_predictions_sha256": digest(prediction_path),
            "calibration_path": str(calibration_path.relative_to(ROOT)),
            "calibration_sha256": digest(calibration_path),
            "eligible_threshold_set_ids": eligible_thresholds,
            "seed_metrics": seed_metrics,
            "seed_stability_r09_diagnostic": _seed_stability(seed_metrics),
            "ensemble_r09_ranking_diagnostic": ranking,
            "probability_r09_diagnostic": calibration_metrics(
                probability_values, ensemble.positive_label
            ),
            "utility_r09_cross_fitted_mae": float(np.mean(np.abs(
                utility.predictions - ensemble.decoded_immediate_utility.to_numpy(dtype=float)
            ))),
            "utility_r09_cross_fitted_rmse": float(np.sqrt(np.mean(
                (utility.predictions - ensemble.decoded_immediate_utility.to_numpy(dtype=float)) ** 2
            ))),
            "utility_r09_cross_fitted_spearman": _finite_correlation(
                utility.predictions, ensemble.decoded_immediate_utility
            ),
        })

    threshold_frame = pd.concat(all_thresholds, ignore_index=True)
    atomic_parquet(threshold_frame, OUT / "r09_threshold_audit.parquet")
    atomic_csv(threshold_frame, OUT / "r09_threshold_audit.csv")
    atomic_csv(pd.DataFrame(ranking_records), OUT / "r09_ensemble_ranking.csv")
    full_bank = _full_bank_audit()
    atomic_json(full_bank, OUT / "r09_full_bank_audit.json")
    continuation = _continuation_diagnostic(
        phase6h_probability, training_protocol_hash
    )

    registry = {
        "schema": "phase6i-mr-pre-r10-candidate-registry-v1.2",
        "status": "FROZEN_BEFORE_R10_ACCESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seeds": list(TRAINING_SEEDS),
        "single_best_seed_selection": "FORBIDDEN",
        "ranking_calibration_separation": (
            "rank by uncalibrated ensemble value; use calibrated values only for gates"
        ),
        "probability_threshold_grid": config["live_policy"]["probability_threshold_grid"],
        "immediate_utility_threshold_grid": config["live_policy"]["immediate_utility_threshold_grid"],
        "threshold_set_id_rule": "zero-based indices Pxx_Uxx into the frozen ordered grids",
        "support_bounds_path": str(support_path.relative_to(ROOT)),
        "support_bounds_sha256": digest(support_path),
        "r10_selection_rule": training_protocol["r10_selection"],
        "training_seed_eligibility": config["training_seed_robustness"]["eligibility"],
        "immediate_candidates": candidate_records,
        "continuation_diagnostic": continuation,
        "u0_comparator": {
            "candidate_id": "U0_PHASE6H_REFERENCE",
            "r10_selectable": False,
            "role": "unchanged comparator only",
            "phase6h_policy_path": str(PHASE6H_POLICY_PATH.relative_to(ROOT)),
            "phase6h_policy_sha256": digest(PHASE6H_POLICY_PATH),
        },
        "full_bank_diagnostic": full_bank,
        "r10_accessed": False,
        "r11_accessed": False,
    }
    registry_path = OUT / "candidate_registry.json"
    atomic_json(registry, registry_path)

    generated = [
        support_path,
        OUT / "r09_threshold_audit.parquet",
        OUT / "r09_threshold_audit.csv",
        OUT / "r09_ensemble_ranking.csv",
        OUT / "r09_full_bank_audit_states.csv",
        OUT / "r09_full_bank_audit.json",
        registry_path,
        *sorted(CALIBRATION_OUT.glob("*.json")),
        *sorted((OUT / "r09_oof").glob("*.parquet")),
    ]
    checks = {
        "r09_collection_pass": load_json(R09_COLLECTION / "collection_integrity.json")["status"] == "PASS",
        "all_seven_immediate_candidates_registered": len(candidate_records) == 7,
        "every_immediate_candidate_has_three_artifacts": all(
            len(candidate["artifacts"]) == 3 for candidate in candidate_records
        ),
        "at_least_one_threshold_per_immediate_candidate": all(
            candidate["eligible_threshold_set_ids"] for candidate in candidate_records
        ),
        "u2h_reported_separately": continuation["r10_selectable"] is False,
        "full_bank_nine_states": full_bank["state_count"] == 9,
        "r10_execution_code_present": all((ROOT / path).is_file() for path in [
            "scripts/run_phase6i_mr_collection.py",
            "scripts/build_phase6i_mr_r10_cache.py",
            "scripts/select_phase6i_mr_on_r10.py",
        ]),
        "r10_not_accessed": True,
        "r11_not_accessed": True,
    }
    code_paths = [
        "rcias_clgri/ni/phase6i_policy.py",
        "scripts/freeze_phase6i_mr_pre_r10.py",
        "scripts/run_phase6i_mr_collection.py",
        "scripts/build_phase6i_mr_r10_cache.py",
        "scripts/select_phase6i_mr_on_r10.py",
    ]
    freeze = {
        "schema": "phase6i-mr-pre-r10-freeze-v1.2",
        "status": "PASS" if all(checks.values()) else "FAILED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "input_hashes": {
            "phase6i_config": digest(CONFIG_PATH),
            "training_protocol": training_protocol_hash,
            "u3_training_protocol": u3_protocol_hash,
            "phase6h_policy": digest(PHASE6H_POLICY_PATH),
            "r09_collection_integrity": digest(R09_COLLECTION / "collection_integrity.json"),
            "training_data_freeze": digest(ROOT / "outputs/phase6i_mr/training_data/training_data_freeze.json"),
        },
        "code_hashes": {path: digest(ROOT / path) for path in code_paths},
        "generated_artifacts": {
            str(path.relative_to(ROOT)): digest(path) for path in generated
        },
        "candidate_registry_sha256": digest(registry_path),
        "r10_accessed": False,
        "r11_accessed": False,
    }
    freeze_path = OUT / "pre_r10_freeze.json"
    atomic_json(freeze, freeze_path)
    if freeze["status"] != "PASS":
        print(json.dumps(freeze, indent=2, sort_keys=True), flush=True)
        raise SystemExit(1)
    authorization = {
        "schema": "phase6i-mr-r10-collection-authorization-v1.2",
        "status": "FROZEN_BEFORE_ONE_TIME_R10_ACCESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pre_r10_freeze_path": str(freeze_path.relative_to(ROOT)),
        "pre_r10_freeze_sha256": digest(freeze_path),
        "candidate_registry_sha256": digest(registry_path),
        "collection_protocol_sha256": digest(
            ROOT / "outputs/phase6i_mr/frozen/collection_protocol.json"
        ),
        "code_hashes": freeze["code_hashes"],
        "access_limit": "ONE_COMPLETE_R10_COLLECTION_AND_SELECTION_PASS_NO_REFIT",
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(
        authorization,
        ROOT / "outputs/phase6i_mr/frozen/r10_collection_authorization.json",
    )
    print(json.dumps(freeze, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
