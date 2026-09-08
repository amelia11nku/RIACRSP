#!/usr/bin/env python3
"""Audit frozen Phase 6L/6N evidence and select the Phase 6O route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6n_architecture_data as predecessor_audit  # noqa: E402


STARTING_COMMIT = "3ec6996123737e034595236b9bea81dc1ca6817a"
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/audit"
REPORT = ROOT / "docs/reports/phase6o_top_selection_audit.md"
PHASE6N = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1"
PHASE6N_PREDICTIONS = PHASE6N / "training/ensemble_oof.parquet"
PHASE6N_LABELS = PHASE6N / "data/combined/r12_expanded_grouped_labels.parquet"
PHASE6N_RAW = PHASE6N / "data/combined/r12_expanded_seed_labels.parquet"
PHASE6L_PREDICTIONS = (
    ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet"
)
TOPK_PATH = OUT / "topk_metrics.csv"
ERROR_PATH = OUT / "top_selection_error.csv"
SHIFT_PATH = OUT / "source_shift_metrics.csv"
RESULT_PATH = OUT / "loss_alignment_audit.json"
PROTECTED_PATH = OUT / "protected_phase6n_evidence.json"
BOUNDARY_PATH = OUT / "starting_boundary.json"
DRIFT = OUT / "relation_block_drift"

KS = (1, 2, 3, 4, 6, 8)
EPSILONS = (0.0025, 0.005, 0.010)
NUMERIC_FEATURES = (
    "origin_rule_count",
    "origin_family_count",
    "destroy_target_cardinality",
    "destroy_target_fraction",
    "fallback_overlap_fraction",
    "fallback_jaccard",
    "critical_overlap_fraction",
    "bottleneck_overlap_fraction",
    "normalized_diversity_rank",
    "is_fallback",
)
CATEGORICAL_FEATURES = (
    "primary_origin_rule",
    "origin_destroy_operator",
    "origin_family",
)
ROUTE_A_THRESHOLDS = {
    "maximum_k": 6,
    "common_exact_best_recall_min": 0.80,
    "expanded_exact_best_recall_min": 0.80,
    "common_near_best_0_005_recall_min": 0.90,
    "expanded_near_best_0_005_recall_min": 0.90,
    "oracle_lift_increment_over_top1_min": 0.0025,
    "minimum_scale_exact_best_recall_min": 0.70,
    "positive_opportunity_fallback_beating_present_min": 0.90,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace changed Phase 6O evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = frame.to_csv(index=False)
    if path.exists() and path.read_text() != value:
        raise RuntimeError(f"refusing to replace changed Phase 6O evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to replace changed Phase 6O evidence: {path}")
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    text = value.rstrip() + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace changed Phase 6O report: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def rank_indices(group: pd.DataFrame, score: str) -> np.ndarray:
    return np.lexsort((
        group.target_set_id.astype(str).to_numpy(),
        -group[score].to_numpy(dtype=float),
    ))


def pairwise_accuracy(group: pd.DataFrame, score: str) -> float:
    truth = group.continuation_advantage_mean.to_numpy(dtype=float)
    prediction = group[score].to_numpy(dtype=float)
    truth_gap = truth[:, None] - truth[None, :]
    prediction_gap = prediction[:, None] - prediction[None, :]
    upper = np.triu(np.ones_like(truth_gap, dtype=bool), k=1)
    comparable = upper & (np.abs(truth_gap) > 1e-12)
    if not comparable.any():
        return 0.5
    product = truth_gap[comparable] * prediction_gap[comparable]
    return float(np.mean(np.where(product > 0, 1.0, np.where(prediction_gap[comparable] == 0, 0.5, 0.0))))


def verify_starting_boundary() -> dict:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    require(ancestor, "Phase 6N terminal commit is not an ancestor of HEAD")
    decision_path = PHASE6N / "final/final_decision.json"
    decision = json.loads(decision_path.read_text())
    require(decision["decision"] == "MODEL_REVISION_REPRESENTATION", "Phase 6N decision changed")
    require(digest(PHASE6N / "training/completion_integrity_audit.json") == decision["n4_audit_sha256"], "Phase 6N N4 audit changed")
    require(digest(PHASE6N / "quality/raw_representation_gate.json") == decision["n5_gate_sha256"], "Phase 6N N5 gate changed")
    ledgers = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        PHASE6N / "r13_selection/access_ledger.json",
        PHASE6N / "r14_holdout/access_ledger.json",
        ROOT / "outputs/phase6o_neural_shortlist_v1/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6o_neural_shortlist_v1/r14_holdout/access_ledger.json",
    ]
    existing = [str(path.relative_to(ROOT)) for path in ledgers if path.exists()]
    require(not existing, f"R13/R14 access detected: {existing}")
    predecessor_chain = predecessor_audit.verify_predecessor_evidence()
    frozen_lm = json.loads(
        (PHASE6N / "audit/protected_phase6l_phase6m_evidence.json").read_text()
    )
    require(frozen_lm == predecessor_audit.protected_phase6l_phase6m(), "Phase 6L/6M evidence changed")
    phase6n_paths = sorted(
        set(
            [path for path in PHASE6N.rglob("*") if path.is_file()]
            + list((ROOT / "docs/reports").glob("phase6n*.md"))
        )
    )
    manifest = {
        str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in phase6n_paths
    }
    atomic_json(PROTECTED_PATH, manifest)
    boundary = {
        "schema": "phase6o-starting-boundary-v1",
        "starting_commit": STARTING_COMMIT,
        "starting_commit_is_ancestor": True,
        "phase6n_decision_path": str(decision_path.relative_to(ROOT)),
        "phase6n_decision_sha256": digest(decision_path),
        "phase6n_decision": decision["decision"],
        "protected_phase6n_manifest_path": str(PROTECTED_PATH.relative_to(ROOT)),
        "protected_phase6n_files": len(manifest),
        "protected_phase6n_bytes": int(sum(row["bytes"] for row in manifest.values())),
        "protected_phase6l_phase6m_files": len(frozen_lm),
        "pre_phase6l_chained_manifest_verification": predecessor_chain,
        "r13_r14_checked_paths": [str(path.relative_to(ROOT)) for path in ledgers],
        "r13_accessed": False,
        "r14_accessed": False,
        "gurobi_run": False,
    }
    atomic_json(BOUNDARY_PATH, boundary)
    return boundary


def input_hashes() -> dict:
    paths = (
        PHASE6N_PREDICTIONS,
        PHASE6N_LABELS,
        PHASE6N_RAW,
        PHASE6L_PREDICTIONS,
        PHASE6N / "training/training_protocol.json",
        ROOT / "rcias_clgri/ni/phase6n_candidate_conditioned.py",
        ROOT / "scripts/train_phase6n_candidate_conditioned.py",
    )
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def make_slices(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    common = frame[frame.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")]
    slices = [
        ("scope", "common_original_288", common),
        ("scope", "expanded_864", frame),
    ]
    for dimension in ("phase6n_data_origin", "scale", "CF_level", "search_stage"):
        slices.extend(
            (dimension, str(value), group)
            for value, group in frame.groupby(dimension, sort=True)
        )
    return slices


def topk_metrics(
    phase6n: pd.DataFrame, phase6l: pd.DataFrame
) -> pd.DataFrame:
    phase6l_groups = {
        str(state): group for state, group in phase6l.groupby("state_id", sort=True)
    }
    rows = []
    for dimension, value, frame in make_slices(phase6n):
        groups = list(frame.groupby("state_id", sort=True))
        for k in KS:
            state_rows = []
            for state_id, group in groups:
                predicted_order = rank_indices(group, "ensemble_advantage_mean")
                truth_order = rank_indices(group, "continuation_advantage_mean")
                selected = predicted_order[:k]
                truth = group.continuation_advantage_mean.to_numpy(dtype=float)
                target_ids = group.target_set_id.astype(str).to_numpy()
                best = float(truth[truth_order[0]])
                oracle = float(truth[selected].max())
                positive_opportunity = best > 1e-12
                phase6l_overlap = math.nan
                if str(state_id) in phase6l_groups:
                    old = phase6l_groups[str(state_id)]
                    old_order = rank_indices(old, "ensemble_advantage_mean")[:k]
                    old_ids = set(old.target_set_id.astype(str).to_numpy()[old_order])
                    phase6l_overlap = len(set(target_ids[selected]) & old_ids) / min(k, len(group))
                state_rows.append({
                    "exact": str(target_ids[truth_order[0]]) in set(target_ids[selected]),
                    "near_0_0025": oracle >= best - 0.0025,
                    "near_0_005": oracle >= best - 0.005,
                    "near_0_010": oracle >= best - 0.010,
                    "oracle": oracle,
                    "regret": best - oracle,
                    "positive_opportunity": positive_opportunity,
                    "fallback_beating": bool((truth[selected] > 1e-12).any()),
                    "origin_diversity": len(set(group.origin_family.astype(str).to_numpy()[selected])),
                    "phase6l_overlap": phase6l_overlap,
                })
            states = pd.DataFrame(state_rows)
            opportunity = states[states.positive_opportunity]
            rows.append({
                "slice_dimension": dimension,
                "slice_value": value,
                "k": k,
                "states": len(states),
                "exact_best_recall": float(states.exact.mean()),
                "near_best_recall_epsilon_0_0025": float(states.near_0_0025.mean()),
                "near_best_recall_epsilon_0_005": float(states.near_0_005.mean()),
                "near_best_recall_epsilon_0_010": float(states.near_0_010.mean()),
                "oracle_shortlist_selected_lift": float(states.oracle.mean()),
                "oracle_shortlist_regret": float(states.regret.mean()),
                "positive_opportunity_states": len(opportunity),
                "fallback_beating_candidate_present": (
                    float(opportunity.fallback_beating.mean()) if len(opportunity) else math.nan
                ),
                "mean_topk_origin_family_diversity": float(states.origin_diversity.mean()),
                "phase6n_phase6l_mean_topk_overlap": float(states.phase6l_overlap.mean()),
                "overlap_states": int(states.phase6l_overlap.notna().sum()),
            })
    return pd.DataFrame(rows).sort_values(
        ["slice_dimension", "slice_value", "k"], kind="stable"
    ).reset_index(drop=True)


def crn_diagnostics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state_id, state in raw.groupby("state_id", sort=True):
        pivot = state.pivot(
            index="target_set_id", columns="continuation_seed", values="continuation_advantage"
        ).sort_index()
        require(pivot.shape[1] == 2 and not pivot.isna().any().any(), f"invalid CRN rows: {state_id}")
        first = pivot.iloc[:, 0].to_numpy(dtype=float)
        second = pivot.iloc[:, 1].to_numpy(dtype=float)
        first_winner = str(pivot.index[np.lexsort((pivot.index.astype(str), -first))[0]])
        second_winner = str(pivot.index[np.lexsort((pivot.index.astype(str), -second))[0]])
        rows.append({
            "state_id": str(state_id),
            "crn_seed_winner_disagreement": first_winner != second_winner,
            "crn_candidate_mean_abs_seed_delta": float(np.abs(first - second).mean()),
            "crn_candidate_max_abs_seed_delta": float(np.abs(first - second).max()),
            "crn_candidate_sign_disagreement_fraction": float(np.mean(first * second < 0)),
        })
    return pd.DataFrame(rows)


def loss_alignment_rows(
    phase6n: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    import torch

    scales_by_fold: dict[int, float] = {}
    for fold in (0, 1, 2):
        values = []
        for seed in (726101, 726102, 726103):
            checkpoint = torch.load(
                PHASE6N / f"training/oof/seed_{seed}/fold_{fold}.pt",
                map_location="cpu",
                weights_only=False,
            )
            values.append(float(checkpoint["objective_scales"]["pair_gap_scale"]))
        require(max(values) - min(values) <= 1e-15, f"objective scale differs by seed in fold {fold}")
        scales_by_fold[fold] = values[0]
    crn = crn_diagnostics(raw).set_index("state_id")
    rows = []
    for state_id, group in phase6n.groupby("state_id", sort=True):
        truth = group.continuation_advantage_mean.to_numpy(dtype=float)
        prediction = group.ensemble_advantage_mean.to_numpy(dtype=float)
        truth_z = (truth - truth.mean()) / max(float(truth.std(ddof=0)), 1e-6)
        prediction_z = (prediction - prediction.mean()) / max(float(prediction.std(ddof=0)), 1e-6)
        gaps = truth[:, None] - truth[None, :]
        better = gaps > 1e-12
        predicted_gaps = prediction_z[:, None] - prediction_z[None, :]
        scale = scales_by_fold[int(group.oof_fold.iloc[0])]
        weights = np.clip(np.abs(gaps) / scale, 0.25, 4.0)
        terms = weights * np.logaddexp(0.0, -predicted_gaps)
        truth_order = rank_indices(group, "continuation_advantage_mean")
        best_index = int(truth_order[0])
        top_mask = np.zeros_like(better)
        top_mask[best_index, :] = True
        top_comparable = better & top_mask
        non_top = better & ~top_mask
        total_loss = float(terms[better].sum())
        listnet = np.exp(truth_z - truth_z.max())
        listnet /= listnet.sum()
        raw_row = crn.loc[str(state_id)]
        rows.append({
            "state_id": str(state_id),
            "all_ordered_candidate_pairs": int(len(group) * (len(group) - 1)),
            "loss_directed_comparable_pairs": int(better.sum()),
            "true_best_vs_rest_pairs": int(top_comparable.sum()),
            "non_top_pairs": int(non_top.sum()),
            "top_vs_rest_pairwise_loss_sum": float(terms[top_comparable].sum()),
            "non_top_pairwise_loss_sum": float(terms[non_top].sum()),
            "top_vs_rest_pairwise_loss_share": (
                float(terms[top_comparable].sum() / total_loss) if total_loss else 0.0
            ),
            "listnet_target_mass_best": float(listnet[truth_order[:1]].sum()),
            "listnet_target_mass_top3": float(listnet[truth_order[:3]].sum()),
            "listnet_target_mass_top6": float(listnet[truth_order[:6]].sum()),
            "state_utility_spread": float(truth.max() - truth.min()),
            **raw_row.to_dict(),
        })
    return pd.DataFrame(rows)


def top_selection_errors(
    phase6n: pd.DataFrame,
    phase6l: pd.DataFrame,
    loss_rows: pd.DataFrame,
) -> pd.DataFrame:
    old_groups = {
        str(state): group for state, group in phase6l.groupby("state_id", sort=True)
    }
    rows = []
    for state_id, group in phase6n.groupby("state_id", sort=True):
        prediction_order = rank_indices(group, "ensemble_advantage_mean")
        truth_order = rank_indices(group, "continuation_advantage_mean")
        target_ids = group.target_set_id.astype(str).to_numpy()
        truth = group.continuation_advantage_mean.to_numpy(dtype=float)
        prediction = group.ensemble_advantage_mean.to_numpy(dtype=float)
        true_best_id = str(target_ids[truth_order[0]])
        winner = int(prediction_order[0])
        row = {
            "state_id": str(state_id),
            "instance_id": str(group.instance_id.iloc[0]),
            "phase6n_data_origin": str(group.phase6n_data_origin.iloc[0]),
            "scale": str(group.scale.iloc[0]),
            "CF_level": str(group.CF_level.iloc[0]),
            "search_stage": str(group.search_stage.iloc[0]),
            "search_progress": float(group.search_progress.iloc[0]),
            "true_continuation_best_target_set_id": true_best_id,
            "true_continuation_best_origin_family": str(group.origin_family.iloc[truth_order[0]]),
            "phase6n_winner_target_set_id": str(target_ids[winner]),
            "phase6n_winner_origin_family": str(group.origin_family.iloc[winner]),
            "phase6n_rank_of_true_best": int(np.flatnonzero(prediction_order == truth_order[0])[0] + 1),
            "phase6n_selected_lift": float(truth[winner]),
            "phase6n_top1_regret": float(truth[truth_order[0]] - truth[winner]),
            "phase6n_predicted_top1_top2_margin": float(prediction[prediction_order[0]] - prediction[prediction_order[1]]),
            "true_top1_top2_continuation_gap": float(truth[truth_order[0]] - truth[truth_order[1]]),
            "phase6n_winner_in_true_near_optimal_0_0025": bool(truth[winner] >= truth[truth_order[0]] - 0.0025),
            "phase6n_winner_in_true_near_optimal_0_005": bool(truth[winner] >= truth[truth_order[0]] - 0.005),
            "phase6n_winner_in_true_near_optimal_0_010": bool(truth[winner] >= truth[truth_order[0]] - 0.010),
            "phase6n_pairwise_accuracy": pairwise_accuracy(group, "ensemble_advantage_mean"),
            "phase6l_winner_target_set_id": None,
            "phase6l_selected_lift": math.nan,
            "phase6l_top1_regret": math.nan,
            "phase6l_pairwise_accuracy": math.nan,
            "phase6n_selected_lift_minus_phase6l": math.nan,
            "phase6n_pairwise_improves_but_top_candidate_worsens": False,
        }
        if str(state_id) in old_groups:
            old = old_groups[str(state_id)]
            old_order = rank_indices(old, "ensemble_advantage_mean")
            old_truth = old.continuation_advantage_mean.to_numpy(dtype=float)
            old_winner = int(old_order[0])
            old_lift = float(old_truth[old_winner])
            old_regret = float(old_truth.max() - old_lift)
            old_pairwise = pairwise_accuracy(old, "ensemble_advantage_mean")
            row.update({
                "phase6l_winner_target_set_id": str(old.target_set_id.iloc[old_winner]),
                "phase6l_selected_lift": old_lift,
                "phase6l_top1_regret": old_regret,
                "phase6l_pairwise_accuracy": old_pairwise,
                "phase6n_selected_lift_minus_phase6l": float(truth[winner] - old_lift),
                "phase6n_pairwise_improves_but_top_candidate_worsens": bool(
                    row["phase6n_pairwise_accuracy"] > old_pairwise
                    and row["phase6n_top1_regret"] > old_regret
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows).merge(loss_rows, on="state_id", validate="one_to_one")


def feature_transform(payload: dict):
    from scripts.train_phase6n_candidate_conditioned import FeatureTransform

    return FeatureTransform(
        {key: tuple(values) for key, values in payload["vocabularies"].items()},
        {key: float(value) for key, value in payload["medians"].items()},
        {key: float(value) for key, value in payload["iqrs"].items()},
    )


def relation_drift_shards(device_name: str, max_new_runs: int | None) -> tuple[pd.DataFrame | None, dict]:
    import torch
    from scripts import train_phase6n_candidate_conditioned as training

    protocol = training.validate_protocol()
    frame = pd.read_parquet(PHASE6N_LABELS)
    frames = training.state_frames(frame)
    samples = training.load_samples()
    device = torch.device(device_name)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested for relation drift audit but unavailable")
    torch.use_deterministic_algorithms(True)
    DRIFT.mkdir(parents=True, exist_ok=True)
    new_runs = 0
    run_records = []
    shards = []
    for seed in (726101, 726102, 726103):
        for fold in (0, 1, 2):
            parquet_path = DRIFT / f"seed_{seed}_fold_{fold}.parquet"
            record_path = DRIFT / f"seed_{seed}_fold_{fold}.json"
            if parquet_path.exists() and record_path.exists():
                record = json.loads(record_path.read_text())
                require(record["parquet_sha256"] == digest(parquet_path), "relation drift shard changed")
                run_records.append(record)
                shards.append(pd.read_parquet(parquet_path))
                continue
            if max_new_runs is not None and new_runs >= max_new_runs:
                continue
            started = time.perf_counter()
            checkpoint_path = PHASE6N / f"training/oof/seed_{seed}/fold_{fold}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            transform = feature_transform(checkpoint["feature_transform"])
            model = training.initialize_model(seed, transform, protocol, device)
            base_relation = {
                key: value.detach().clone()
                for key, value in model.state_encoder.layers[-1].state_dict().items()
            }
            result = model.load_state_dict(checkpoint["trainable_model_state"], strict=False)
            require(not result.unexpected_keys, f"unexpected checkpoint keys: {result.unexpected_keys}")
            state_ids = sorted(
                state_id for state_id, group in frames.items()
                if int(group.oof_fold.iloc[0]) == fold
            )
            trained = training.predict(model, state_ids, samples, frames, transform, protocol, device)
            archived = pd.read_parquet(
                PHASE6N / f"training/oof/seed_{seed}/fold_{fold}.parquet"
            )[["state_id", "target_set_id", "predicted_continuation_advantage"]]
            replay = trained[[
                "state_id", "target_set_id", "predicted_continuation_advantage"
            ]].merge(
                archived,
                on=["state_id", "target_set_id"],
                suffixes=("_replayed", "_archived"),
                validate="one_to_one",
            )
            replay_error = float(np.max(np.abs(
                replay.predicted_continuation_advantage_replayed
                - replay.predicted_continuation_advantage_archived
            )))
            require(replay_error <= 1e-5, f"checkpoint replay drifted by {replay_error}")
            trained_relation = model.state_encoder.layers[-1].state_dict()
            numerator = math.sqrt(sum(float(((trained_relation[key].detach().cpu() - value.cpu()) ** 2).sum()) for key, value in base_relation.items()))
            denominator = math.sqrt(sum(float((value.cpu() ** 2).sum()) for value in base_relation.values()))
            model.state_encoder.layers[-1].load_state_dict(base_relation)
            counterfactual = training.predict(
                model, state_ids, samples, frames, transform, protocol, device
            )[["state_id", "target_set_id", "predicted_continuation_advantage"]]
            shard = trained[[
                "state_id", "target_set_id", "phase6n_data_origin", "scale", "CF_level",
                "continuation_advantage_mean", "predicted_continuation_advantage",
            ]].merge(
                counterfactual,
                on=["state_id", "target_set_id"],
                suffixes=("_fine_tuned", "_frozen_relation"),
                validate="one_to_one",
            )
            shard["prediction_drift"] = (
                shard.predicted_continuation_advantage_fine_tuned
                - shard.predicted_continuation_advantage_frozen_relation
            )
            shard["absolute_prediction_drift"] = shard.prediction_drift.abs()
            shard["training_seed"] = seed
            shard["held_fold"] = fold
            atomic_parquet(parquet_path, shard)
            record = {
                "schema": "phase6o-relation-block-counterfactual-drift-v1",
                "training_seed": seed,
                "held_fold": fold,
                "states": len(state_ids),
                "candidates": len(shard),
                "device": str(device),
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": digest(checkpoint_path),
                "maximum_checkpoint_replay_error": replay_error,
                "relation_parameter_relative_l2_delta": numerator / max(denominator, 1e-12),
                "elapsed_seconds": time.perf_counter() - started,
                "parquet_path": str(parquet_path.relative_to(ROOT)),
                "parquet_sha256": digest(parquet_path),
                "historical_score_calls": 0,
                "r13_accessed": False,
                "r14_accessed": False,
            }
            atomic_json(record_path, record)
            run_records.append(record)
            shards.append(shard)
            new_runs += 1
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_json(OUT / "progress.json", {
                "schema": "phase6o-o0-progress-v1",
                "status": "RUNNING" if len(shards) < 9 else "DRIFT_COMPLETE",
                "relation_drift_runs_complete": len(shards),
                "relation_drift_runs_expected": 9,
                "latest_seed": seed,
                "latest_fold": fold,
                "historical_score_calls": 0,
                "r13_accessed": False,
                "r14_accessed": False,
            })
    if len(shards) != 9:
        return None, {
            "status": "PARTIAL",
            "runs_complete": len(shards),
            "runs_expected": 9,
            "new_runs": new_runs,
            "records": run_records,
        }
    joined = pd.concat(shards, ignore_index=True)
    return joined, {
        "status": "PASS",
        "runs_complete": 9,
        "runs_expected": 9,
        "new_runs": new_runs,
        "records": run_records,
        "maximum_checkpoint_replay_error": max(row["maximum_checkpoint_replay_error"] for row in run_records),
        "mean_relation_parameter_relative_l2_delta": float(np.mean([
            row["relation_parameter_relative_l2_delta"] for row in run_records
        ])),
    }


def source_shift_metrics(
    phase6n: pd.DataFrame,
    errors: pd.DataFrame,
    drift: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(source: str, category: str, metric: str, value: float, count: int) -> None:
        rows.append({
            "source": source,
            "category": category,
            "metric": metric,
            "value": float(value),
            "observations": int(count),
        })

    for source, group in phase6n.groupby("phase6n_data_origin", sort=True):
        source = str(source)
        for column in NUMERIC_FEATURES:
            values = group[column].to_numpy(dtype=float)
            for statistic, value in (
                ("mean", values.mean()),
                ("std", values.std(ddof=0)),
                ("q10", np.quantile(values, 0.10)),
                ("q50", np.quantile(values, 0.50)),
                ("q90", np.quantile(values, 0.90)),
            ):
                add(source, f"candidate_feature:{column}", statistic, value, len(values))
        for column in CATEGORICAL_FEATURES:
            fractions = group[column].astype(str).value_counts(normalize=True)
            for name, value in fractions.sort_index().items():
                add(source, f"candidate_feature:{column}", f"fraction:{name}", value, len(group))
        for column in ("origin_family",):
            fractions = group[column].astype(str).value_counts(normalize=True)
            for name, value in fractions.sort_index().items():
                add(source, "candidate_origin_distribution", f"fraction:{name}", value, len(group))
        advantage = group.continuation_advantage_mean.to_numpy(dtype=float)
        for statistic, value in (
            ("mean", advantage.mean()),
            ("std", advantage.std(ddof=0)),
            ("q10", np.quantile(advantage, 0.10)),
            ("q50", np.quantile(advantage, 0.50)),
            ("q90", np.quantile(advantage, 0.90)),
        ):
            add(source, "continuation_advantage", statistic, value, len(group))
        states = errors[errors.phase6n_data_origin.eq(source)]
        add(source, "model_error", "mean_top1_regret", states.phase6n_top1_regret.mean(), len(states))
        add(source, "model_error", "exact_top1_hit_rate", states.phase6n_rank_of_true_best.eq(1).mean(), len(states))
        add(source, "model_error", "mean_pairwise_accuracy", states.phase6n_pairwise_accuracy.mean(), len(states))
        add(source, "crn_disagreement", "winner_disagreement_rate", states.crn_seed_winner_disagreement.mean(), len(states))
        add(source, "crn_disagreement", "mean_candidate_abs_seed_delta", states.crn_candidate_mean_abs_seed_delta.mean(), len(states))
        for name, value in states.true_continuation_best_origin_family.value_counts(normalize=True).sort_index().items():
            add(source, "true_top_candidate_origin", f"fraction:{name}", value, len(states))
        for dimension in ("scale", "search_stage"):
            for name, value in states[dimension].value_counts(normalize=True).sort_index().items():
                add(source, f"state_composition:{dimension}", f"fraction:{name}", value, len(states))
        add(source, "state_composition", "state_count", len(states), len(states))
        add(source, "effective_phase6n_training_weight", "equal_state_loss_aggregate_fraction", len(states) / len(errors), len(states))
        source_drift = drift[drift.phase6n_data_origin.eq(source)]
        values = source_drift.absolute_prediction_drift.to_numpy(dtype=float)
        add(source, "fine_tuned_relation_block_prediction_drift", "mean_absolute", values.mean(), len(values))
        add(source, "fine_tuned_relation_block_prediction_drift", "q50_absolute", np.quantile(values, 0.50), len(values))
        add(source, "fine_tuned_relation_block_prediction_drift", "q90_absolute", np.quantile(values, 0.90), len(values))
        add(source, "fine_tuned_relation_block_prediction_drift", "q99_absolute", np.quantile(values, 0.99), len(values))
    return pd.DataFrame(rows).sort_values(
        ["category", "metric", "source"], kind="stable"
    ).reset_index(drop=True)


def route_decision(metrics: pd.DataFrame) -> dict:
    overall = metrics[
        metrics.slice_dimension.eq("scope")
        & metrics.slice_value.isin(["common_original_288", "expanded_864"])
        & metrics.k.le(ROUTE_A_THRESHOLDS["maximum_k"])
    ]
    top1 = overall[overall.k.eq(1)].set_index("slice_value")
    candidates = []
    for k in (2, 3, 4, 6):
        selected = overall[overall.k.eq(k)].set_index("slice_value")
        scales = metrics[
            metrics.slice_dimension.eq("scale") & metrics.k.eq(k)
        ]
        checks = {
            "common_exact_best_recall": selected.loc["common_original_288", "exact_best_recall"] >= ROUTE_A_THRESHOLDS["common_exact_best_recall_min"],
            "expanded_exact_best_recall": selected.loc["expanded_864", "exact_best_recall"] >= ROUTE_A_THRESHOLDS["expanded_exact_best_recall_min"],
            "common_near_best_0_005_recall": selected.loc["common_original_288", "near_best_recall_epsilon_0_005"] >= ROUTE_A_THRESHOLDS["common_near_best_0_005_recall_min"],
            "expanded_near_best_0_005_recall": selected.loc["expanded_864", "near_best_recall_epsilon_0_005"] >= ROUTE_A_THRESHOLDS["expanded_near_best_0_005_recall_min"],
            "common_oracle_lift_increment": selected.loc["common_original_288", "oracle_shortlist_selected_lift"] - top1.loc["common_original_288", "oracle_shortlist_selected_lift"] >= ROUTE_A_THRESHOLDS["oracle_lift_increment_over_top1_min"],
            "expanded_oracle_lift_increment": selected.loc["expanded_864", "oracle_shortlist_selected_lift"] - top1.loc["expanded_864", "oracle_shortlist_selected_lift"] >= ROUTE_A_THRESHOLDS["oracle_lift_increment_over_top1_min"],
            "no_severe_scale_exact_recall_collapse": scales.exact_best_recall.min() >= ROUTE_A_THRESHOLDS["minimum_scale_exact_best_recall_min"],
            "common_fallback_beating_presence": selected.loc["common_original_288", "fallback_beating_candidate_present"] >= ROUTE_A_THRESHOLDS["positive_opportunity_fallback_beating_present_min"],
            "expanded_fallback_beating_presence": selected.loc["expanded_864", "fallback_beating_candidate_present"] >= ROUTE_A_THRESHOLDS["positive_opportunity_fallback_beating_present_min"],
        }
        candidates.append({
            "k": k,
            "checks": {name: bool(value) for name, value in checks.items()},
            "all_checks_pass": bool(all(checks.values())),
        })
    eligible = [row["k"] for row in candidates if row["all_checks_pass"]]
    return {
        "decision": "PROCEED_NEURAL_SCREENING" if eligible else "PROCEED_TOP_UTILITY_RETRAIN",
        "selected_shortlist_k": min(eligible) if eligible else None,
        "route_a_thresholds": ROUTE_A_THRESHOLDS,
        "route_a_candidates": candidates,
        "reason": (
            "the smallest k satisfying the frozen screening-quality frontier is selected"
            if eligible else
            "no k <= 6 has strong exact/near-best recall across common and expanded evidence"
        ),
    }


def concentration_summary(errors: pd.DataFrame) -> dict:
    common = errors[errors.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")].copy()
    negative = common[common.phase6n_selected_lift_minus_phase6l.lt(0)].sort_values(
        "phase6n_selected_lift_minus_phase6l"
    )
    deficit = -negative.phase6n_selected_lift_minus_phase6l.to_numpy(dtype=float)
    cumulative = np.cumsum(deficit)

    def count_for(fraction: float) -> int:
        if not len(deficit) or cumulative[-1] == 0:
            return 0
        return int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left") + 1)

    return {
        "common_states": len(common),
        "pairwise_improves_but_top_candidate_worsens_states": int(common.phase6n_pairwise_improves_but_top_candidate_worsens.sum()),
        "negative_paired_delta_states": len(negative),
        "mean_paired_selected_lift_delta": float(common.phase6n_selected_lift_minus_phase6l.mean()),
        "paired_delta_by_scale": {
            str(name): float(group.phase6n_selected_lift_minus_phase6l.mean())
            for name, group in common.groupby("scale", sort=True)
        },
        "worst_states_for_50_percent_of_negative_deficit": count_for(0.50),
        "worst_states_for_80_percent_of_negative_deficit": count_for(0.80),
        "negative_deficit_total": float(deficit.sum()),
        "worst_10_states_negative_deficit_fraction": (
            float(deficit[:10].sum() / deficit.sum()) if deficit.sum() else 0.0
        ),
    }


def metric_summary(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    result = {}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        result[column] = {
            "mean": float(values.mean()),
            "q10": float(np.quantile(values, 0.10)),
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
        }
    return result


def render_report(result: dict, topk: pd.DataFrame) -> str:
    route = result["route"]
    rows = topk[
        topk.slice_dimension.eq("scope")
        & topk.slice_value.isin(["common_original_288", "expanded_864"])
    ]
    table = [
        "| 范围 | k | Exact recall | Near-best@0.005 | Oracle lift | Regret | Positive-opportunity presence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        table.append(
            f"| {row.slice_value} | {row.k} | {row.exact_best_recall:.3%} | "
            f"{row.near_best_recall_epsilon_0_005:.3%} | "
            f"{row.oracle_shortlist_selected_lift:.6f} | {row.oracle_shortlist_regret:.6f} | "
            f"{row.fallback_beating_candidate_present:.3%} |"
        )
    errors = result["top_selection_error_summary"]
    loss = result["loss_alignment_summary"]
    drift = result["relation_block_drift_summary"]
    crn = result["crn_summary"]
    return f"""# Phase 6O O0 Top-Selection 审计

## 路线决策

**`{route['decision']}`**。

冻结 Phase 6N critic 在 k≤6 内未达到强 shortlist recall。Route A 的 oracle evaluation 潜力很高，但真实最优候选经常不在 shortlist 中；用 solver outcomes 调整 k 会违反冻结边界。因此 O1 应预注册 Route B：保留 candidate-conditioned pooling，冻结整个 Phase 6F encoder，以 top-utility-aligned、source-balanced、noise-aware objective 重训。

{chr(10).join(table)}

没有 k≤6 同时满足 common/expanded exact recall 80%、ε=0.005 near-best recall 90% 及 scale-collapse 条件。k=6 的 common exact/near recall 分别为 {rows[(rows.slice_value.eq('common_original_288')) & rows.k.eq(6)].exact_best_recall.iloc[0]:.3%}/{rows[(rows.slice_value.eq('common_original_288')) & rows.k.eq(6)].near_best_recall_epsilon_0_005.iloc[0]:.3%}；expanded 分别为 {rows[(rows.slice_value.eq('expanded_864')) & rows.k.eq(6)].exact_best_recall.iloc[0]:.3%}/{rows[(rows.slice_value.eq('expanded_864')) & rows.k.eq(6)].near_best_recall_epsilon_0_005.iloc[0]:.3%}。

## Top-selection 与标签噪声

common 288 中，Phase 6N 的 state-level pairwise accuracy 改善、但 top candidate regret 同时变差的状态为 {errors['pairwise_improves_but_top_candidate_worsens_states']}。paired selected-lift delta 为 {errors['mean_paired_selected_lift_delta']:.6f}；最差 {errors['worst_states_for_50_percent_of_negative_deficit']} 个负向状态贡献全部负 deficit 的 50%，最差 10 个贡献 {errors['worst_10_states_negative_deficit_fraction']:.3%}。按 scale 的 delta 为 `{errors['paired_delta_by_scale']}`。

两条 CRN continuation seeds 在 {crn['winner_disagreement_states']}/{crn['states']} 个状态上给出不同 top candidate（{crn['winner_disagreement_rate']:.3%}）。这超过 O1 建议采用 targeted high-fidelity relabeling 的 25% 预注册触发线；原 Phase 6L/6N truth 不得覆盖。

## 目标函数与来源偏移

Phase 6N 每 state 平均有 {loss['all_ordered_candidate_pairs']['mean']:.1f} 个 ordered pairs，但 true-best-vs-rest 仅 {loss['true_best_vs_rest_pairs']['mean']:.1f} 个；其 pairwise-loss contribution 均值占比 {loss['top_vs_rest_pairwise_loss_share']['mean']:.3%}。ListNet 给 best/top-3/top-6 的平均 target mass 为 {loss['listnet_target_mass_best']['mean']:.3%}/{loss['listnet_target_mass_top3']['mean']:.3%}/{loss['listnet_target_mass_top6']['mean']:.3%}。现有 pairwise + standard-z ListNet objective 因而主要优化 broad ordering，而非 top-1 expected utility。

Phase 6N 训练以 state 等权处理 288 个 ORIGINAL_PHASE6J_CAUR 与 576 个 NEW_ALNS_EXPANSION，实际 aggregate weight 为 1/3 与 2/3。反事实恢复 Phase 6F relation block 后，fine-tuned block 引起的 mean absolute prediction drift 在 ORIGINAL 为 {drift['by_source']['ORIGINAL_PHASE6J_CAUR']['mean_absolute_prediction_drift']:.6f}，NEW 为 {drift['by_source']['NEW_ALNS_EXPANSION']['mean_absolute_prediction_drift']:.6f}；`larger_on_original={drift['larger_on_original']}`。完整 feature/origin/advantage/error/composition 分布见 `source_shift_metrics.csv`。

## 边界

O0 没有 optimizer step、live solver、Gurobi、R13 或 R14 访问。Phase 6I-MR/6J/6K/6L/6M/6N 保护链通过。路线在 O1 配置与协议冻结前不得开始训练或 targeted relabeling。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-drift-runs", type=int)
    args = parser.parse_args()

    boundary = verify_starting_boundary()
    hashes = input_hashes()
    phase6n = pd.read_parquet(PHASE6N_PREDICTIONS)
    labels = pd.read_parquet(PHASE6N_LABELS)
    raw = pd.read_parquet(PHASE6N_RAW)
    phase6l = pd.read_parquet(PHASE6L_PREDICTIONS)
    require(len(phase6n) == len(labels) == 20441, "Phase 6N candidate evidence is incomplete")
    require(len(raw) == 2 * len(labels), "Phase 6N CRN evidence is incomplete")
    identity = ["state_id", "target_set_id", "continuation_advantage_mean"]
    require(
        phase6n[identity].sort_values(identity[:2]).reset_index(drop=True).equals(
            labels[identity].sort_values(identity[:2]).reset_index(drop=True)
        ),
        "Phase 6N prediction/label identity changed",
    )
    common = phase6n[phase6n.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")]
    require(len(common) == len(phase6l) == 6809, "common Phase 6L/6N evidence is incomplete")
    require(
        set(zip(common.state_id, common.target_set_id))
        == set(zip(phase6l.state_id, phase6l.target_set_id)),
        "Phase 6L/6N common candidate identities differ",
    )

    drift, drift_audit = relation_drift_shards(args.device, args.max_new_drift_runs)
    if drift is None:
        print(json.dumps(drift_audit, indent=2, sort_keys=True))
        raise SystemExit(2)

    topk = topk_metrics(phase6n, phase6l)
    alignment = loss_alignment_rows(phase6n, raw)
    errors = top_selection_errors(phase6n, phase6l, alignment)
    shift = source_shift_metrics(phase6n, errors, drift)
    route = route_decision(topk)
    concentration = concentration_summary(errors)
    crn_summary = {
        "states": len(errors),
        "winner_disagreement_states": int(errors.crn_seed_winner_disagreement.sum()),
        "winner_disagreement_rate": float(errors.crn_seed_winner_disagreement.mean()),
        "targeted_relabeling_trigger_rate": 0.25,
        "targeted_relabeling_triggered": bool(errors.crn_seed_winner_disagreement.mean() > 0.25),
    }
    loss_summary = metric_summary(alignment, (
        "all_ordered_candidate_pairs",
        "true_best_vs_rest_pairs",
        "top_vs_rest_pairwise_loss_share",
        "listnet_target_mass_best",
        "listnet_target_mass_top3",
        "listnet_target_mass_top6",
        "state_utility_spread",
        "crn_candidate_mean_abs_seed_delta",
    ))
    drift_by_source = {}
    for source, group in drift.groupby("phase6n_data_origin", sort=True):
        values = group.absolute_prediction_drift.to_numpy(dtype=float)
        drift_by_source[str(source)] = {
            "candidates_x_seeds": len(group),
            "mean_absolute_prediction_drift": float(values.mean()),
            "median_absolute_prediction_drift": float(np.median(values)),
            "p90_absolute_prediction_drift": float(np.quantile(values, 0.90)),
            "p99_absolute_prediction_drift": float(np.quantile(values, 0.99)),
        }
    relation_summary = {
        **drift_audit,
        "by_source": drift_by_source,
        "larger_on_original": drift_by_source["ORIGINAL_PHASE6J_CAUR"]["mean_absolute_prediction_drift"] > drift_by_source["NEW_ALNS_EXPANSION"]["mean_absolute_prediction_drift"],
    }

    atomic_csv(TOPK_PATH, topk)
    atomic_csv(ERROR_PATH, errors)
    atomic_csv(SHIFT_PATH, shift)
    result = {
        "schema": "phase6o-top-selection-loss-alignment-audit-v1",
        "status": "PASS",
        "route": route,
        "top_selection_error_summary": concentration,
        "loss_alignment_summary": loss_summary,
        "crn_summary": crn_summary,
        "source_state_counts": {
            str(name): int(group.state_id.nunique())
            for name, group in phase6n.groupby("phase6n_data_origin", sort=True)
        },
        "effective_training_source_weight": {
            "ORIGINAL_PHASE6J_CAUR": 1 / 3,
            "NEW_ALNS_EXPANSION": 2 / 3,
            "basis": "Phase 6N equal-weight whole-state loss over 288 original and 576 new states",
        },
        "relation_block_drift_summary": relation_summary,
        "input_hashes": hashes,
        "artifacts": {
            str(TOPK_PATH.relative_to(ROOT)): digest(TOPK_PATH),
            str(ERROR_PATH.relative_to(ROOT)): digest(ERROR_PATH),
            str(SHIFT_PATH.relative_to(ROOT)): digest(SHIFT_PATH),
        },
        "starting_boundary": boundary,
        "optimizer_steps_started": False,
        "live_solver_runs_started": False,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(RESULT_PATH, result)
    atomic_text(REPORT, render_report(result, topk))
    write_json(OUT / "progress.json", {
        "schema": "phase6o-o0-progress-v1",
        "status": "COMPLETE",
        "route_decision": route["decision"],
        "relation_drift_runs_complete": 9,
        "relation_drift_runs_expected": 9,
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    })
    print(json.dumps({
        "status": result["status"],
        "route": route,
        "crn_summary": crn_summary,
        "top_selection_error_summary": concentration,
        "relation_block_drift_summary": relation_summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
