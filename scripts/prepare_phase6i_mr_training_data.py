#!/usr/bin/env python3
"""Audit and freeze Phase 6I-MR R09-only model-training inputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
COLLECTION = ROOT / "outputs/phase6i_mr/collection/r09"
CONTINUATION = ROOT / "outputs/phase6i_mr/continuation"
PILOT = ROOT / "outputs/phase6i_mr/pilot_v12"
OLD_CACHE_MANIFEST = ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv"
OLD_DATASET = ROOT / "outputs/phase6c/dataset/train"
OUT = ROOT / "outputs/phase6i_mr/training_data"


CONTEXT_SOURCES = {
    "log1p_operation_count_r09_robust_z": lambda row: np.log1p(row.operation_count),
    "log1p_graph_node_count_r09_robust_z": lambda row: np.log1p(row.graph_node_count),
    "log1p_graph_edge_count_r09_robust_z": lambda row: np.log1p(row.graph_edge_count),
    "edge_per_node_ratio": lambda row: row.graph_edge_count / row.graph_node_count,
    "dag_depth_per_operation": lambda row: row.dag_depth_proxy / row.operation_count,
    "dag_width_per_operation": lambda row: row.dag_width_proxy / row.operation_count,
    "eligibility_density": lambda row: row.eligibility_density,
    "resource_load_cv": lambda row: row.resource_load_cv,
    "current_makespan_over_h1_makespan": lambda row: row.current_makespan_over_h1_makespan,
    "critical_path_over_h1_critical_path": lambda row: row.critical_path_over_h1_critical_path,
    "w_delay_over_current_makespan": lambda row: row.w_delay_over_current_makespan,
    "f_delay_over_current_makespan": lambda row: row.f_delay_over_current_makespan,
    "reconfiguration_over_current_makespan": lambda row: row.reconfiguration_over_current_makespan,
    "mean_slack_ratio": lambda row: row.mean_slack_ratio,
    "mean_w_delay_ratio": lambda row: row.mean_w_delay_ratio,
    "mean_f_delay_ratio": lambda row: row.mean_f_delay_ratio,
    "mean_island_relative_load": lambda row: row.mean_island_relative_load,
    "mean_local_reconfiguration_ratio": lambda row: row.mean_local_reconfiguration_ratio,
    "search_progress": lambda row: row.search_progress,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        ) + "\n"
    )
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


def broad_stage(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "EARLY"
    if progress < 2.0 / 3.0:
        return "MIDDLE"
    return "LATE"


def fold_map(config: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for fold, cells in config["hard_state_aggregation"]["grouped_oof_folds"].items():
        for cell in cells:
            if cell in mapping:
                raise RuntimeError(f"duplicate OOF cell: {cell}")
            mapping[cell] = fold
    expected = {f"{scale}_CF{cf}" for scale in "SML" for cf in (1, 2, 3)}
    if set(mapping) != expected:
        raise RuntimeError("grouped OOF folds do not cover the nine structural cells")
    return mapping


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name, function in CONTEXT_SOURCES.items():
        result[f"raw_context__{name}"] = function(result)
    values = result[[f"raw_context__{name}" for name in CONTEXT_SOURCES]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError("R09 context contains non-finite values")
    return result


def fit_normalization(states: pd.DataFrame) -> dict[str, dict[str, float]]:
    constants = {}
    for name in CONTEXT_SOURCES:
        values = states[f"raw_context__{name}"].to_numpy(dtype=float)
        q005, q25, median, q75, q995 = np.quantile(
            values, [0.005, 0.25, 0.50, 0.75, 0.995]
        )
        robust_scale = max(float((q75 - q25) / 1.349), 1e-6)
        constants[name] = {
            "median": float(median),
            "robust_scale": robust_scale,
            "winsor_lower": float(q005),
            "winsor_upper": float(q995),
            "support_lower": float(q005),
            "support_upper": float(q995),
            "fit_split": "R09_ONLY",
        }
    return constants


def apply_normalization(
    frame: pd.DataFrame,
    constants: dict[str, dict[str, float]],
) -> pd.DataFrame:
    result = frame.copy()
    for name, record in constants.items():
        values = result[f"raw_context__{name}"].clip(
            record["winsor_lower"], record["winsor_upper"]
        )
        result[f"context__{name}"] = (
            values - record["median"]
        ) / record["robust_scale"]
    return result


def select_old_train_states() -> pd.DataFrame:
    cache_manifest = pd.read_csv(OLD_CACHE_MANIFEST)
    records = []
    for scale in "SML":
        for cf in (1, 2, 3):
            instance_id = f"CB1_TRAIN_{scale}_CF{cf}_RI2_TI2_R02"
            cache = cache_manifest[
                cache_manifest.instance_id.eq(instance_id)
                & cache_manifest.training_split.eq("TRAIN")
                & cache_manifest.status.eq("COMPLETE")
            ]
            if len(cache) != 1:
                raise RuntimeError(f"missing unique old TRAIN cache: {instance_id}")
            cache_row = cache.iloc[0]
            cache_path = Path(str(cache_row.cache_path))
            if digest(cache_path) != str(cache_row.cache_sha256):
                raise RuntimeError(f"old TRAIN cache hash mismatch: {instance_id}")
            states_path = OLD_DATASET / instance_id / "states.parquet"
            states = pd.read_parquet(states_path)
            states["training_stage"] = states.search_progress.map(broad_stage)
            for stage in ("EARLY", "MIDDLE", "LATE"):
                candidates = states[states.training_stage.eq(stage)].copy()
                candidates["selection_key"] = candidates.state_id.map(
                    lambda state_id: hashlib.sha256(
                        f"phase6i-old-subset-v1.2|{state_id}".encode()
                    ).hexdigest()
                )
                chosen = candidates.sort_values(
                    ["selection_key", "state_id"]
                ).head(20)
                if len(chosen) != 20:
                    raise RuntimeError(
                        f"old TRAIN shard lacks 20 {stage} states: {instance_id}"
                    )
                for row in chosen.itertuples(index=False):
                    records.append({
                        "instance_id": instance_id,
                        "state_id": row.state_id,
                        "scale": scale,
                        "CF_level": f"CF{cf}",
                        "search_stage": stage,
                        "cache_path": str(cache_path),
                        "cache_sha256": str(cache_row.cache_sha256),
                        "source_states_path": str(states_path),
                        "source_states_sha256": digest(states_path),
                        "selection_key": row.selection_key,
                        "training_split": "EARLIER_PHASE6F_TRAIN_ONLY",
                    })
    result = pd.DataFrame(records)
    if len(result) != 540 or result.state_id.nunique() != 540:
        raise RuntimeError("old TRAIN selection must contain 540 unique states")
    return result.sort_values(["scale", "CF_level", "search_stage", "state_id"])


def grouped_summary(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return frame.groupby(key).agg(
        states=("state_id", "nunique"),
        actions=("state_id", "size"),
        positive_rate=("positive_label", "mean"),
        mean_immediate_utility=("decoded_immediate_utility", "mean"),
        mean_regret=("regret_to_best", "mean"),
        sign_error_rate=("sign_error", "mean"),
    ).reset_index()


def main() -> None:
    config = load_json(CONFIG_PATH)
    collection_integrity_path = COLLECTION / "collection_integrity.json"
    collection_integrity = load_json(collection_integrity_path)
    continuation_integrity_path = CONTINUATION / "continuation_integrity.json"
    continuation_integrity = load_json(continuation_integrity_path)
    if not all([
        collection_integrity.get("status") == "PASS",
        collection_integrity.get("r10_accessed") is False,
        collection_integrity.get("r11_accessed") is False,
        continuation_integrity.get("status") == "PASS",
        continuation_integrity.get("r10_accessed") is False,
        continuation_integrity.get("r11_accessed") is False,
    ]):
        raise RuntimeError("R09 collection and continuation integrity must pass")

    source_path = COLLECTION / "forced_action_labels.parquet"
    actions = pd.read_parquet(source_path)
    if not all([
        len(actions) == 6480,
        actions.state_id.nunique() == 1620,
        actions.instance_id.nunique() == 18,
        set(actions.split) == {"R09"},
        not actions.duplicated(["state_id", "target_set_id"]).any(),
        bool(actions.candidate_feasible.all()),
        bool(actions.labels_post_trajectory.all()),
    ]):
        raise RuntimeError("R09 broad label table failed training-input checks")
    folds = fold_map(config)
    actions["structural_cell"] = actions.scale + "_" + actions.CF_level
    actions["oof_fold"] = actions.structural_cell.map(folds)
    if actions.oof_fold.isna().any():
        raise RuntimeError("R09 action lacks grouped OOF fold")
    actions = add_context(actions)
    state_context = actions.drop_duplicates("state_id").copy()
    normalization = fit_normalization(state_context)
    actions = apply_normalization(actions, normalization)

    p99 = float(actions.decoded_immediate_utility.abs().quantile(0.99))
    actions["normalized_immediate_utility"] = (
        actions.decoded_immediate_utility / p99
    ).clip(-1.0, 1.0)
    positive_count = int(actions.positive_label.sum())
    negative_count = len(actions) - positive_count
    positive_weight = negative_count / max(positive_count, 1)
    state_std = actions.groupby("state_id").decoded_immediate_utility.std(ddof=0)
    listwise_temperature = max(
        float(state_std[state_std > 0].median()), 0.001
    )

    pilot_actions = pd.read_parquet(PILOT / "forced_action_failure_table.parquet")
    continuation = pd.read_parquet(
        CONTINUATION / "continuation_action_table.parquet"
    )
    continuation_target = continuation.groupby(
        ["state_id", "target_set_id"], as_index=False
    ).agg(
        continuation_value=("continuation_value", "mean"),
        continuation_value_std=("continuation_value", "std"),
        continuation_seed_count=("continuation_seed", "nunique"),
        continuation_all_feasible=("continuation_feasible", "all"),
    )
    continuation_actions = pilot_actions.merge(
        continuation_target,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    if not all([
        len(continuation_actions) == 108,
        continuation_actions.state_id.nunique() == 27,
        continuation_actions.continuation_seed_count.eq(2).all(),
        continuation_actions.continuation_all_feasible.all(),
    ]):
        raise RuntimeError("continuation training target join is incomplete")
    continuation_actions["structural_cell"] = (
        continuation_actions.scale + "_" + continuation_actions.CF_level
    )
    continuation_actions["oof_fold"] = continuation_actions.structural_cell.map(folds)
    continuation_actions = add_context(continuation_actions)
    continuation_actions = apply_normalization(continuation_actions, normalization)

    old_selection = select_old_train_states()
    context_columns = [f"context__{name}" for name in CONTEXT_SOURCES]
    required_context_exact = config["scale_invariance"]["context_features"]
    if list(CONTEXT_SOURCES) != required_context_exact:
        raise RuntimeError("implemented context feature order differs from preregistration")

    OUT.mkdir(parents=True, exist_ok=True)
    r09_output = OUT / "r09_actions.parquet"
    continuation_output = OUT / "continuation_actions.parquet"
    old_output = OUT / "old_train_state_selection.csv"
    atomic_parquet(actions, r09_output)
    atomic_parquet(continuation_actions, continuation_output)
    atomic_csv(old_selection, old_output)
    atomic_csv(grouped_summary(actions, "scale"), OUT / "r09_by_scale.csv")
    atomic_csv(grouped_summary(actions, "search_stage"), OUT / "r09_by_stage.csv")
    atomic_csv(grouped_summary(actions, "candidate_role"), OUT / "r09_by_role.csv")

    constants = {
        "schema": "phase6i-mr-r09-training-constants-v1.2",
        "status": "FROZEN_R09_ONLY_BEFORE_MODEL_FIT_AND_R10_ACCESS",
        "fit_split": "R09_ONLY",
        "immediate_utility_p99_absolute": p99,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_weight": positive_weight,
        "listwise_temperature": listwise_temperature,
        "huber_delta": config["objective"]["huber_delta"],
        "context_feature_order": list(CONTEXT_SOURCES),
        "context_normalization": normalization,
        "context_output_columns": context_columns,
        "support_out_of_range_rule": "raw value outside R09 q0.005-q0.995 for any required context feature",
        "grouped_oof_folds": config["hard_state_aggregation"]["grouped_oof_folds"],
        "mixed_old_new_state_batch_ratio": [1, 3],
        "mixed_source_loss_aggregation": "mean loss per state, then 25% old plus 75% R09",
        "old_train_selection": {
            "states": len(old_selection),
            "shards": old_selection.instance_id.nunique(),
            "states_per_shard": 60,
            "states_per_stage_per_shard": 20,
            "fixed_shard_pattern": "CB1_TRAIN_{S,M,L}_CF{1,2,3}_RI2_TI2_R02",
        },
        "continuation_target": {
            "states": continuation_actions.state_id.nunique(),
            "actions": len(continuation_actions),
            "aggregation": "arithmetic mean over CRN seeds 685101 and 685102",
            "field": "continuation_value",
            "kept_separate_from_immediate_utility": True,
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }
    constants_path = OUT / "training_constants.json"
    atomic_json(constants, constants_path)
    manifest = {
        "schema": "phase6i-mr-training-data-freeze-v1.2",
        "status": "FROZEN_BEFORE_MODEL_FIT_AND_R10_ACCESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "config_sha256": digest(CONFIG_PATH),
            "r09_collection_integrity_sha256": digest(collection_integrity_path),
            "r09_forced_action_labels_sha256": digest(source_path),
            "continuation_integrity_sha256": digest(continuation_integrity_path),
            "continuation_action_table_sha256": digest(
                CONTINUATION / "continuation_action_table.parquet"
            ),
            "pilot_forced_action_table_sha256": digest(
                PILOT / "forced_action_failure_table.parquet"
            ),
            "old_cache_manifest_sha256": digest(OLD_CACHE_MANIFEST),
        },
        "frozen_outputs": {
            "r09_actions": {"path": str(r09_output), "sha256": digest(r09_output)},
            "continuation_actions": {
                "path": str(continuation_output),
                "sha256": digest(continuation_output),
            },
            "old_train_state_selection": {
                "path": str(old_output),
                "sha256": digest(old_output),
            },
            "training_constants": {
                "path": str(constants_path),
                "sha256": digest(constants_path),
            },
        },
        "cardinality": {
            "r09_instances": actions.instance_id.nunique(),
            "r09_states": actions.state_id.nunique(),
            "r09_actions": len(actions),
            "continuation_states": continuation_actions.state_id.nunique(),
            "continuation_actions": len(continuation_actions),
            "old_train_states": len(old_selection),
        },
        "checks": {
            "r09_only_fit": True,
            "three_grouped_folds_exact": actions.oof_fold.nunique() == 3,
            "no_state_target_duplicates": not actions.duplicated(
                ["state_id", "target_set_id"]
            ).any(),
            "all_context_finite": np.isfinite(
                actions[context_columns].to_numpy(dtype=float)
            ).all(),
            "old_source_train_only": set(old_selection.training_split)
            == {"EARLIER_PHASE6F_TRAIN_ONLY"},
            "continuation_labels_separate": True,
            "r10_accessed": False,
            "r11_accessed": False,
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(manifest, OUT / "training_data_freeze.json")
    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
