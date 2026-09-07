#!/usr/bin/env python3
"""Derive the immutable Phase 6L score-free dataset from frozen Phase 6J outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import grouped_oof_fold  # noqa: E402
from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FORBIDDEN_OUTCOME_COLUMNS,
    NUMERIC_COLUMNS,
    ONLINE_INPUT_COLUMNS,
    REMOVED_LEGACY_INPUT_COLUMNS,
    build_score_free_candidate_source_features,
)
from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402
from scripts.run_phase6j_caur_collection import read_alns_config  # noqa: E402
from scripts.run_phase6j_caur_pilot import candidate_from_dict  # noqa: E402


CONFIG = ROOT / "configs/phase6l_legacy_score_decoupling_v1.json"
PREREG = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/preregistration/preregistration.json"
SOURCE_GROUPED = ROOT / "outputs/phase6j_caur/r12_collection/r12_grouped_labels.parquet"
SOURCE_RAW = ROOT / "outputs/phase6j_caur/r12_collection/r12_seed_labels.parquet"
REPLAYS = ROOT / "outputs/phase6j_caur/r12_collection/state_replays"
OUT = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def fit_transform_record(frame: pd.DataFrame) -> dict:
    vocabularies = {
        column: sorted(frame[column].astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }
    medians = {}
    iqrs = {}
    for column in NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite score-free feature: {column}")
        medians[column] = float(np.median(values))
        iqrs[column] = max(
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-6
        )
    return {
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "vocabularies": vocabularies,
        "medians": medians,
        "iqrs": iqrs,
        "unknown_category_index": 0,
        "numeric_clip": [-8.0, 8.0],
    }


def validate_preregistration(config: dict) -> None:
    record = json.loads(PREREG.read_text())
    hashes = record["source_hashes"]
    if hashes[str(CONFIG.relative_to(ROOT))] != digest(CONFIG):
        raise RuntimeError("Phase 6L config changed after preregistration")
    if digest(SOURCE_GROUPED) != config["locked_inputs"]["phase6j_grouped_labels_sha256"]:
        raise RuntimeError("frozen grouped labels changed")
    if digest(SOURCE_RAW) != config["locked_inputs"]["phase6j_raw_seed_labels_sha256"]:
        raise RuntimeError("frozen raw labels changed")
    if any((ROOT / path).exists() for path in (
        "outputs/phase6j_caur/r13_selection/access_ledger.json",
        "outputs/phase6j_caur/r14_holdout/access_ledger.json",
    )):
        raise RuntimeError("R13/R14 access is forbidden")


def derive() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    grouped = pd.read_parquet(SOURCE_GROUPED)
    raw = pd.read_parquet(SOURCE_RAW)
    replay_current = {
        path.stem: float(json.loads(path.read_text())["snapshot"]["current_makespan"])
        for path in REPLAYS.glob("*.json")
    }
    fallback_ids = {}
    target_operations = {}
    for state_id, group in grouped.groupby("state_id", sort=True):
        direct = group[
            group.origin_rules.map(lambda value: "operator_related" in json.loads(value))
        ]
        if len(direct) != 1:
            raise RuntimeError(f"operator_related is not unique: {state_id}")
        fallback_ids[str(state_id)] = str(direct.target_set_id.iloc[0])
        target_operations[str(state_id)] = {
            str(row.target_set_id): set(json.loads(row.target_operation_ids))
            for row in group.itertuples(index=False)
        }

    raw = raw.copy()
    raw["fallback_target_set_id"] = raw.state_id.map(fallback_ids)
    fallback_results = raw[
        raw.target_set_id.eq(raw.fallback_target_set_id)
    ][["state_id", "continuation_seed", "candidate_continuation_best_makespan"]].rename(
        columns={"candidate_continuation_best_makespan": "new_fallback_best"}
    )
    if len(fallback_results) != 576:
        raise RuntimeError("score-free fallback results are incomplete")
    raw = raw.drop(columns=["fallback_continuation_best_makespan", "continuation_advantage"])
    raw = raw.merge(
        fallback_results,
        on=["state_id", "continuation_seed"],
        validate="many_to_one",
    )
    incumbent = raw.state_id.map(replay_current).to_numpy(dtype=float)
    raw["fallback_continuation_best_makespan"] = raw.pop("new_fallback_best")
    raw["continuation_advantage"] = (
        raw.fallback_continuation_best_makespan
        - raw.candidate_continuation_best_makespan
    ) / incumbent
    raw["is_fallback"] = raw.target_set_id.eq(raw.fallback_target_set_id)

    aggregates = raw.groupby(["state_id", "target_set_id"], sort=True).agg(
        continuation_seed_count=("continuation_seed", "size"),
        continuation_advantage_mean=("continuation_advantage", "mean"),
        continuation_advantage_std=("continuation_advantage", lambda values: float(np.std(values, ddof=0))),
        beats_fallback=("continuation_advantage", lambda values: float(np.mean(np.asarray(values) > 0))),
        continuation_best_makespan=("candidate_continuation_best_makespan", "mean"),
        fallback_continuation_best_makespan=("fallback_continuation_best_makespan", "mean"),
    ).reset_index()
    grouped = grouped.drop(columns=[
        "fallback_target_set_id", "fallback_overlap_fraction", "fallback_jaccard",
        "is_fallback", "continuation_seed_count", "continuation_advantage_mean",
        "continuation_advantage_std", "beats_fallback", "continuation_best_makespan",
        "fallback_continuation_best_makespan",
    ])
    grouped["fallback_target_set_id"] = grouped.state_id.map(fallback_ids)
    grouped["is_fallback"] = grouped.target_set_id.eq(grouped.fallback_target_set_id)
    overlap = []
    for row in grouped.itertuples(index=False):
        targets = target_operations[str(row.state_id)]
        target = targets[str(row.target_set_id)]
        fallback = targets[str(row.fallback_target_set_id)]
        overlap.append((len(target & fallback) / len(target), jaccard(target, fallback)))
    grouped["fallback_overlap_fraction"] = [row[0] for row in overlap]
    grouped["fallback_jaccard"] = [row[1] for row in overlap]
    grouped = grouped.merge(aggregates, on=["state_id", "target_set_id"], validate="one_to_one")
    grouped["split"] = "R12_CAUR_FIT"
    grouped["oof_fold"] = [
        grouped_oof_fold(str(scale), str(cf))
        for scale, cf in zip(grouped.scale, grouped.CF_level)
    ]

    model_columns = [
        "instance_id", "instance_relative_path", "instance_sha256", "scale", "CF_level",
        "RI_level", "TI_level", "cell_replicate", "trajectory_seed", "state_id",
        "target_set_id", "label_scope", "is_reduced_top8_audit", *ONLINE_INPUT_COLUMNS,
        "split", "target_progress", "search_progress", "search_stage", "source_iteration",
        "fallback_target_set_id", "target_operation_ids", "origin_rules", "origin_families",
        "immediate_utility", "candidate_feasible", "requested_bank_count",
        "full_bank_unique_count", "duplicate_bank_count", "bottleneck_proxy",
        "critical_operation_ids", "bottleneck_operation_ids", "labels_post_source_trajectory",
        "horizon", "continuation_seed_count", "continuation_advantage_mean",
        "continuation_advantage_std", "beats_fallback", "continuation_best_makespan",
        "fallback_continuation_best_makespan", "oof_fold",
    ]
    raw_columns = [
        "instance_id", "scale", "CF_level", "state_id", "target_set_id",
        "fallback_target_set_id", "continuation_seed", "horizon",
        "candidate_continuation_best_makespan", "fallback_continuation_best_makespan",
        "continuation_advantage", "paired_derived_seed", "is_fallback",
    ]
    grouped = grouped[model_columns].sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    raw = raw[raw_columns].sort_values(
        ["state_id", "target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    old = pd.read_parquet(SOURCE_GROUPED, columns=[
        "state_id", "target_set_id", "fallback_target_set_id", "continuation_advantage_mean"
    ])
    compared = grouped[["state_id", "target_set_id", "fallback_target_set_id", "continuation_advantage_mean"]].merge(
        old, on=["state_id", "target_set_id"], suffixes=("_new", "_old"), validate="one_to_one"
    )
    same_states = compared.fallback_target_set_id_new.eq(compared.fallback_target_set_id_old)
    max_same_delta = float(np.max(np.abs(
        compared.loc[same_states, "continuation_advantage_mean_new"]
        - compared.loc[same_states, "continuation_advantage_mean_old"]
    )))
    audit = {
        "states": int(grouped.state_id.nunique()),
        "candidates": len(grouped),
        "raw_seed_rows": len(raw),
        "fallback_rows": int(grouped.is_fallback.sum()),
        "fallback_raw_advantage_max_abs": float(raw.loc[raw.is_fallback, "continuation_advantage"].abs().max()),
        "reanchored_states": int(compared.loc[
            compared.fallback_target_set_id_new.ne(compared.fallback_target_set_id_old), "state_id"
        ].nunique()),
        "unchanged_fallback_states": int(compared.loc[same_states, "state_id"].nunique()),
        "unchanged_state_label_max_abs_delta": max_same_delta,
    }
    return grouped, raw, audit


def validate_live_builder(grouped: pd.DataFrame, config: dict) -> dict:
    phase6j = json.loads((ROOT / config["locked_inputs"]["phase6j_config"]).read_text())
    alns = read_alns_config(phase6j)
    instances = {}
    checked = 0
    for state_id, frame in grouped.groupby("state_id", sort=True):
        replay = json.loads((REPLAYS / f"{state_id}.json").read_text())
        relative = replay["instance_relative_path"]
        if relative not in instances:
            instances[relative] = load_phase6j_instance(
                ROOT / phase6j["instance_suite"]["root"] / relative
            )
        instance = instances[relative]
        snapshot = replay["snapshot"]
        current = decode_candidate(instance, candidate_from_dict(snapshot["current_candidate"]))
        count = min(max(2, round(instance.num_operations * alns.destroy_fraction)), instance.num_operations)
        generated = generate_revised_target_arms(
            instance, current, str(state_id), count, int(phase6j["rng"]["proposal_namespace"])
        )
        built = pd.DataFrame(build_score_free_candidate_source_features(
            generated,
            state_id=str(state_id),
            operation_count=instance.num_operations,
            critical_operations=json.loads(frame.critical_operation_ids.iloc[0]),
            bottleneck_operations=json.loads(frame.bottleneck_operation_ids.iloc[0]),
        )).sort_values("target_set_id").reset_index(drop=True)
        expected = frame.sort_values("target_set_id").reset_index(drop=True)
        if tuple(built.target_set_id) != tuple(expected.target_set_id):
            raise RuntimeError(f"live/dataset candidate drift: {state_id}")
        for column in ONLINE_INPUT_COLUMNS:
            if column in CATEGORICAL_COLUMNS:
                equal = built[column].astype(str).equals(expected[column].astype(str))
            else:
                equal = np.allclose(
                    built[column].to_numpy(dtype=float),
                    expected[column].to_numpy(dtype=float), atol=1e-12, rtol=0.0,
                )
            if not equal:
                raise RuntimeError(f"live/dataset feature drift: {state_id}/{column}")
        checked += 1
    return {"states_checked": checked, "candidate_feature_pairs_checked": len(grouped)}


def main() -> None:
    config = json.loads(CONFIG.read_text())
    validate_preregistration(config)
    grouped, raw, derivation = derive()
    if derivation != {
        **derivation,
        "states": 288,
        "candidates": 6809,
        "raw_seed_rows": 13618,
        "fallback_rows": 288,
        "fallback_raw_advantage_max_abs": 0.0,
        "reanchored_states": 10,
        "unchanged_fallback_states": 278,
    } or derivation["unchanged_state_label_max_abs_delta"] > 1e-15:
        raise RuntimeError(f"score-free derivation integrity failed: {derivation}")
    live = validate_live_builder(grouped, config)
    grouped_path = OUT / "r12_score_free_grouped_labels.parquet"
    raw_path = OUT / "r12_score_free_seed_labels.parquet"
    atomic_parquet(grouped_path, grouped)
    atomic_parquet(raw_path, raw)

    transforms = {}
    for held in (0, 1, 2):
        outer = grouped[grouped.oof_fold.ne(held)]
        inner_fit = grouped[grouped.oof_fold.eq((held + 2) % 3)]
        transforms[f"outer_held_fold_{held}"] = {
            "outer_training": fit_transform_record(outer),
            "inner_epoch_training": fit_transform_record(inner_fit),
            "outer_training_states": int(outer.state_id.nunique()),
            "inner_training_states": int(inner_fit.state_id.nunique()),
        }
    full_transform = fit_transform_record(grouped)
    normalization = {
        "schema": "phase6l-score-free-normalization-manifest-v1",
        "status": "FROZEN_FROM_AUTHORIZED_TRAINING_FOLDS",
        "fit_policy": "separate inner and outer training-fold transforms; full transform only for final deployment after R12 qualification",
        "fold_transforms": transforms,
        "full_r12_transform": full_transform,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    normalization_path = OUT / "normalization_manifest.json"
    atomic_json(normalization_path, normalization)
    schema = {
        "schema": "phase6l-score-free-feature-schema-v1",
        "online_input_columns": list(ONLINE_INPUT_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "removed_legacy_columns": list(REMOVED_LEGACY_INPUT_COLUMNS),
        "forbidden_outcome_columns": list(FORBIDDEN_OUTCOME_COLUMNS),
        "historical_score_forward_calls": 0,
        "fallback": config["fallback"],
    }
    provenance = {
        "schema": "phase6l-score-free-feature-provenance-v1",
        "source": "frozen Phase 6J R12_CAUR_FIT outcomes",
        "source_grouped_sha256": digest(SOURCE_GROUPED),
        "source_raw_sha256": digest(SOURCE_RAW),
        "derivation": derivation,
        "live_builder_parity": live,
        "historical_model_used_for_derivation": False,
        "new_decoder_or_continuation_runs": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    schema_path = OUT / "feature_schema.json"
    provenance_path = OUT / "feature_provenance.json"
    atomic_json(schema_path, schema)
    atomic_json(provenance_path, provenance)
    manifest = {
        "schema": "phase6l-score-free-dataset-manifest-v1",
        "status": "PASS",
        "grouped_path": str(grouped_path.relative_to(ROOT)),
        "grouped_sha256": digest(grouped_path),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": digest(raw_path),
        "feature_schema_sha256": digest(schema_path),
        "feature_provenance_sha256": digest(provenance_path),
        "normalization_manifest_sha256": digest(normalization_path),
        "preregistration_sha256": digest(PREREG),
        "derivation": derivation,
        "live_builder_parity": live,
        "online_historical_score_forward_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
