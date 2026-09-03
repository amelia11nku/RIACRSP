#!/usr/bin/env python3
"""Build the single authorized Phase 6I-MR R09 hard-state round."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)


TRAINING = ROOT / "outputs/phase6i_mr/model_training"
PREDICTIONS_PATH = TRAINING / "immediate_oof_predictions.parquet"
PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
CONSTANTS_PATH = ROOT / "outputs/phase6i_mr/training_data/training_constants.json"
OUT = ROOT / "outputs/phase6i_mr/hard_state_round1"


def selected_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "model_family", "data_variant", "state_id", "target_set_id"
    ]
    averaged = (
        predictions.groupby(identity, as_index=False)
        .agg(
            predicted_normalized_value=("predicted_normalized_value", "mean"),
            normalized_immediate_utility=("normalized_immediate_utility", "first"),
            decoded_immediate_utility=("decoded_immediate_utility", "first"),
            frozen_reference_score=("frozen_reference_score", "first"),
            instance_id=("instance_id", "first"),
            scale=("scale", "first"),
            CF_level=("CF_level", "first"),
            search_stage=("search_stage", "first"),
            candidate_role=("candidate_role", "first"),
            fallback_target_set_id=("fallback_target_set_id", "first"),
        )
    )
    rows = []
    candidate_keys = ["model_family", "data_variant"]
    for candidate, frame in averaged.groupby(candidate_keys, sort=True):
        work = frame.copy()
        work["residual"] = (
            work.predicted_normalized_value - work.normalized_immediate_utility
        )
        scale_stats = work.groupby("scale").residual.agg(["mean", "std"])
        work = work.join(scale_stats, on="scale", rsuffix="_scale")
        work["residual_z"] = (
            (work.residual - work["mean"])
            / work["std"].replace(0, math.nan)
        )
        selected = (
            work.sort_values(
                ["state_id", "predicted_normalized_value", "frozen_reference_score", "target_set_id"],
                ascending=[True, False, False, True],
                kind="stable",
            )
            .groupby("state_id", sort=True)
            .head(1)
            .copy()
        )
        best = work.groupby("state_id").decoded_immediate_utility.max()
        selected["selected_regret"] = (
            selected.state_id.map(best) - selected.decoded_immediate_utility
        )
        regret_threshold = float(selected.selected_regret.quantile(0.90))
        selected["top_decile_regret"] = selected.selected_regret.ge(regret_threshold)
        selected["sign_error"] = (
            selected.predicted_normalized_value.ge(0)
            != selected.decoded_immediate_utility.ge(0)
        )
        selected["neural_fallback_disagreement"] = (
            selected.target_set_id != selected.fallback_target_set_id
        )
        selected["absolute_scale_residual_z_at_least_2"] = selected.residual_z.abs().ge(2)
        selected["candidate_id"] = f"{candidate[0]}__{candidate[1]}"
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def low_support_states(
    predictions: pd.DataFrame, constants: dict
) -> set[str]:
    first = predictions.drop_duplicates("state_id")
    flag = np.zeros(len(first), dtype=bool)
    for name, spec in constants["context_normalization"].items():
        values = first[f"raw_context__{name}"].to_numpy(dtype=float)
        flag |= values < float(spec["support_lower"])
        flag |= values > float(spec["support_upper"])
    return set(first.loc[flag, "state_id"])


def select_states(
    predictions: pd.DataFrame, diagnostics: pd.DataFrame, constants: dict
) -> pd.DataFrame:
    flags = [
        "top_decile_regret",
        "sign_error",
        "neural_fallback_disagreement",
        "absolute_scale_residual_z_at_least_2",
    ]
    aggregated = diagnostics.groupby("state_id", as_index=False).agg(
        instance_id=("instance_id", "first"),
        scale=("scale", "first"),
        CF_level=("CF_level", "first"),
        search_stage=("search_stage", "first"),
        regret=("selected_regret", "max"),
        **{name: (name, "max") for name in flags},
    )
    low_support = low_support_states(predictions, constants)
    aggregated["low_support"] = aggregated.state_id.isin(low_support)
    all_flags = flags + ["low_support"]
    aggregated["priority_count"] = aggregated[all_flags].sum(axis=1).astype(int)
    eligible = aggregated[aggregated.priority_count.gt(0)].copy()
    return (
        eligible.sort_values(
            ["instance_id", "priority_count", "regret", "state_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("instance_id", sort=True)
        .head(20)
        .reset_index(drop=True)
    )


def main() -> None:
    protocol = load_json(PROTOCOL_PATH)
    constants = load_json(CONSTANTS_PATH)
    if not all([
        protocol.get("status") == "FROZEN_BEFORE_FIRST_MODEL_OPTIMIZER_STEP_AND_R10_R11_ACCESS",
        protocol.get("r10_accessed") is False,
        protocol.get("r11_accessed") is False,
        protocol["hard_state_aggregation"]["rounds"] == 1,
    ]):
        raise RuntimeError("hard-state protocol is not frozen and leakage-safe")
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    if not all([
        len(predictions) == 77_760,
        predictions.state_id.nunique() == 1_620,
        predictions.instance_id.nunique() == 18,
        set(predictions.model_family) == {"U1", "U2"},
        set(predictions.data_variant) == {"R09_ONLY", "MIXED_OLD_NEW"},
        set(predictions.training_seed) == {686101, 686102, 686103},
    ]):
        raise RuntimeError("incomplete immediate OOF prediction matrix")
    diagnostics = selected_diagnostics(predictions)
    selected = select_states(predictions, diagnostics, constants)
    source = predictions.drop_duplicates(["state_id", "target_set_id"]).copy()
    selected_columns = [
        "state_id", "priority_count", "regret", "top_decile_regret",
        "sign_error", "neural_fallback_disagreement",
        "absolute_scale_residual_z_at_least_2", "low_support",
    ]
    hard_actions = source.merge(selected[selected_columns], on="state_id", how="inner")
    checks = {
        "r09_only": set(hard_actions.split) == {"R09"},
        "maximum_360_states": selected.state_id.nunique() <= 360,
        "maximum_20_states_per_instance": selected.groupby("instance_id").size().max() <= 20,
        "all_selected_have_priority": bool(selected.priority_count.gt(0).all()),
        "four_unique_actions_per_state": bool(
            hard_actions.groupby("state_id").target_set_id.nunique().eq(4).all()
        ),
        "all_actions_feasible": bool(hard_actions.candidate_feasible.all()),
        "one_round_only": True,
        "r10_not_accessed": True,
        "r11_not_accessed": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(selected, OUT / "hard_state_manifest.csv")
    atomic_parquet(diagnostics, OUT / "candidate_ensemble_diagnostics.parquet")
    atomic_parquet(hard_actions, OUT / "hard_state_actions.parquet")
    integrity = {
        "schema": "phase6i-mr-hard-state-round1-integrity-v1.2",
        "status": "PASS" if all(checks.values()) else "FAILED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "round": 1,
        "selected_states": selected.state_id.nunique(),
        "selected_actions": len(hard_actions),
        "instances": selected.instance_id.nunique(),
        "priority_counts": selected.priority_count.value_counts().sort_index().to_dict(),
        "flag_counts": {
            name: int(selected[name].sum())
            for name in [
                "top_decile_regret", "sign_error", "neural_fallback_disagreement",
                "absolute_scale_residual_z_at_least_2", "low_support",
            ]
        },
        "checks": checks,
        "source_hashes": {
            "training_protocol": digest(PROTOCOL_PATH),
            "immediate_oof_predictions": digest(PREDICTIONS_PATH),
            "training_constants": digest(CONSTANTS_PATH),
        },
        "output_hashes": {
            "hard_state_manifest": digest(OUT / "hard_state_manifest.csv"),
            "candidate_ensemble_diagnostics": digest(OUT / "candidate_ensemble_diagnostics.parquet"),
            "hard_state_actions": digest(OUT / "hard_state_actions.parquet"),
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(integrity, OUT / "hard_state_integrity.json")
    print(integrity)
    if integrity["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
