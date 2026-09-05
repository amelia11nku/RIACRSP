#!/usr/bin/env python3
"""Reproduce J3 evidence and certify R12 data gates for all activated families."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase6j_caur_j3 as j3  # noqa: E402
from scripts.audit_phase6j_caur_training import audit_regular_training  # noqa: E402

r = j3.regular
OUT = ROOT / "outputs/phase6j_caur/r12_acceptance"


def gate_mask(selected, gate, harm_floor):
    return (
        ~selected.is_fallback.astype(bool)
        & selected.calibrated_probability.ge(gate["p_min"])
        & (selected.ensemble_advantage_mean - gate["lcb_lambda"] * selected.ensemble_advantage_std).gt(gate["delta_min"])
        & selected.supported.astype(bool)
        & selected.ensemble_immediate_utility.ge(harm_floor)
    )


def data_gate_checks(summary, config, *, origin_diversity):
    metrics = summary["metrics"]
    essential = config["r12_acceptance"]["essential"]
    gate = summary["selected_gate"]
    return {
        "positive_overall_spearman": metrics["overall_spearman"] > essential["overall_spearman_gt"],
        "nonnegative_scale_spearman": min(metrics["mean_spearman_by_scale"].values()) >= essential["minimum_scale_mean_spearman"],
        "positive_raw_selected_lift": metrics["selected_lift"] > essential["selected_lift_gt"],
        "positive_raw_selected_lift_lcb": metrics["selected_lift_lcb"] > essential["selected_bootstrap_lcb_gt"],
        "selected_winner_ece": summary["calibration"]["metrics"][0]["expected_calibration_error"] <= essential["selected_winner_ece_max"],
        "retained_gate_with_scale_coverage": gate is not None and bool(gate["retained"]) and bool(gate["coverage_pass"]),
        "positive_gated_lift_lcb": gate is not None and gate["selected_lift"] > 0 and gate["selected_lift_lcb"] > 0,
        "nonnegative_gated_scale_lift": gate is not None and all(gate[f"scale_{s}_lift"] >= 0 for s in ("S", "M", "L")),
        "observed_origin_diversity": origin_diversity,
    }


def audit_j3():
    protocol = j3.validate_protocol()
    sha = r.digest(j3.PROTOCOL_PATH)
    worker = r.load_json(j3.OUT / "worker_status.json")
    progress = r.load_json(j3.OUT / "progress.json")
    if not (worker["exit_code"] == 0 and worker["status"] == progress["status"] == "COMPLETE_J3_OOF"
            and progress["completed_runs"] == progress["expected_runs"] == 9):
        raise RuntimeError("J3 completion boundary failed")
    done = j3.completed_runs(sha)
    if len(done) != 9:
        raise RuntimeError("J3 requires nine valid checkpoint/prediction/record triples")
    source = pd.read_parquet(r.SOURCE_PATH)
    source["oof_fold"] = [r.grouped_oof_fold(str(s), str(c)) for s, c in zip(source.scale, source.CF_level)]
    lookup = source.set_index(["state_id", "target_set_id"]).sort_index()
    artifacts, frames, run_rows = {}, [], []
    for seed, fold, paths in done:
        frame = pd.read_parquet(paths[1])
        expected = lookup[lookup.oof_fold.eq(fold)]
        actual = frame.set_index(["state_id", "target_set_id"]).sort_index()
        pd.testing.assert_index_equal(actual.index, expected.index)
        for name in expected:
            pd.testing.assert_series_equal(actual[name], expected[name])
        if not (frame.held_fold.eq(fold).all() and frame.training_seed.eq(seed).all()
                and frame.model_family.eq(j3.FAMILY).all() and frame.state_id.nunique() == 96):
            raise RuntimeError("J3 run identity failed")
        if not np.isfinite(frame.filter(regex="^predicted_").to_numpy(float)).all():
            raise RuntimeError("non-finite J3 predictions")
        checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
        transform = r.fit_feature_transform(source[source.oof_fold.ne(fold)])
        if checkpoint["feature_transform"] != transform.to_dict():
            raise RuntimeError("J3 normalization did not come from outer training folds")
        model = j3.initialize(seed, transform, protocol, torch.device("cpu"))
        j3.restore_trainable(model, checkpoint["trainable_model_state"])
        for name, value in {"training_seed": seed, "held_fold": fold, "model_family": j3.FAMILY,
                            "training_protocol_sha256": sha,
                            "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"]}.items():
            if checkpoint[name] != value:
                raise RuntimeError(f"J3 checkpoint identity mismatch: {name}")
        record = r.load_json(paths[2])
        h = record["history"]
        if {h["inner_training_fold"], h["inner_validation_fold"], fold} != {0, 1, 2}:
            raise RuntimeError("J3 nested folds overlap")
        if record["best_epoch"] != len(h["outer_final_fit"]):
            raise RuntimeError("J3 selected epoch was not used for the outer fit")
        run_rows.append({"seed": seed, "fold": fold, "best_epoch": record["best_epoch"],
                         "runtime_seconds": record["runtime_seconds"]})
        for path in paths:
            artifacts[str(path.relative_to(ROOT))] = r.digest(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["training_seed", "state_id", "target_set_id"]
    pd.testing.assert_frame_equal(combined.sort_values(keys).reset_index(drop=True),
                                  pd.read_parquet(j3.OUT / "oof_predictions.parquet").sort_values(keys).reset_index(drop=True))
    result = r.ensemble_family(combined, j3.FAMILY, protocol)
    summary = r.load_json(j3.OUT / "oof_summary.json")
    for field in ("metrics", "calibration", "selected_gate"):
        if summary[field] != result[field]:
            raise RuntimeError(f"J3 summary cannot be reproduced: {field}")
    for name, field in (("ensemble_oof", "ensemble"), ("state_metrics", "states"),
                        ("selected_winners", "selected_winners")):
        pd.testing.assert_frame_equal(pd.read_parquet(j3.OUT / f"{name}.parquet"), result[field])
    for name in ("oof_predictions.parquet", "oof_summary.json", "ensemble_oof.parquet",
                 "state_metrics.parquet", "selected_winners.parquet", "calibration_metrics.csv",
                 "gate_grid.csv", "three_seed_stability.csv", "progress.json", "worker_status.json"):
        path = j3.OUT / name
        artifacts[str(path.relative_to(ROOT))] = r.digest(path)
    return {"schema": "phase6j-caur-j3-completion-audit-v1", "status": "PASS", "runs": 9,
            "states": 288, "prediction_rows": len(combined), "training_protocol_sha256": sha,
            "checks": {"checkpoint_hashes_and_parameters": True, "predictions_and_labels_exact": True,
                       "normalization_and_nested_folds": True, "aggregate_and_summaries_reproduced": True,
                       "worker_complete_exit_zero": True, "r13_r14_locked": True},
            "artifact_sha256": artifacts, "run_details": run_rows, "r13_accessed": False, "r14_accessed": False}


def main():
    regular_audit = audit_regular_training()
    if regular_audit != r.load_json(j3.REGULAR_AUDIT):
        raise RuntimeError("previous regular-family completion audit changed")
    j3_audit = audit_j3()
    config = r.load_json(r.CONFIG_PATH)
    families, origin_rows, intervention_rows = {}, [], []
    for family in (*r.FAMILIES, j3.FAMILY):
        root = j3.OUT if family == j3.FAMILY else r.OUT
        summary = r.load_json(root / ("oof_summary.json" if family == j3.FAMILY else f"{family}_oof_summary.json"))
        selected = pd.read_parquet(root / ("selected_winners.parquet" if family == j3.FAMILY else f"{family}_selected_winners.parquet"))
        gate = summary["selected_gate"]
        selected["intervened"] = False if gate is None else gate_mask(selected, gate, config["gate"]["immediate_harm_floor"])
        selected["gate_lift"] = selected.continuation_advantage_mean.where(selected.intervened, 0.0)
        diverse = True
        for scale, group in selected.groupby("scale", sort=True):
            for scope, values in (("argmax", group), ("intervened", group[group.intervened])):
                counts = values.origin_family.value_counts()
                if scope == "argmax" and len(counts) <= 1:
                    diverse = False
                if scope == "intervened" and len(values) and len(counts) <= 1:
                    diverse = False
                for origin, count in counts.items():
                    origin_rows.append({"family": family, "scale": scale, "scope": scope,
                                        "origin": origin, "count": int(count), "share": float(count / len(values))})
        for dimension in ("scale", "CF_level", "search_stage"):
            for value, group in selected.groupby(dimension, sort=True):
                intervention_rows.append({"family": family, "dimension": dimension, "group": value,
                                          "states": len(group), "has_retained_gate": gate is not None,
                                          "interventions": int(group.intervened.sum()),
                                          "gated_mean_lift": float(group.gate_lift.mean()) if gate is not None else None,
                                          "raw_selected_mean_lift": float(group.continuation_advantage_mean.mean())})
        checks = data_gate_checks(summary, config, origin_diversity=diverse)
        families[family] = {
            "data_gate_pass": all(checks.values()), "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "metrics": summary["metrics"], "selected_gate": gate,
            "selected_winner_ece": summary["calibration"]["metrics"][0]["expected_calibration_error"],
            "r13_eligible": False,
        }
    payload = {"schema": "phase6j-caur-r12-data-readiness-v1", "status": "DATA_AUDIT_COMPLETE",
               "regular_completion_audit_sha256": r.digest(j3.REGULAR_AUDIT),
               "families": families, "data_eligible_families": [f for f, x in families.items() if x["data_gate_pass"]],
               "origin_audit_scope": "observed single-origin degeneration by scale; shares reported without posthoc entropy thresholds",
               "remaining": ["full-R12 three-seed deployable fit", "inference parity", "latency", "R13 artifact bundle"],
               "r13_accessed": False, "r14_accessed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    for path, value in ((OUT / "j3_completion_integrity_audit.json", j3_audit),
                        (OUT / "data_readiness.json", payload)):
        if path.exists() and r.load_json(path) != value:
            raise RuntimeError(f"existing readiness evidence changed: {path}")
        r.atomic_json(value, path)
    r.atomic_csv(pd.DataFrame(origin_rows), OUT / "origin_selection_by_scale.csv")
    r.atomic_csv(pd.DataFrame(intervention_rows), OUT / "intervention_lift_by_group.csv")
    print(json.dumps({"status": payload["status"], "j3_integrity": j3_audit["status"],
                      "data_eligible_families": payload["data_eligible_families"],
                      "family_failed_checks": {f: x["failed_checks"] for f, x in families.items()}}, indent=2))


if __name__ == "__main__":
    main()
