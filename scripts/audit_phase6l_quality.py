#!/usr/bin/env python3
"""Audit the frozen Phase 6L L4 development-quality boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_phase6l_score_free as primary  # noqa: E402


NAMESPACE = ROOT / "outputs/phase6l_legacy_score_decoupling_v1"
TRAINING = NAMESPACE / "training"
ABLATION = NAMESPACE / "ablation/no_fallback_context"
QUALITY = NAMESPACE / "quality"
FAMILY = primary.FAMILY
J1_FAMILY = "J1_CONT_FROZEN"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace frozen evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace frozen evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def verify_manifest(path: Path, *, records: bool) -> dict:
    manifest = json.loads(path.read_text())
    total_bytes = 0
    for relative, expected in manifest.items():
        target = ROOT / relative
        require(target.is_file(), f"protected file missing: {relative}")
        expected_hash = expected["sha256"] if records else expected
        require(digest(target) == expected_hash, f"protected file changed: {relative}")
        if records:
            require(target.stat().st_size == expected["bytes"], f"protected size changed: {relative}")
            total_bytes += expected["bytes"]
    return {"files": len(manifest), "bytes": total_bytes, "status": "PASS"}


def verify_protected_evidence() -> dict:
    start = json.loads((NAMESPACE / "audit/starting_audit.json").read_text())
    boundary = start["protected_evidence"]
    paths = {
        "phase6i_phase6j": ROOT / boundary["phase6i_phase6j_manifest_path"],
        "phase6k": ROOT / boundary["phase6k_manifest_path"],
        "tracked_predecessor": ROOT / boundary["tracked_predecessor_manifest_path"],
    }
    require(digest(paths["phase6i_phase6j"]) == boundary["phase6i_phase6j_manifest_sha256"],
            "Phase 6I/6J protection manifest changed")
    require(digest(paths["phase6k"]) == boundary["phase6k_manifest_sha256"],
            "Phase 6K protection manifest changed")
    require(digest(paths["tracked_predecessor"]) == boundary["tracked_predecessor_manifest_sha256"],
            "tracked predecessor manifest changed")
    return {
        "phase6i_phase6j": verify_manifest(paths["phase6i_phase6j"], records=True),
        "phase6k": verify_manifest(paths["phase6k"], records=True),
        "tracked_predecessor": verify_manifest(paths["tracked_predecessor"], records=False),
        "manifest_hashes_match_l0": True,
    }


def metrics(states: pd.DataFrame, *, bootstrap: dict) -> dict:
    lower, upper = primary.base.grouped_bootstrap_interval(
        states, "selected_lift", seed=int(bootstrap["seed"]),
        resamples=int(bootstrap["resamples"]),
    )
    return {
        "state_count": len(states),
        "overall_spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower,
        "selected_lift_ucb": upper,
        "selection_regret": float(states.selection_regret.mean()),
        "mean_spearman_by_scale": {
            scale: float(states.loc[states.scale.eq(scale), "spearman"].mean())
            for scale in ("S", "M", "L")
        },
    }


def audit_primary(protocol: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completion = json.loads((TRAINING / "completion_integrity_audit.json").read_text())
    require(completion["status"] == "PASS" and all(completion["checks"].values()),
            "L3 completion audit failed")
    predictions = pd.read_parquet(TRAINING / "oof_predictions.parquet")
    reproduced = primary.base.ensemble_family(predictions, FAMILY, protocol)
    saved_summary = json.loads((TRAINING / "oof_summary.json").read_text())
    for field in ("metrics", "calibration", "selected_gate"):
        require(reproduced[field] == saved_summary[field], f"primary {field} changed")
    for filename, field in (
        ("ensemble_oof.parquet", "ensemble"),
        ("state_metrics.parquet", "states"),
        ("selected_winners.parquet", "selected_winners"),
    ):
        pd.testing.assert_frame_equal(pd.read_parquet(TRAINING / filename), reproduced[field])
    return saved_summary, predictions, reproduced["ensemble"], reproduced["selected_winners"]


def audit_candidate_bank(ensemble: pd.DataFrame) -> dict:
    groups = ensemble.groupby("state_id", sort=True)
    counts = groups.size()
    checks = {
        "states_288": ensemble.state_id.nunique() == 288,
        "candidates_6809": len(ensemble) == 6809,
        "requested_rules_24_every_state": ensemble.requested_bank_count.eq(24).all(),
        "unique_count_matches_rows": groups.full_bank_unique_count.first().eq(counts).all(),
        "one_score_free_fallback_per_state": groups.is_fallback.sum().eq(1).all(),
        "candidate_identity_unique": not ensemble.duplicated(["state_id", "target_set_id"]).any(),
        "all_candidate_repairs_feasible": ensemble.candidate_feasible.astype(bool).all(),
    }
    require(all(bool(value) for value in checks.values()), f"candidate-bank audit failed: {checks}")
    return {
        "status": "PASS",
        "checks": {key: bool(value) for key, value in checks.items()},
        "states": 288,
        "candidates": 6809,
        "requested_rules_per_state": 24,
        "unique_candidates_min": int(counts.min()),
        "unique_candidates_max": int(counts.max()),
    }


def stratified_tables(states: pd.DataFrame, ensemble: pd.DataFrame,
                      selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    support_rows = []
    for dimension in ("scale", "CF_level", "search_stage"):
        for value, group in states.groupby(dimension, sort=True):
            metric_rows.append({
                "dimension": dimension, "group": str(value), "states": len(group),
                "spearman": float(group.spearman.mean()),
                "pairwise_accuracy": float(group.pairwise_accuracy.mean()),
                "ndcg_at_1": float(group.ndcg_at_1.mean()),
                "selected_lift": float(group.selected_lift.mean()),
                "selection_regret": float(group.selection_regret.mean()),
            })
        for value, group in ensemble.groupby(dimension, sort=True):
            winners = selected[selected[dimension].eq(value)]
            support_rows.append({
                "dimension": dimension, "group": str(value),
                "candidates": len(group), "candidate_support_rate": float(group.supported.mean()),
                "selected_winners": len(winners),
                "selected_winner_support_rate": float(winners.supported.mean()),
            })
    origin_rows = []
    for scale, group in selected.groupby("scale", sort=True):
        for origin, count in group.origin_family.value_counts().sort_index().items():
            origin_rows.append({
                "scale": scale, "scope": "neural_argmax",
                "origin_family": origin, "count": int(count),
                "share": float(count / len(group)),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(support_rows), pd.DataFrame(origin_rows)


def audit_ablation(protocol: dict, predictions: pd.DataFrame) -> dict:
    frozen = json.loads((ABLATION / "ablation_protocol.json").read_text())
    require(frozen["status"] == "FROZEN_BEFORE_ABLATION_OPTIMIZER_STEP",
            "ablation protocol was not frozen")
    require(frozen["eligible_for_selection_or_promotion"] is False,
            "ablation cannot be selectable")
    for relative, expected in frozen["source_hashes"].items():
        require(digest(ROOT / relative) == expected, f"ablation source changed: {relative}")
    frames = []
    for fold in range(3):
        stem = ABLATION / "oof" / f"fold_{fold}"
        checkpoint, prediction, record_path = (
            stem.with_suffix(".pt"), stem.with_suffix(".parquet"), stem.with_suffix(".json")
        )
        record = json.loads(record_path.read_text())
        require(record["status"] == "COMPLETE" and record["held_fold"] == fold,
                f"ablation fold {fold} incomplete")
        require(record["checkpoint_sha256"] == digest(checkpoint),
                f"ablation checkpoint hash failed: fold {fold}")
        require(record["predictions_sha256"] == digest(prediction),
                f"ablation prediction hash failed: fold {fold}")
        frame = pd.read_parquet(prediction)
        require(frame.held_fold.eq(fold).all() and frame.training_seed.eq(706101).all(),
                f"ablation identity failed: fold {fold}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    saved = pd.read_parquet(ABLATION / "oof_predictions.parquet")
    key = ["state_id", "target_set_id"]
    pd.testing.assert_frame_equal(
        combined.sort_values(key).reset_index(drop=True),
        saved.sort_values(key).reset_index(drop=True),
    )
    ablation_states = primary.base.ranking_state_metrics(
        combined, "predicted_continuation_advantage"
    )
    pd.testing.assert_frame_equal(
        ablation_states, pd.read_parquet(ABLATION / "state_metrics.parquet")
    )
    ablation_metrics = metrics(ablation_states, bootstrap=protocol["bootstrap"])
    result = json.loads((ABLATION / "result.json").read_text())
    for key_name, value in ablation_metrics.items():
        if key_name == "mean_spearman_by_scale":
            require(value == result["metrics"][key_name], "ablation scale metrics changed")
        else:
            require(value == result["metrics"][key_name], f"ablation metric changed: {key_name}")
    primary_seed = predictions[predictions.training_seed.eq(706101)]
    primary_states = primary.base.ranking_state_metrics(
        primary_seed, "predicted_continuation_advantage"
    )
    primary_metrics = metrics(primary_states, bootstrap=protocol["bootstrap"])
    delta = {
        name: float(ablation_metrics[name] - primary_metrics[name])
        for name in (
            "overall_spearman", "pairwise_accuracy", "ndcg_at_1",
            "selected_lift", "selected_lift_lcb", "selection_regret",
        )
    }
    return {
        "status": "PASS_NONSELECTABLE_AUDIT",
        "comparison_scope": "matching seed 706101 outer OOF predictions",
        "removed_online_inputs": frozen["removed_online_inputs"],
        "primary_seed_706101": primary_metrics,
        "no_fallback_context_seed_706101": ablation_metrics,
        "ablation_minus_primary": delta,
        "eligible_for_selection_or_promotion": False,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def j1_comparison(protocol: dict, primary_summary: dict) -> dict:
    predecessor_path = ROOT / "outputs/phase6j_caur/training/J1_CONT_FROZEN_oof_summary.json"
    predecessor = json.loads(predecessor_path.read_text())
    direct_metrics = {}
    for name in (
        "overall_spearman", "pairwise_accuracy", "ndcg_at_1",
        "selected_lift", "selected_lift_lcb", "selection_regret",
    ):
        direct_metrics[name] = {
            "phase6j_j1": predecessor["metrics"][name],
            "phase6l_score_free": primary_summary["metrics"][name],
            "phase6l_minus_phase6j": (
                primary_summary["metrics"][name] - predecessor["metrics"][name]
            ),
        }
    old_predictions = pd.read_parquet(
        ROOT / "outputs/phase6j_caur/training/oof_predictions.parquet"
    )
    old_predictions = old_predictions[old_predictions.model_family.eq(J1_FAMILY)].copy()
    new_labels = pd.read_parquet(primary.SOURCE)
    keys = ["state_id", "target_set_id"]
    label_columns = [
        "continuation_advantage_mean", "continuation_advantage_std", "beats_fallback",
        "continuation_best_makespan", "fallback_continuation_best_makespan",
        "is_fallback", "fallback_target_set_id",
    ]
    aligned = old_predictions.drop(columns=label_columns).merge(
        new_labels[[*keys, *label_columns]], on=keys, how="inner", validate="many_to_one"
    )
    require(len(aligned) == 3 * 6809, "J1 common-target alignment is incomplete")
    diagnostic_protocol = copy.deepcopy(protocol)
    diagnostic_protocol["training"]["seeds"] = sorted(
        int(value) for value in aligned.training_seed.unique()
    )
    common = primary.base.ensemble_family(aligned, J1_FAMILY, diagnostic_protocol)
    return {
        "authoritative_existing_oof": {
            "scope": "Phase 6J original frozen fallback labels and existing OOF predictions",
            "metrics": predecessor["metrics"],
            "selected_winner_ece": predecessor["calibration"]["metrics"][0]["expected_calibration_error"],
            "selected_gate": predecessor["selected_gate"],
            "source_sha256": digest(predecessor_path),
        },
        "direct_reported_metric_comparison": direct_metrics,
        "common_reanchored_target_diagnostic": {
            "scope": "existing Phase 6J J1 OOF predictions re-scored only against Phase 6L canonical-fallback labels",
            "limitation": "no J1 reinference; ten changed fallback states retain their historical J1 input context",
            "metrics": common["metrics"],
            "selected_winner_ece": common["calibration"]["metrics"][0]["expected_calibration_error"],
            "eligible_gate_count": int(common["gate_table"].retained.sum()),
            "selected_gate": common["selected_gate"],
        },
    }


def main() -> None:
    protocol = primary.validate_protocol()
    config = json.loads(primary.CONFIG.read_text())
    protected = verify_protected_evidence()
    primary_summary, predictions, ensemble, selected = audit_primary(protocol)
    states = pd.read_parquet(TRAINING / "state_metrics.parquet")
    candidate_bank = audit_candidate_bank(ensemble)
    stratified, support, origins = stratified_tables(states, ensemble, selected)
    origin_diversity = all(
        group.origin_family.nunique() > 1 for _, group in selected.groupby("scale")
    )
    essential = config["quality_gates"]["essential"]
    raw_checks = {
        "replay_feasibility": candidate_bank["status"] == essential["replay_feasibility"],
        "full_bank_completeness": candidate_bank["status"] == essential["full_bank_completeness"],
        "positive_overall_spearman": primary_summary["metrics"]["overall_spearman"] > essential["overall_spearman_gt"],
        "nonnegative_scale_spearman": min(primary_summary["metrics"]["mean_spearman_by_scale"].values()) >= essential["minimum_scale_mean_spearman"],
        "positive_raw_selected_lift": primary_summary["metrics"]["selected_lift"] > essential["selected_lift_gt"],
        "positive_raw_selected_lift_lcb": primary_summary["metrics"]["selected_lift_lcb"] > essential["selected_bootstrap_lcb_gt"],
        "selected_winner_ece": primary_summary["calibration"]["metrics"][0]["expected_calibration_error"] <= essential["selected_winner_ece_max"],
        "observed_origin_diversity": origin_diversity and not essential["candidate_origin_collapse"],
    }
    gate = primary_summary["selected_gate"]
    gate_checks = {
        "retained_gate_with_scale_coverage": gate is not None and bool(gate["retained"]) and bool(gate["coverage_pass"]),
        "positive_gated_lift_lcb": gate is not None and gate["selected_lift"] > 0 and gate["selected_lift_lcb"] > 0,
        "nonnegative_gated_scale_lift": gate is not None and all(gate[f"scale_{scale}_lift"] >= 0 for scale in ("S", "M", "L")),
    }
    preferred = config["quality_gates"]["preferred"]
    preferred_checks = {
        "overall_spearman": primary_summary["metrics"]["overall_spearman"] >= preferred["overall_spearman_min"],
        "pairwise_accuracy": primary_summary["metrics"]["pairwise_accuracy"] >= preferred["pairwise_accuracy_min"],
        "ndcg_at_1": primary_summary["metrics"]["ndcg_at_1"] >= preferred["ndcg_at_1_min"],
    }
    gate_grid = pd.read_csv(TRAINING / "gate_grid.csv")
    require(len(gate_grid) == 18 and int(gate_grid.retained.sum()) == 0,
            "frozen gate grid changed")
    ablation = audit_ablation(protocol, predictions)
    comparison = j1_comparison(protocol, primary_summary)
    all_checks = {**raw_checks, **gate_checks}
    decision = "PROCEED_TO_L5" if all(all_checks.values()) else "MODEL_REVISION_QUALITY"
    require(decision == "MODEL_REVISION_QUALITY", "terminal L4 finalizer is only valid for failed quality")
    result = {
        "schema": "phase6l-development-quality-audit-v1",
        "status": "QUALITY_AUDIT_COMPLETE",
        "decision": decision,
        "stop_boundary": "L4_BEFORE_DEVELOPMENT_RUNTIME_AND_R12_QUALIFICATION",
        "model_family": FAMILY,
        "metrics": primary_summary["metrics"],
        "calibration": primary_summary["calibration"],
        "selected_gate": gate,
        "eligible_gate_count": primary_summary["eligible_gate_count"],
        "raw_essential_checks": raw_checks,
        "phase6j_intervention_readiness_checks": gate_checks,
        "failed_checks": [name for name, passed in all_checks.items() if not passed],
        "preferred_diagnostics": preferred_checks,
        "candidate_bank": candidate_bank,
        "support": {
            "candidate_support_rate": float(ensemble.supported.mean()),
            "selected_winner_support_rate": float(selected.supported.mean()),
        },
        "action_frequency": {
            "neural_argmax_is_fallback": int(selected.is_fallback.astype(bool).sum()),
            "neural_argmax_nonfallback": int((~selected.is_fallback.astype(bool)).sum()),
            "deployable_interventions": 0,
            "fallback_decisions_without_retained_gate": 288,
        },
        "ablation": ablation,
        "phase6j_j1_comparison": comparison,
        "protected_evidence": protected,
        "runtime_stage": "NOT_RUN_STOPPED_ON_L4_QUALITY",
        "r12_qualification": "NOT_RUN_STOPPED_ON_L4_QUALITY",
        "solver_gate": "NOT_RUN",
        "r13_accessed": False,
        "r14_accessed": False,
        "r13_locked": True,
        "r14_locked": True,
        "artifact_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in (
                primary.CONFIG, primary.PROTOCOL, primary.SOURCE,
                TRAINING / "completion_integrity_audit.json",
                TRAINING / "oof_predictions.parquet",
                TRAINING / "oof_summary.json",
                TRAINING / "gate_grid.csv",
                ABLATION / "ablation_protocol.json",
                ABLATION / "oof_predictions.parquet",
                ABLATION / "result.json",
                ROOT / "outputs/phase6j_caur/training/J1_CONT_FROZEN_oof_summary.json",
            )
        },
        "audit_code_sha256": digest(Path(__file__)),
    }
    QUALITY.mkdir(parents=True, exist_ok=True)
    atomic_csv(stratified, QUALITY / "stratified_metrics.csv")
    atomic_csv(support, QUALITY / "support_by_regime.csv")
    atomic_csv(origins, QUALITY / "origin_selection_by_scale.csv")
    atomic_json(QUALITY / "ablation_comparison.json", ablation)
    atomic_json(QUALITY / "phase6j_j1_comparison.json", comparison)
    atomic_json(QUALITY / "development_quality.json", result)
    print(json.dumps({
        "status": result["status"], "decision": decision,
        "raw_essential_pass": all(raw_checks.values()),
        "intervention_readiness_pass": all(gate_checks.values()),
        "failed_checks": result["failed_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
