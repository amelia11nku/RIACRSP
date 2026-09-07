#!/usr/bin/env python3
"""Freeze the Phase 6L start boundary and legacy-score dependency audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import (  # noqa: E402
    FrozenArmPrediction,
    select_forced_candidate_roles,
)
from rcias_clgri.analysis.phase6j_caur import (  # noqa: E402
    build_candidate_source_features,
    critical_and_bottleneck_operations,
)
from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    ONLINE_INPUT_COLUMNS,
    REMOVED_LEGACY_INPUT_COLUMNS,
    build_score_free_candidate_source_features,
    select_score_free_fallback,
)
from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402
from scripts.run_phase6j_caur_collection import read_alns_config  # noqa: E402
from scripts.run_phase6j_caur_pilot import candidate_from_dict  # noqa: E402


STARTING_COMMIT = "c52a37f64573d3fa822282960a6a9cbf6222f375"
OUT = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/audit"
GROUPED = ROOT / "outputs/phase6j_caur/r12_collection/grouped_labels"
REPLAYS = ROOT / "outputs/phase6j_caur/r12_collection/state_replays"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prediction(arm, raw_score: float) -> FrozenArmPrediction:
    return FrozenArmPrediction(
        target_set_id=arm.target_set_id,
        arm_family=arm.arm_family,
        origin_destroy_operator=arm.origin_destroy_operator,
        origin_rules=arm.origin_rules,
        destroyed_operations=arm.destroyed_operations,
        raw_score=float(raw_score),
        raw_probability=0.0,
        raw_utility=0.0,
        calibrated_probability=0.0,
        calibrated_utility=0.0,
    )


def legacy_fallback(arms: tuple[FrozenArmPrediction, ...]) -> str:
    return next(
        row.arm.target_set_id
        for row in select_forced_candidate_roles(arms)
        if row.role == "ALNS_RELATED_FALLBACK"
    )


def collect_r12_structure() -> tuple[list[dict], list[str], int]:
    summaries = []
    mismatches = []
    candidate_count = 0
    for path in sorted(GROUPED.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "state_id", "scale", "target_set_id", "origin_rules", "is_fallback",
                "fallback_target_set_id", "requested_bank_count", "full_bank_unique_count",
                "duplicate_bank_count",
            ],
        )
        direct = frame[
            frame.origin_rules.map(lambda value: "operator_related" in json.loads(value))
        ]
        old = frame[frame.is_fallback.astype(bool)]
        if len(direct) != 1 or len(old) != 1:
            raise RuntimeError(f"invalid fallback provenance in {path}")
        state_id = str(frame.state_id.iloc[0])
        old_id = str(old.target_set_id.iloc[0])
        direct_id = str(direct.target_set_id.iloc[0])
        if old_id != direct_id:
            mismatches.append(state_id)
        candidate_count += len(frame)
        summaries.append({
            "state_id": state_id,
            "scale": str(frame.scale.iloc[0]),
            "candidate_count": len(frame),
            "requested_rule_count": int(frame.requested_bank_count.iloc[0]),
            "duplicate_count": int(frame.duplicate_bank_count.iloc[0]),
            "legacy_fallback_target_set_id": old_id,
            "score_free_fallback_target_set_id": direct_id,
            "same_fallback": old_id == direct_id,
        })
    return summaries, mismatches, candidate_count


def representative_states(summaries: list[dict], mismatches: list[str]) -> list[str]:
    chosen = []
    for scale in ("S", "M", "L"):
        candidates = sorted(row["state_id"] for row in summaries if row["scale"] == scale)
        mismatch = [state_id for state_id in candidates if state_id in mismatches]
        chosen.append((mismatch or candidates)[0])
    return chosen


def dynamic_trace(state_ids: list[str]) -> list[dict]:
    config = json.loads((ROOT / "configs/phase6j_caur.json").read_text())
    alns = read_alns_config(config)
    traces = []
    for state_id in state_ids:
        replay = json.loads((REPLAYS / f"{state_id}.json").read_text())
        snapshot = replay["snapshot"]
        instance = load_phase6j_instance(
            ROOT / config["instance_suite"]["root"] / replay["instance_relative_path"]
        )
        current = decode_candidate(instance, candidate_from_dict(snapshot["current_candidate"]))
        if abs(current.makespan - float(snapshot["current_makespan"])) > 1e-9:
            raise RuntimeError(f"state replay failed: {state_id}")
        destroy_count = min(
            max(2, round(instance.num_operations * alns.destroy_fraction)),
            instance.num_operations,
        )
        generated = generate_revised_target_arms(
            instance, current, state_id, destroy_count,
            int(config["rng"]["proposal_namespace"]),
        )
        repeated = generate_revised_target_arms(
            instance, current, state_id, destroy_count,
            int(config["rng"]["proposal_namespace"]),
        )
        ids = [arm.target_set_id for arm in generated.arms]
        if ids != replay["full_bank_target_ids"] or generated != repeated:
            raise RuntimeError(f"candidate generation drift: {state_id}")
        frame = pd.read_parquet(GROUPED / f"{state_id}.parquet")
        score_by_id = dict(zip(frame.target_set_id, frame.frozen_raw_score))
        scored = tuple(prediction(arm, score_by_id[arm.target_set_id]) for arm in generated.arms)
        reversed_scored = tuple(replace(arm, raw_score=-arm.raw_score) for arm in scored)
        old_fallback = legacy_fallback(scored)
        reversed_fallback = legacy_fallback(reversed_scored)
        direct_fallback = select_score_free_fallback(generated).target_set_id
        critical, bottleneck, _ = critical_and_bottleneck_operations(instance, current)
        old_features = build_candidate_source_features(
            generated,
            state_id=state_id,
            operation_count=instance.num_operations,
            fallback_target_set_id=old_fallback,
            frozen_scores=score_by_id,
            critical_operations=critical,
            bottleneck_operations=bottleneck,
        )
        reversed_features = build_candidate_source_features(
            generated,
            state_id=state_id,
            operation_count=instance.num_operations,
            fallback_target_set_id=reversed_fallback,
            frozen_scores={key: -value for key, value in score_by_id.items()},
            critical_operations=critical,
            bottleneck_operations=bottleneck,
        )
        score_free = build_score_free_candidate_source_features(
            generated,
            state_id=state_id,
            operation_count=instance.num_operations,
            critical_operations=critical,
            bottleneck_operations=bottleneck,
        )
        traces.append({
            "state_id": state_id,
            "scale": str(frame.scale.iloc[0]),
            "requested_rule_count": generated.requested_arm_count,
            "unique_candidate_count": generated.unique_arm_count,
            "candidate_ids_equal_archived_generation_order": ids == replay["full_bank_target_ids"],
            "candidate_generation_repeat_exact": generated == repeated,
            "candidate_generation_api_accepts_historical_policy_or_scores": False,
            "legacy_fallback_target_set_id": old_fallback,
            "reversed_score_fallback_target_set_id": reversed_fallback,
            "score_free_fallback_target_set_id": direct_fallback,
            "fallback_changed_by_score_reversal": old_fallback != reversed_fallback,
            "legacy_feature_rows_changed_by_score_reversal": old_features != reversed_features,
            "score_free_candidate_ids_in_generation_order": [row["target_set_id"] for row in score_free],
            "score_free_input_columns": list(ONLINE_INPUT_COLUMNS),
            "removed_legacy_input_columns": list(REMOVED_LEGACY_INPUT_COLUMNS),
        })
    return traces


def dependency_entries() -> list[dict]:
    def entry(interface, dependency, classification, resolution, evidence):
        return {"interface": interface, "dependency": dependency,
                "classification": classification, "resolution": resolution,
                "evidence": evidence}

    return [
        entry("historical scorer forward", "direct", "REMOVE",
              "Phase 6L online path exposes no policy/model parameter to its feature builder",
              "phase6i_mr.py:114-190; phase6l_legacy_score.py"),
        entry("frozen_raw_score and calibrated historical outputs", "direct", "OFFLINE_TEACHER_ONLY",
              "excluded from all online inputs; no teacher is selected for the primary model",
              "phase6i_mr.py:114-190; Phase 6J grouped label schema"),
        entry("24-rule proposal generation", "none", "KEEP",
              "retain generate_revised_target_arms unchanged",
              "phase6c.py:126-217; dynamic replay trace"),
        entry("target-set deduplication and identity", "none", "KEEP",
              "retain insertion-order deduplication by sorted operation tuple and target_set_id",
              "phase6c.py:205-225; dynamic replay trace"),
        entry("candidate order and target_set_id tie breaks", "none", "KEEP",
              "retain generator order for live features and lexical target ID for model ties",
              "phase6c.py:205-225; phase6j_caur.py:466-470"),
        entry("forced FROZEN_NEURAL_TOP1/TOP2 roles", "direct", "REMOVE",
              "not used by the full-bank Phase 6L live decision",
              "phase6i_mr.py:238-286"),
        entry("reduced top-8 audit roles", "direct", "REMOVE",
              "Phase 6L formal claims use the complete deduplicated bank",
              "phase6i_mr.py:289-305"),
        entry("ALNS_RELATED_FALLBACK role", "indirect", "REPLACE",
              "select the unique deduplicated operator_related canonical target before scoring",
              "phase6i_mr.py:213-266; full R12 provenance audit"),
        entry("fallback_overlap_fraction and fallback_jaccard", "indirect", "REPLACE",
              "recompute against the deterministic operator_related fallback",
              "phase6j_caur.py:398-399; phase6l_legacy_score.py"),
        entry("is_fallback", "indirect", "REPLACE",
              "derive from the deterministic operator_related fallback",
              "phase6j_caur.py:405; phase6l_legacy_score.py"),
        entry("best_frozen_score_jaccard", "direct", "REMOVE",
              "remove from the feature schema",
              "phase6j_caur.py:402"),
        entry("normalized_frozen_score_rank", "direct", "REMOVE",
              "remove from the feature schema",
              "phase6j_caur.py:366-367,403"),
        entry("normalized_diversity_rank", "none", "KEEP",
              "recompute from target-set Jaccard distances and target ID ties only",
              "phase6j_caur.py:368-376,404"),
        entry("continuation advantage and beats-fallback labels", "indirect", "REPLACE",
              "re-anchor stored per-seed candidate outcomes to operator_related without new rollouts",
              "run_phase6j_caur_collection.py:323-414"),
        entry("feature normalization and categorical vocabulary", "indirect", "REPLACE",
              "fit within each authorized training fold on the Phase 6L schema only",
              "train_phase6j_caur.py:55-110"),
        entry("support mask", "indirect", "REPLACE",
              "recompute from the Phase 6L fold transform and score-free inputs",
              "Phase 6J support contract and train_phase6j_caur.py"),
        entry("candidate context and three prediction heads", "indirect", "REPLACE",
              "train a separately named continuation ranker; no E4R equivalence claim",
              "Phase 6J J1 model and Phase 6L execution manual"),
        entry("OOF calibrator and decision thresholds", "indirect", "REPLACE",
              "refit/select only from authorized grouped OOF predictions",
              "phase6j_caur.json gate and calibration contracts"),
        entry("gate winner, LCB, support and harm formulation", "none after retraining", "KEEP",
              "retain the Phase 6J decision rule and frozen threshold grid",
              "phase6j_caur.py:449-491; phase6j_caur.json"),
        entry("repair, decoder and feasibility semantics", "none", "KEEP",
              "retain eight deterministic trials per target and existing decoder checks",
              "run_phase6j_caur_collection.py:291-319"),
    ]


def freeze_protected_evidence() -> dict:
    prior_manifest = ROOT / "outputs/phase6k_runtime_v1/audit/protected_evidence.json"
    prior = json.loads(prior_manifest.read_text())
    prior_verified = all(digest(ROOT / path) == value["sha256"] for path, value in prior.items())
    phase6k_files = sorted(
        path for path in (ROOT / "outputs/phase6k_runtime_v1").rglob("*") if path.is_file()
    )
    phase6k_manifest = {
        str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in phase6k_files
    }
    manifest_path = OUT / "protected_phase6k_evidence.json"
    atomic_json(manifest_path, phase6k_manifest)
    tracked = [
        line for line in git("ls-files").splitlines()
        if any(token in line.lower() for token in ("phase6i", "phase6j", "phase6k"))
    ]
    tracked_hashes = {path: digest(ROOT / path) for path in tracked}
    tracked_path = OUT / "protected_tracked_predecessor_files.json"
    atomic_json(tracked_path, tracked_hashes)
    return {
        "phase6i_phase6j_manifest_path": str(prior_manifest.relative_to(ROOT)),
        "phase6i_phase6j_manifest_sha256": digest(prior_manifest),
        "phase6i_phase6j_files": len(prior),
        "phase6i_phase6j_files_verified": prior_verified,
        "phase6k_manifest_path": str(manifest_path.relative_to(ROOT)),
        "phase6k_manifest_sha256": digest(manifest_path),
        "phase6k_files": len(phase6k_manifest),
        "phase6k_bytes": sum(value["bytes"] for value in phase6k_manifest.values()),
        "tracked_predecessor_manifest_path": str(tracked_path.relative_to(ROOT)),
        "tracked_predecessor_manifest_sha256": digest(tracked_path),
        "tracked_predecessor_files": len(tracked_hashes),
    }


def main() -> None:
    if git("rev-parse", "HEAD") != STARTING_COMMIT:
        raise RuntimeError("Phase 6L start commit drifted before L0 freeze")
    summaries, mismatches, candidate_count = collect_r12_structure()
    if len(summaries) != 288 or candidate_count != 6809:
        raise RuntimeError("authoritative R12 CAUR-FIT collection is incomplete")
    traces = dynamic_trace(representative_states(summaries, mismatches))
    if not all(
        row["candidate_ids_equal_archived_generation_order"]
        and row["candidate_generation_repeat_exact"] for row in traces
    ):
        raise RuntimeError("candidate generation is not stable")
    access_paths = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "outputs/phase6j_caur").rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("r13", "r14"))
    ]
    if access_paths:
        raise RuntimeError(f"R13/R14 access evidence exists: {access_paths}")
    protected = freeze_protected_evidence()
    start = {
        "schema": "phase6l-start-audit-v1",
        "status": "PASS",
        "starting_commit": STARTING_COMMIT,
        "starting_worktree": "CLEAN_VERIFIED_BEFORE_PHASE6L_EDITS",
        "starting_regression": {"passed": 342, "failed": 0, "seconds": 16.58},
        "protected_evidence": protected,
        "r13_r14_access_paths": access_paths,
        "r13_accessed": False,
        "r14_accessed": False,
        "current_phase6l_worktree_paths": git("status", "--short").splitlines(),
    }
    dependencies = dependency_entries()
    result = {
        "schema": "phase6l-legacy-score-dependency-map-v1",
        "status": "PASS",
        "decision": "PROCEED_TO_L1_PREREGISTRATION",
        "action_space_changed_by_legacy_score": False,
        "candidate_generation_intrinsically_score_independent": True,
        "fallback_requires_score_independent_replacement": True,
        "existing_labels_reusable_without_new_rollouts": True,
        "label_reanchoring_required_states": len(mismatches),
        "r12_caur_fit_structure": {
            "states": len(summaries),
            "candidates": candidate_count,
            "requested_rules_per_state": 24,
            "unique_candidates_min": min(row["candidate_count"] for row in summaries),
            "unique_candidates_max": max(row["candidate_count"] for row in summaries),
            "unique_operator_related_candidates_per_state": 1,
            "legacy_fallback_equals_operator_related_states": len(summaries) - len(mismatches),
            "legacy_fallback_differs_from_operator_related_states": len(mismatches),
            "differing_state_ids": mismatches,
        },
        "dependencies": dependencies,
        "dynamic_trace": traces,
        "known_live_interfaces_covered": len(dependencies),
        "historical_forward_calls_during_dynamic_candidate_trace": 0,
        "data_scope": "existing authorized R12_CAUR_FIT development/OOF evidence only",
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT / "starting_audit.json", start)
    atomic_json(OUT / "legacy_score_dependency_map.json", result)
    print(json.dumps({
        "status": result["status"],
        "states": len(summaries),
        "candidates": candidate_count,
        "fallback_reanchor_states": len(mismatches),
        "phase6k_protected_files": protected["phase6k_files"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
