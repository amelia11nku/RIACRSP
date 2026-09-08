#!/usr/bin/env python3
"""Audit and compact the completed Phase 6O targeted relabeling evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase6o_targeted_relabeling as worker  # noqa: E402


OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling"
ADDITIONAL = OUT / "additional_seed_labels.parquet"
MANIFEST = OUT / "completion_shard_manifest.csv"
AUDIT = OUT / "completion_audit.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    config, _, plan, union, _ = worker.validate_boundary()
    progress = worker.load_json(worker.PROGRESS)
    integrity = worker.load_json(OUT / "integrity.json")
    amendment = worker.load_json(worker.RECOVERY_AMENDMENT)
    diagnosis = worker.load_json(worker.RECOVERY_DIAGNOSIS)
    require(progress["status"] == "COMPLETE", "relabeling is incomplete")
    require(progress["states_complete"] == 864, "state count is incomplete")
    require(progress["additional_rows_complete"] == 14358, "additional row count is incomplete")
    require(integrity["status"] == "PASS", "worker integrity did not pass")
    require(amendment["scientific_calculation_changed"] is False, "recovery changed scientific calculation")

    accepted_commits = {
        amendment["base_implementation_commit"],
        amendment["amended_implementation_commit"],
    }
    accepted_workers = {
        amendment["base_worker_sha256"],
        amendment["amended_code_sha256"][worker.WORKER_KEY],
    }
    raw_paths = sorted(worker.RAW_SHARDS.glob("*.parquet"))
    status_paths = sorted(worker.STATUS_SHARDS.glob("*.json"))
    require(len(raw_paths) == len(status_paths) == 864, "relabel shard count changed")
    statuses = []
    manifest_rows = []
    for raw_path, status_path in zip(raw_paths, status_paths):
        require(raw_path.stem == status_path.stem, "raw/status state identity mismatch")
        status = worker.load_json(status_path)
        require(status["status"] == "COMPLETE", "incomplete state status")
        require(status["implementation_commit"] in accepted_commits, "unknown implementation commit")
        require(status["worker_sha256"] in accepted_workers, "unknown worker hash")
        require(status["raw_sha256"] == digest(raw_path), "state raw hash mismatch")
        require(status["historical_score_calls"] == 0, "historical scorer was called")
        require(status["r13_accessed"] is False and status["r14_accessed"] is False, "holdout was accessed")
        statuses.append(status)
        for path in (raw_path, status_path):
            manifest_rows.append({
                "state_id": raw_path.stem,
                "artifact_kind": "RAW" if path == raw_path else "STATUS",
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            })

    additional = pd.concat(
        [pd.read_parquet(path) for path in raw_paths], ignore_index=True
    ).sort_values(
        ["state_id", "target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    require(len(additional) == int(plan["actual_additional_continuation_rows"]), "additional labels changed")
    require(additional.state_id.nunique() == 864, "additional state coverage changed")
    require(set(additional.continuation_seed.unique()) == set(plan["additional_crn_seeds"]), "additional seed set changed")
    require(additional.groupby(["state_id", "target_set_id"]).continuation_seed.nunique().eq(3).all(), "targeted seed coverage changed")
    require(np.isfinite(additional.continuation_advantage).all(), "non-finite additional label")
    actual_keys = set(zip(additional.state_id.astype(str), additional.target_set_id.astype(str)))
    frozen_keys = set(zip(union.state_id.astype(str), union.target_set_id.astype(str)))
    require(actual_keys == frozen_keys, "additional labels escaped frozen union")
    require(additional.groupby("state_id").is_fallback.sum().eq(3).all(), "additional fallback rows changed")
    require(additional.loc[additional.is_fallback, "continuation_advantage"].abs().max() == 0, "fallback advantage is nonzero")
    atomic_parquet(ADDITIONAL, additional)

    combined = pd.read_parquet(OUT / "combined_seed_labels.parquet")
    training = pd.read_parquet(OUT / "training_grouped_labels.parquet")
    predecessor = pd.read_parquet(worker.PHASE6N_GROUPED)
    require(len(combined) == 55240, "combined seed-label count changed")
    require(len(training) == len(predecessor) == int(config["locked_inputs"]["candidates"]), "full-bank candidate count changed")
    keys = ["state_id", "target_set_id"]
    counts = combined.groupby(keys).continuation_seed.nunique()
    require(int(counts.eq(5).sum()) == len(union), "five-seed targeted coverage changed")
    require(int(counts.eq(2).sum()) == len(training) - len(union), "two-seed non-targeted coverage changed")
    require(training.phase6o_targeted_relabel.astype(bool).sum() == len(union), "training targeted marker changed")
    require(
        list(zip(training.state_id.astype(str), training.target_set_id.astype(str)))
        == list(zip(predecessor.state_id.astype(str), predecessor.target_set_id.astype(str))),
        "training candidate identity/order changed",
    )
    recomputed = combined.groupby(keys, sort=True).agg(
        mean=("continuation_advantage", "mean"),
        std=("continuation_advantage", lambda x: float(np.std(x, ddof=0))),
        beats=("continuation_advantage", lambda x: float(np.mean(np.asarray(x) > 0))),
    ).reset_index()
    checked = training.merge(recomputed, on=keys, validate="one_to_one")
    require(np.array_equal(checked.continuation_advantage_mean.to_numpy(), checked["mean"].to_numpy()), "training mean changed")
    require(np.array_equal(checked.continuation_advantage_std.to_numpy(), checked["std"].to_numpy()), "training std changed")
    require(np.array_equal(checked.beats_fallback.to_numpy(), checked.beats.to_numpy()), "training beats label changed")

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["state_id", "artifact_kind"], kind="stable"
    ).reset_index(drop=True)
    manifest.to_csv(MANIFEST, index=False)
    reanchored = [row for row in statuses if row.get("fallback_reanchored_from_historical_replay")]
    require(len(reanchored) == len(diagnosis["known_mismatches"]) == 10, "fallback reanchoring audit changed")
    payload = {
        "schema": "phase6o-targeted-relabel-completion-audit-v1",
        "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "states": 864,
        "instances": int(training.instance_id.nunique()),
        "full_bank_candidates": len(training),
        "targeted_candidates": len(union),
        "additional_seed_rows": len(additional),
        "combined_seed_rows": len(combined),
        "targeted_five_seed_candidates": int(counts.eq(5).sum()),
        "non_targeted_two_seed_candidates": int(counts.eq(2).sum()),
        "reanchored_original_states": len(reanchored),
        "base_implementation_states": sum(row["implementation_commit"] == amendment["base_implementation_commit"] for row in statuses),
        "amended_implementation_states": sum(row["implementation_commit"] == amendment["amended_implementation_commit"] for row in statuses),
        "candidate_identity_and_order_preserved": True,
        "candidate_feasibility": bool(training.candidate_feasible.astype(bool).all()),
        "canonical_fallback_per_state": bool(training.groupby("state_id").is_fallback.sum().eq(1).all()),
        "all_shard_hashes_valid": True,
        "scientific_calculation_changed_by_recovery": False,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "artifacts": {
            str(ADDITIONAL.relative_to(ROOT)): digest(ADDITIONAL),
            str((OUT / "combined_seed_labels.parquet").relative_to(ROOT)): digest(OUT / "combined_seed_labels.parquet"),
            str((OUT / "training_grouped_labels.parquet").relative_to(ROOT)): digest(OUT / "training_grouped_labels.parquet"),
            str(MANIFEST.relative_to(ROOT)): digest(MANIFEST),
        },
    }
    atomic_json(AUDIT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
