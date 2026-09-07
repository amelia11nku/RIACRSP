#!/usr/bin/env python3
"""Verify Phase 6L L3 OOF training completion and artifact hashes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training"
FAMILY = "L1_SCORE_FREE_CONT_FROZEN"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    protocol_path = TRAINING / "training_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    progress_path = TRAINING / "progress.json"
    progress = json.loads(progress_path.read_text())
    launch = json.loads((TRAINING / "launch_record.json").read_text())
    errors = []
    runs = []
    predictions = []
    for seed in protocol["training"]["seeds"]:
        for fold in range(3):
            stem = TRAINING / "oof" / FAMILY / f"seed_{seed}" / f"fold_{fold}"
            checkpoint = stem.with_suffix(".pt")
            prediction = stem.with_suffix(".parquet")
            record_path = stem.with_suffix(".json")
            if not all(path.is_file() for path in (checkpoint, prediction, record_path)):
                errors.append(f"missing:{stem}")
                continue
            record = json.loads(record_path.read_text())
            checks = {
                "status": record.get("status") == "COMPLETE",
                "seed": record.get("training_seed") == seed,
                "fold": record.get("held_fold") == fold,
                "protocol": record.get("training_protocol_sha256") == digest(protocol_path),
                "checkpoint": record.get("checkpoint_sha256") == digest(checkpoint),
                "prediction": record.get("predictions_sha256") == digest(prediction),
                "r13": record.get("r13_accessed") is False,
                "r14": record.get("r14_accessed") is False,
            }
            if not all(checks.values()):
                errors.append({str(stem): checks})
            frame = pd.read_parquet(prediction)
            if set(frame.held_fold) != {fold} or set(frame.training_seed) != {seed}:
                errors.append(f"prediction identity:{stem}")
            predictions.append(frame)
            runs.append({
                "training_seed": seed,
                "held_fold": fold,
                "best_epoch": int(record["best_epoch"]),
                "inner_epochs_run": int(record["inner_epochs_run"]),
                "runtime_seconds": float(record["runtime_seconds"]),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "predictions_sha256": record["predictions_sha256"],
            })
    combined = pd.concat(predictions, ignore_index=True)
    identity_counts = combined.groupby(["state_id", "target_set_id"]).training_seed.nunique()
    summary_path = TRAINING / "oof_summary.json"
    summary = json.loads(summary_path.read_text())
    checks = {
        "no_errors": not errors,
        "progress_complete": progress.get("status") == "COMPLETE",
        "nine_runs": len(runs) == 9 == progress.get("completed_runs") == progress.get("expected_runs"),
        "prediction_rows": len(combined) == 3 * 6809,
        "three_predictions_per_candidate": bool(identity_counts.eq(3).all()),
        "states": combined.state_id.nunique() == 288,
        "candidates_per_seed": len(combined) == 3 * 6809,
        "summary_complete": summary.get("status") == "COMPLETE",
        "summary_actions": summary.get("metrics", {}).get("action_count") == 6809,
        "process_exited": not alive(int(launch["pid"])),
        "r13_locked": not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        "r14_locked": not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 6L L3 integrity failed: checks={checks}, errors={errors}")
    result = {
        "schema": "phase6l-l3-training-completion-audit-v1",
        "status": "PASS",
        "checks": checks,
        "runs": runs,
        "total_run_runtime_seconds": sum(row["runtime_seconds"] for row in runs),
        "worker_elapsed_seconds": progress["elapsed_seconds"],
        "training_protocol_sha256": digest(protocol_path),
        "oof_predictions_sha256": digest(TRAINING / "oof_predictions.parquet"),
        "oof_summary_sha256": digest(summary_path),
        "metrics": summary["metrics"],
        "calibration": summary["calibration"],
        "eligible_gate_count": summary["eligible_gate_count"],
        "selected_gate": summary["selected_gate"],
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(TRAINING / "completion_integrity_audit.json", result)
    print(json.dumps({
        "status": result["status"], "runs": len(runs),
        "prediction_rows": len(combined),
        "eligible_gate_count": result["eligible_gate_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
