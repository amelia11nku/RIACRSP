#!/usr/bin/env python3
"""Independent completion, replay, and integrity audit for frozen R11 evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from rcias_clgri.data.phase6i_access import load_phase6i_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.analysis.phase6h import validate_incumbent_trace
from scripts.run_phase6i_mr_r11_validation import (
    ARTIFACT_PATH,
    FREEZE_RECORD,
    OUT,
    build_tasks,
    forced_path,
    live_log_path,
    result_path,
    valid_result,
    verify_unlock,
)


AUDIT_PATH = OUT / "completion_integrity_audit.json"
MANIFEST_PATH = OUT / "r11_result_hash_manifest.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    config, _artifact, artifact_hash, freeze_hash = verify_unlock()
    tasks = build_tasks(config)
    instance_root = Path(__file__).resolve().parents[1] / config["instance_suite"]["root"]
    instances = {}
    rows = []
    replayed = trace_valid = feasible = 0
    records = []
    for task in tasks:
        payload = valid_result(task, artifact_hash, freeze_hash)
        if payload is None:
            raise RuntimeError(f"invalid frozen result: {task}")
        records.append(payload)
        instance_id = task["instance_id"]
        if instance_id not in instances:
            instances[instance_id] = load_phase6i_instance(
                instance_root / task["instance_relative_path"],
                freeze_record_path=FREEZE_RECORD,
                artifact_path=ARTIFACT_PATH,
            )
        instance = instances[instance_id]
        environment = RCIASConstructionEnv(instance)
        for raw_action in payload["best_actions"]:
            environment.step(Action(**raw_action))
        schedule_audit = check_schedule(instance, environment.schedule)
        replay_makespan = float(environment.objective().makespan)
        if not environment.done or replay_makespan != float(payload["final_best"]):
            raise RuntimeError(f"action replay mismatch: {result_path(task)}")
        replayed += 1
        if not schedule_audit["feasible"]:
            raise RuntimeError(f"replayed schedule infeasible: {result_path(task)}")
        feasible += 1
        validate_incumbent_trace(payload["incumbent_trace"], final_best=payload["final_best"])
        trace_valid += 1
        path = result_path(task)
        rows.append({
            "method": task["method"],
            "instance_id": instance_id,
            "seed": task["seed"],
            "result_path": str(path.relative_to(Path(__file__).resolve().parents[1])),
            "result_sha256": digest(path),
            "replay_makespan": replay_makespan,
            "feasible": True,
            "trace_valid": True,
        })

    revised_tasks = [task for task in tasks if task["method"] == "PHASE6I_MR_CSGNI"]
    forced_frames = [pd.read_parquet(forced_path(task)) for task in revised_tasks]
    forced_expected = pd.concat(forced_frames, ignore_index=True)
    forced_aggregate = pd.read_parquet(OUT / "forced_diagnostics.parquet")
    pd.testing.assert_frame_equal(forced_aggregate, forced_expected, check_exact=True)
    summary = pd.read_csv(OUT / "validation_run_summary.csv")
    temporary_files = sorted(
        str(path.relative_to(OUT))
        for path in OUT.rglob("*")
        if path.is_file() and (".tmp." in path.name or path.suffix in {".part", ".partial"})
    )
    checks = {
        "progress_complete_288": json.loads((OUT / "progress.json").read_text())["completed_runs"] == 288,
        "task_identity_and_frozen_hashes": len(records) == 288,
        "unique_task_keys": len({(row["method"], row["instance_id"], row["seed"]) for row in rows}) == 288,
        "accepted_feasibility": all(record.get("feasible") is True for record in records),
        "action_replay_exact": replayed == 288,
        "replay_feasibility": feasible == 288,
        "monotone_trace_integrity": trace_valid == 288,
        "summary_rows_and_decoder_seconds": bool(
            len(summary) == 288 and summary.decoder_seconds.notna().all()
        ),
        "live_log_count": sum(live_log_path(task).is_file() for task in tasks) == 180,
        "forced_file_count": sum(forced_path(task).is_file() for task in revised_tasks) == 90,
        "forced_aggregate_exact": len(forced_aggregate) == 3600 and forced_aggregate.state_id.nunique() == 900,
        "no_temporary_fragments": not temporary_files,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "temporary_files": temporary_files})
    atomic_csv(MANIFEST_PATH, pd.DataFrame(rows))
    audit = {
        "schema": "phase6i-mr-r11-completion-integrity-audit-v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "instances": len(instances),
        "results": len(records),
        "live_logs": 180,
        "forced_diagnostic_files": 90,
        "forced_states": int(forced_aggregate.state_id.nunique()),
        "forced_actions": len(forced_aggregate),
        "selected_artifact_sha256": artifact_hash,
        "artifact_freeze_sha256": freeze_hash,
        "summary_compatibility_audit": "outputs/phase6i_mr/r11_validation/summary_compatibility_audit.json",
        "result_hash_manifest": str(MANIFEST_PATH.relative_to(Path(__file__).resolve().parents[1])),
        "gurobi_executed": False,
    }
    atomic_json(AUDIT_PATH, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
