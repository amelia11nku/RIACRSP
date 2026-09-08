#!/usr/bin/env python3
"""Launch or verify the persistent Phase 6O targeted relabel worker."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling"
LAUNCH = OUT / "launch_record.json"
PROGRESS = OUT / "progress.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def verify() -> None:
    record = json.loads(LAUNCH.read_text())
    pid = int(record["pid"])
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else None
    log_path = ROOT / record["log_path"]
    is_alive = alive(pid)
    record.update({
        "status": "RUNNING_VERIFIED"
        if is_alive and progress is not None and log_path.exists() and log_path.stat().st_size > 0
        else "VERIFICATION_FAILED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_alive": is_alive,
        "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "progress_observed": progress,
    })
    atomic_json(LAUNCH, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if record["status"] != "RUNNING_VERIFIED":
        raise RuntimeError("Phase 6O relabel worker verification failed")


def launch() -> None:
    if LAUNCH.exists():
        previous = json.loads(LAUNCH.read_text())
        if alive(int(previous["pid"])):
            raise RuntimeError(f"Phase 6O relabel worker already active: {previous['pid']}")
        if PROGRESS.exists() and json.loads(PROGRESS.read_text()).get("status") == "COMPLETE":
            raise RuntimeError("Phase 6O targeted relabeling is already complete")
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"formal_targeted_relabeling_{stamp}.log"
    command = [sys.executable, "-u", "scripts/run_phase6o_targeted_relabeling.py"]
    OUT.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    projected = 3600.0
    if PROGRESS.exists():
        observed = json.loads(PROGRESS.read_text())
        remaining = observed.get("measured_remaining_state_seconds")
        if remaining is not None and float(remaining) > 0:
            projected = float(remaining)
    record = {
        "schema": "phase6o-targeted-relabel-launch-v1",
        "status": "STARTED_PENDING_VERIFICATION",
        "pid": process.pid,
        "command": command,
        "working_directory": str(ROOT),
        "started_at_utc": started.isoformat(),
        "projected_remaining_seconds": projected,
        "projected_completion_at_utc": (started + timedelta(seconds=projected)).isoformat(),
        "log_path": str(log_path.relative_to(ROOT)),
        "output_path": str(OUT.relative_to(ROOT)),
        "progress_path": str(PROGRESS.relative_to(ROOT)),
        "resume_command": command,
        "checkpoint_resume": "completed state shards are hash-validated and skipped",
        "session_mode": "start_new_session",
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(LAUNCH, record)
    print(json.dumps(record, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else launch()


if __name__ == "__main__":
    main()
