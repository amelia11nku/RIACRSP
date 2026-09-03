#!/usr/bin/env python3
"""Run the Phase 6I-MR continuation worker and persist its exit status."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6i_mr/continuation"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    status_path = OUT / "worker_status.json"
    command = [
        PYTHON,
        "scripts/run_phase6i_mr_continuation.py",
        "--device",
        "cuda",
    ]
    started = datetime.now(timezone.utc)
    atomic_json({
        "schema": "phase6i-mr-continuation-worker-status-v1.2",
        "status": "RUNNING",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
        "exit_code": None,
        "r10_accessed": False,
        "r11_accessed": False,
    }, status_path)
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    atomic_json({
        "schema": "phase6i-mr-continuation-worker-status-v1.2",
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": completed.returncode,
        "r10_accessed": False,
        "r11_accessed": False,
    }, status_path)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
