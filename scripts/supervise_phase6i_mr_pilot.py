#!/usr/bin/env python3
"""Run the Phase 6I-MR pilot worker and persist its exact exit status."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6i_mr_live_utility_revision.json"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    status_path = ROOT / config["pilot"]["output_root"] / "worker_status.json"
    command = [
        PYTHON,
        "scripts/run_phase6i_mr_pilot.py",
        "--device",
        "cuda",
        "--protocol-revision",
        "1.2",
    ]
    started = datetime.now(timezone.utc)
    atomic_json({
        "schema": "phase6i-mr-pilot-worker-status-v1.2",
        "status": "RUNNING",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
        "exit_code": None,
    }, status_path)
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    atomic_json({
        "schema": "phase6i-mr-pilot-worker-status-v1.2",
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": completed.returncode,
    }, status_path)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
