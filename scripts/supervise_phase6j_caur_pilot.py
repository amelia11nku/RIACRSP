#!/usr/bin/env python3
"""Run the Phase 6J R12 pilot worker and persist its exit status."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
STATUS = ROOT / "outputs/phase6j_caur/r12_pilot/worker_status.json"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    command = [PYTHON, "scripts/run_phase6j_caur_pilot.py", "--device", "cuda"]
    started = datetime.now(timezone.utc)
    base = {
        "schema": "phase6j-caur-r12-pilot-worker-status-v1",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
    }
    atomic_json({**base, "status": "RUNNING", "exit_code": None}, STATUS)
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    atomic_json({
        **base,
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": completed.returncode,
    }, STATUS)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
