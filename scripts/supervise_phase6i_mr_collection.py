#!/usr/bin/env python3
"""Run one Phase 6I-MR collection worker and persist its exact exit status."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["R09", "R10"], required=True)
    args = parser.parse_args()
    out = ROOT / f"outputs/phase6i_mr/collection/{args.split.lower()}"
    status_path = out / "worker_status.json"
    command = [
        PYTHON,
        "scripts/run_phase6i_mr_collection.py",
        "--split",
        args.split,
        "--device",
        "cuda",
    ]
    started = datetime.now(timezone.utc)
    atomic_json({
        "schema": "phase6i-mr-collection-worker-status-v1.2",
        "split": args.split,
        "status": "RUNNING",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
        "exit_code": None,
        "r10_accessed": args.split == "R10",
        "r11_accessed": False,
    }, status_path)
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    atomic_json({
        "schema": "phase6i-mr-collection-worker-status-v1.2",
        "split": args.split,
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "supervisor_pid": os.getpid(),
        "command": command,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": completed.returncode,
        "r10_accessed": args.split == "R10",
        "r11_accessed": False,
    }, status_path)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
