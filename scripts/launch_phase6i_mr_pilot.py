#!/usr/bin/env python3
"""Launch the remaining Phase 6I-MR R09 pilot as a verified detached job."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-revision", choices=["1.2"], default="1.2")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out = ROOT / config["pilot"]["output_root"]
    launch = out / "launch_record.json"
    if launch.exists():
        previous = json.loads(launch.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(f"R09 pilot is already alive: {previous['pid']}")
    metrics = pd.read_csv(
        ROOT
        / config["instance_suite"]["root"]
        / "manifests/r09_r10_structural_metrics.csv"
    )
    fit = metrics[metrics.live_revision_split == "LIVE_REV_FIT"]
    fit = fit[fit.cell_replicate == config["pilot"]["cell_replicate"]]
    completed_ids = {
        path.stem for path in (out / "runs").glob("*.json")
    }
    remaining = fit[~fit.instance_id.isin(completed_ids)]
    nominal_seconds = float(
        remaining.number_of_operations.sum()
        * config["search"]["pilot_wall_clock_seconds_per_operation"]
    )
    expected_seconds = nominal_seconds * 1.15 + 90.0
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = out / f"pilot_{timestamp}.log"
    command = [PYTHON, "scripts/supervise_phase6i_mr_pilot.py"]
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    out.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(5.0)
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            f"R09 pilot exited during verification: {return_code}; inspect {log_path}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "PHASE6I_MR_R09_PILOT_START" not in log_text:
        process.terminate()
        raise RuntimeError("R09 pilot did not emit its start marker")
    payload = {
        "schema": "phase6i-mr-detached-r09-pilot-launch-v1.2",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "session_mode": "start_new_session",
        "command": command,
        "working_directory": str(ROOT),
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(out.relative_to(ROOT)),
        "completed_before_launch": len(completed_ids),
        "remaining_runs_at_launch": len(remaining),
        "nominal_remaining_search_budget_seconds": nominal_seconds,
        "started_at_utc": started.isoformat(),
        "expected_elapsed_seconds": expected_seconds,
        "expected_completion_at_utc": (
            started + timedelta(seconds=expected_seconds)
        ).isoformat(),
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 40 {log_path.relative_to(ROOT)}",
            f"cat {(out / 'progress.json').relative_to(ROOT)}",
            f"cat {(out / 'worker_status.json').relative_to(ROOT)}",
        ],
    }
    atomic_json(payload, launch)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
