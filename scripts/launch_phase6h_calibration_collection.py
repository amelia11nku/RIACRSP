#!/usr/bin/env python3
"""Launch the long CAL-FIT collection as a verified detached process."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6h_calibration/collection"
LAUNCH = OUT / "launch_record.json"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    if LAUNCH.exists():
        previous = json.loads(LAUNCH.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(f"collection process is already alive: {previous['pid']}")
    config = json.loads(
        (ROOT / "configs/phase6h_live_calibration.json").read_text(encoding="utf-8")
    )
    manifest = pd.read_csv(ROOT / config["calibration_instances"]["manifest"])
    fit = manifest[manifest.calibration_split == "CAL_FIT"]
    operations = []
    for row in fit.itertuples(index=False):
        raw = json.loads((
            ROOT / config["calibration_instances"]["root"] / row.relative_path
        ).read_text(encoding="utf-8"))
        operations.append(len(raw["operations"]))
    nominal_seconds = (
        sum(operations)
        * len(config["seeds"]["CAL_FIT_COLLECTION"])
        * float(config["search"]["wall_clock_seconds_per_operation"])
    )
    expected_seconds = nominal_seconds * 1.10 + 120.0
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"collection_{timestamp}.log"
    command = [PYTHON, "scripts/run_phase6h_calibration_collection.py", "--device", "cuda"]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    OUT.mkdir(parents=True, exist_ok=True)
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
            f"collection process exited during launch verification: {return_code}; "
            f"inspect {log_path}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "PHASE6H_CAL_FIT_COLLECTION_START" not in log_text:
        process.terminate()
        raise RuntimeError("collection process did not emit its start marker")
    payload = {
        "schema": "phase6h-detached-collection-launch-v1",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "session_mode": "start_new_session",
        "command": command,
        "working_directory": str(ROOT),
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "started_at_utc": started.isoformat(),
        "nominal_search_budget_seconds": nominal_seconds,
        "expected_completion_at_utc": (
            started + timedelta(seconds=expected_seconds)
        ).isoformat(),
        "expected_elapsed_seconds": expected_seconds,
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 40 {log_path.relative_to(ROOT)}",
            "cat outputs/phase6h_calibration/collection/progress.json",
        ],
    }
    LAUNCH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
