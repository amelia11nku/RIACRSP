#!/usr/bin/env python3
"""Launch the formal Phase 6I-MR continuation diagnostic as a verified job."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6i_mr/continuation"
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
    launch_path = OUT / "launch_record.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(
                f"continuation diagnostic is already alive: {previous['pid']}"
            )
    completed_before = len(list((OUT / "state_runs").glob("*.json")))
    remaining = max(0, 27 - completed_before)
    # Formal one-state R09 smoke took 12.1 s; retain a conservative margin.
    expected_seconds = remaining * 15.0 + 60.0
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"continuation_{timestamp}.log"
    command = [PYTHON, "scripts/supervise_phase6i_mr_continuation.py"]
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0",
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
            f"continuation diagnostic exited during verification: "
            f"{return_code}; inspect {log_path}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "PHASE6I_MR_CONTINUATION_START" not in log_text:
        process.terminate()
        raise RuntimeError("continuation diagnostic did not emit its start marker")
    payload = {
        "schema": "phase6i-mr-detached-continuation-launch-v1.2",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "session_mode": "start_new_session",
        "command": command,
        "working_directory": str(ROOT),
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "completed_states_before_launch": completed_before,
        "remaining_states_at_launch": remaining,
        "estimate_basis": "12.1-second formal R09 one-state smoke plus margin",
        "started_at_utc": started.isoformat(),
        "expected_elapsed_seconds": expected_seconds,
        "expected_completion_at_utc": (
            started + timedelta(seconds=expected_seconds)
        ).isoformat(),
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 40 {log_path.relative_to(ROOT)}",
            f"cat {(OUT / 'progress.json').relative_to(ROOT)}",
            f"cat {(OUT / 'worker_status.json').relative_to(ROOT)}",
        ],
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(payload, launch_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
