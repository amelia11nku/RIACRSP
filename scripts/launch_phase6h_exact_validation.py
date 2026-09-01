#!/usr/bin/env python3
"""Launch the resumable Phase 6H exact-validation batch detached."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6h_exact_validation"
LAUNCH = OUT / "launch_record.json"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
            raise RuntimeError(f"exact-validation process already alive: {previous['pid']}")
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"exact_validation_{timestamp}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "scripts/run_phase6h_exact_validation.py"]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
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
    if process.poll() is not None:
        raise RuntimeError(f"exact-validation process exited early; inspect {log_path}")
    if "PHASE6H_EXACT_VALIDATION_START" not in log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        process.terminate()
        raise RuntimeError("exact-validation start marker missing")
    expected_seconds = 5400.0
    payload = {
        "schema": "phase6h-detached-exact-validation-launch-v1",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "command": command,
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "started_at_utc": started.isoformat(),
        "expected_completion_at_utc": (
            started + timedelta(seconds=expected_seconds)
        ).isoformat(),
        "expected_elapsed_seconds": expected_seconds,
        "duration_note": (
            "Normally a few minutes with the known size-limited license; up to "
            "90 minutes if an adequate license enables the conditional small comparison."
        ),
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 50 {log_path.relative_to(ROOT)}",
            "cat outputs/phase6h_exact_validation/audit/exact_validation_integrity.json",
        ],
    }
    atomic_json(payload, LAUNCH)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
