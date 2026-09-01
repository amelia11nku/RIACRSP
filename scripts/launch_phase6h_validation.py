#!/usr/bin/env python3
"""Freeze concurrency and launch CAL-HOLDOUT validation in a detached session."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
OUT = ROOT / "outputs/phase6h_validation"
LAUNCH = OUT / "launch_record.json"
PROTOCOL = OUT / "environment/concurrency_protocol.json"
FROZEN_POLICY = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
FREEZE_RECORD = ROOT / "outputs/phase6h_calibration/frozen/freeze_record.json"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if LAUNCH.exists():
        previous = json.loads(LAUNCH.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(f"validation orchestrator is already alive: {previous['pid']}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_BEFORE_CAL_HOLDOUT"
        or freeze.get("cal_holdout_opened") is not False
        or digest(FROZEN_POLICY) != freeze.get("policy_sha256")
    ):
        raise RuntimeError("Phase 6H policy freeze is invalid")
    existing_results = list((OUT / "runs").rglob("*.json"))
    if existing_results and not PROTOCOL.exists():
        raise RuntimeError("validation results exist without a frozen concurrency protocol")
    if not PROTOCOL.exists():
        atomic_json({
            "schema": "phase6h-cal-holdout-concurrency-protocol-v1",
            "status": "FROZEN_BEFORE_FIRST_CAL_HOLDOUT_SOLVER_RUN",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_sha256": freeze["policy_sha256"],
            "methods": {
                "H1": {"runs_per_instance": 1, "execution": "serial before stochastic workers"},
                "ALNS": {"runs_per_instance": 5, "worker_count": 1, "gpu_usage": False},
                "GA": {"runs_per_instance": 5, "worker_count": 1, "gpu_usage": False},
                "DCGA": {"runs_per_instance": 5, "worker_count": 1, "gpu_usage": False},
                "PHASE6G_CSGNI": {"runs_per_instance": 5, "shared_gpu_worker": "CSGNI"},
                "PHASE6H_CSGNI": {"runs_per_instance": 5, "shared_gpu_worker": "CSGNI"}
            },
            "shared_protocol": {
                "maximum_concurrent_stochastic_runs": 4,
                "cpu_threads_per_worker": 1,
                "available_logical_cpu_count": os.cpu_count(),
                "gpu_worker_count": 1,
                "wall_clock_budget_per_run": "2 * N_operations seconds",
                "seed_ids": config["seeds"]["CAL_HOLDOUT"],
                "atomic_resume": True,
                "duplicate_task_execution": False,
            },
            "measurement_caveat": (
                "Four workers share host resources. Every numerical library is restricted "
                "to one thread; the two CSG-NI methods execute sequentially in the sole GPU worker."
            ),
            "cal_holdout_used_for_selection": False,
        }, PROTOCOL)

    manifest = pd.read_csv(ROOT / config["calibration_instances"]["manifest"])
    holdout = manifest[manifest.calibration_split == "CAL_HOLDOUT"]
    operations = []
    for row in holdout.itertuples(index=False):
        raw = json.loads((
            ROOT / config["calibration_instances"]["root"] / row.relative_path
        ).read_text(encoding="utf-8"))
        operations.append(len(raw["operations"]))
    one_method_seconds = (
        sum(operations)
        * len(config["seeds"]["CAL_HOLDOUT"])
        * float(config["search"]["wall_clock_seconds_per_operation"])
    )
    critical_path_seconds = 2 * one_method_seconds
    expected_seconds = critical_path_seconds * 1.12 + 300.0
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"validation_{timestamp}.log"
    command = [PYTHON, "scripts/run_phase6h_validation_parallel.py"]
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
            f"validation process exited during verification: {return_code}; inspect {log_path}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "PHASE6H_VALIDATION_PARALLEL_START" not in log_text:
        process.terminate()
        raise RuntimeError("validation process did not emit its start marker")
    payload = {
        "schema": "phase6h-detached-validation-launch-v1",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "session_mode": "start_new_session",
        "command": command,
        "working_directory": str(ROOT),
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "concurrency_protocol_sha256": digest(PROTOCOL),
        "policy_sha256": freeze["policy_sha256"],
        "started_at_utc": started.isoformat(),
        "critical_path_nominal_seconds": critical_path_seconds,
        "expected_completion_at_utc": (
            started + timedelta(seconds=expected_seconds)
        ).isoformat(),
        "expected_elapsed_seconds": expected_seconds,
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 50 {log_path.relative_to(ROOT)}",
            "cat outputs/phase6h_validation/progress.json",
            "tail -n 20 outputs/phase6h_validation/worker_CSGNI.log",
        ],
    }
    atomic_json(payload, LAUNCH)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
