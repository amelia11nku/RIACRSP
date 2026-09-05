#!/usr/bin/env python3
"""Launch one locked, detached J3 worker and verify actual epoch progress."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase6j_caur_j3 as j3  # noqa: E402

PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def is_training_command(arguments: list[str]) -> bool:
    if len(arguments) < 2 or not Path(arguments[0]).name.startswith("python"):
        return False
    script = Path(arguments[1]).name
    if script in {"train_phase6j_caur.py", "supervise_phase6j_caur_training.py"}:
        return True
    if script == "run_phase6j_caur_stage.py":
        return "r12-train" in arguments
    if script == "run_phase6j_caur_j3.py":
        return "--mode" in arguments and any(mode in arguments for mode in ("train", "smoke"))
    return False


def active_training_processes():
    output = subprocess.run(["ps", "-eo", "pid=,args="], check=True,
                            text=True, capture_output=True).stdout
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if fields and is_training_command(fields[1:]):
            rows.append({"pid": int(fields[0]), "arguments": fields[1:]})
    return rows


def main():
    j3.OUT.mkdir(parents=True, exist_ok=True)
    with (j3.OUT / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch()


def launch():
    active = active_training_processes()
    if active:
        raise RuntimeError(f"existing Phase 6J training processes: {active}")
    # Also detects direct workers that have not yet appeared in the process snapshot.
    with (j3.OUT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = j3.validate_protocol()
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                      capture_output=True, text=True).stdout.strip():
        raise RuntimeError("J3 launch requires committed implementation and clean worktree")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("J3 launch requires working host CUDA")
    torch.ones(1, device="cuda").sum().item()
    sha = j3.regular.digest(j3.PROTOCOL_PATH)
    done = j3.completed_runs(sha)
    if len(done) == protocol["expected_runs"]:
        raise RuntimeError("all J3 runs are complete; audit outputs instead")
    records = [j3.regular.load_json(paths[2]) for _, _, paths in done]
    if not records:
        raise RuntimeError("complete one bounded formal J3 fold before persistent launch")
    mean_seconds = sum(row["runtime_seconds"] for row in records) / len(records)
    remaining_seconds = mean_seconds * (protocol["expected_runs"] - len(done))
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = j3.OUT / f"r12_j3_{stamp}.log"
    command = [PYTHON, "scripts/run_phase6j_caur_j3.py", "--mode", "train", "--device", "cuda"]
    environment = os.environ.copy()
    environment.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    payload = {
        "schema": "phase6j-caur-j3-launch-v1", "status": "STARTING", "pid": process.pid,
        "command": command, "log_path": str(log_path.relative_to(ROOT)),
        "output_root": str(j3.OUT.relative_to(ROOT)), "training_protocol_sha256": sha,
        "implementation_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                 text=True, capture_output=True, check=True).stdout.strip(),
        "gpu": torch.cuda.get_device_name(0), "completed_before_launch": len(done),
        "expected_runs": protocol["expected_runs"], "observed_mean_run_seconds": mean_seconds,
        "projected_remaining_seconds": remaining_seconds, "started_at_utc": started.isoformat(),
        "projected_completion_at_utc": (started + timedelta(seconds=remaining_seconds)).isoformat(),
        "r13_accessed": False, "r14_accessed": False,
    }
    record_path = j3.OUT / "launch_record.json"
    attempt_path = j3.OUT / f"launch_{stamp}.json"
    j3.regular.atomic_json(payload, record_path)
    j3.regular.atomic_json(payload, attempt_path)
    deadline = time.monotonic() + 30
    verified = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if '"event": "phase6j_j3_epoch"' in log:
            status = j3.regular.load_json(j3.OUT / "progress.json")
            verified = status["worker_pid"] == process.pid and "current" in status
            if verified:
                break
        time.sleep(1)
    if not verified:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        payload.update({"status": "FAILED_STARTUP", "exit_code": process.returncode})
    else:
        payload["status"] = "RUNNING_VERIFIED"
    j3.regular.atomic_json(payload, record_path)
    j3.regular.atomic_json(payload, attempt_path)
    print(json.dumps(payload, indent=2), flush=True)
    if not verified:
        raise RuntimeError(f"J3 startup failed; inspect {log_path}")


if __name__ == "__main__":
    main()
