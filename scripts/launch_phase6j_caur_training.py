#!/usr/bin/env python3
"""Launch and verify one persistent Phase 6J R12 OOF training worker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase6j_caur_pilot import digest  # noqa: E402
from scripts.train_phase6j_caur import validate_protocol  # noqa: E402


PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
OUT = ROOT / "outputs/phase6j_caur/training"
PROTOCOL = ROOT / "outputs/phase6j_caur/frozen/r12_training_protocol.json"


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
    worktree = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if worktree:
        raise RuntimeError("R12 training requires a clean committed implementation boundary")
    protocol = validate_protocol()
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    gpu = subprocess.run(
        [PYTHON, "-c", "import torch; assert torch.cuda.is_available(); "
         "print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda').item())"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    progress_path = OUT / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    completed = int(progress.get("completed_runs", 0))
    expected = 18
    if completed >= expected and progress.get("status") == "COMPLETE_J1_J2":
        raise RuntimeError("R12 J1/J2 OOF training is already complete")
    launch_path = OUT / "launch_record.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(f"R12 training worker is already alive: {previous['pid']}")

    completed_records = sorted(OUT.glob("oof/*/seed_*/fold_*.json"))
    observed = []
    for path in completed_records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "COMPLETE":
            observed.append(float(record["runtime_seconds"]))
    per_run_seconds = float(sum(observed) / len(observed)) if observed else 900.0
    projected_seconds = max(60.0, per_run_seconds * (expected - completed))
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"r12_training_{timestamp}.log"
    command = [PYTHON, "scripts/supervise_phase6j_caur_training.py"]
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(5.0)
    return_code = process.poll()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if return_code is not None:
        raise RuntimeError(f"R12 training exited with {return_code}; inspect {log_path}")
    if "phase6j_caur_epoch" not in log_text:
        process.terminate()
        raise RuntimeError("R12 training did not emit an epoch liveness event")
    payload = {
        "schema": "phase6j-caur-r12-training-launch-v1",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "session_mode": "start_new_session",
        "command": command,
        "worker_command": [
            PYTHON, "scripts/run_phase6j_caur_stage.py", "--stage", "r12-train", "--execute"
        ],
        "working_directory": str(ROOT),
        "implementation_commit": implementation_commit,
        "training_protocol_sha256": digest(PROTOCOL),
        "gpu": gpu[0],
        "cuda_tensor_smoke": gpu[1],
        "log_path": str(log_path.relative_to(ROOT)),
        "output_root": str(OUT.relative_to(ROOT)),
        "completed_runs_before_launch": completed,
        "remaining_runs_at_launch": expected - completed,
        "observed_completed_run_seconds": observed,
        "started_at_utc": started.isoformat(),
        "projected_elapsed_seconds": projected_seconds,
        "projected_completion_at_utc": (
            started + timedelta(seconds=projected_seconds)
        ).isoformat(),
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 40 {log_path.relative_to(ROOT)}",
            "cat outputs/phase6j_caur/training/progress.json",
            "cat outputs/phase6j_caur/training/worker_status.json",
        ],
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(payload, launch_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
