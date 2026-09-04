#!/usr/bin/env python3
"""Launch and verify one persistent Phase 6J R12 OOF training worker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase6j_caur_pilot import digest  # noqa: E402
from scripts.train_phase6j_caur import (  # noqa: E402
    FAMILIES,
    run_paths,
    valid_run,
    validate_protocol,
)


PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
OUT = ROOT / "outputs/phase6j_caur/training"
PROTOCOL = ROOT / "outputs/phase6j_caur/frozen/r12_training_protocol.json"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def active_training_processes() -> list[int]:
    listing = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    markers = (
        "scripts/supervise_phase6j_caur_training.py",
        "scripts/run_phase6j_caur_stage.py --stage r12-train --execute",
        "scripts/train_phase6j_caur.py --device cuda",
    )
    return sorted({
        int(line.strip().split(maxsplit=1)[0])
        for line in listing.splitlines()
        if line.strip() and any(marker in line for marker in markers)
    })


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def wait_for_epoch_liveness(
    process: subprocess.Popen, log_path: Path, timeout_seconds: float = 30.0
) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    log_text = ""
    while time.monotonic() < deadline:
        return_code = process.poll()
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "phase6j_caur_epoch" in log_text:
            return True, log_text
        if return_code is not None:
            return False, log_text
        time.sleep(1.0)
    return False, log_text


def main() -> None:
    worktree = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if worktree:
        raise RuntimeError("R12 training requires a clean committed implementation boundary")
    protocol = validate_protocol()
    active = active_training_processes()
    if active:
        raise RuntimeError(f"R12 training worker already exists: {active}")
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    gpu = subprocess.run(
        [PYTHON, "-c", "import torch; assert torch.cuda.is_available(); "
         "print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda').item())"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    protocol_sha256 = digest(PROTOCOL)
    completed_paths = []
    for family in FAMILIES:
        for seed in protocol["training"]["seeds"]:
            for held_fold in range(3):
                paths = run_paths(family, int(seed), held_fold)
                if valid_run(paths, protocol_sha256):
                    completed_paths.append(paths)
    completed = len(completed_paths)
    expected = len(FAMILIES) * len(protocol["training"]["seeds"]) * 3
    if completed >= expected:
        raise RuntimeError("R12 J1/J2 OOF training is already complete")
    launch_path = OUT / "launch_record.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(f"R12 training worker is already alive: {previous['pid']}")

    observed = [
        float(json.loads(paths[2].read_text(encoding="utf-8"))["runtime_seconds"])
        for paths in completed_paths
    ]
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
    alive, _ = wait_for_epoch_liveness(process, log_path)
    if not alive:
        return_code = process.poll()
        if return_code is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise RuntimeError(
            f"R12 training failed its 30-second epoch liveness check; inspect {log_path}"
        )
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
