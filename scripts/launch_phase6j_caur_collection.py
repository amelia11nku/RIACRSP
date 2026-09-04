#!/usr/bin/env python3
"""Launch and verify the single persistent Phase 6J R12 collection worker."""

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

from rcias_clgri.data.phase6j_access import verify_r12_collection_authorization  # noqa: E402
from scripts.run_phase6j_caur_pilot import digest  # noqa: E402


PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
CONFIG = ROOT / "configs/phase6j_caur.json"
COLLECTOR = ROOT / "scripts/run_phase6j_caur_collection.py"
FREEZE = ROOT / "outputs/phase6j_caur/frozen/r12_horizon_freeze.json"
OUT = ROOT / "outputs/phase6j_caur/r12_collection"


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
        raise RuntimeError("R12 collection requires a clean committed implementation boundary")
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    authorization = verify_r12_collection_authorization(
        FREEZE,
        project_root=ROOT,
        config_path=CONFIG,
        collection_script_path=COLLECTOR,
    )
    gpu = subprocess.run(
        [PYTHON, "-c", "import torch; assert torch.cuda.is_available(); "
         "print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda').item())"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    launch_path = OUT / "launch_record.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text(encoding="utf-8"))
        if process_alive(int(previous["pid"])):
            raise RuntimeError(f"R12 collection worker is already alive: {previous['pid']}")
    progress_path = OUT / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    complete = int(progress.get("states_complete", 0))
    if complete >= 288:
        raise RuntimeError("R12 collection is already complete")
    projection = authorization["full_collection_projection"]
    expected_seconds = float(projection["projected_elapsed_seconds"]) * (
        (288 - complete) / 288
    )
    expected_seconds = max(expected_seconds, 60.0)
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"r12_collection_{timestamp}.log"
    command = [PYTHON, "scripts/supervise_phase6j_caur_collection.py"]
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
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if return_code is not None:
        raise RuntimeError(f"R12 collection exited with {return_code}; inspect {log_path}")
    if "PHASE6J_CAUR_R12_COLLECTION_START" not in log_text:
        process.terminate()
        raise RuntimeError("R12 collection did not emit its start marker")
    payload = {
        "schema": "phase6j-caur-r12-collection-launch-v1",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "session_mode": "start_new_session",
        "command": command,
        "worker_command": [
            PYTHON,
            "scripts/run_phase6j_caur_stage.py",
            "--stage",
            "r12-collect",
            "--execute",
        ],
        "working_directory": str(ROOT),
        "implementation_commit": implementation_commit,
        "r12_horizon_freeze_sha256": digest(FREEZE),
        "selected_horizon": int(authorization["selected_horizon"]),
        "collection_scope": authorization["collection_scope"],
        "gpu": gpu[0],
        "cuda_tensor_smoke": gpu[1],
        "log_path": str(log_path.relative_to(ROOT)),
        "output_root": str(OUT.relative_to(ROOT)),
        "states_complete_before_launch": complete,
        "states_remaining_at_launch": 288 - complete,
        "started_at_utc": started.isoformat(),
        "projected_elapsed_seconds": expected_seconds,
        "projected_completion_at_utc": (
            started + timedelta(seconds=expected_seconds)
        ).isoformat(),
        "status_commands": [
            f"ps -p {process.pid} -o pid,ppid,etime,stat,cmd",
            f"tail -n 40 {log_path.relative_to(ROOT)}",
            "cat outputs/phase6j_caur/r12_collection/progress.json",
            "cat outputs/phase6j_caur/r12_collection/worker_status.json",
        ],
    }
    atomic_json(payload, launch_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
