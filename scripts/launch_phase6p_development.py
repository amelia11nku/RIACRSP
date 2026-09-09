#!/usr/bin/env python3
"""Launch the resumable Phase 6P P3 campaign as a detached local worker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/development"
PROTOCOL = OUT / "protocol.json"
PROGRESS = OUT / "progress.json"
LAUNCH = OUT / "launch_record.json"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def active_workers() -> list[int]:
    listing = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    return sorted({
        int(line.strip().split(maxsplit=1)[0])
        for line in listing.splitlines()
        if "scripts/run_phase6p_development.py" in line
        and "launch_phase6p_development.py" not in line
    })


def main() -> None:
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Phase 6P P3 launch requires a clean committed worktree")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_P3_SOLVER_OUTCOMES":
        raise RuntimeError("Phase 6P P3 protocol is not frozen")
    if active_workers():
        raise RuntimeError(f"Phase 6P development already active: {active_workers()}")
    if PROGRESS.exists() and json.loads(PROGRESS.read_text()).get("status") == "COMPLETE":
        raise RuntimeError("Phase 6P development is already complete")
    subprocess.run([
        PYTHON,
        "-c",
        "import torch; assert torch.cuda.is_available(); "
        "print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda').item())",
    ], cwd=ROOT, check=True)
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"formal_development_{stamp}.log"
    command = [
        PYTHON, "-u", "scripts/run_phase6p_development.py", "--device", "cuda"
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    stream = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 55.0
    observed = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if PROGRESS.exists():
            try:
                candidate = json.loads(PROGRESS.read_text())
                if candidate.get("status") == "RUNNING":
                    observed = candidate
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(1.0)
    stream.close()
    if process.poll() is not None or observed is None:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(
            f"Phase 6P development launch verification failed: "
            f"returncode={process.poll()} progress={observed}\n{tail}"
        )
    projected_seconds = max(
        68_400.0,
        float(observed["nominal_remaining_budget_seconds"]) * 1.05 + 600.0,
    )
    record = {
        "schema": "phase6p-development-launch-v1",
        "status": "RUNNING_VERIFIED",
        "pid": process.pid,
        "command": command,
        "working_directory": str(ROOT),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "started_at_utc": started.isoformat(),
        "projected_seconds": projected_seconds,
        "expected_completion_utc": (
            started + timedelta(seconds=projected_seconds)
        ).isoformat(),
        "eta_basis": "18.0 h exact serial solver budgets plus 5 percent inter-run overhead and 10 min setup margin; lower bound raised to 19 h",
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "progress_path": str(PROGRESS.relative_to(ROOT)),
        "resume_command": command,
        "resume_semantics": "rerun identical command; complete per-run payloads are hash-validated and skipped",
        "session_mode": "start_new_session",
        "execution_concurrency": 1,
        "liveness_evidence": {
            "process_alive": True,
            "progress_observed": observed,
        },
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(LAUNCH, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
