#!/usr/bin/env python3
"""Launch and verify persistent Phase 6N formal OOF training."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def active_workers() -> list[int]:
    listing = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    return sorted({
        int(line.strip().split(maxsplit=1)[0])
        for line in listing.splitlines()
        if "scripts/train_phase6n_candidate_conditioned.py --device cuda" in line
        and "launch_phase6n_training.py" not in line
    })


def main() -> None:
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Phase 6N formal training requires a clean committed worktree")
    if active_workers():
        raise RuntimeError(f"Phase 6N training already active: {active_workers()}")
    subprocess.run(
        [PYTHON, "-c", "import torch; assert torch.cuda.is_available(); "
         "print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda').item())"],
        cwd=ROOT,
        check=True,
    )
    smoke = json.loads((OUT / "smoke/progress.json").read_text())
    if smoke.get("status") != "SMOKE_COMPLETE":
        raise RuntimeError("Phase 6N smoke result is missing")
    projected_seconds = max(600.0, float(smoke["elapsed_seconds"]) * 135.0)
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT / f"formal_training_{stamp}.log"
    command = [
        PYTHON,
        "scripts/train_phase6n_candidate_conditioned.py",
        "--device",
        "cuda",
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    log = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 55.0
    observed = False
    while time.monotonic() < deadline:
        log.flush()
        content = log_path.read_text(encoding="utf-8", errors="replace")
        if '"event": "phase6n_first_batch"' in content and '"finite": true' in content:
            observed = True
            break
        if process.poll() is not None:
            break
        time.sleep(1.0)
    log.close()
    alive = process.poll() is None
    if not observed or not alive:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(
            f"Phase 6N worker failed liveness/sanity verification: alive={alive}\n{tail}"
        )
    record = {
        "schema": "phase6n-formal-training-launch-v1",
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
        "eta_basis": "one-run one-inner-plus-one-outer epoch smoke multiplied by 9 runs and conservative 15 epoch-pair factor",
        "log_path": str(log_path.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "progress_path": str((OUT / "progress.json").relative_to(ROOT)),
        "checkpoint_resume": "rerun identical command; completed fold checkpoint/prediction/record hashes are validated and skipped",
        "liveness_evidence": "finite first-batch loss observed and detached process alive",
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT / "launch_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
