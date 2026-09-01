#!/usr/bin/env python3
"""Orchestrate the frozen four-worker Phase 6H CAL-HOLDOUT protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6h_validation"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"
RUNNER = "scripts/run_phase6h_validation.py"


def main() -> None:
    protocol = json.loads((OUT / "environment/concurrency_protocol.json").read_text())
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_CAL_HOLDOUT_SOLVER_RUN":
        raise RuntimeError("Phase 6H validation concurrency protocol is not frozen")
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    workers: list[subprocess.Popen] = []

    def terminate_workers(_signum=None, _frame=None) -> None:
        for process in workers:
            if process.poll() is None:
                process.terminate()
        for process in workers:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, terminate_workers)
    signal.signal(signal.SIGTERM, terminate_workers)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    print("PHASE6H_VALIDATION_PARALLEL_START", flush=True)

    h1_log = OUT / "worker_H1.log"
    with h1_log.open("ab", buffering=0) as stream:
        completed = subprocess.run(
            [PYTHON, RUNNER, "--methods", "H1"],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"H1 validation worker failed: {completed.returncode}")
    print("PHASE6H_VALIDATION_H1_COMPLETE", flush=True)

    worker_specs = {
        "ALNS": ["ALNS"],
        "GA": ["GA"],
        "DCGA": ["DCGA"],
        "CSGNI": ["PHASE6G_CSGNI", "PHASE6H_CSGNI"],
    }
    streams = []
    try:
        for label, methods in worker_specs.items():
            stream = (OUT / f"worker_{label}.log").open("ab", buffering=0)
            streams.append(stream)
            command = [PYTHON, RUNNER, "--device", "cuda", "--methods", *methods]
            workers.append(subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
            ))
            print(
                f"PHASE6H_VALIDATION_WORKER_STARTED label={label} pid={workers[-1].pid}",
                flush=True,
            )
        failures = []
        for (label, _), process in zip(worker_specs.items(), workers):
            return_code = process.wait()
            print(
                f"PHASE6H_VALIDATION_WORKER_RETURNED label={label} status={return_code}",
                flush=True,
            )
            if return_code != 0:
                failures.append((label, return_code))
        if failures:
            raise RuntimeError(f"Phase 6H validation workers failed: {failures}")
    finally:
        for stream in streams:
            stream.close()

    completed = subprocess.run(
        [PYTHON, RUNNER, "--summarize-only"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"validation summarization failed: {completed.returncode}")
    print("PHASE6H_VALIDATION_PARALLEL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
