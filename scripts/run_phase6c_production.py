#!/usr/bin/env python3
"""Resume-safe orchestration of the Phase 6C reservoir and full dataset run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "outputs/phase6c/environment/production_job_status.json"


def write_status(stage: str, status: str, **extra):
    payload = {
        "schema": "phase6c-production-job-status-v1",
        "pid": os.getpid(),
        "stage": stage,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    partial = STATUS.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, STATUS)


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    freeze = ROOT / "outputs/phase6c/environment/production_config_freeze.json"
    if not freeze.exists():
        raise RuntimeError("production configuration must be frozen before launch")
    frozen = json.loads(freeze.read_text())
    current_hashes = {
        "config_sha256": hashlib.sha256((ROOT / "configs/phase6c_counterfactual.json").read_bytes()).hexdigest(),
        "counterfactual_evaluator_sha256": hashlib.sha256((ROOT / "rcias_clgri/search/counterfactual.py").read_bytes()).hexdigest(),
        "revised_arm_design_sha256": hashlib.sha256((ROOT / "rcias_clgri/search/phase6c.py").read_bytes()).hexdigest(),
        "reconstruction_contract_sha256": hashlib.sha256((ROOT / "rcias_clgri/data/phase6c.py").read_bytes()).hexdigest(),
        "field_contract_sha256": hashlib.sha256((ROOT / "rcias_clgri/data/phase6c_contract.py").read_bytes()).hexdigest(),
        "shard_io_sha256": hashlib.sha256((ROOT / "rcias_clgri/data/phase6c_io.py").read_bytes()).hexdigest(),
        "reservoir_runner_sha256": hashlib.sha256((ROOT / "scripts/run_phase6c_reservoir.py").read_bytes()).hexdigest(),
        "dataset_runner_sha256": hashlib.sha256((ROOT / "scripts/run_phase6c_dataset.py").read_bytes()).hexdigest(),
    }
    if any(frozen[key] != value for key, value in current_hashes.items()):
        raise RuntimeError("production code/config no longer matches the frozen hashes")
    try:
        write_status("RESERVOIR", "RUNNING", workers=args.workers)
        run("scripts/run_phase6c_reservoir.py", "--workers", str(args.workers))
        run("scripts/run_phase6c_reservoir.py", "--verify-only")
        write_status("COUNTERFACTUAL_DATASET", "RUNNING", workers=args.workers)
        run("scripts/run_phase6c_dataset.py", "--resume", "--workers", str(args.workers))
        run("scripts/run_phase6c_dataset.py", "--verify-only")
        write_status("DATASET_VERIFIED", "COMPLETE", workers=args.workers)
    except Exception as error:
        write_status("FAILED", "FAILED", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
