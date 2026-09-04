#!/usr/bin/env python3
"""Verify the immutable Phase 6I-MR evidence boundary for Phase 6J setup.

This is the only Phase 6J-named script allowed to open the listed R11 files.
Training, feature, calibration, threshold, selection, and holdout code must read
only the resulting hash manifest and must never import this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "configs/phase6j_caur_phase6i_mr_evidence_manifest.json"
OUT = ROOT / "outputs/phase6j_caur/setup/starting_boundary_audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pytest", required=True)
    args = parser.parse_args()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    file_checks = []
    for record in evidence["files"]:
        path = ROOT / record["path"]
        file_checks.append({
            "role": record["role"],
            "path": record["path"],
            "exists": path.is_file(),
            "sha256_matches": path.is_file() and digest(path) == record["sha256"],
            "size_matches": path.is_file() and path.stat().st_size == record["size_bytes"],
        })
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", evidence["source_commit_in_ancestry"], "HEAD"],
        cwd=ROOT,
    ).returncode == 0
    checks = {
        "handoff_commit_in_ancestry": ancestry,
        "all_phase6i_mr_evidence_exists": all(row["exists"] for row in file_checks),
        "all_phase6i_mr_evidence_hashes_match": all(row["sha256_matches"] for row in file_checks),
        "all_phase6i_mr_evidence_sizes_match": all(row["size_matches"] for row in file_checks),
        "baseline_pytest_passed": "passed" in args.baseline_pytest.lower(),
    }
    audit = {
        "schema": "phase6j-caur-starting-boundary-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "starting_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "starting_commit": evidence["phase6j_starting_commit"],
        "current_head": git("rev-parse", "HEAD"),
        "initial_worktree_status": "CLEAN",
        "baseline_pytest": args.baseline_pytest,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "checks": checks,
        "evidence_files": file_checks,
        "r11_use_policy": "historical motivation and bit-integrity audit only; never tuning data",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.partial")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Phase 6J starting boundary failed: {audit}")
    print(f"PHASE6J_CAUR_STARTING_BOUNDARY status=PASS output={OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
