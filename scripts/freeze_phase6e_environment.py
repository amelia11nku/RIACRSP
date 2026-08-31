#!/usr/bin/env python3
"""Verify and freeze the Phase 6C/6D boundary before Phase 6E modeling."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6e/environment"
PHASE6C_FREEZE_HASH = "695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def run(name: str, arguments: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    log = OUT / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "command": [sys.executable, *arguments],
        "return_code": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log": str(log.relative_to(ROOT)),
        "passed": completed.returncode == 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase6c = json.loads((ROOT / "outputs/phase6c/audit/dataset_freeze_record.json").read_text())
    phase6d = json.loads((ROOT / "outputs/phase6d/schema_freeze_record.json").read_text())
    phase6d_gate = json.loads((ROOT / "outputs/phase6d/audit/completion_gate.json").read_text())
    commands = [
        run("phase6d_revalidation", [
            "scripts/validate_phase6d_csg.py", "--sample-size", "30", "--workers", "4",
            "--progress-every", "30", "--output", "outputs/phase6e/environment/csg_revalidation",
        ]),
        run("historical_pytest", ["-m", "pytest", "-q"]),
    ]
    revalidation = json.loads((OUT / "csg_revalidation/csg_validation_summary.json").read_text())
    csg_source_hashes = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "rcias_clgri/csg").glob("*.py"))
    }
    checks = {
        "phase6c_dataset_hash_verified": phase6c["freeze_hash"] == PHASE6C_FREEZE_HASH,
        "phase6d_schema_hash_verified": phase6d["csg_schema_sha256"] == digest(ROOT / "configs/csg_v1_schema.json"),
        "phase6d_source_hashes_verified": phase6d["csg_source_sha256"] == csg_source_hashes,
        "phase6d_all_gates_passed": bool(phase6d["all_acceptance_gates_passed"]) and bool(phase6d_gate["PHASE6D_COMPLETE"]),
        "representative_csg_revalidation_passed": revalidation["completed_state_count"] == 30 and bool(revalidation["all_structural_checks_passed"]),
        "historical_tests_passed": all(command["passed"] for command in commands),
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    worktree = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    payload = {
        "schema": "phase6e-environment-freeze-v1",
        "status": "FROZEN",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "worktree_status_at_freeze": worktree,
        "phase6c_dataset_version": "phase6c-v1",
        "phase6c_dataset_freeze_hash": phase6c["freeze_hash"],
        "phase6d_csg_version": phase6d["csg_schema_version"],
        "phase6d_schema_sha256": phase6d["csg_schema_sha256"],
        "phase6d_freeze_record_sha256": digest(ROOT / "outputs/phase6d/schema_freeze_record.json"),
        "checks": checks,
        "commands": commands,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": gpu,
            "torch_geometric_available": find_spec("torch_geometric") is not None,
            "selected_graph_framework": "native PyTorch relation-aware heterogeneous tensors",
            "framework_decision": "PyG is not installed; native PyTorch avoids an unnecessary dependency and preserves all CSG edge features.",
        },
    }
    (OUT / "freeze_record.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PHASE6E_FROZEN_BOUNDARY_VERIFIED")


if __name__ == "__main__":
    main()
