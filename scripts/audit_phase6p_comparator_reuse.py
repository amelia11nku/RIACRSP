#!/usr/bin/env python3
"""Initialize the canonical 2|O| registry and audit Phase 6P comparators."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6p_adaptive_portfolio_v1.json"
PREREG = ROOT / "outputs/phase6p_adaptive_portfolio_v1/preregistration"
REGISTRY_ROOT = ROOT / "outputs/frozen_2o_baselines"
AUDIT_OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit"
REGISTRY_REPORT = ROOT / "docs/reports/frozen_2o_baseline_registry_report.md"
AUDIT_REPORT = ROOT / "docs/reports/phase6p_comparator_reuse_audit.md"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "UNKNOWN"


def environment() -> dict:
    machine_id = Path("/etc/machine-id").read_text().strip()
    cuda_available = torch.cuda.is_available()
    return {
        "schema": "frozen-2o-environment-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine_identifier": {
            "hostname": socket.gethostname(),
            "machine_id_sha256": hashlib.sha256(machine_id.encode()).hexdigest(),
        },
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "python": platform.python_version(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "project_environment": Path(sys.prefix).name,
        "python_executable": sys.executable,
        "conda_default_env_variable": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def source_hashes(paths: list[str]) -> dict[str, str]:
    return {path: digest(ROOT / path) for path in paths}


def main() -> None:
    if REGISTRY_ROOT.exists():
        raise RuntimeError("canonical 2|O| registry already exists; audit it instead of replacing it")
    config = load_json(CONFIG)
    instances = load_json(PREREG / "development_instance_manifest.json")
    if config["development"]["seeds"] != [746101, 746102, 746103]:
        raise RuntimeError("Phase 6P development seeds drifted")
    if len(instances["instances"]) != 18:
        raise RuntimeError("Phase 6P development instance scope drifted")
    for row in instances["instances"]:
        path = (
            ROOT / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14"
            / row["relative_path"]
        )
        if digest(path) != row["sha256"]:
            raise RuntimeError(f"development instance changed: {path}")

    for name in ("alns", "phase6h", "phase6n", "lg_hga_2o", "ga", "dcga", "dabc"):
        (REGISTRY_ROOT / name).mkdir(parents=True, exist_ok=True)
    environment_record = environment()
    atomic_json(REGISTRY_ROOT / "environment.json", environment_record)
    instance_record = {
        **instances,
        "schema": "frozen-2o-instance-manifest-v1",
        "status": "FROZEN_CANONICAL_SCOPE",
        "source_phase6p_manifest": str(
            (PREREG / "development_instance_manifest.json").relative_to(ROOT)
        ),
        "source_phase6p_manifest_sha256": digest(
            PREREG / "development_instance_manifest.json"
        ),
    }
    atomic_json(REGISTRY_ROOT / "instance_manifest.json", instance_record)

    shared = source_hashes([
        "rcias_clgri/search/common.py",
        "rcias_clgri/env/insertion_decoder.py",
        "rcias_clgri/env/feasibility.py",
        "rcias_clgri/heuristic/dispatching.py",
    ])
    checkpoint_manifest = PREREG / "phase6n_checkpoint_manifest.json"
    phase6h_policy = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
    lghga_manifest = ROOT / "outputs/baselines/lghga_kb_v2/models/model_manifest.json"
    entries = [
        {
            "algorithm_id": "ALNS",
            "algorithm_display_name": "ALNS-H1",
            "status": "RERUN_REQUIRED",
            "source_files": source_hashes(["rcias_clgri/search/alns.py"]),
            "model_checkpoint_hashes": {},
            "historical_evidence": [
                "outputs/phase6h_validation/validation_run_summary.csv",
                "outputs/phase6j_caur/r12_collection/source_runs/"
                "CB1_CAUR_S_CF1_RI2_TI2_R12__seed691201.json",
            ],
            "mismatches": [
                "no canonical frozen_2o registry result exists",
                "historical 2|O| ALNS validation used CAL R08 instances and seeds 671301-671305",
                "historical R12 collection used different seeds and a 15.25 s budget for 61 operations, not 2|O|=122 s",
            ],
            "raw_result_paths": ["outputs/frozen_2o_baselines/alns/runs"],
            "aggregate_summary_path": "outputs/frozen_2o_baselines/alns/summary.json",
        },
        {
            "algorithm_id": "PHASE6H",
            "algorithm_display_name": "Phase 6H CSG-NI",
            "status": "RERUN_REQUIRED",
            "source_files": source_hashes([
                "rcias_clgri/search/csgni.py",
                "rcias_clgri/ni/live_inference.py",
                "rcias_clgri/ni/live_policy.py",
            ]),
            "model_checkpoint_hashes": {
                "phase6h_policy": digest(phase6h_policy),
            },
            "historical_evidence": [
                "outputs/phase6h_validation/validation_run_summary.csv",
                "outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json",
            ],
            "mismatches": [
                "no canonical frozen_2o registry result exists",
                "Phase 6H validation used CAL R08 instances and seeds 671301-671305",
                "Phase 6I R11 evidence used different LIVE_REV instances and seeds 681401-681405",
            ],
            "raw_result_paths": ["outputs/frozen_2o_baselines/phase6h/runs"],
            "aggregate_summary_path": "outputs/frozen_2o_baselines/phase6h/summary.json",
        },
        {
            "algorithm_id": "PHASE6N_TOP1",
            "algorithm_display_name": "Phase 6N deterministic top-1",
            "status": "RERUN_REQUIRED",
            "source_files": source_hashes([
                "rcias_clgri/search/phase6p_adaptive.py",
                "rcias_clgri/ni/phase6p_live_inference.py",
            ]),
            "model_checkpoint_hashes": {
                "phase6n_checkpoint_manifest": digest(checkpoint_manifest),
            },
            "historical_evidence": [],
            "mismatches": [
                "no prior live deterministic top-1 solver result exists",
                "the comparator was first implemented after Phase 6P preregistration",
            ],
            "raw_result_paths": ["outputs/frozen_2o_baselines/phase6n/runs"],
            "aggregate_summary_path": "outputs/frozen_2o_baselines/phase6n/summary.json",
        },
        {
            "algorithm_id": "LG_HGA_2O",
            "algorithm_display_name": "LG_HGA-2O",
            "status": "RERUN_REQUIRED",
            "source_files": source_hashes([
                "rcias_clgri/search/lghga.py",
                "rcias_clgri/search/lghga_v2.py",
                "rcias_clgri/search/lghga_2o.py",
                "rcias_clgri/search/lghga_learning.py",
                "rcias_clgri/search/lghga_neighborhoods.py",
                "rcias_clgri/search/lghga_neighborhoods_v2.py",
            ]),
            "model_checkpoint_hashes": {
                "lghga_v2_model_manifest": digest(lghga_manifest),
            },
            "historical_evidence": [
                "outputs/phase6k_runtime_v1/audit/lghga_2o_real_clock_smoke.json"
            ],
            "mismatches": [
                "no canonical frozen_2o registry result exists",
                "the only 2|O| evidence is a 12 s tiny_01 smoke",
                "the smoke used seed 696101 rather than the Phase 6P development seeds",
            ],
            "raw_result_paths": ["outputs/frozen_2o_baselines/lg_hga_2o/runs"],
            "aggregate_summary_path": "outputs/frozen_2o_baselines/lg_hga_2o/summary.json",
        },
    ]
    common = {
        "source_commit": git("rev-parse", "HEAD"),
        "shared_source_files": shared,
        "config_sha256": digest(CONFIG),
        "instance_manifest_sha256": digest(REGISTRY_ROOT / "instance_manifest.json"),
        "instance_ids": [row["instance_id"] for row in instances["instances"]],
        "seed_list": config["development"]["seeds"],
        "budget_formula": "2 * instance.num_operations seconds",
        "initialization_accounting": "inside wall-clock budget",
        "decoder_accounting": "all decoder calls inside wall-clock budget",
        "machine_environment_sha256": digest(REGISTRY_ROOT / "environment.json"),
        "started_at_utc": None,
        "completed_at_utc": None,
        "feasibility_evidence": None,
        "regression_integrity_result": None,
        "supersedes": None,
        "superseded_by": None,
    }
    entries = [{**entry, **common} for entry in entries]
    registry = {
        "schema": "frozen-2o-baseline-registry-v1",
        "status": "INITIALIZED_ALL_REQUIRED_RUNS_PENDING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 6P R12 CAUR-FIT development and reusable future 2|O| comparators",
        "no_favorable_reruns": True,
        "valid_actions": [
            "REUSE_FROZEN_RESULT", "EXTEND_MISSING_SEEDS", "RERUN_REQUIRED"
        ],
        "entries": entries,
    }
    atomic_json(REGISTRY_ROOT / "registry.json", registry)

    audit_entries = []
    for entry in entries:
        audit_entries.append({
            "algorithm_id": entry["algorithm_id"],
            "action": "RERUN_REQUIRED",
            "prior_2o_run_exists": bool(entry["historical_evidence"]),
            "canonical_registry_run_exists": False,
            "instance_hashes_match": False,
            "algorithm_checkpoint_config_match": False,
            "seed_list_matches": False,
            "budget_accounting_matches": False,
            "machine_runtime_matches": None,
            "shared_decoder_search_invalidates_reuse": None,
            "historical_evidence": entry["historical_evidence"],
            "reasons": entry["mismatches"],
        })
    audit = {
        "schema": "phase6p-comparator-reuse-audit-v1",
        "status": "COMPLETE_BEFORE_COMPARATOR_EXECUTION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str((REGISTRY_ROOT / "registry.json").relative_to(ROOT)),
        "registry_sha256": digest(REGISTRY_ROOT / "registry.json"),
        "instance_manifest_sha256": digest(REGISTRY_ROOT / "instance_manifest.json"),
        "development_seeds": config["development"]["seeds"],
        "budget_formula": config["development"]["budget_seconds"],
        "entries": audit_entries,
        "comparators_may_start": True,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(AUDIT_OUT / "comparator_reuse_audit.json", audit)

    REGISTRY_REPORT.write_text("""# Frozen 2|O| baseline registry

The canonical registry has been initialized for the 18 frozen R12 CAUR-FIT instances, development seeds 746101–746103, and a wall-clock budget of `2 * num_operations` seconds with initialization and decoder work included. The machine, dependency, instance, config, checkpoint, and relevant source hashes are frozen under `outputs/frozen_2o_baselines/`.

No completed historical run satisfies the full canonical contract. All four Phase 6P comparators are currently `RERUN_REQUIRED`; entries remain pending until their complete three-seed results pass feasibility and integrity checks. Future phases must reuse a resulting `FROZEN_CANONICAL` entry when all ten compatibility conditions match, and may extend only missing seeds when the existing seed list is a strict valid subset.

Valid results may not be rerun based on whether a new method performs well or poorly. Superseded entries must remain preserved with an explicit reason and link.
""", encoding="utf-8")
    lines = [
        "# Phase 6P comparator reuse audit",
        "",
        "The audit completed before any development comparator launch. No canonical `outputs/frozen_2o_baselines/` registry existed beforehand.",
        "",
        "| Comparator | Action | Blocking mismatch |",
        "| --- | --- | --- |",
    ]
    for entry in audit_entries:
        lines.append(
            f"| {entry['algorithm_id']} | `RERUN_REQUIRED` | "
            + "; ".join(entry["reasons"])
            + " |"
        )
    lines.extend([
        "",
        "The existing ALNS R12 collection used different seeds and a non-2|O| collection budget. Phase 6H evidence uses CAL R08 or LIVE_REV R11 instances. Phase 6N top-1 has no prior live solver run. LG_HGA-2O has only a `tiny_01` real-clock smoke. None can be promoted by relabeling.",
        "",
        "All four comparators may now run once under the canonical development protocol. Their existing evidence remains preserved and will not be overwritten.",
    ])
    AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "actions": {entry["algorithm_id"]: entry["action"] for entry in audit_entries},
        "registry": str((REGISTRY_ROOT / "registry.json").relative_to(ROOT)),
    }))


if __name__ == "__main__":
    main()
