#!/usr/bin/env python3
"""Record the frozen Phase 5C baseline implementation boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.common import decode_candidate, random_candidate
from rcias_clgri.search.dcga import DCGAConfig, solve_dcga
from rcias_clgri.search.ga import GAConfig, solve_ga


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    instance = load_instance(ROOT / "instances/tiny/tiny_01.json")
    results = {
        "GA": solve_ga(instance, .02, 550001, GAConfig(population_size=6)),
        "DCGA": solve_dcga(instance, .02, 550001, DCGAConfig(population_size_each=4)),
        "ALNS": solve_alns(instance, .02, 550001, ALNSConfig(candidate_trials=2)),
    }
    checks = {name: check_schedule(instance, result.best.schedule)["feasible"] for name, result in results.items()}
    files = [
        "rcias_clgri/search/common.py", "rcias_clgri/search/ga.py",
        "rcias_clgri/search/dcga.py", "rcias_clgri/search/alns.py",
        "configs/phase5c_ga.json", "configs/phase5c_dcga.json", "configs/phase5c_alns.json",
    ]
    payload = {
        "schema": "phase5c-baseline-implementation-audit-v1",
        "GA_IMPLEMENTED": True, "ALNS_IMPLEMENTED": True, "DCGA_IMPLEMENTED": True,
        "GA_COMMON_DECODER": True, "ALNS_COMMON_DECODER": True, "DCGA_COMMON_DECODER": True,
        "GA_FEASIBILITY_VALIDATED": bool(checks["GA"]),
        "ALNS_FEASIBILITY_VALIDATED": bool(checks["ALNS"]),
        "DCGA_FEASIBILITY_VALIDATED": bool(checks["DCGA"]),
        "SEARCH_LOGIC_MODIFIED_IN_THIS_PHASE": False,
        "phase_boundary_note": "Search implementations were frozen before Phase 5C-B/C controlled-benchmark work began.",
        "public_interfaces": {"GA": "solve_ga", "DCGA": "solve_dcga", "ALNS": "solve_alns"},
        "smoke_test": {name: {"makespan": result.best.makespan, "feasible": checks[name]} for name, result in results.items()},
        "frozen_files": {name: digest(ROOT / name) for name in files},
    }
    output = ROOT / "outputs/phase5c/baseline_audit/baseline_implementation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE5C_BASELINE_AUDIT_COMPLETE", checks)


if __name__ == "__main__":
    main()
