#!/usr/bin/env python3
"""Freeze accepted RCIAS-CB1 files after the Stage-A validation gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB1 = ROOT / "instances/controlled/RCIAS-CB1"
OUT = ROOT / "outputs/phase5c/controlled_benchmark_audit"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    coverage = json.loads((OUT / "coverage_diagnostics.json").read_text())
    required = [coverage["all_108_loadable"], coverage["all_108_feasible"],
                coverage["core_cells_balanced"], coverage["dev_cells_balanced"],
                coverage["sensitivity_cells_balanced"], coverage["sensitivity_pairing_verified"],
                not coverage["core_acceptance_failures"]]
    if not all(required):
        raise RuntimeError(f"Stage-A validation gate not satisfied: {coverage}")
    files = sorted([*CB1.glob("dev/*.json"), *CB1.glob("core/*.json"), *CB1.glob("sensitivity/*.json")])
    files += [CB1 / "manifests/generation_spec.json", CB1 / "manifests/dev_manifest.csv",
              CB1 / "manifests/core_manifest.csv", CB1 / "manifests/sensitivity_manifest.csv",
              CB1 / "manifests/benchmark_manifest.csv"]
    if len([path for path in files if path.suffix == ".json" and path.parent.name in {"dev", "core", "sensitivity"}]) != 108:
        raise RuntimeError("expected exactly 108 instance JSON files")
    lines = [f"{digest(path)}  {path.relative_to(CB1)}" for path in files]
    checksum_path = CB1 / "manifests/checksums.sha256"
    checksum_path.write_text("\n".join(lines) + "\n")
    record = {
        "schema": "rcias-cb1-freeze-v1", "benchmark": "RCIAS-CB1",
        "controlled_instances": 108, "dev_instances": 18, "core_instances": 45,
        "sensitivity_instances": 45, "formal_new_test_instances": 90,
        "checksums_file_sha256": digest(checksum_path), "files_frozen": len(files),
        "coverage_diagnostics_sha256": digest(OUT / "coverage_diagnostics.json"),
        "legacy_manifest_sha256": digest(ROOT / "instances/canonical/RCIAS-2.0/manifest.json"),
        "test_core_and_sensitivity_immutable": True,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["freeze_hash"] = hashlib.sha256(canonical).hexdigest()
    (OUT / "freeze_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("RCIAS_CB1_FROZEN", record["freeze_hash"], "files", len(files))


if __name__ == "__main__": main()
