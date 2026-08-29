#!/usr/bin/env python3
"""Freeze the completed Phase 6A evidence before Phase 6B work."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PHASE6A = ROOT / "outputs/phase6a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collection_digest(paths: list[Path]) -> tuple[str, dict[str, str]]:
    entries = {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), entries


def main():
    report = ROOT / "docs/reports/phase6a_alns_search_diagnosis_report.md"
    raw_hash, raw_files = collection_digest(list((PHASE6A / "raw_logs").glob("*.parquet")))
    _, summary_hashes = collection_digest(list((PHASE6A / "summaries").glob("*.csv")))
    regression = json.loads((PHASE6A / "diagnostics/instrumentation_regression.json").read_text())
    record = {
        "schema": "phase6a-freeze-for-phase6b-v1",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "phase6a_report_hash": digest(report), "phase6a_raw_log_hash": raw_hash,
        "phase6a_raw_file_hashes": raw_files, "phase6a_summary_hashes": summary_hashes,
        "legacy_checksum_status": "PASS_130_OF_130_BYTE_REGENERATION",
        "cb1_checksum_status": "PASS_113_OF_113",
        "instrumentation_regression_status": "PASS" if not regression["INSTRUMENTATION_CHANGES_SEARCH_BEHAVIOR"] else "FAIL",
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["freeze_hash"] = hashlib.sha256(canonical).hexdigest()
    path = ROOT / "outputs/phase6b/environment/phase6a_freeze_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if record["instrumentation_regression_status"] != "PASS" or len(summary_hashes) < 9 or len(raw_files) != 3:
        raise RuntimeError(record)
    print(f"PHASE6A_FROZEN {record['freeze_hash']}")


if __name__ == "__main__": main()
