#!/usr/bin/env python3
"""Repair the frozen R11 summary's omitted decoder-seconds projection.

The immutable per-run payloads already contain this field.  This compatibility
step changes only the derived CSV consumed by the frozen finalizer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from scripts.run_phase6i_mr_r11_validation import (
    OUT,
    build_tasks,
    valid_result,
    verify_unlock,
)


SUMMARY_PATH = OUT / "validation_run_summary.csv"
AUDIT_PATH = OUT / "summary_compatibility_audit.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def add_decoder_seconds(summary: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    if len(summary) != len(records):
        raise RuntimeError("summary and immutable payload counts differ")
    result = summary.copy()
    decoder_seconds: list[float] = []
    for row, record in zip(result.itertuples(index=False), records):
        seed_matches = (
            (record["seed"] is None and pd.isna(row.seed))
            or (record["seed"] is not None and int(record["seed"]) == int(row.seed))
        )
        if not (
            record["method"] == row.method
            and record["instance_id"] == row.instance_id
            and seed_matches
        ):
            raise RuntimeError("summary row order or identity differs from frozen task order")
        decoder_seconds.append(float(record["runtime_components"]["decoder_seconds"]))
    if "decoder_seconds" in result.columns:
        existing = result["decoder_seconds"].to_numpy(dtype=float)
        if not pd.Series(existing).equals(pd.Series(decoder_seconds, dtype=float)):
            raise RuntimeError("existing decoder_seconds differs from immutable payloads")
        return result
    position = result.columns.get_loc("repair_seconds") + 1
    result.insert(position, "decoder_seconds", decoder_seconds)
    return result


def main() -> int:
    config, _artifact, artifact_hash, freeze_hash = verify_unlock()
    tasks = build_tasks(config)
    records = [valid_result(task, artifact_hash, freeze_hash) for task in tasks]
    if any(record is None for record in records):
        raise RuntimeError("all 288 frozen R11 payloads must pass integrity checks")
    typed_records = [record for record in records if record is not None]
    summary = pd.read_csv(SUMMARY_PATH)
    before_hash = digest(SUMMARY_PATH)
    repaired = add_decoder_seconds(summary, typed_records)
    temporary = SUMMARY_PATH.with_name(SUMMARY_PATH.name + f".tmp.{os.getpid()}")
    repaired.to_csv(temporary, index=False)
    temporary.replace(SUMMARY_PATH)
    audit = {
        "schema": "phase6i-mr-r11-summary-compatibility-audit-v1",
        "status": "PASS_REPAIRED_DERIVED_SUMMARY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cause": "frozen runner summary projection omitted decoder_seconds although all immutable per-run payloads contain it",
        "scientific_result_payloads_modified": False,
        "validated_payloads": len(typed_records),
        "source_field_present": sum(
            "decoder_seconds" in record["runtime_components"] for record in typed_records
        ),
        "summary_sha256_before": before_hash,
        "summary_sha256_after": digest(SUMMARY_PATH),
        "selected_artifact_sha256": artifact_hash,
        "artifact_freeze_sha256": freeze_hash,
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
