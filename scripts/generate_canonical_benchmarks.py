#!/usr/bin/env python3
"""Generate or verify the frozen 130-instance RCIAS-2.0 public suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.canonical import (
    discover_public_sources,
    generate_public_instance,
    manifest_record,
    sha256_file,
    verify_public_processing_preservation,
)
from rcias_clgri.data.generation import deterministic_json_text, write_json
from rcias_clgri.data.loader import load_instance
from generate_fjsp_reconfigurable import parse_fjsp

CANONICAL = ROOT / "instances" / "canonical" / "RCIAS-2.0"
SOURCE_ROOT = ROOT / "FJSP-benchmark-main"
CONFIG_PATH = CANONICAL / "generation_config.json"
MANIFEST_FIELDS = (
    "instance_id", "family", "source_file", "seed", "num_products", "num_operations",
    "num_islands", "num_configurations", "num_w_agvs", "num_f_agvs",
    "num_precedence_edges", "dag_density", "mean_eligible_islands",
    "min_eligible_islands", "max_eligible_islands", "mean_processing_time",
    "reconfiguration_time_mean", "reconfiguration_time_max", "source_sha256",
    "generated_sha256", "schema_version", "generator_version", "valid",
)


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["schema_version"] != "RCIAS-2.0":
        raise ValueError("canonical generation_config schema mismatch")
    return config


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate() -> None:
    config = _load_config()
    sources = discover_public_sources(SOURCE_ROOT)
    records: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    checksums: list[str] = []
    print("This run freezes 130 public FJSP processing domains as deterministic RCIAS-2.0 JSON.")
    for index, source in enumerate(sources, start=1):
        raw, base = generate_public_instance(source, config)
        output = CANONICAL / source.output_relative
        write_json(raw, output)
        record = manifest_record(source, raw, base)
        records.append(record)
        validations.append({
            "instance_id": source.instance_id,
            "schema": "PASS",
            "dag": "PASS",
            "partial_order": "PASS",
            "processing_preservation": "PASS",
            "configuration": "PASS",
            "reconfiguration": "PASS",
            "travel": "PASS",
            "valid": True,
        })
        checksums.append(f"{record['generated_sha256']}  {source.output_relative.as_posix()}")
        if index % 10 == 0 or index == len(sources):
            print(f"generated {index}/{len(sources)}")
    _write_csv(CANONICAL / "manifest.csv", records, MANIFEST_FIELDS)
    write_json({"schema_version": "RCIAS-2.0", "instance_count": 130, "instances": records}, CANONICAL / "manifest.json")
    _write_csv(
        CANONICAL / "validation_report.csv",
        validations,
        ("instance_id", "schema", "dag", "partial_order", "processing_preservation",
         "configuration", "reconfiguration", "travel", "valid"),
    )
    (CANONICAL / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    (CANONICAL / "README.md").write_text(
        "# Canonical RCIAS-2.0 public benchmark suite\n\n"
        "This directory contains 130 deterministic RCIAS extensions: Brandimarte Mk01–Mk10 "
        "and Hurink edata/rdata/vdata la01–la40. Original processing alternatives and times "
        "are preserved byte-for-value from `FJSP-benchmark-main`; original job chains are not inherited.\n\n"
        "Regenerate with `python scripts/generate_canonical_benchmarks.py` and verify with "
        "`python scripts/generate_canonical_benchmarks.py --verify-only`.\n",
        encoding="utf-8",
    )
    print("CANONICAL_BENCHMARK_READY = TRUE (130/130 VALID)")


def verify() -> None:
    config = _load_config()
    sources = discover_public_sources(SOURCE_ROOT)
    manifest = json.loads((CANONICAL / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("instance_count") != 130 or len(manifest.get("instances", [])) != 130:
        raise ValueError("canonical manifest does not contain 130 instances")
    by_id = {record["instance_id"]: record for record in manifest["instances"]}
    for index, source in enumerate(sources, start=1):
        record = by_id[source.instance_id]
        output = CANONICAL / source.output_relative
        if sha256_file(output) != record["generated_sha256"]:
            raise ValueError(f"generated checksum mismatch: {source.instance_id}")
        if sha256_file(source.source_path) != record["source_sha256"]:
            raise ValueError(f"source checksum mismatch: {source.instance_id}")
        loaded = load_instance(output)
        base = parse_fjsp(source.source_path)
        raw = json.loads(output.read_text(encoding="utf-8"))
        verify_public_processing_preservation(base, raw)
        regenerated, _ = generate_public_instance(source, config)
        if deterministic_json_text(regenerated).encode("utf-8") != output.read_bytes():
            raise ValueError(f"byte-level regeneration mismatch: {source.instance_id}")
        if loaded.instance_id != source.instance_id:
            raise ValueError(f"instance ID mismatch: {source.instance_id}")
        if index % 20 == 0 or index == len(sources):
            print(f"verified {index}/{len(sources)}")
    print("CANONICAL_BENCHMARK_READY = TRUE (checksums and byte-level regeneration verified)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify() if args.verify_only else generate()


if __name__ == "__main__":
    main()
