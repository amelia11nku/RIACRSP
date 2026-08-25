from __future__ import annotations

import csv
import json
from pathlib import Path

from generate_fjsp_reconfigurable import build_instance
from rcias_clgri.data.canonical import (
    assert_has_incomparable_pair,
    discover_public_sources,
    generate_public_instance,
    sha256_file,
    verify_public_processing_preservation,
)
from rcias_clgri.data.generation import deterministic_json_text
from rcias_clgri.data.loader import load_instance

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "FJSP-benchmark-main"
CANONICAL = ROOT / "instances" / "canonical" / "RCIAS-2.0"


def _config():
    return json.loads((CANONICAL / "generation_config.json").read_text(encoding="utf-8"))


def test_public_processing_time_preservation():
    for source in discover_public_sources(SOURCE_ROOT):
        raw = json.loads((CANONICAL / source.output_relative).read_text(encoding="utf-8"))
        from generate_fjsp_reconfigurable import parse_fjsp
        verify_public_processing_preservation(parse_fjsp(source.source_path), raw)


def test_public_machine_eligibility_preservation():
    # The same strict comparison checks ordered eligibility separately from times.
    test_public_processing_time_preservation()


def test_two_operation_product_has_no_forced_chain():
    base = {"n_jobs": 1, "n_machines": 2, "jobs": [[[(1, 2)], [(2, 3)]]]}
    raw = build_instance(base, seed=1, family="unit", generation_config=_config())
    assert raw["products"]["J1"]["precedence"] == []


def test_generated_dag_is_acyclic():
    # Strict loading performs a topological sort and rejects every cycle.
    for source in discover_public_sources(SOURCE_ROOT):
        load_instance(CANONICAL / source.output_relative)


def test_generated_dag_has_incomparable_pair():
    for source in discover_public_sources(SOURCE_ROOT):
        instance = load_instance(CANONICAL / source.output_relative)
        for product in instance.product_data.values():
            if len(product.operations) >= 2:
                assert_has_incomparable_pair(product.operations, product.precedence)


def test_canonical_generation_is_reproducible():
    source = discover_public_sources(SOURCE_ROOT)[0]
    first, _ = generate_public_instance(source, _config())
    second, _ = generate_public_instance(source, _config())
    expected = (CANONICAL / source.output_relative).read_bytes()
    assert deterministic_json_text(first).encode("utf-8") == deterministic_json_text(second).encode("utf-8")
    assert deterministic_json_text(first).encode("utf-8") == expected


def test_manifest_and_checksums_cover_130_valid_instances():
    manifest = json.loads((CANONICAL / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["instance_count"] == len(manifest["instances"]) == 130
    assert all(record["valid"] for record in manifest["instances"])
    checksums = (CANONICAL / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksums) == 130
    for line in checksums:
        digest, relative = line.split("  ", 1)
        assert sha256_file(CANONICAL / relative) == digest
    with (CANONICAL / "validation_report.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 130
    assert all(row["valid"] == "True" for row in rows)
