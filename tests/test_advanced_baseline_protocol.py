from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from rcias_clgri.search.dabc import DABCConfig
from rcias_clgri.search.lghga import LGHGAConfig


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_manifest_is_frozen_and_record_count_is_exact():
    manifest = json.loads(
        (ROOT / "configs/baselines/advanced_formal_manifest.json").read_text()
    )
    instances = ROOT / manifest["instance_manifest"]
    checksums = ROOT / manifest["instance_checksums"]
    seeds = ROOT / manifest["seed_manifest"]
    assert _sha256(instances) == manifest["instance_manifest_sha256"]
    assert _sha256(checksums) == manifest["instance_checksums_sha256"]
    assert _sha256(seeds) == manifest["seed_manifest_sha256"]
    with instances.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    seed_values = json.loads(seeds.read_text())["seeds"]
    assert len(rows) == manifest["instance_count"] == 45
    assert len(seed_values) == manifest["runs_per_instance"] == 10
    assert manifest["expected_records"] == 45 * 10 * len(manifest["methods"])


def test_knowledge_instances_are_hash_verified_and_disjoint_from_core():
    manifest = json.loads(
        (ROOT / "configs/baselines/lghga_knowledge_training_manifest.json").read_text()
    )
    training_root = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
    training_ids = set()
    training_hashes = set()
    cells = set()
    for row in manifest["instances"]:
        path = training_root / row["relative_path"]
        assert _sha256(path) == row["sha256"]
        training_ids.add(row["instance_id"])
        training_hashes.add(row["sha256"])
        cells.add((row["scale"], row["CF_level"]))
    assert len(training_ids) == manifest["instance_count"] == 9
    assert len(cells) == 9

    core_root = ROOT / "instances/controlled/RCIAS-CB1"
    with (core_root / "manifests/core_manifest.csv").open(newline="") as handle:
        core_rows = list(csv.DictReader(handle))
    core_ids = {row["instance_id"] for row in core_rows}
    core_hashes = {_sha256(core_root / row["relative_path"]) for row in core_rows}
    assert not training_ids & core_ids
    assert not training_hashes & core_hashes


def test_frozen_json_configs_construct_algorithm_configs():
    dabc_raw = json.loads(
        (ROOT / "configs/baselines/dabc_riacrsp.json").read_text()
    )
    lghga_raw = json.loads(
        (ROOT / "configs/baselines/lghga_riacrsp.json").read_text()
    )
    dabc = DABCConfig(**{
        key: value for key, value in dabc_raw.items()
        if key in DABCConfig.__dataclass_fields__
    })
    lghga = LGHGAConfig(**{
        key: value for key, value in lghga_raw.items()
        if key in LGHGAConfig.__dataclass_fields__
    })
    assert dabc.source_clipping_mode == "shadow"
    assert lghga.local_search_threshold_pct == 50
    assert lghga.knowledge_generation_runs == 20


def test_implementation_manifest_matches_every_formal_source_file():
    manifest = json.loads(
        (ROOT / "configs/baselines/advanced_implementation_manifest.json").read_text()
    )
    assert manifest["status"] == "FROZEN_BEFORE_FORMAL_DATA_COLLECTION"
    assert len(manifest["files"]) >= 10
    for record in manifest["files"]:
        assert _sha256(ROOT / record["path"]) == record["sha256"]
