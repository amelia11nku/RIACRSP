from __future__ import annotations

import json
from pathlib import Path

from rcias_clgri.data.generation import deterministic_json_text
from rcias_clgri.training import TrainingInstanceFactory

ROOT = Path(__file__).resolve().parents[1]


def _factory():
    config = json.loads((ROOT / "configs/phase3_training.json").read_text(encoding="utf-8"))
    return TrainingInstanceFactory(
        config["curriculum_levels"],
        ROOT / "instances/canonical/RCIAS-2.0/manifest.csv",
    )


def test_training_instance_factory_is_deterministic_and_independent():
    factory = _factory()
    first = factory.sample_raw(12345, "S")
    second = factory.sample_raw(12345, "S")
    assert deterministic_json_text(first) == deterministic_json_text(second)
    assert first["meta"]["canonical_operation_records_used"] is False
    assert first["meta"]["generator"] == "phase3-independent-synthetic-distribution"


def test_training_levels_respect_ranges_and_l_uses_manifest_quantiles():
    factory = _factory()
    for level, seed in (("S", 11), ("M", 12), ("L", 13)):
        instance = factory.sample(seed, level)
        spec = factory.levels[level]
        assert spec.products[0] <= len(instance.products) <= spec.products[1]
        assert spec.islands[0] <= len(instance.islands) <= spec.islands[1]
        assert instance.metadata["curriculum_level"] == level
    assert factory.levels["L"].products == (10, 30)
    assert factory.levels["L"].total_operations == (50, 300)
