from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rcias_clgri.data.loader import load_instance, load_instance_dict
from rcias_clgri.data.validator import InstanceValidationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "instances/tiny/tiny_01.json",
        "instances/tiny/tiny_02.json",
        "instances/tiny/tiny_03.json",
        "instances/canonical/RCIAS-2.0/brandimarte/BR_Mk01.json",
        "instances/canonical/RCIAS-2.0/hurink/edata/HU_E_la01.json",
    ],
)
def test_all_generated_instances_load(relative):
    instance = load_instance(ROOT / relative)
    assert instance.schema == "RCIAS-2.0"
    assert instance.nodes == ("WH",) + instance.islands


def test_processing_keys_must_equal_eligibility():
    raw = json.loads((ROOT / "instances/tiny/tiny_02.json").read_text(encoding="utf-8"))
    broken = copy.deepcopy(raw)
    op_id = broken["sets"]["operations"][0]
    broken["operations"][op_id]["processing_time"].pop(
        broken["operations"][op_id]["eligible_islands"][0]
    )
    with pytest.raises(InstanceValidationError, match="exactly equal"):
        load_instance_dict(broken)


def test_legacy_schema_is_rejected_without_fallback():
    with pytest.raises(InstanceValidationError, match="legacy"):
        load_instance_dict({"meta": {"schema": "RAIS-1.0"}})
