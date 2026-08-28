import copy
import csv
import json
from pathlib import Path

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.loader import load_instance, load_instance_dict
from rcias_clgri.instances.controlled_generator import (
    acceptance_failures, generate_candidate, scale_sensitivity_variant,
)

ROOT = Path(__file__).resolve().parents[1]
CB1 = ROOT / "instances/controlled/RCIAS-CB1"
SPEC = json.loads((CB1 / "manifests/generation_spec.json").read_text())


def _manifest(name="benchmark_manifest.csv"):
    with (CB1 / "manifests" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_suite_counts_and_factorial_balance():
    rows = _manifest()
    assert sum(row["suite"] == "DEV" for row in rows) == 18
    assert sum(row["suite"] == "CORE" for row in rows) == 45
    assert sum(row["suite"] == "SENS" for row in rows) == 45
    core = {(scale, cf): 0 for scale in ("S", "M", "L") for cf in ("CF1", "CF2", "CF3")}
    sensitivity = {(ri, ti): 0 for ri in ("RI1", "RI2", "RI3") for ti in ("TI1", "TI2", "TI3")}
    for row in rows:
        if row["suite"] == "CORE": core[(row["scale"], row["CF_level"])] += 1
        if row["suite"] == "SENS": sensitivity[(row["RI_level"], row["TI_level"])] += 1
    assert set(core.values()) == {5}
    assert set(sensitivity.values()) == {5}


def test_core_acceptance_gates_and_manifest_completeness():
    required = {"instance_id", "suite", "scale", "CF_level", "RI_level", "TI_level", "replicate",
                "base_seed", "accepted_attempt", "final_seed", "target_F_cap", "realized_F_cap",
                "target_F_route", "realized_F_route", "target_processing_CV", "realized_processing_CV",
                "target_RI", "realized_RI", "realized_W_TI", "realized_F_TI", "configuration_entropy",
                "R_full_op", "R_full_island", "relative_path"}
    for row in _manifest("core_manifest.csv"):
        assert required <= set(row)
        raw = json.loads((CB1 / row["relative_path"]).read_text())
        assert acceptance_failures(raw, row["scale"], row["CF_level"], SPEC) == []


def test_generation_is_deterministic_for_frozen_seed():
    row = _manifest("core_manifest.csv")[0]
    generated = generate_candidate(row["instance_id"], "CORE", row["scale"], row["CF_level"], int(row["final_seed"]), SPEC)
    saved = json.loads((CB1 / row["relative_path"]).read_text())
    assert generated == saved


def test_seed_ranges_are_isolated():
    rows = _manifest()
    bases = {suite: {int(row["base_seed"]) for row in rows if row["suite"] == suite} for suite in ("DEV", "CORE", "SENS")}
    assert not (bases["DEV"] & bases["CORE"] or bases["DEV"] & bases["SENS"] or bases["CORE"] & bases["SENS"])


def test_sensitivity_pairing_and_realized_levels():
    rows = _manifest("sensitivity_manifest.csv")
    for replicate in ("R01", "R02", "R03", "R04", "R05"):
        selected = [row for row in rows if row["replicate"] == replicate]
        assert len(selected) == 9
        instances = [load_instance(CB1 / row["relative_path"]) for row in selected]
        first = instances[0]
        for instance in instances[1:]:
            assert instance.products == first.products
            assert instance.operations == first.operations
            assert instance.predecessors == first.predecessors
            assert instance.processing_time == first.processing_time
            assert instance.island_data == first.island_data
            assert instance.distance == first.distance
        by_ri = {}
        by_ti = {}
        for row, instance in zip(selected, instances):
            metrics = benchmark_metrics(instance)
            by_ri.setdefault(row["RI_level"], []).append(metrics["RI"])
            by_ti.setdefault(row["TI_level"], []).append(metrics["W_transport_intensity"])
        assert max(by_ri["RI1"]) < min(by_ri["RI2"])
        assert max(by_ri["RI2"]) < min(by_ri["RI3"])
        assert max(by_ti["TI1"]) < min(by_ti["TI2"])
        assert max(by_ti["TI2"]) < min(by_ti["TI3"])


def test_all_instances_load_and_positive_times():
    for row in _manifest():
        instance = load_instance(CB1 / row["relative_path"])
        assert all(value > 0 for value in instance.processing_time.values())
