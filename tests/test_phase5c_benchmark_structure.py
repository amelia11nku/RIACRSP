from pathlib import Path

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.loader import load_instance


ROOT = Path(__file__).resolve().parents[1]


def test_audit_metrics_are_bounded_on_canonical_instance():
    path = ROOT / "instances/canonical/RCIAS-2.0/hurink/vdata/HU_V_la01.json"
    metrics = benchmark_metrics(load_instance(path))
    for name in (
        "R_full_op",
        "R_high_op",
        "F_route_mean",
        "F_cap_mean",
        "R_full_island",
    ):
        assert 0.0 <= metrics[name] <= 1.0
    assert metrics["number_of_operations"] > 0


def test_audit_is_deterministic():
    path = ROOT / "instances/canonical/RCIAS-2.0/brandimarte/BR_Mk01.json"
    assert benchmark_metrics(load_instance(path)) == benchmark_metrics(load_instance(path))
