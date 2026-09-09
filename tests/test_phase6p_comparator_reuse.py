import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs/frozen_2o_baselines/registry.json"
AUDIT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit/comparator_reuse_audit.json"


def load(path):
    return json.loads(path.read_text())


def test_phase6p_comparator_audit_precedes_execution():
    audit = load(AUDIT)
    assert audit["status"] == "COMPLETE_BEFORE_COMPARATOR_EXECUTION"
    assert audit["comparators_may_start"] is True
    assert audit["r13_accessed"] is False and audit["r14_accessed"] is False
    assert {entry["algorithm_id"]: entry["action"] for entry in audit["entries"]} == {
        "ALNS": "RERUN_REQUIRED",
        "PHASE6H": "RERUN_REQUIRED",
        "PHASE6N_TOP1": "RERUN_REQUIRED",
        "LG_HGA_2O": "RERUN_REQUIRED",
    }


def test_frozen_2o_registry_has_complete_pending_contracts():
    registry = load(REGISTRY)
    assert registry["status"] == "INITIALIZED_ALL_REQUIRED_RUNS_PENDING"
    assert registry["no_favorable_reruns"] is True
    assert len(registry["entries"]) == 4
    for entry in registry["entries"]:
        assert entry["status"] == "RERUN_REQUIRED"
        assert len(entry["instance_ids"]) == 18
        assert entry["seed_list"] == [746101, 746102, 746103]
        assert entry["budget_formula"] == "2 * instance.num_operations seconds"
        assert entry["initialization_accounting"] == "inside wall-clock budget"
        assert entry["decoder_accounting"] == "all decoder calls inside wall-clock budget"
        assert entry["source_files"]
        assert entry["shared_source_files"]
        assert entry["raw_result_paths"]
        assert entry["aggregate_summary_path"]
