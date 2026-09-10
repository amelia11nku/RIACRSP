import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs/frozen_2o_baselines/registry.json"
AUDIT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit/comparator_reuse_audit.json"


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def test_frozen_2o_registry_has_complete_three_seed_contracts():
    registry = load(REGISTRY)
    assert registry["status"] == "FROZEN_CANONICAL_THREE_SEED_DEVELOPMENT"
    assert registry["no_favorable_reruns"] is True
    assert len(registry["entries"]) == 4
    for entry in registry["entries"]:
        assert entry["status"] == "FROZEN_CANONICAL"
        assert len(entry["instance_ids"]) == 18
        assert entry["seed_list"] == [746101, 746102, 746103]
        assert entry["budget_formula"] == "2 * instance.num_operations seconds"
        assert entry["initialization_accounting"] == "inside wall-clock budget"
        assert entry["decoder_accounting"] == "all decoder calls inside wall-clock budget"
        assert entry["source_files"]
        assert entry["algorithm_config_hashes"]
        assert entry["shared_source_files"]
        assert entry["raw_result_paths"]
        assert entry["aggregate_summary_path"]
        assert entry["feasibility_evidence"] == {"rate": 1.0, "run_count": 54}
        assert entry["regression_integrity_result"] == "PASS_PHASE6P_DEVELOPMENT_AUDIT"

        summary_path = ROOT / entry["aggregate_summary_path"]
        manifest_path = ROOT / entry["run_manifest_path"]
        assert digest(summary_path) == entry["aggregate_summary_sha256"]
        assert digest(manifest_path) == entry["run_manifest_sha256"]
        summary = load(summary_path)
        manifest = load(manifest_path)
        assert summary["status"] == "FROZEN_CANONICAL"
        assert summary["run_count"] == 54
        assert summary["seeds"] == [746101, 746102, 746103]
        assert manifest["status"] == "FROZEN_CANONICAL"
        assert len(manifest["runs"]) == 54
