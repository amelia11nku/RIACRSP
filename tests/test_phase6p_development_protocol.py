import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/phase6p_adaptive_portfolio_v1/development/protocol.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_phase6p_development_protocol_is_frozen_and_complete():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "FROZEN_BEFORE_P3_SOLVER_OUTCOMES"
    assert protocol["development_seeds"] == [746101, 746102, 746103]
    assert protocol["task_count"] == 270
    assert protocol["budget_formula"] == "2 * instance.num_operations seconds"
    assert protocol["nominal_total_budget_seconds"] == 64800
    assert protocol["execution_concurrency"] == 1
    assert protocol["result_provenance"] == "NEW_RUN"
    assert protocol["gurobi"] is False
    assert protocol["r13_accessed"] is False and protocol["r14_accessed"] is False
    assert all(digest(ROOT / path) == expected for path, expected in protocol["source_hashes"].items())
    assert all(digest(ROOT / path) == expected for path, expected in protocol["artifact_hashes"].items())
