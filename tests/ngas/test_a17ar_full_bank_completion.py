import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'


def test_full_bank_completion_audit_passes_every_frozen_boundary():
    audit = json.loads((OUT / 'audit/full_bank_completion_audit.json').read_text())
    assert audit['status'] == 'PASS'
    assert all(audit['checks'].values())
    assert audit['completed_states'] == 36
    assert audit['completed_actions'] == 12_775
    assert audit['dataset_origin_counts'] == {
        'CLEAN_NON_R12_DEVELOPMENT': 18,
        'R12_DEVELOPMENT_EXPOSED': 18,
    }
    assert audit['boundaries']['R13'] == 'LOCKED_FINAL_EVAL_NO_ACCESS'
    assert audit['boundaries']['R14'] == 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS'
    assert audit['boundaries']['RCIAS_CB1_CORE45'] == (
        'EXTERNAL_BASELINE_ONLY_NO_ACCESS')
    assert audit['boundaries']['production_solver_changed'] is False
    assert audit['boundaries']['gurobi_run'] is False


def test_full_bank_raw_manifest_hashes_every_formal_state():
    manifest = json.loads(
        (OUT / 'diagnostics/full_bank_raw_manifest.json').read_text())
    assert manifest['completed_states'] == len(manifest['files']) == 36
    assert manifest['completed_actions'] == 12_775
    for relative_path, expected_sha256 in manifest['files'].items():
        path = ROOT / relative_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
