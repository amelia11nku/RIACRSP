import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
TERMINAL = 'NGAS_A1_7AR_PASS_CONTAMINATION_RECOVERED'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_decision_passes_and_preserves_locked_boundaries():
    decision = json.loads((OUT / 'final_decision.json').read_text())
    assert decision['terminal_classification'] == TERMINAL
    assert all(decision['checks'].values())
    assert decision['contamination_recovery']['R12_role'] == 'DEVELOPMENT_EXPOSED'
    assert decision['contamination_recovery']['final_C1_states'] == 72
    assert decision['contamination_recovery']['final_C1_joint_action_labels'] == 6465
    assert decision['future_data'] == {
        'collected_states': 540,
        'reserved_id_and_hash_disjoint': True,
        'train_instances': 81,
        'train_states': 405,
        'validation_instances': 27,
        'validation_states': 135,
    }
    assert decision['boundaries'] == {
        'C1_retrained': False,
        'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
        'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
        'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_ACCESS',
        'adaptive_refresh_implemented': False,
        'adaptive_trial_racing_implemented': False,
        'gurobi_run': False,
        'production_solver_changed': False,
    }
    assert decision['recommendation']['automatic_start'] is False


def test_result_manifest_hashes_key_final_evidence():
    manifest = json.loads((OUT / 'result_manifest.json').read_text())
    assert manifest['terminal_classification'] == TERMINAL
    assert manifest['artifact_count'] == len(manifest['files'])
    assert manifest['artifact_count'] == 327
    assert manifest['excluded_self_path'] == (
        'outputs/ngas_a1/trajectory_utility_a17ar_v1/result_manifest.json')

    key_paths = (
        'artifacts/dataset_exposure_ledger.jsonl',
        'configs/dataset_role_registry.json',
        'artifacts/ngas_a17ar/trajectory_protocol_manifest.json',
        'outputs/ngas_a1/trajectory_utility_a17ar_v1/final_decision.json',
        'outputs/ngas_a1/trajectory_utility_a17ar_v1/audit/collection_completion_audit.json',
        'outputs/ngas_a1/trajectory_utility_a17ar_v1/audit/full_bank_completion_audit.json',
        'outputs/ngas_a1/trajectory_utility_a17ar_v1/audit/prior_staleness_completion_audit.json',
        'reports/figures/ngas_a17ar/figure_qa.json',
        'reports/ngas_a17ar_final_report.md',
    )
    for relative_path in key_paths:
        path = ROOT / relative_path
        assert path.is_file()
        assert manifest['files'][relative_path]['sha256'] == sha256(path)
        assert manifest['files'][relative_path]['bytes'] == path.stat().st_size


def test_figure_qa_and_recorded_full_regression_pass():
    figure_qa = json.loads(
        (ROOT / 'reports/figures/ngas_a17ar/figure_qa.json').read_text())
    regression = json.loads(
        (OUT / 'audit/final_regression_test.json').read_text())
    assert figure_qa['status'] == 'PASS'
    assert all(figure_qa['checks'].values())
    assert len(figure_qa['figures']) == 5
    assert regression['status'] == 'PASS'
    assert regression['tests_failed'] == 0
    assert regression['tests_passed'] >= 542
