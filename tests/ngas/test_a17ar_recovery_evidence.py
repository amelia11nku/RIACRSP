import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports'
ARTIFACTS = ROOT / 'artifacts/ngas_a17ar'


def test_preflight_keeps_r13_r14_locked_and_core45_reserved():
    payload = json.loads((ARTIFACTS / 'preflight.json').read_text())
    assert payload['a16r']['terminal_decision'] == 'NGAS_A1_6R_PASS_REVALIDATED'
    assert payload['locked_evaluation']['R13'] == {
        'role': 'LOCKED_FINAL_EVAL', 'instances': 18,
        'performance_accessed': False,
    }
    assert payload['locked_evaluation']['R14'] == {
        'role': 'LOCKED_GENERALIZATION_EVAL', 'instances': 18,
        'performance_accessed': False,
    }
    assert set(payload['split_metadata']['reserved_from_training_and_model_selection']) == {
        'RCIAS_CB1_CORE45', 'RCIAS_CB1_R13', 'RCIAS_CB1_R14',
    }


def test_r12_exposure_reconstruction_matches_frozen_evidence():
    payload = json.loads((REPORTS / 'ngas_a17ar_r12_exposure_audit.json').read_text())
    assert payload['status'] == 'PASS'
    assert payload['c1_training']['instances'] == 18
    assert payload['c1_training']['states'] == 72
    assert payload['c1_training']['joint_action_labels'] == 6465
    assert payload['comparison'] == {
        'a16_invalid_runs': 54,
        'a16r_comparator_runs': 216,
        'a16r_ngas_runs': 54,
        'instances': 18,
        'seeds': [746101, 746102, 746103],
    }
    assert payload['overlap']['instance_content_hash_percent'] == 100.0
    assert payload['overlap']['instance_seed_percent'] == 100.0


def test_state_overlap_is_conservatively_classified():
    with (REPORTS / 'ngas_a17ar_state_overlap.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert {row['classification'] for row in rows} == {
        'SAME_INSTANCE_SEED_NON_IDENTICAL_STATE'}
    assert {int(row['seed']) for row in rows} == {746102}
    assert all(row['training_state_fingerprint'] != row['a16r_state_fingerprint']
               for row in rows)


def test_comparator_exposure_table_is_complete_and_explicit():
    with (REPORTS / 'ngas_a17ar_comparator_exposure.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row['algorithm'] for row in rows} == {
        'GA', 'DCGA', 'DABC', 'LG_HGA_2O', 'ALNS', 'PHASE6H',
        'PHASE6N_TOP1', 'NGAS_C1',
    }
    tuned = {row['algorithm'] for row in rows if row['r12_specific_tuning'] == 'True'}
    assert tuned == {'PHASE6N_TOP1', 'NGAS_C1'}


def test_terminology_audit_reports_no_prohibited_r12_claims():
    text = (REPORTS / 'ngas_a17ar_terminology_audit.md').read_text()
    assert 'No prohibited R12 independent/unseen/final-test wording was found' in text
