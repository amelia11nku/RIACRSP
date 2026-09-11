import json
from pathlib import Path

import pytest

from rcias_ngas.governance.dataset_roles import (
    AccessDenied,
    DatasetRegistry,
    append_exposure,
    sha256_file,
    validate_manifest,
    validate_training_records,
    verify_exposure_ledger,
)
from scripts.build_ngas_a17ar_registry import build_registry, serialized


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / 'configs/dataset_role_registry.json'


def registry():
    return DatasetRegistry(REGISTRY_PATH, ROOT)


def row_for(role):
    return next(row for row in registry().payload['instances'] if row['role'] == role)


def request(row, role=None):
    return {
        'instance_id': row['instance_id'],
        'content_sha256': row['content_sha256'],
        'dataset_role': role or row['role'],
    }


def test_registry_is_deterministic_complete_and_revision_visible():
    payload = json.loads(REGISTRY_PATH.read_text())
    assert REGISTRY_PATH.read_text() == serialized()
    assert payload['revision'] == 1
    assert payload['family_counts']['RCIAS_CB1_CORE45'] == 45
    assert payload['family_counts']['RCIAS_CB1_R12'] == 18
    assert payload['family_counts']['RCIAS_CB1_R13'] == 18
    assert payload['family_counts']['RCIAS_CB1_R14'] == 18
    assert len(build_registry()['instances']) == len(payload['instances'])


def test_r12_is_permanently_development_exposed():
    r12 = [row for row in registry().payload['instances']
           if row['family_id'] == 'RCIAS_CB1_R12']
    assert len(r12) == 18
    assert {row['role'] for row in r12} == {'DEVELOPMENT_EXPOSED'}
    with pytest.raises(AccessDenied, match='not independent'):
        registry().authorize(request(r12[0]), 'independent_test',
                             allow_locked_evaluation=True)
    for purpose in ('training', 'model_selection'):
        with pytest.raises(AccessDenied, match='forbidden'):
            registry().authorize(request(r12[0]), purpose)


@pytest.mark.parametrize('family', ['RCIAS_CB1_R13', 'RCIAS_CB1_R14'])
@pytest.mark.parametrize('purpose', ['training', 'model_selection'])
def test_locked_eval_instances_fail_before_training_or_selection(family, purpose):
    row = next(row for row in registry().payload['instances']
               if row['family_id'] == family)
    with pytest.raises(AccessDenied):
        validate_manifest([request(row)], registry(), purpose)


@pytest.mark.parametrize('purpose', ['training', 'model_selection'])
def test_core45_is_never_training_or_model_selection_data(purpose):
    core = next(row for row in registry().payload['instances']
                if row['family_id'] == 'RCIAS_CB1_CORE45')
    assert core['role'] == 'EXTERNAL_BASELINE_ONLY'
    with pytest.raises(AccessDenied):
        validate_manifest([request(core)], registry(), purpose)


def test_filename_rename_and_exact_copy_cannot_bypass_hash_guard(tmp_path):
    locked = row_for('LOCKED_FINAL_EVAL')
    copied = tmp_path / 'harmless_training_name.json'
    copied.write_bytes((ROOT / locked['relative_path']).read_bytes())
    assert sha256_file(copied) == locked['content_sha256']
    disguised = {
        'instance_id': 'RENAMED_TRAINING_INSTANCE',
        'content_sha256': sha256_file(copied),
        'dataset_role': 'TRAIN',
    }
    with pytest.raises(AccessDenied, match='disagrees'):
        registry().authorize(disguised, 'training')


def test_conflicting_id_and_content_hash_are_rejected():
    train = row_for('TRAIN')
    locked = row_for('LOCKED_GENERALIZATION_EVAL')
    disguised = request(train)
    disguised['content_sha256'] = locked['content_sha256']
    with pytest.raises(AccessDenied, match='different registry rows'):
        registry().authorize(disguised, 'training')


def test_future_manifest_requires_explicit_registered_role():
    train = row_for('TRAIN')
    incomplete = request(train)
    incomplete.pop('dataset_role')
    with pytest.raises(AccessDenied, match='omits dataset role'):
        validate_manifest([incomplete], registry(), 'training')


def test_audit_only_states_are_excluded_from_c1_v2_training():
    audit = row_for('AUDIT_ONLY')
    record = {**request(audit), 'audit_only': True}
    with pytest.raises(AccessDenied, match='audit_only=false'):
        validate_training_records([record], registry())


def test_governed_training_record_is_accepted():
    train = row_for('TRAIN')
    record = {**request(train), 'audit_only': False}
    assert validate_training_records([record], registry()) == [record]


def test_exposure_ledger_is_hash_chained_and_tamper_evident(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    first = append_exposure(path, {'phase': 'test', 'instance_id': 'a', 'permitted': True})
    second = append_exposure(path, {'phase': 'test', 'instance_id': 'b', 'permitted': False})
    entries = verify_exposure_ledger(path)
    assert entries[1]['previous_entry_sha256'] == first['entry_sha256']
    assert entries[1]['entry_sha256'] == second['entry_sha256']
    path.write_text(path.read_text().replace('"instance_id":"a"', '"instance_id":"x"'))
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_exposure_ledger(path)
