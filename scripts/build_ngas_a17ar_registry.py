#!/usr/bin/env python3
"""Build the authoritative NGAS A1.7A-R dataset-role registry."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.governance.dataset_roles import ROLES, sha256_file


OUTPUT = ROOT / 'configs/dataset_role_registry.json'
INSTANCE_ROOT = ROOT / 'instances'
SOURCE_MANIFESTS = (
    ROOT / 'instances/canonical/RCIAS-2.0/manifest.csv',
    ROOT / 'instances/controlled/RCIAS-CB1/manifests/core_manifest.csv',
    ROOT / 'instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv',
    ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_instance_manifest.csv',
)


def _classification(relative: Path) -> tuple[str, str, str]:
    value = relative.as_posix()
    if value.startswith('instances/canonical/RCIAS-2.0/'):
        return 'RCIAS_2_0_CANONICAL', 'EXTERNAL_BASELINE_ONLY', 'PRE_NGAS_CANONICAL_IMPORT'
    if value.startswith('instances/tiny/'):
        return 'TINY_UNIT_TEST', 'AUDIT_ONLY', 'UNIT_TEST_FIXTURES'
    if '/RCIAS-CB1/core/' in value:
        return 'RCIAS_CB1_CORE45', 'EXTERNAL_BASELINE_ONLY', 'PHASE5C_FORMAL_BASELINE'
    if '/RCIAS-CB1/sensitivity/' in value:
        return 'RCIAS_CB1_SENSITIVITY45', 'DEVELOPMENT_EXPOSED', 'PHASE5C_SENSITIVITY'
    if '/RCIAS-CB1/dev/' in value:
        return 'RCIAS_CB1_DEV18', 'DEVELOPMENT_EXPOSED', 'PHASE5C_DEVELOPMENT'
    if '/RCIAS-CB1-TRAIN/train/' in value:
        return 'RCIAS_CB1_TRAIN243', 'TRAIN', 'PHASE6B_TRAIN'
    if '/RCIAS-CB1-TRAIN/train_validation/' in value:
        return 'RCIAS_CB1_VALIDATION81', 'VALIDATION', 'PHASE6B_VALIDATION'
    if '/RCIAS-CB1-TRAIN/train_internal_holdout/' in value:
        return 'RCIAS_CB1_INTERNAL_HOLDOUT81', 'DEVELOPMENT_EXPOSED', 'PHASE6B_INTERNAL_HOLDOUT_ACCESSED'
    if '/RCIAS-CB1-TRAIN-R06/revision_holdout/' in value:
        return 'RCIAS_CB1_R06', 'DEVELOPMENT_EXPOSED', 'PHASE6F_REVISION_HOLDOUT_ACCESSED'
    if '/RCIAS-CB1-CAL/cal_fit/' in value:
        return 'RCIAS_CB1_R07', 'DEVELOPMENT_EXPOSED', 'PHASE6H_CALIBRATION_FIT'
    if '/RCIAS-CB1-CAL/cal_holdout/' in value:
        return 'RCIAS_CB1_R08', 'DEVELOPMENT_EXPOSED', 'PHASE6H_CALIBRATION_HOLDOUT_ACCESSED'
    if '/r09_live_rev_fit/' in value:
        return 'RCIAS_CB1_R09', 'DEVELOPMENT_EXPOSED', 'PHASE6I_MR_FIT'
    if '/r10_live_rev_select/' in value:
        return 'RCIAS_CB1_R10', 'DEVELOPMENT_EXPOSED', 'PHASE6I_MR_SELECTION_ACCESSED'
    if '/r11_live_rev_holdout/' in value:
        return 'RCIAS_CB1_R11', 'DEVELOPMENT_EXPOSED', 'PHASE6I_MR_HOLDOUT_ACCESSED'
    if '/r12_caur_fit/' in value:
        return 'RCIAS_CB1_R12', 'DEVELOPMENT_EXPOSED', 'PHASE6J_AND_NGAS_A1_DEVELOPMENT'
    if '/r13_caur_select/' in value:
        return 'RCIAS_CB1_R13', 'LOCKED_FINAL_EVAL', 'UNTOUCHED_LOCKED'
    if '/r14_caur_holdout/' in value:
        return 'RCIAS_CB1_R14', 'LOCKED_GENERALIZATION_EVAL', 'UNTOUCHED_LOCKED'
    raise ValueError(f'No dataset-role classification for {relative}')


def _operations(role: str) -> tuple[list[str], list[str]]:
    if role == 'TRAIN':
        return ['training', 'diagnostic', 'regression'], [
            'model_selection', 'independent_test', 'final_evaluation',
        ]
    if role == 'VALIDATION':
        return ['model_selection', 'validation', 'diagnostic', 'regression'], [
            'training', 'independent_test', 'final_evaluation',
        ]
    if role == 'DEVELOPMENT_EXPOSED':
        return ['diagnostic', 'regression', 'development_analysis'], [
            'independent_test', 'final_evaluation', 'generalization_claim',
        ]
    if role == 'AUDIT_ONLY':
        return ['audit', 'diagnostic'], ['training', 'model_selection', 'independent_test']
    if role == 'EXTERNAL_BASELINE_ONLY':
        return ['external_baseline_evaluation', 'audit'], [
            'training', 'model_selection', 'validation', 'architecture_selection',
        ]
    if role in {'LOCKED_FINAL_EVAL', 'LOCKED_GENERALIZATION_EVAL'}:
        return ['metadata_integrity_while_locked'], [
            'training', 'model_selection', 'validation', 'diagnostic',
            'performance_inspection', 'solver_execution_before_final_freeze',
        ]
    return ['metadata_integrity'], [
        'training', 'model_selection', 'validation', 'evaluation',
    ]


def build_registry() -> dict:
    caur_manifest_path = (
        ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/'
        'phase6j_instance_manifest.csv')
    with caur_manifest_path.open() as stream:
        caur_metadata = {row['relative_path']: row for row in csv.DictReader(stream)}
    rows = []
    for path in sorted(INSTANCE_ROOT.rglob('*.json')):
        if 'manifests' in path.parts or path.name in {'generation_config.json', 'manifest.json'}:
            continue
        relative = path.relative_to(ROOT)
        family, role, first_exposure = _classification(relative)
        locked_metadata = caur_metadata.get('/'.join(relative.parts[-2:])) \
            if role in {'LOCKED_FINAL_EVAL', 'LOCKED_GENERALIZATION_EVAL'} else None
        if locked_metadata is not None:
            metadata = {
                'instance_id': locked_metadata['instance_id'],
                'generator': 'RCIAS-CB1-native-v1 (manifest metadata only)',
                'seed': int(locked_metadata['final_generation_seed']),
                'scale': locked_metadata['scale'],
                'CF_level': locked_metadata['CF_level'],
                'RI_level': locked_metadata['RI_level'],
                'TI_level': locked_metadata['TI_level'],
            }
        else:
            payload = json.loads(path.read_text())
            metadata = payload.get('meta')
            if not isinstance(metadata, dict) or not metadata.get('instance_id'):
                continue
        allowed, forbidden = _operations(role)
        rows.append({
            'instance_id': metadata['instance_id'],
            'family_id': family,
            'relative_path': relative.as_posix(),
            'content_sha256': sha256_file(path),
            'role': role,
            'historical_roles': [role],
            'generation_provenance': metadata.get('generator', metadata.get('source', 'UNKNOWN')),
            'creation_seed': metadata.get('seed'),
            'scale': metadata.get('scale'),
            'CF_level': metadata.get('CF_level'),
            'RI_level': metadata.get('RI_level'),
            'TI_level': metadata.get('TI_level'),
            'first_exposure_stage': first_exposure,
            'allowed_operations': allowed,
            'forbidden_operations': forbidden,
        })
    counts = Counter(row['family_id'] for row in rows)
    expected = {
        'RCIAS_CB1_CORE45': 45,
        'RCIAS_CB1_TRAIN243': 243,
        'RCIAS_CB1_VALIDATION81': 81,
        'RCIAS_CB1_R12': 18,
        'RCIAS_CB1_R13': 18,
        'RCIAS_CB1_R14': 18,
    }
    for family, count in expected.items():
        if counts[family] != count:
            raise RuntimeError(f'{family} count mismatch: {counts[family]} != {count}')
    content_hashes = [row['content_sha256'] for row in rows]
    if len(content_hashes) != len(set(content_hashes)):
        raise RuntimeError('Duplicate instance content exists across registry families')
    return {
        'schema': 'ngas-dataset-role-registry-v1',
        'revision': 1,
        'revision_date': '2026-09-11',
        'authority': 'NGAS_A1_7A_R',
        'roles': sorted(ROLES),
        'policy': {
            'identity': 'instance_id_and_file_content_sha256',
            'unknown_instances': 'FAIL_CLOSED',
            'core45_training_or_model_selection': 'FORBIDDEN',
            'r12': 'PERMANENT_DEVELOPMENT_EXPOSED',
            'r13': 'LOCKED_FINAL_EVAL',
            'r14': 'LOCKED_GENERALIZATION_EVAL',
            'locked_metadata_access': 'HASH_AND_MANIFEST_INTEGRITY_ONLY',
        },
        'source_manifests': [
            {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256_file(path)}
            for path in SOURCE_MANIFESTS
        ],
        'family_counts': dict(sorted(counts.items())),
        'instances': sorted(rows, key=lambda row: row['instance_id']),
    }


def serialized() -> str:
    return json.dumps(build_registry(), indent=2, sort_keys=True) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    value = serialized()
    if args.verify:
        if not OUTPUT.exists() or OUTPUT.read_text() != value:
            raise RuntimeError('Dataset-role registry differs from deterministic reconstruction')
        print(json.dumps({'status': 'PASS', 'path': str(OUTPUT.relative_to(ROOT)),
                          'sha256': sha256_file(OUTPUT)}))
        return
    OUTPUT.write_text(value)
    print(json.dumps({'status': 'CREATED', 'path': str(OUTPUT.relative_to(ROOT)),
                      'sha256': sha256_file(OUTPUT),
                      'instances': len(build_registry()['instances'])}))


if __name__ == '__main__':
    main()
