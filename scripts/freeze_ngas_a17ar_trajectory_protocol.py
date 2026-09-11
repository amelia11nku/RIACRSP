#!/usr/bin/env python3
"""Freeze the clean A1.7A-R trajectory split before formal collection."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.governance.dataset_roles import (
    DatasetRegistry, sha256_file, validate_manifest,
)


CONFIG = ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml'
REGISTRY_PATH = ROOT / 'configs/dataset_role_registry.json'
OUTPUT = ROOT / 'artifacts/ngas_a17ar/trajectory_protocol_manifest.json'
REPORT = ROOT / 'reports/ngas_a17ar_trajectory_protocol.md'
SCALES = ('S', 'M', 'L')
CFS = ('CF1', 'CF2', 'CF3')
RIS = ('RI1', 'RI2', 'RI3')
TIS = ('TI1', 'TI2', 'TI3')


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_rows(config: dict) -> list[dict]:
    path = ROOT / config['source_instance_manifest']
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 405:
        raise RuntimeError('Expected the frozen 405-instance training distribution')
    return rows


def select_rows(config: dict, registry: DatasetRegistry) -> tuple[list[dict], list[dict]]:
    source = _source_rows(config)
    lookup = {(row['training_split'], row['scale'], row['CF_level'], row['RI_level'],
               row['TI_level'], row['replicate']): row for row in source}
    train = []
    for scale_index, scale in enumerate(SCALES):
        for cf_index, cf in enumerate(CFS):
            for ri_index, ri in enumerate(RIS):
                for ti_index, ti in enumerate(TIS):
                    replicate = f'R0{1 + (scale_index + cf_index + ri_index + ti_index) % 3}'
                    train.append(lookup[('TRAIN', scale, cf, ri, ti, replicate)])
    validation = []
    for scale_index, scale in enumerate(SCALES):
        for cf_index, cf in enumerate(CFS):
            for ri_index, ri in enumerate(RIS):
                ti = TIS[(scale_index + cf_index + ri_index) % 3]
                validation.append(lookup[
                    ('TRAIN_VALIDATION', scale, cf, ri, ti, 'R04')])
    if len(train) != config['split']['train']['expected_instances'] \
            or len(validation) != config['split']['validation']['expected_instances']:
        raise RuntimeError('Clean trajectory split count mismatch')

    def governed(rows: list[dict], role: str, purpose: str) -> list[dict]:
        requests = [{
            'instance_id': row['instance_id'],
            'content_sha256': row['sha256'],
            'dataset_role': role,
        } for row in rows]
        validate_manifest(requests, registry, purpose)
        return rows

    return governed(train, 'TRAIN', 'training'), governed(
        validation, 'VALIDATION', 'model_selection')


def _selected_record(row: dict, role: str, trajectory_seed: int) -> dict:
    path = ROOT / 'instances/controlled/RCIAS-CB1-TRAIN' / row['relative_path']
    payload = json.loads(path.read_text())
    if len(payload['operations']) <= 0:
        raise RuntimeError(f'Invalid operation count for {row["instance_id"]}')
    return {
        'instance_id': row['instance_id'], 'dataset_role': role,
        'audit_only': False,
        'relative_path': relative(path), 'content_sha256': row['sha256'],
        'scale': row['scale'], 'CF_level': row['CF_level'],
        'RI_level': row['RI_level'], 'TI_level': row['TI_level'],
        'replicate': row['replicate'], 'base_structure': row['base_structure'],
        'base_generation_seed': int(row['base_generation_seed']),
        'final_generation_seed': int(row['final_generation_seed']),
        'trajectory_seed': trajectory_seed,
        'num_operations': len(payload['operations']),
        'budget_seconds': 2 * len(payload['operations']),
    }


def build_manifest() -> dict:
    config = json.loads(CONFIG.read_text())
    registry = DatasetRegistry(REGISTRY_PATH, ROOT)
    train_rows, validation_rows = select_rows(config, registry)
    selected = []
    namespace = int(config['trajectory']['seed_namespace'])
    for index, (role, row) in enumerate(
            [('TRAIN', row) for row in sorted(train_rows, key=lambda item: item['instance_id'])]
            + [('VALIDATION', row) for row in sorted(validation_rows, key=lambda item: item['instance_id'])]):
        selected.append(_selected_record(row, role, namespace + index))
    train = [row for row in selected if row['dataset_role'] == 'TRAIN']
    validation = [row for row in selected if row['dataset_role'] == 'VALIDATION']
    reserved = [row for row in registry.payload['instances'] if row['family_id'] in config['reserved_families']]
    selected_ids = {row['instance_id'] for row in selected}
    selected_hashes = {row['content_sha256'] for row in selected}
    reserved_ids = {row['instance_id'] for row in reserved}
    reserved_hashes = {row['content_sha256'] for row in reserved}
    checks = {
        'train_count': len(train) == 81,
        'validation_count': len(validation) == 27,
        'selected_ids_unique': len(selected_ids) == len(selected),
        'selected_hashes_unique': len(selected_hashes) == len(selected),
        'train_validation_id_disjoint': not (
            {row['instance_id'] for row in train} & {row['instance_id'] for row in validation}),
        'train_validation_hash_disjoint': not (
            {row['content_sha256'] for row in train}
            & {row['content_sha256'] for row in validation}),
        'reserved_id_disjoint': not (selected_ids & reserved_ids),
        'reserved_content_hash_disjoint': not (selected_hashes & reserved_hashes),
        'train_all_81_factor_cells': len({
            (row['scale'], row['CF_level'], row['RI_level'], row['TI_level'])
            for row in train}) == 81,
        'validation_balanced_scale_cf_ri_ti': all(
            Counter(row[key] for row in validation) == Counter({value: 9 for value in values})
            for key, values in (('scale', SCALES), ('CF_level', CFS),
                                ('RI_level', RIS), ('TI_level', TIS))),
        'state_target_540': len(selected) * len(config['trajectory']['capture_fractions']) == 540,
        'production_checkpoint_hash': sha256_file(
            ROOT / config['production_solver']['checkpoint_path'])
            == config['production_solver']['checkpoint_sha256'],
        'candidate_trials_is_trial_count_8': (
            config['production_solver']['candidate_trials'] == 8
            and config['production_solver']['search']['candidate_trials'] == 8),
        'fixed_refresh_20': (
            config['production_solver']['search']['refresh']['fixed_horizon'] == 20),
        'r13_r14_core45_not_selected': not selected_ids & {
            row['instance_id'] for row in reserved
            if row['family_id'] in {'RCIAS_CB1_R13', 'RCIAS_CB1_R14', 'RCIAS_CB1_CORE45'}},
    }
    if not all(checks.values()):
        raise RuntimeError(f'Trajectory protocol checks failed: {checks}')
    sources = [
        CONFIG,
        REGISTRY_PATH,
        ROOT / config['source_instance_manifest'],
        ROOT / 'scripts/run_ngas_a17ar_trajectory_collection.py',
        ROOT / 'scripts/smoke_ngas_a17ar_trajectory_collection.py',
        ROOT / 'scripts/recover_ngas_a17ar_stale_lock.py',
        ROOT / 'rcias_ngas/governance/dataset_roles.py',
        ROOT / 'rcias_ngas/search/ngas_solver.py',
        ROOT / 'rcias_ngas/bank/ngas_bank_v1.py',
        ROOT / 'rcias_ngas/runtime/production_refresh.py',
        ROOT / 'rcias_ngas/critic/inference.py',
    ]
    source_hashes = {relative(path): sha256_file(path) for path in sources}
    full_bank_tasks = []
    for role, values in (('TRAIN', train), ('VALIDATION', validation)):
        for scale in SCALES:
            for cf in CFS:
                candidates = sorted(
                    (row for row in values if row['scale'] == scale and row['CF_level'] == cf),
                    key=lambda row: row['instance_id'])
                full_bank_tasks.append({
                    'instance_id': candidates[0]['instance_id'],
                    'dataset_role': role, 'capture_fraction': 0.5,
                })
    candidate_bank_identity = {
        'bank_source_sha256': source_hashes['rcias_ngas/bank/ngas_bank_v1.py'],
        'rules_per_size_before_deduplication': 24,
        'sizes': ['small', 'medium', 'large'],
        'repairs': 5,
    }
    return {
        'schema': 'ngas-a17ar-trajectory-protocol-manifest-v1',
        'revision': config['revision'],
        'supersedes_protocol_sha256': config['supersedes_protocol_sha256'],
        'supersession_reason': config['supersession_reason'],
        'status': 'FROZEN_BEFORE_FORMAL_COLLECTION',
        'freeze_source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config_path': relative(CONFIG), 'config_sha256': sha256_file(CONFIG),
        'registry_path': relative(REGISTRY_PATH),
        'registry_sha256': sha256_file(REGISTRY_PATH),
        'checkpoint_path': config['production_solver']['checkpoint_path'],
        'checkpoint_sha256': config['production_solver']['checkpoint_sha256'],
        'source_hashes': source_hashes,
        'candidate_bank_identity': candidate_bank_identity,
        'candidate_bank_hash': canonical_hash(candidate_bank_identity),
        'checks': checks,
        'reserved_family_counts': dict(sorted(Counter(
            row['family_id'] for row in reserved).items())),
        'instances': selected,
        'train_instances': [row['instance_id'] for row in train],
        'validation_instances': [row['instance_id'] for row in validation],
        'full_bank_audit_tasks': full_bank_tasks,
        'expected_runs': len(selected),
        'expected_clean_states': 540,
        'capture_fractions': config['trajectory']['capture_fractions'],
        'R12': 'DEVELOPMENT_EXPOSED_AUDIT_ONLY',
        'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
        'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
        'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_TRAINING_OR_MODEL_SELECTION',
        'production_solver_changed': False,
        'train_c1_v2': False,
    }


def main() -> None:
    if subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError(
            'Freeze A1.7A-R trajectory protocol only from a clean committed worktree')
    manifest = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    train = [row for row in manifest['instances'] if row['dataset_role'] == 'TRAIN']
    validation = [row for row in manifest['instances'] if row['dataset_role'] == 'VALIDATION']
    nominal_seconds = sum(row['budget_seconds'] for row in manifest['instances'])
    REPORT.write_text(f'''# NGAS A1.7A-R clean trajectory protocol

Protocol revision **{manifest['revision']}** is frozen before formal collection. It supersedes rejected revision 1 (`{manifest['supersedes_protocol_sha256']}`). It does not train C1-v2 or change the production solver.

- TRAIN: **{len(train)} instances**, one balanced selection across all 81 scale×CF×RI×TI cells.
- VALIDATION: **{len(validation)} instances**, balanced across S/M/L, CF1/2/3, RI1/2/3, and TI1/2/3.
- Each instance contributes five snapshots at normalized budget fractions `{manifest['capture_fractions']}` for **540 expected clean states**.
- Every trajectory uses the frozen `PERSISTENT_FIXED_REFRESH` C1 solver, refresh interval 20, five repairs, and `candidate_trials=8` stochastic realizations per selected joint action.
- Budget: `2 * |O|` seconds per trajectory; nominal serial budget is {nominal_seconds / 3600:.2f} hours before setup and final audit work.
- Candidate-bank hash: `{manifest['candidate_bank_hash']}`.
- R12 is development-exposed and audit-only. R13/R14 remain locked with no solver access.
- RCIAS-CB1-CORE45 is external-baseline-only and excluded from both training and model selection.

Instance IDs and file-content SHA256 values are mutually disjoint between TRAIN and VALIDATION and from R12, R13, R14, and RCIAS-CB1-CORE45. The full per-instance list, generation seeds, trajectory seeds, budgets, checkpoint hash, collection-code hashes, and audit subset are in `{relative(OUTPUT)}`.
''')
    print(json.dumps({
        'status': manifest['status'], 'instances': len(manifest['instances']),
        'states': manifest['expected_clean_states'],
        'manifest_sha256': sha256_file(OUTPUT),
        'candidate_bank_hash': manifest['candidate_bank_hash'],
    }))


if __name__ == '__main__':
    main()
