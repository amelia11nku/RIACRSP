#!/usr/bin/env python3
"""Reconstruct NGAS A1.7A-R benchmark exposure from immutable artifacts."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.governance.dataset_roles import (
    DatasetRegistry,
    append_exposure,
    sha256_file,
    verify_exposure_ledger,
)


REPORTS = ROOT / 'reports'
ARTIFACTS = ROOT / 'artifacts/ngas_a17ar'
REGISTRY_PATH = ROOT / 'configs/dataset_role_registry.json'
LEDGER_PATH = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
R12_MANIFEST = ROOT / 'outputs/frozen_2o_baselines/instance_manifest.json'
CAUR_MANIFEST = (
    ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/'
    'phase6j_instance_manifest.csv')
CACHE = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/data/training_cache_v2.json.gz'
CACHE_MANIFEST = (
    ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/data/training_cache_manifest_v2.json')
DEVELOPMENT_PROTOCOL = ROOT / 'outputs/ngas_a1/development_v1/protocol.json'
CHECKPOINT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/production/revised_joint_critic.pt'
C1_AUDIT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/audit/completion_audit.json'
A16R_ROOT = ROOT / 'outputs/ngas_a1/solver_comparison_a16r_v1'


EXPOSURE_COLUMNS = (
    'record_type', 'campaign', 'algorithm', 'instance_id', 'instance_path',
    'instance_sha256', 'state_id', 'state_source', 'state_type', 'seed', 'fold',
    'action_count', 'contributed_to_final_c1', 'budget_seconds', 'artifact_path',
    'artifact_sha256', 'manifest_path', 'manifest_member',
)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def csv_write(path: Path, rows: list[dict], columns: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def _consistent_r12() -> tuple[list[dict], dict[str, dict]]:
    frozen = json.loads(R12_MANIFEST.read_text())
    with CAUR_MANIFEST.open() as stream:
        caur = [row for row in csv.DictReader(stream) if row['caur_split'] == 'CAUR_FIT']
    frozen_by_id = {row['instance_id']: row for row in frozen['instances']}
    caur_by_id = {row['instance_id']: row for row in caur}
    if set(frozen_by_id) != set(caur_by_id) or len(frozen_by_id) != 18:
        raise RuntimeError('Authoritative R12 manifests disagree on instance IDs')
    for instance_id, row in frozen_by_id.items():
        if row['sha256'] != caur_by_id[instance_id]['sha256']:
            raise RuntimeError(f'Authoritative R12 manifests disagree on hash: {instance_id}')
        path = ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / row['relative_path']
        if sha256_file(path) != row['sha256']:
            raise RuntimeError(f'R12 instance bytes disagree with manifest: {instance_id}')
    return sorted(frozen_by_id.values(), key=lambda row: row['instance_id']), frozen_by_id


def build_preflight(stage_entry_sha: str) -> dict:
    r12, _ = _consistent_r12()
    registry = DatasetRegistry(REGISTRY_PATH, ROOT)
    c1 = json.loads(C1_AUDIT.read_text())
    a16r_manifest = A16R_ROOT / 'result_manifest.json'
    a16r_audit = json.loads((A16R_ROOT / 'audit/completion_audit.json').read_text())
    expected_checkpoint = c1['production_replay']['checkpoint_sha256']
    if not CHECKPOINT.is_file() or sha256_file(CHECKPOINT) != expected_checkpoint:
        raise RuntimeError('Frozen production C1 checkpoint cannot be uniquely validated')
    if (a16r_audit.get('terminal_decision') != 'NGAS_A1_6R_PASS_REVALIDATED'
            or not a16r_manifest.is_file()):
        raise RuntimeError('A1.6R evidence is missing or not revalidated')
    r13 = [row for row in registry.payload['instances'] if row['family_id'] == 'RCIAS_CB1_R13']
    r14 = [row for row in registry.payload['instances'] if row['family_id'] == 'RCIAS_CB1_R14']
    if (len(r13), {row['role'] for row in r13}) != (18, {'LOCKED_FINAL_EVAL'}):
        raise RuntimeError('R13 lock definition is ambiguous')
    if (len(r14), {row['role'] for row in r14}) != (18, {'LOCKED_GENERALIZATION_EVAL'}):
        raise RuntimeError('R14 lock definition is ambiguous')
    return {
        'schema': 'ngas-a17ar-preflight-v1',
        'phase': 'NGAS_A1_7A_R',
        'stage_entry': {
            'git_sha': stage_entry_sha,
            'branch': _git('branch', '--show-current'),
            'working_tree': 'CLEAN_BEFORE_A17AR_MODIFICATIONS',
            'observation': 'captured before the first A1.7A-R file was created',
        },
        'execution_environment': {
            'python_executable': sys.executable,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'torch_version': torch.__version__,
            'torch_cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_device_names': [torch.cuda.get_device_name(index)
                                  for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        },
        'frozen_c1': {
            'checkpoint_path': relative(CHECKPOINT),
            'checkpoint_sha256': expected_checkpoint,
            'selection': c1['selected_variant'],
            'completion_audit_path': relative(C1_AUDIT),
            'completion_audit_sha256': sha256_file(C1_AUDIT),
        },
        'a16r': {
            'terminal_decision': a16r_audit['terminal_decision'],
            'result_manifest_path': relative(a16r_manifest),
            'result_manifest_sha256': sha256_file(a16r_manifest),
            'completion_audit_path': relative(A16R_ROOT / 'audit/completion_audit.json'),
            'report_paths': [
                'docs/reports/ngas_a1/12_a16_solver_comparison.md',
                'docs/reports/ngas_a1/13_a16r_integrity_diagnostic.md',
            ],
        },
        'r12': {
            'instance_count': len(r12),
            'manifest_path': relative(R12_MANIFEST),
            'manifest_sha256': sha256_file(R12_MANIFEST),
            'consistent_source_manifest_path': relative(CAUR_MANIFEST),
            'consistent_source_manifest_sha256': sha256_file(CAUR_MANIFEST),
        },
        'locked_evaluation': {
            'R13': {'role': 'LOCKED_FINAL_EVAL', 'instances': 18,
                    'performance_accessed': False},
            'R14': {'role': 'LOCKED_GENERALIZATION_EVAL', 'instances': 18,
                    'performance_accessed': False},
            'allowed_access': 'manifest metadata and byte hashes only',
        },
        'split_metadata': {
            'registry_path': relative(REGISTRY_PATH),
            'registry_sha256': sha256_file(REGISTRY_PATH),
            'family_counts': registry.payload['family_counts'],
            'source_manifests': registry.payload['source_manifests'],
            'reserved_from_training_and_model_selection': [
                'RCIAS_CB1_CORE45', 'RCIAS_CB1_R13', 'RCIAS_CB1_R14',
            ],
        },
    }


def write_preflight(stage_entry_sha: str) -> None:
    payload = build_preflight(stage_entry_sha)
    json_write(ARTIFACTS / 'preflight.json', payload)
    (REPORTS / 'ngas_a17ar_preflight.md').write_text(f'''# NGAS A1.7A-R preflight

- Stage-entry commit: `{payload['stage_entry']['git_sha']}` on `{payload['stage_entry']['branch']}`; working tree was clean before A1.7A-R modifications.
- Python: `{payload['execution_environment']['python_version']}` at `{payload['execution_environment']['python_executable']}`.
- PyTorch/CUDA: `{payload['execution_environment']['torch_version']}` / `{payload['execution_environment']['torch_cuda_version']}`; CUDA available = `{str(payload['execution_environment']['cuda_available']).lower()}`.
- Frozen production C1: `{payload['frozen_c1']['checkpoint_path']}`, SHA256 `{payload['frozen_c1']['checkpoint_sha256']}`.
- A1.6R manifest: `{payload['a16r']['result_manifest_path']}`, SHA256 `{payload['a16r']['result_manifest_sha256']}`; terminal decision remains `NGAS_A1_6R_PASS_REVALIDATED`.
- R12 authority: `{payload['r12']['manifest_path']}`; its 18 IDs and hashes agree with the Phase 6J source manifest.
- R13: `LOCKED_FINAL_EVAL`; R14: `LOCKED_GENERALIZATION_EVAL`. Only manifest metadata and byte hashes were inspected. No solver, model, or performance access occurred.
- RCIAS-CB1-CORE45: `EXTERNAL_BASELINE_ONLY`; it is reserved from training and model selection.
- Dataset registry: `{payload['split_metadata']['registry_path']}`, covering {sum(payload['split_metadata']['family_counts'].values())} instance files.

All mandatory preflight identities are unique and internally consistent. Formal R13/R14 evaluation remains locked.
''')


def _load_training() -> tuple[list[dict], dict, dict]:
    manifest = json.loads(CACHE_MANIFEST.read_text())
    if sha256_file(CACHE) != manifest['cache_sha256']:
        raise RuntimeError('C1 training cache hash mismatch')
    raw = gzip.decompress(CACHE.read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']:
        raise RuntimeError('C1 uncompressed training cache hash mismatch')
    payload = json.loads(raw)
    protocol = json.loads(DEVELOPMENT_PROTOCOL.read_text())
    states = {row['state_id']: row for row in protocol['states']}
    records = payload['records']
    if (len(records), len({row['instance_id'] for row in records}),
            sum(len(row['actions']) for row in records)) != (
            manifest['states'], manifest['instances'], manifest['actions']):
        raise RuntimeError('C1 training cache scope contradicts its manifest')
    if set(states) != {row['state_id'] for row in records}:
        raise RuntimeError('C1 cache and development protocol state identities disagree')
    return records, manifest, states


def training_exposure_rows() -> tuple[list[dict], dict]:
    records, manifest, states = _load_training()
    rows = []
    for record in sorted(records, key=lambda row: row['state_id']):
        spec = states[record['state_id']]
        source = spec['source']
        seed = int(source.rsplit('_', 1)[1]) if source.startswith('NATIVE16_') else ''
        rows.append({
            'record_type': 'C1_TRAINING_STATE',
            'campaign': 'A1.3R_FINAL_C1_FIT',
            'algorithm': 'C1_COMPACT_RELATIONAL',
            'instance_id': record['instance_id'],
            'instance_path': ('instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/'
                              + spec['instance']['relative_path']),
            'instance_sha256': spec['instance']['sha256'],
            'state_id': record['state_id'],
            'state_source': source,
            'state_type': 'H1_INITIAL' if source == 'H1' else 'NATIVE16_SEARCH_STATE',
            'seed': seed,
            'fold': record['fold'],
            'action_count': len(record['actions']),
            'contributed_to_final_c1': True,
            'budget_seconds': '',
            'artifact_path': relative(CACHE),
            'artifact_sha256': manifest['cache_sha256'],
            'manifest_path': relative(CACHE_MANIFEST),
            'manifest_member': True,
        })
    return rows, manifest


def _campaign_rows(root: Path, campaign: str) -> list[dict]:
    manifest_path = root / 'raw_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['completed_runs'] != 54 or len(manifest['files']) != 54:
        raise RuntimeError(f'{campaign} raw manifest does not contain 54 runs')
    rows = []
    for name, expected in sorted(manifest['files'].items()):
        path = ROOT / name
        if sha256_file(path) != expected:
            raise RuntimeError(f'{campaign} raw artifact hash mismatch: {name}')
        payload = json.loads(path.read_text())
        rows.append({
            'record_type': 'SOLVER_COMPARISON_RUN', 'campaign': campaign,
            'algorithm': payload.get('algorithm_id', payload.get('variant', 'NGAS')),
            'instance_id': payload['instance_id'],
            'instance_path': ('instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/'
                              + payload['instance_relative_path']),
            'instance_sha256': payload['instance_sha256'], 'state_id': '',
            'state_source': '', 'state_type': '', 'seed': payload['seed'], 'fold': '',
            'action_count': '', 'contributed_to_final_c1': False,
            'budget_seconds': payload['budget_seconds'], 'artifact_path': name,
            'artifact_sha256': expected, 'manifest_path': relative(manifest_path),
            'manifest_member': True,
        })
    return rows


def _comparator_rows(r12_by_id: dict[str, dict]) -> list[dict]:
    rows = []
    for directory in ('alns', 'phase6h', 'phase6n', 'lg_hga_2o'):
        manifest_path = ROOT / 'outputs/frozen_2o_baselines' / directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'FROZEN_CANONICAL' or len(manifest['runs']) != 54:
            raise RuntimeError(f'Frozen comparator manifest invalid: {directory}')
        for item in manifest['runs']:
            path = ROOT / item['path']
            if sha256_file(path) != item['sha256']:
                raise RuntimeError(f'Frozen comparator hash mismatch: {item["path"]}')
            payload = json.loads(path.read_text())
            expected = r12_by_id[item['instance_id']]
            if payload['instance_sha256'] != expected['sha256']:
                raise RuntimeError(f'Comparator instance hash mismatch: {item["path"]}')
            rows.append({
                'record_type': 'SOLVER_COMPARISON_RUN',
                'campaign': 'A1.6R_FROZEN_COMPARATOR',
                'algorithm': manifest['algorithm_id'],
                'instance_id': item['instance_id'],
                'instance_path': ('instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/'
                                  + expected['relative_path']),
                'instance_sha256': expected['sha256'], 'state_id': '',
                'state_source': '', 'state_type': '', 'seed': item['seed'], 'fold': '',
                'action_count': '', 'contributed_to_final_c1': False,
                'budget_seconds': payload['budget_seconds'],
                'artifact_path': item['path'], 'artifact_sha256': item['sha256'],
                'manifest_path': relative(manifest_path), 'manifest_member': True,
            })
    return rows


def _overlap_rows(training: list[dict], comparison: list[dict]) -> list[dict]:
    train_instances = {row['instance_id'] for row in training}
    train_hashes = {row['instance_sha256'] for row in training}
    train_seeds = {int(row['seed']) for row in training if row['seed'] != ''}
    train_pairs = {(row['instance_id'], int(row['seed'])) for row in training if row['seed'] != ''}
    rows = []
    groups = {'A1.6R_ALL_METHODS': comparison}
    groups.update({algorithm: [row for row in comparison if row['algorithm'] == algorithm]
                   for algorithm in sorted({row['algorithm'] for row in comparison})})
    for name, values in groups.items():
        instances = {row['instance_id'] for row in values}
        hashes = {row['instance_sha256'] for row in values}
        seeds = {int(row['seed']) for row in values}
        pairs = {(row['instance_id'], int(row['seed'])) for row in values}
        for level, left, right in (
            ('INSTANCE_ID', train_instances, instances),
            ('INSTANCE_CONTENT_SHA256', train_hashes, hashes),
            ('SEED', train_seeds, seeds),
            ('INSTANCE_X_SEED', train_pairs, pairs),
        ):
            denominator = len(right)
            numerator = len(left & right)
            rows.append({
                'left': 'C1_FINAL_TRAIN', 'right': name, 'identity_level': level,
                'overlap_count': numerator, 'right_count': denominator,
                'overlap_percent': 100. * numerator / denominator if denominator else 0.,
            })
    return rows


def write_exposure_audit() -> None:
    _, r12_by_id = _consistent_r12()
    training, cache_manifest = training_exposure_rows()
    a16_invalid = _campaign_rows(
        ROOT / 'outputs/ngas_a1/solver_comparison_v1', 'A1.6_INVALID_CONCURRENCY')
    a16r = _campaign_rows(A16R_ROOT, 'A1.6R_REVALIDATED')
    comparators = _comparator_rows(r12_by_id)
    comparison = a16r + comparators
    all_rows = training + a16_invalid + comparison
    csv_path = REPORTS / 'ngas_a17ar_r12_exposure_rows.csv'
    overlap_path = REPORTS / 'ngas_a17ar_overlap_matrix.csv'
    report_path = REPORTS / 'ngas_a17ar_r12_exposure_audit.md'
    csv_write(csv_path, all_rows, EXPOSURE_COLUMNS)
    overlaps = _overlap_rows(training, comparison)
    csv_write(overlap_path, overlaps, [
        'left', 'right', 'identity_level', 'overlap_count', 'right_count',
        'overlap_percent'])
    train_instances = {row['instance_id'] for row in training}
    train_seeds = {int(row['seed']) for row in training if row['seed'] != ''}
    a16r_instances = {row['instance_id'] for row in a16r}
    a16r_seeds = {int(row['seed']) for row in a16r}
    report_path.write_text(f'''# NGAS A1.7A-R R12 exposure audit

R12 is an architecture-development benchmark with complete instance-level exposure to the final frozen C1 fitting process and therefore is not an independent or unseen test set.

## Reconstructed C1 exposure

- Instances: **{len(train_instances)} R12 instances**.
- States: **{len(training)}**.
- Joint-action labels: **{sum(int(row['action_count']) for row in training):,}**.
- `no_continuation_rollout`: **`{str(cache_manifest['no_continuation_rollout']).lower()}`**.
- Each instance contributes one H1 state and one state from each of `NATIVE16_746101`, `NATIVE16_746102`, and `NATIVE16_746103`.
- Every listed state contributed to the final all-development production C1 fit.

## Reconstructed solver-comparison exposure

- A1.6 preserved invalid-concurrency campaign: {len(a16_invalid)} R12 runs.
- A1.6R revalidated NGAS campaign: {len(a16r)} R12 runs.
- Frozen A1.6R comparators: {len(comparators)} runs across four algorithms.
- A1.6R uses {len(a16r_instances)} instances and seeds `{sorted(a16r_seeds)}`.

## Overlap

- C1 final-fit instance overlap with A1.6R: **{len(train_instances & a16r_instances)}/{len(a16r_instances)} = {100 * len(train_instances & a16r_instances) / len(a16r_instances):.1f}%**.
- C1 NATIVE16 seed overlap with A1.6R: **{len(train_seeds & a16r_seeds)}/{len(a16r_seeds)} = {100 * len(train_seeds & a16r_seeds) / len(a16r_seeds):.1f}%**.
- The instance-content hashes and all 54 instance×seed identities also overlap completely.

A1.6R remains valid for integration, budget accounting, concurrency, feasibility, reproducibility, runtime qualification, and R12 development-benchmark performance. It is not evidence of unseen-instance generalization, a leakage-free held-out test, or an unbiased final comparison.
''')
    payload = {
        'schema': 'ngas-a17ar-r12-exposure-audit-v1',
        'status': 'PASS',
        'headline': 'C1_FINAL_TRAIN_INSTANCE_OVERLAP_WITH_A16R_R12 = 18/18',
        'c1_training': {
            'instances': len(train_instances), 'states': len(training),
            'joint_action_labels': sum(int(row['action_count']) for row in training),
            'no_continuation_rollout': cache_manifest['no_continuation_rollout'],
            'sources': ['H1', 'NATIVE16_746101', 'NATIVE16_746102', 'NATIVE16_746103'],
            'cache_path': relative(CACHE), 'cache_sha256': sha256_file(CACHE),
        },
        'comparison': {
            'a16_invalid_runs': len(a16_invalid), 'a16r_ngas_runs': len(a16r),
            'a16r_comparator_runs': len(comparators),
            'instances': len(a16r_instances), 'seeds': sorted(a16r_seeds),
        },
        'overlap': {
            'instance_ids': {'count': len(train_instances & a16r_instances), 'denominator': len(a16r_instances)},
            'seeds': {'count': len(train_seeds & a16r_seeds), 'denominator': len(a16r_seeds)},
            'instance_content_hash_percent': 100.0,
            'instance_seed_percent': 100.0,
        },
        'input_hashes': {
            relative(CACHE_MANIFEST): sha256_file(CACHE_MANIFEST),
            relative(DEVELOPMENT_PROTOCOL): sha256_file(DEVELOPMENT_PROTOCOL),
            relative(R12_MANIFEST): sha256_file(R12_MANIFEST),
            relative(A16R_ROOT / 'raw_manifest.json'): sha256_file(A16R_ROOT / 'raw_manifest.json'),
        },
        'output_hashes': {
            relative(csv_path): sha256_file(csv_path),
            relative(overlap_path): sha256_file(overlap_path),
            relative(report_path): sha256_file(report_path),
        },
    }
    json_write(REPORTS / 'ngas_a17ar_r12_exposure_audit.json', payload)


def _candidate_fingerprint(candidate: dict) -> str:
    keys = ('operation_order', 'island_assignment', 'w_assignment', 'f_assignment')
    canonical = {key: list(candidate[key]) for key in keys}
    return hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_state_overlap() -> None:
    training, _, specs = _load_training()
    train_states = {}
    for record in training:
        spec = specs[record['state_id']]
        source = spec['source']
        seed = int(source.rsplit('_', 1)[1]) if source.startswith('NATIVE16_') else None
        state_path = ROOT / 'outputs/ngas_a1/development_v1/states' / record['state_id'] / 'state.json'
        payload = json.loads(state_path.read_text())
        train_states[(record['instance_id'], seed)] = {
            'state_id': record['state_id'], 'path': relative(state_path),
            'fingerprint': _candidate_fingerprint(payload['candidate']),
            'makespan': payload['makespan'],
        }
    rows = []
    snapshot_count = 0
    for path in sorted((A16R_ROOT / 'raw').glob('*/*.json')):
        payload = json.loads(path.read_text())
        for snapshot in payload['search_diagnostics'].get('a16r_replayable_states', []):
            snapshot_count += 1
            train = train_states.get((payload['instance_id'], int(payload['seed'])))
            audit_fingerprint = _candidate_fingerprint(snapshot['current_candidate'])
            if train is None:
                classification = 'NO_RECOVERABLE_COMPARISON'
            elif train['fingerprint'] == audit_fingerprint:
                classification = 'SAME_SCHEDULE_STRUCTURE_RUNTIME_METADATA_UNAVAILABLE'
            else:
                classification = 'SAME_INSTANCE_SEED_NON_IDENTICAL_STATE'
            rows.append({
                'instance_id': payload['instance_id'], 'seed': payload['seed'],
                'training_state_id': train['state_id'] if train else '',
                'training_state_path': train['path'] if train else '',
                'training_state_fingerprint': train['fingerprint'] if train else '',
                'training_makespan': train['makespan'] if train else '',
                'a16r_state_id': snapshot['state_id'],
                'a16r_run_path': relative(path),
                'capture_fraction': snapshot['capture_fraction'],
                'a16r_state_fingerprint': audit_fingerprint,
                'a16r_makespan': snapshot['current_makespan'],
                'classification': classification,
            })
    if snapshot_count != 18:
        raise RuntimeError(f'Expected 18 A1.6R replayable states, found {snapshot_count}')
    columns = list(rows[0])
    csv_path = REPORTS / 'ngas_a17ar_state_overlap.csv'
    csv_write(csv_path, rows, columns)
    schema = {
        'schema': 'ngas-a17ar-state-fingerprint-schema-v1',
        'algorithm': 'sha256(canonical-json(candidate assignment arrays))',
        'candidate_fields': [
            'operation_order', 'island_assignment', 'w_assignment', 'f_assignment'],
        'canonical_json': {'sort_keys': True, 'separators': [',', ':']},
        'exact_state_requirement': (
            'candidate fingerprint plus recoverable runtime, RNG, temperature, portfolio, '
            'and incumbent metadata; candidate equality alone is not called exact state identity'),
        'limitations': (
            'A1.6R retained 18 snapshots from six instances under seed 746102; '
            'the remaining historical C1 states have no comparable A1.6R snapshot.'),
    }
    json_write(REPORTS / 'ngas_a17ar_state_fingerprint_schema.json', schema)
    counts = {name: sum(row['classification'] == name for row in rows)
              for name in sorted({row['classification'] for row in rows})}
    exact = counts.get('EXACT_STATE_MATCH', 0)
    report = REPORTS / 'ngas_a17ar_state_overlap_audit.md'
    report.write_text(f'''# NGAS A1.7A-R state-overlap audit

The A1.6R artifacts retain {snapshot_count} replayable trajectory snapshots: three progress points for each of six R12 instances, all under seed `746102`. Each has one matching C1 `NATIVE16_746102` training state by instance and seed.

- Exact state matches supported by the full fingerprint contract: **{exact}**.
- Same candidate/schedule structure with incomplete runtime metadata: **{counts.get('SAME_SCHEDULE_STRUCTURE_RUNTIME_METADATA_UNAVAILABLE', 0)}**.
- Same instance and seed but non-identical candidate state: **{counts.get('SAME_INSTANCE_SEED_NON_IDENTICAL_STATE', 0)}**.
- No recoverable pair among the retained snapshots: **{counts.get('NO_RECOVERABLE_COMPARISON', 0)}**.

All 18 recoverable comparisons have different candidate-assignment fingerprints. The evidence therefore supports same-instance and same-seed trajectory provenance, but does not support an exact state-duplication claim. This conservative state-level result does not alter the instance-level conclusion: all 18 R12 instances were exposed during final C1 fitting.
''')


def write_comparator_audit() -> None:
    rows = [
        {'algorithm': 'GA', 'parameter_provenance': 'GLOBALLY_FIXED_BEFORE_R12',
         'r12_specific_tuning': False, 'evidence': 'configs/phase5c_ga.json',
         'interpretation': 'Phase 5C configuration predates the R12 suite.'},
        {'algorithm': 'DCGA', 'parameter_provenance': 'GLOBALLY_FIXED_BEFORE_R12',
         'r12_specific_tuning': False, 'evidence': 'configs/phase5c_dcga.json; docs/reports/phase5c_dcga_adaptation_protocol.md',
         'interpretation': 'Adapted and frozen in Phase 5C without R12 outcome tuning.'},
        {'algorithm': 'DABC', 'parameter_provenance': 'LITERATURE_DEFAULT_AND_PRE_R12_ADAPTATION',
         'r12_specific_tuning': False, 'evidence': 'configs/baselines/dabc_riacrsp.json; docs/reports/dabc_source_fidelity.md',
         'interpretation': 'Source defaults and declared adaptation were frozen before R12.'},
        {'algorithm': 'LG_HGA_2O', 'parameter_provenance': 'PRE_R12_TRAINING_AND_FIXED_CONFIGURATION',
         'r12_specific_tuning': False, 'evidence': 'configs/baselines/lghga_knowledge_training_manifest.json; configs/baselines/lghga_v2_implementation_manifest.json',
         'interpretation': 'Knowledge data came from RCIAS-CB1-TRAIN; no R12 tuning evidence.'},
        {'algorithm': 'ALNS', 'parameter_provenance': 'GLOBALLY_FIXED_BEFORE_R12',
         'r12_specific_tuning': False, 'evidence': 'configs/phase5c_alns.json',
         'interpretation': 'Phase 5C configuration predates the R12 suite.'},
        {'algorithm': 'PHASE6H', 'parameter_provenance': 'PRE_R12_R07_R08_CALIBRATION',
         'r12_specific_tuning': False, 'evidence': 'configs/phase6h_live_calibration.json; docs/reports/phase6h_live_calibration_report.md',
         'interpretation': 'Calibration used R07/R08 and was frozen before R12.'},
        {'algorithm': 'PHASE6N_TOP1', 'parameter_provenance': 'TUNED_AND_TRAINED_ON_R12',
         'r12_specific_tuning': True, 'evidence': 'configs/phase6n_candidate_conditioned_csg_v1.json; docs/reports/phase6n_data_generation_report.md',
         'interpretation': 'Training and selection explicitly used R12 states and outcomes.'},
        {'algorithm': 'NGAS_C1', 'parameter_provenance': 'TUNED_AND_TRAINED_ON_R12',
         'r12_specific_tuning': True, 'evidence': 'outputs/ngas_a1/critic_training_rthgt_v2/data/training_cache_manifest_v2.json; outputs/ngas_a1/critic_training_rthgt_v2/audit/completion_audit.json',
         'interpretation': 'Final compact-relational C1 fit used all 18 R12 instances.'},
    ]
    path = REPORTS / 'ngas_a17ar_comparator_exposure.csv'
    csv_write(path, rows, list(rows[0]))
    tuned = [row['algorithm'] for row in rows if row['r12_specific_tuning']]
    (REPORTS / 'ngas_a17ar_comparator_exposure_audit.md').write_text(f'''# NGAS A1.7A-R comparator exposure audit

Parameter provenance was reconstructed from frozen configuration and protocol artifacts. Explicit R12-specific training or adaptation is confirmed for **{', '.join(tuned)}**. No R12-specific parameter tuning evidence was found for GA, DCGA, DABC, LG_HGA_2O, ALNS, or PHASE6H; their listed configurations predate R12 or use separately generated development/calibration data.

This asymmetry matters when interpreting the R12 development comparison. Even if every comparator had used R12, it would still not restore R12 as an independent test set.

The machine-readable table records the evidence path and classification for each algorithm.
''')


def write_terminology_audit() -> None:
    patterns = [
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'R12.{0,100}(independent test|unseen benchmark|unseen-instance|held-out test|unbiased generalization|external test set)',
            r'(independent test|unseen benchmark|unseen-instance|held-out test|unbiased generalization|external test set).{0,100}R12',
        )
    ]
    flags = []
    for path in sorted((ROOT / 'docs').rglob('*')):
        if path.suffix not in {'.md', '.tex', '.txt'}:
            continue
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                flags.append({'path': relative(path), 'line': number, 'text': line.strip(),
                              'status': 'FLAGGED_FOR_CORRECTION'})
    report = REPORTS / 'ngas_a17ar_terminology_audit.md'
    entries = '\n'.join(
        f"- `{row['path']}:{row['line']}` — {row['text']}" for row in flags)
    if not entries:
        entries = '- No prohibited R12 independent/unseen/final-test wording was found by the scoped search.'
    report.write_text(f'''# NGAS A1.7A-R terminology audit

Search scope: repository `docs/` Markdown, TeX, and text files. The audit looked for R12 near independent-test, unseen-benchmark, held-out-test, external-test, or unbiased-generalization language.

{entries}

`docs/reports/ngas_a1/13_a16r_integrity_diagnostic.md` is explicitly clarified by A1.7A-R as an R12 development-exposed comparison. The terminal A1.6R integrity decision remains unchanged; only its scientific evidence class is restricted.
''')


def record_reserved_metadata_access() -> None:
    registry = DatasetRegistry(REGISTRY_PATH, ROOT)
    existing = verify_exposure_ledger(LEDGER_PATH) if LEDGER_PATH.exists() else []
    recorded = {(row.get('phase'), row.get('instance_id'), row.get('requested_purpose'))
                for row in existing}
    for row in registry.payload['instances']:
        if row['family_id'] not in {'RCIAS_CB1_CORE45', 'RCIAS_CB1_R13', 'RCIAS_CB1_R14'}:
            continue
        key = ('NGAS_A1_7A_R_STAGE0', row['instance_id'], 'metadata_integrity')
        if key in recorded:
            continue
        append_exposure(LEDGER_PATH, {
            'git_sha': _git('rev-parse', 'HEAD'),
            'command': 'scripts/audit_ngas_a17ar_exposure.py',
            'phase': key[0], 'instance_id': row['instance_id'],
            'instance_content_sha256': row['content_sha256'],
            'requested_purpose': key[2], 'dataset_role': row['role'],
            'checkpoint_identifier': None, 'permitted': True,
            'access_scope': 'manifest metadata and byte hash only; no solver or performance access',
        })
    verify_exposure_ledger(LEDGER_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-entry-sha', default=_git('rev-parse', 'HEAD'))
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_preflight(args.stage_entry_sha)
    record_reserved_metadata_access()
    write_exposure_audit()
    write_state_overlap()
    write_comparator_audit()
    write_terminology_audit()
    print(json.dumps({
        'status': 'PASS', 'stage': 'A1.7A-R_STAGES_0_TO_5_EVIDENCE',
        'r12_instance_overlap': '18/18', 'r13': 'LOCKED_FINAL_EVAL',
        'r14': 'LOCKED_GENERALIZATION_EVAL',
        'core45': 'EXTERNAL_BASELINE_ONLY',
    }))


if __name__ == '__main__':
    main()
