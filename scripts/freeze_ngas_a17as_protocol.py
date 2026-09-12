#!/usr/bin/env python3
"""Deterministically freeze the 27 A1.7A-S clean supplemental states."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a17as import UTILITY_TOLERANCE  # noqa: E402
from rcias_ngas.evaluation.a17ar import canonical_hash  # noqa: E402
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    DatasetRegistry, sha256_file,
)


CONFIG = ROOT / 'configs/ngas_a17as_supplemental_protocol.yaml'
PREFLIGHT = ROOT / 'artifacts/ngas_a17as/preflight.json'
OUTPUT = ROOT / 'artifacts/ngas_a17as/supplemental_protocol_manifest.json'
SELECTION_CSV = ROOT / 'reports/ngas_a17as_state_selection.csv'
COVERAGE_CSV = ROOT / 'reports/ngas_a17as_state_selection_coverage.csv'
REPORT = ROOT / 'reports/ngas_a17as_state_selection.md'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def stage_for(fraction: float) -> str | None:
    if 0. <= fraction < .2:
        return 'EARLY'
    if .4 <= fraction < .6:
        return 'MIDDLE'
    if .8 <= fraction <= 1.:
        return 'LATE'
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_eligible(config: dict, trajectory: dict, raw_manifest: dict) -> list[dict]:
    task_by_id = {row['instance_id']: row for row in trajectory['instances']}
    eligible = []
    for instance_id, task in sorted(task_by_id.items()):
        role = task['dataset_role']
        if role not in {'TRAIN', 'VALIDATION'}:
            continue
        path = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/raw' \
            / role.lower() / f'{instance_id}.json'
        expected_hash = raw_manifest['files'].get(relative(path))
        if expected_hash is None or sha256_file(path) != expected_hash:
            raise RuntimeError(f'A1.7A-R clean raw hash mismatch: {relative(path)}')
        payload = json.loads(path.read_text())
        if payload['instance_content_sha256'] != task['content_sha256']:
            raise RuntimeError(f'instance identity mismatch: {instance_id}')
        for state in payload['states']:
            fraction = float(state['normalized_budget_fraction'])
            stage = stage_for(fraction)
            if stage is None:
                continue
            tags = tuple(sorted(state['search_condition_tags']))
            if not set(tags) <= set(config['state_selection']['condition_tags']):
                raise RuntimeError(f'unknown state condition tag: {state["state_id"]}')
            eligible.append({
                'state_key': f'CLEAN|{state["state_id"]}',
                'instance_id': instance_id,
                'instance_relative_path': task['relative_path'],
                'instance_sha256': task['content_sha256'],
                'dataset_role': role,
                'scale': task['scale'],
                'observed_budget_fraction': fraction,
                'stage': stage,
                'capture_fraction': float(state['target_capture_fraction']),
                'capture_target': float(state['target_capture_fraction']),
                'search_iteration': int(state['search_iteration']),
                'seed': int(task['trajectory_seed']),
                'condition_tags': list(tags),
                'source_a17ar_raw_file': relative(path),
                'source_a17ar_raw_sha256': expected_hash,
                'state_payload_sha256': canonical_hash(state),
                'formal_raw_path': relative(path),
                'formal_raw_sha256': expected_hash,
                'state_sha256': canonical_hash(state),
                'dataset_origin': 'CLEAN_NON_R12_DEVELOPMENT',
                'audit_only': True,
                'eligible_for_future_training': False,
            })
    return eligible


def select_states(eligible: list[dict]) -> list[dict]:
    selected = []
    for scale in ('S', 'M', 'L'):
        for stage in ('EARLY', 'MIDDLE', 'LATE'):
            cell = [row for row in eligible
                    if row['scale'] == scale and row['stage'] == stage]
            frequencies = Counter(tag for row in cell for tag in row['condition_tags'])
            covered: set[str] = set()
            for role, count in (('TRAIN', 2), ('VALIDATION', 1)):
                pool = [row for row in cell if row['dataset_role'] == role]
                if len(pool) < count:
                    raise RuntimeError(f'insufficient states for {scale}/{stage}/{role}')
                for _ in range(count):
                    candidates = [row for row in pool if row not in selected]
                    if not candidates:
                        raise RuntimeError(f'selection exhausted for {scale}/{stage}/{role}')

                    def key(row: dict):
                        new_tags = set(row['condition_tags']) - covered
                        rarity = sum((Fraction(1, frequencies[tag])
                                      for tag in row['condition_tags']), Fraction())
                        return (-len(new_tags), -rarity, row['state_key'])

                    choice = min(candidates, key=key)
                    selected.append(choice)
                    covered.update(choice['condition_tags'])
    return selected


def build_protocol() -> dict:
    config = json.loads(CONFIG.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    trajectory_path = ROOT / config['source_trajectory_protocol']
    collection_path = ROOT / config['source_collection_audit']
    raw_manifest_path = ROOT / config['source_raw_manifest']
    trajectory = json.loads(trajectory_path.read_text())
    collection = json.loads(collection_path.read_text())
    raw_manifest = json.loads(raw_manifest_path.read_text())
    registry = DatasetRegistry(ROOT / config['dataset_role_registry'], ROOT)
    if preflight['status'] != 'PASS' or collection['status'] != 'PASS':
        raise RuntimeError('A1.7A-S preflight or source collection audit failed')
    eligible = load_eligible(config, trajectory, raw_manifest)
    selected = select_states(eligible)
    for row in selected:
        registry.authorize({
            'instance_id': row['instance_id'],
            'content_sha256': row['instance_sha256'],
            'dataset_role': row['dataset_role'],
        }, 'diagnostic')
    cell_counts = Counter((row['scale'], row['stage']) for row in selected)
    cell_roles = Counter(
        (row['scale'], row['stage'], row['dataset_role']) for row in selected)
    state_keys = {row['state_key'] for row in selected}
    state_hashes = {row['state_payload_sha256'] for row in selected}
    checks = {
        'preflight_pass': preflight['status'] == 'PASS',
        'source_collection_108_runs_540_states': (
            collection['completed_runs'] == 108 and collection['completed_states'] == 540),
        'exact_27_selected_states': len(selected) == 27,
        'exact_3_per_scale_stage_cell': cell_counts == Counter({
            (scale, stage): 3 for scale in ('S', 'M', 'L')
            for stage in ('EARLY', 'MIDDLE', 'LATE')}),
        'exact_2_train_1_validation_per_cell': cell_roles == Counter({
            **{(scale, stage, 'TRAIN'): 2 for scale in ('S', 'M', 'L')
               for stage in ('EARLY', 'MIDDLE', 'LATE')},
            **{(scale, stage, 'VALIDATION'): 1 for scale in ('S', 'M', 'L')
               for stage in ('EARLY', 'MIDDLE', 'LATE')},
        }),
        'state_keys_unique': len(state_keys) == len(selected),
        'state_payload_hashes_unique': len(state_hashes) == len(selected),
        'only_train_validation_roles': {
            row['dataset_role'] for row in selected} == {'TRAIN', 'VALIDATION'},
        'all_supplemental_labels_audit_only': all(
            row['audit_only'] is True
            and row['eligible_for_future_training'] is False for row in selected),
        'checkpoint_sha_exact': sha256_file(ROOT / config['checkpoint_path'])
            == config['checkpoint_sha256'],
        'utility_tolerance_exact': config['full_bank']['utility_tolerance']
            == UTILITY_TOLERANCE,
    }
    if not all(checks.values()):
        raise RuntimeError({'selection_protocol_failed': checks})
    source_paths = (
        CONFIG,
        PREFLIGHT,
        trajectory_path,
        collection_path,
        raw_manifest_path,
        ROOT / config['dataset_role_registry'],
        ROOT / 'rcias_ngas/evaluation/a17as.py',
        ROOT / 'rcias_ngas/evaluation/a17ar.py',
        ROOT / 'scripts/freeze_ngas_a17as_protocol.py',
        ROOT / 'scripts/run_ngas_a17as_full_bank.py',
        ROOT / 'scripts/audit_ngas_a17as_full_bank.py',
        ROOT / 'scripts/run_ngas_a17ar_full_bank_audit.py',
        ROOT / 'rcias_ngas/search/ngas_solver.py',
        ROOT / 'rcias_ngas/runtime/production_refresh.py',
        ROOT / 'rcias_ngas/actions/repair.py',
        ROOT / 'rcias_ngas/bank/ngas_bank_v1.py',
        ROOT / 'rcias_clgri/env/insertion_decoder.py',
    )
    return {
        'schema': 'ngas-a17as-supplemental-protocol-v1',
        'status': 'FROZEN_BEFORE_SUPPLEMENTAL_OUTCOMES',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_head': config['baseline_head'],
        'freeze_source_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config_path': relative(CONFIG),
        'config_sha256': sha256_file(CONFIG),
        'checkpoint_path': config['checkpoint_path'],
        'checkpoint_sha256': config['checkpoint_sha256'],
        'source_trajectory_protocol_sha256': sha256_file(trajectory_path),
        'source_collection_audit_sha256': sha256_file(collection_path),
        'source_raw_manifest_sha256': sha256_file(raw_manifest_path),
        'selection_rule': config['state_selection']['selection_rule'],
        'selection_role_order': config['state_selection']['role_order'],
        'eligible_state_count': len(eligible),
        'states': selected,
        'state_count': len(selected),
        'matched_trials_per_action': 8,
        'continuation_steps': 2,
        'counterfactual_scope': 'FULL_UNIQUE_PRODUCTION_JOINT_BANK',
        'utility_tolerance': UTILITY_TOLERANCE,
        'source_hashes': {relative(path): sha256_file(path) for path in source_paths},
        'checks': checks,
        'boundaries': config['boundaries'],
    }


def emit_selection(protocol: dict) -> None:
    rows = protocol['states']
    csv_rows = [{
        key: '|'.join(row[key]) if key == 'condition_tags' else row[key]
        for key in (
            'state_key', 'instance_id', 'instance_sha256', 'dataset_role', 'scale',
            'observed_budget_fraction', 'stage', 'capture_fraction', 'capture_target',
            'search_iteration', 'seed', 'condition_tags', 'source_a17ar_raw_file',
            'source_a17ar_raw_sha256', 'state_payload_sha256', 'audit_only',
            'eligible_for_future_training')
    } for row in rows]
    write_csv(SELECTION_CSV, csv_rows)
    coverage_rows = []
    for scale in ('S', 'M', 'L'):
        for stage in ('EARLY', 'MIDDLE', 'LATE'):
            values = [row for row in rows
                      if row['scale'] == scale and row['stage'] == stage]
            coverage_rows.append({
                'scale': scale, 'stage': stage, 'states': len(values),
                'train_states': sum(row['dataset_role'] == 'TRAIN' for row in values),
                'validation_states': sum(
                    row['dataset_role'] == 'VALIDATION' for row in values),
                **{f'tag_{tag}': sum(tag in row['condition_tags'] for row in values)
                   for tag in json.loads(CONFIG.read_text())[
                       'state_selection']['condition_tags']},
            })
    write_csv(COVERAGE_CSV, coverage_rows)
    REPORT.write_text(f'''# NGAS A1.7A-S deterministic state selection

The frozen supplemental set contains **27 existing clean non-R12 states**. Each
S/M/L × EARLY/MIDDLE/LATE cell contains exactly three states: two governed TRAIN
states and one governed VALIDATION state. No 20–40% or 60–80% state is included.

Selection is deterministic. Within each cell, TRAIN is selected before VALIDATION;
each choice maximizes newly covered condition tags, then the sum of reciprocal tag
frequencies in the eligible scale-stage pool, then uses lexicographic `state_key`.
All 27 state keys and payload hashes are unique. Supplemental labels are audit-only
and ineligible for future training loaders.

- Selection manifest: `{relative(OUTPUT)}`
- Selected rows: `{relative(SELECTION_CSV)}`
- Coverage table: `{relative(COVERAGE_CSV)}`
''')


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError('A1.7A-S protocol already exists and is immutable')
    protocol = build_protocol()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    emit_selection(protocol)
    print(json.dumps({
        'status': protocol['status'], 'states': protocol['state_count'],
        'train': sum(row['dataset_role'] == 'TRAIN' for row in protocol['states']),
        'validation': sum(
            row['dataset_role'] == 'VALIDATION' for row in protocol['states']),
        'protocol_sha256': sha256_file(OUTPUT),
    }, indent=2))


if __name__ == '__main__':
    main()
