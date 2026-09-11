"""Integrity checks for immutable inputs to the NGAS A1.6 comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


COMPARATOR_DIRECTORIES = {
    'ALNS': 'alns',
    'PHASE6H': 'phase6h',
    'PHASE6N_TOP1': 'phase6n',
    'LG_HGA_2O': 'lg_hga_2o',
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def audit_frozen_inputs(root: Path, config: dict) -> dict:
    """Fully hash and validate the A1.6 instance, comparator, and BKS inputs."""
    scope = config['scope']
    manifest_path = root / scope['instance_manifest_path']
    manifest = load_json(manifest_path)
    instances = manifest['instances']
    seeds = scope['seeds']
    expected_keys = {
        (row['instance_id'], int(seed)) for row in instances for seed in seeds
    }
    checks = {
        'instance_manifest_sha256': digest(manifest_path) == scope['instance_manifest_sha256'],
        'instance_scope_exact': len(instances) == scope['instance_count'] == 18
                                and len(expected_keys) == scope['formal_run_count'] == 54,
        'instance_files_match': all(
            digest(root / scope['instance_root'] / row['relative_path']) == row['sha256']
            for row in instances
        ),
    }
    instance_hashes = {row['instance_id']: row['sha256'] for row in instances}
    operation_counts = {row['instance_id']: int(row['num_operations']) for row in instances}

    registry_path = root / config['comparators']['registry_path']
    registry = load_json(registry_path)
    entries = {entry['algorithm_id']: entry for entry in registry['entries']}
    checks['registry_status'] = (
        registry.get('status') == config['comparators']['required_registry_status']
        and set(entries) == set(COMPARATOR_DIRECTORIES)
    )
    comparator_audits = []
    for algorithm_id, directory in COMPARATOR_DIRECTORIES.items():
        entry = entries[algorithm_id]
        method_root = root / 'outputs/frozen_2o_baselines' / directory
        summary_path = root / entry['aggregate_summary_path']
        run_manifest_path = root / entry['run_manifest_path']
        run_manifest = load_json(run_manifest_path)
        rows = run_manifest['runs']
        listed_paths = {root / row['path'] for row in rows}
        actual_paths = set((method_root / 'runs').glob('*/*.json'))
        row_keys = {(row['instance_id'], int(row['seed'])) for row in rows}
        hashes_match = all(
            path.is_file() and digest(path) == row['sha256']
            for path, row in ((root / row['path'], row) for row in rows)
        )
        payloads_match = True
        for row in rows:
            payload = load_json(root / row['path'])
            instance_id = row['instance_id']
            payloads_match &= all((
                payload.get('status') == 'COMPLETE',
                payload.get('instance_id') == instance_id,
                payload.get('instance_sha256') == instance_hashes.get(instance_id),
                int(payload.get('seed', -1)) == int(row['seed']),
                float(payload.get('budget_seconds', -1.)) == 2. * operation_counts.get(instance_id, -1),
                float(payload.get('time_limit_seconds', -1.)) == 2. * operation_counts.get(instance_id, -1),
                payload.get('feasible') is True,
                payload.get('feasibility_replay', {}).get('feasible') is True,
                payload.get('r13_accessed') is False,
                payload.get('r14_accessed') is False,
                payload.get('gurobi_run') is False,
            ))
        audit = {
            'algorithm_id': algorithm_id,
            'status_frozen_canonical': entry.get('status') == 'FROZEN_CANONICAL',
            'registry_manifest_hash_match': digest(run_manifest_path) == entry['run_manifest_sha256'],
            'registry_summary_hash_match': digest(summary_path) == entry['aggregate_summary_sha256'],
            'instance_manifest_hash_match': entry.get('instance_manifest_sha256') == scope['instance_manifest_sha256'],
            'scope_exact_54': len(rows) == 54 and row_keys == expected_keys,
            'raw_file_set_exact': actual_paths == listed_paths,
            'raw_hashes_match': hashes_match,
            'payload_contract_match': payloads_match,
            'feasibility_rate_one': load_json(summary_path).get('feasibility_rate') == 1.,
            'manifest_path': entry['run_manifest_path'],
            'manifest_sha256': digest(run_manifest_path),
            'summary_path': entry['aggregate_summary_path'],
            'summary_sha256': digest(summary_path),
        }
        audit['pass'] = all(value for key, value in audit.items()
                            if key not in {'algorithm_id', 'manifest_path', 'manifest_sha256',
                                           'summary_path', 'summary_sha256', 'pass'})
        comparator_audits.append(audit)
    checks['all_comparators_pass'] = all(row['pass'] for row in comparator_audits)

    bks_path = root / config['bks']['path']
    bks = load_json(bks_path)
    bks_entries = bks.get('instances', {})
    bks_sources_match = True
    for instance_id, entry in bks_entries.items():
        source = root / entry['raw_result_path']
        raw = load_json(source)
        bks_sources_match &= all((
            source.is_file(),
            digest(source) == entry['raw_sha256'],
            entry['instance_sha256'] == instance_hashes.get(instance_id),
            float(entry['makespan']) == float(raw['final_makespan']),
            raw.get('feasible') is True,
        ))
    checks.update({
        'bks_schema_and_scope': bks.get('schema') == 'ngas-bks-v1'
                                and bks.get('version') == config['bks']['version']
                                and set(bks_entries) == set(instance_hashes),
        'bks_sources_match': bks_sources_match,
        'checkpoint_sha256': digest(root / config['production_solver']['checkpoint_path'])
                             == config['production_solver']['checkpoint_sha256'],
    })
    return {
        'schema': 'ngas-a16-frozen-input-audit-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'instance_manifest_path': scope['instance_manifest_path'],
        'instance_manifest_sha256': digest(manifest_path),
        'registry_path': config['comparators']['registry_path'],
        'registry_sha256': digest(registry_path),
        'comparators': comparator_audits,
        'bks_path': config['bks']['path'],
        'bks_sha256': digest(bks_path),
        'checkpoint_path': config['production_solver']['checkpoint_path'],
        'checkpoint_sha256': digest(root / config['production_solver']['checkpoint_path']),
    }
