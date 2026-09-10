#!/usr/bin/env python3
"""Import immutable comparator and Phase6P raw evidence; never rerun solvers."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ngas_starting_state import digest
from rcias_ngas.evaluation.bks import make_manifest, write_immutable
from rcias_ngas.evaluation.rpd import derive_rpd


def main():
    boundary = json.loads((ROOT / 'outputs/ngas_a1/audit/starting_state.json').read_text())
    for path, expected in boundary['protected_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Protected evidence changed: {path}')
    registry = boundary['registry']
    manifest = json.loads((ROOT / boundary['canonical_manifest_path']).read_text())
    instance_hashes = {r['instance_id']: r['sha256'] for r in manifest['instances']}
    imports = []
    for entry in registry['entries']:
        for row in json.loads((ROOT / entry['run_manifest_path']).read_text())['runs']:
            imports.append((entry, row['path']))
    for path in sorted((ROOT / 'outputs/phase6p_adaptive_portfolio_v1/development/runs').rglob('seed_*.json')):
        imports.append(({'algorithm_id': 'PHASE6P', 'source_commit': boundary['starting_commit'],
                         'config_sha256': digest('configs/phase6p_adaptive_portfolio_v1.json'),
                         'model_checkpoint_hashes': registry['entries'][2]['model_checkpoint_hashes'],
                         'source_files': {p: h for p, h in boundary['protected_hashes'].items() if p in registry['entries'][2]['source_files']},
                         'budget_formula': registry['entries'][2]['budget_formula'],
                         'initialization_accounting': 'inside wall-clock budget'}, str(path.relative_to(ROOT))))
    rows = []
    for entry, path in imports:
        raw = json.loads((ROOT / path).read_text())
        last = raw['incumbent_trace'][-1]
        rows.append({
            'instance_id': raw['instance_id'], 'instance_sha256': instance_hashes[raw['instance_id']],
            'algorithm_id': entry['algorithm_id'], 'algorithm_commit': entry['source_commit'],
            'algorithm_version': entry['source_files'], 'checkpoint_sha256': entry['model_checkpoint_hashes'],
            'config_sha256': entry['config_sha256'], 'seed': raw['seed'],
            'budget_formula': entry['budget_formula'], 'initialization_accounting': entry['initialization_accounting'],
            'final_makespan': raw['final_makespan'], 'feasible': raw['feasible'],
            'decoder_evals': raw['decoder_evaluations'], 'total_runtime_sec': raw['runtime_seconds'],
            'last_best_time_sec': raw['best_found_seconds'],
            'last_best_decoder_evals': last['decoder_evaluations'],
            'last_best_iteration': None, 'last_best_iteration_status': 'NOT_RECORDED_IN_FROZEN_RAW',
            'raw_result_path': path, 'raw_sha256': digest(path),
        })
    assert len(rows) == 270
    out = ROOT / 'outputs/ngas_a1/provenance'
    raw_registry = {'schema': 'ngas-raw-registry-v1', 'runs': rows,
                    'historical_anytime_deprecations': [{
                        'path': 'outputs/phase6p_adaptive_portfolio_v1/development/anytime_summary.csv',
                        'field': 'mean_decoder_evaluations',
                        'reason': 'Last-best work count, not checkpoint work count; frozen file preserved'}]}
    bks = make_manifest(rows, 1)
    write_immutable(out / 'raw_run_registry.json', raw_registry)
    write_immutable(out / 'bks_manifest_v001.json', bks)
    metrics = derive_rpd(rows, bks)
    metrics['bks_manifest_path'] = 'outputs/ngas_a1/provenance/bks_manifest_v001.json'
    write_immutable(out / 'derived_metric_manifest_v001.json', metrics)
    print(json.dumps({'raw_runs': len(rows), 'bks_instances': len(bks['instances'])}))


if __name__ == '__main__':
    main()
