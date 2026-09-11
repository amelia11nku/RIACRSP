#!/usr/bin/env python3
"""Run the frozen matched C1/R1 explanatory solver ablation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import decode_candidate
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.search.ngas_solver import search_config_from_dict, solve_ngas
from scripts.run_ngas_a14_integration import (
    anytime_auc, candidate_from_payload, candidate_payload,
)

OUT = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
PRIMARY_PROTOCOL = ROOT / 'outputs/ngas_a1/search_integration_v1/preregistration/protocol.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def load_boundary() -> tuple[dict, dict, str]:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'FROZEN_BEFORE_OUTCOMES':
        raise RuntimeError('C1/R1 protocol is not frozen')
    if digest(PRIMARY_PROTOCOL) != protocol['primary_protocol_sha256']:
        raise RuntimeError('Inherited A1.4 primary protocol drift')
    primary = json.loads(PRIMARY_PROTOCOL.read_text())
    for relative, expected in primary['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Inherited A1.4 source drift: {relative}')
    for relative, expected in protocol['ablation_source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'C1/R1 ablation source drift: {relative}')
    decision_path = ROOT / protocol['primary_decision_path']
    if digest(decision_path) != protocol['primary_decision_sha256']:
        raise RuntimeError('Frozen A1.4 mechanism decision drift')
    config = json.loads((ROOT / primary['config_path']).read_text())
    return protocol, config, digest(PROTOCOL)


def raw_path(variant: str, instance_id: str, seed: int) -> Path:
    return OUT / 'raw' / variant / instance_id / f'seed_{seed}.json'


def validate_raw(path: Path, protocol_sha256: str, config: dict) -> dict:
    row = json.loads(path.read_text())
    if row.get('protocol_sha256') != protocol_sha256:
        raise RuntimeError(f'C1/R1 protocol mismatch in {path}')
    instance = load_instance(
        ROOT / config['instance_source']['root'] / row['instance']['relative_path'])
    replay = decode_candidate(instance, candidate_from_payload(row['best_candidate']))
    if not replay.feasible or replay.makespan != row['best_makespan']:
        raise RuntimeError(f'C1/R1 final replay mismatch in {path}')
    row['_anytime_auc'] = anytime_auc(row)
    return row


def serialize(result, protocol_sha256: str, instance_row: dict, seed: int,
              variant: str, checkpoint_path: str, time_limit: float) -> dict:
    return {
        'schema': 'ngas-a14-c1-r1-run-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha256,
        'selected_mechanism': 'PERSISTENT_FIXED_REFRESH',
        'representation_variant': variant,
        'representation_role': ('PRODUCTION' if variant == 'C1' else 'EXPLANATORY_ONLY'),
        'checkpoint_path': checkpoint_path,
        'seed': seed, 'held_fold': 0,
        'instance': instance_row, 'time_limit_seconds': time_limit,
        'best_makespan': result.best.makespan,
        'best_found_time_seconds': result.best_found_time,
        'runtime_seconds': result.runtime,
        'decoder_evaluations': result.decoder_evaluations,
        'iterations': result.iterations,
        'best_candidate': candidate_payload(result.best.candidate),
        'best_objective': asdict(result.best.objective),
        'convergence_trace': [asdict(point) for point in result.convergence_trace],
        'diagnostics': result.diagnostics,
        'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False},
    }


def run_all(device: str) -> None:
    protocol, config, protocol_sha256 = load_boundary()
    search_config = search_config_from_dict(config['search'])
    completed = 0
    for seed in protocol['seeds']:
        critics = {}
        for variant in protocol['variants']:
            checkpoint = protocol['checkpoints'][variant][str(seed)]
            critics[variant] = FrozenJointCritic(
                ROOT / checkpoint['path'], device,
                checkpoint['sha256'], variant)
        for instance_row in config['development_subset']:
            instance = load_instance(
                ROOT / config['instance_source']['root'] / instance_row['relative_path'])
            for variant in protocol['variants']:
                path = raw_path(variant, instance.instance_id, seed)
                if path.exists():
                    validate_raw(path, protocol_sha256, config)
                    completed += 1
                    continue
                time_limit = config['budget']['multiplier'] * instance.num_operations
                print(json.dumps({
                    'event': 'ablation_run_started', 'variant': variant,
                    'instance_id': instance.instance_id, 'seed': seed,
                    'time_limit_seconds': time_limit,
                    'completed': completed, 'expected': protocol['expected_runs'],
                }), flush=True)
                result = solve_ngas(
                    instance, time_limit, seed, protocol['selected_mechanism'],
                    critics[variant], search_config)
                checkpoint_path = protocol['checkpoints'][variant][str(seed)]['path']
                payload = serialize(
                    result, protocol_sha256, instance_row, seed, variant,
                    checkpoint_path, time_limit)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as stream:
                    stream.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
                completed += 1
                atomic_json(OUT / 'progress.json', {
                    'schema': 'ngas-a14-c1-r1-progress-v1', 'status': 'RUNNING',
                    'completed_runs': completed, 'expected_runs': protocol['expected_runs'],
                    'last_completed': str(path.relative_to(ROOT)),
                    'protocol_sha256': protocol_sha256,
                    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'production_variant': 'C1', 'R1_role': 'EXPLANATORY_ONLY',
                    'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
                })
                print(json.dumps({
                    'event': 'ablation_run_completed', 'variant': variant,
                    'instance_id': instance.instance_id, 'seed': seed,
                    'best_makespan': result.best.makespan,
                    'iterations': result.iterations, 'completed': completed,
                }), flush=True)
        del critics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finalize(protocol, config, protocol_sha256)


def finalize(protocol: dict, config: dict, protocol_sha256: str) -> None:
    rows, files = [], {}
    for seed in protocol['seeds']:
        for instance in config['development_subset']:
            for variant in protocol['variants']:
                path = raw_path(variant, instance['instance_id'], seed)
                if not path.exists():
                    raise RuntimeError(f'Missing C1/R1 result: {path}')
                rows.append(validate_raw(path, protocol_sha256, config))
                files[str(path.relative_to(ROOT))] = digest(path)
    by_variant = {}
    for variant in protocol['variants']:
        selected = [row for row in rows if row['representation_variant'] == variant]
        refresh_seconds = [refresh['refresh_seconds'] for row in selected
                           for refresh in row['diagnostics']['refreshes']]
        by_variant[variant] = {
            'run_count': len(selected), 'feasible_replay_fraction': 1.,
            'mean_final_over_H1': statistics.fmean(
                row['best_makespan'] / row['convergence_trace'][0]['current_best_makespan']
                for row in selected),
            'mean_anytime_auc_over_H1': statistics.fmean(row['_anytime_auc'] for row in selected),
            'mean_decoder_evaluations': statistics.fmean(row['decoder_evaluations'] for row in selected),
            'mean_iterations': statistics.fmean(row['iterations'] for row in selected),
            'mean_critic_calls': statistics.fmean(
                row['diagnostics']['telemetry']['termination']['neural_calls'] for row in selected),
            'mean_effective_neural_overhead_seconds': statistics.fmean(
                row['diagnostics']['effective_neural_overhead_seconds'] for row in selected),
            'mean_refresh_seconds': statistics.fmean(refresh_seconds),
            'p90_observed_refresh_seconds': sorted(refresh_seconds)[
                math.ceil(.9 * len(refresh_seconds)) - 1],
        }
    c1 = {(row['instance']['instance_id'], row['seed']): row for row in rows
          if row['representation_variant'] == 'C1'}
    pairs = []
    for row in rows:
        if row['representation_variant'] != 'R1':
            continue
        base = c1[row['instance']['instance_id'], row['seed']]
        pairs.append({
            'instance_id': row['instance']['instance_id'], 'seed': row['seed'],
            'R1_final_gain_over_C1': (base['best_makespan'] - row['best_makespan']) / base['best_makespan'],
            'R1_anytime_auc_gain_over_C1': (base['_anytime_auc'] - row['_anytime_auc']) / base['_anytime_auc'],
        })
    gains = [row['R1_final_gain_over_C1'] for row in pairs]
    auc_gains = [row['R1_anytime_auc_gain_over_C1'] for row in pairs]
    summary = {
        'schema': 'ngas-a14-c1-r1-summary-v1', 'protocol_sha256': protocol_sha256,
        'run_count': len(rows), 'selected_mechanism': protocol['selected_mechanism'],
        'variant_summaries': by_variant,
        'paired_R1_over_C1': {
            'pairs': pairs, 'mean_final_gain': statistics.fmean(gains),
            'mean_anytime_auc_gain': statistics.fmean(auc_gains),
            'wins': sum(value > 1e-12 for value in gains),
            'ties': sum(abs(value) <= 1e-12 for value in gains),
            'losses': sum(value < -1e-12 for value in gains),
        },
        'interpretation_boundary': 'R1 is explanatory only and cannot replace production C1',
    }
    decision = {
        'schema': 'ngas-a14-c1-r1-decision-v1',
        'decision': 'NGAS_A1_4_REPRESENTATION_ABLATION_COMPLETE_PENDING_AUDIT',
        'selected_mechanism': protocol['selected_mechanism'],
        'production_variant': 'C1', 'R1_role': 'EXPLANATORY_ONLY',
        'next_gate': 'A1_4_COMPLETION_AUDIT',
        'protocol_sha256': protocol_sha256,
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'raw_manifest.json', {
        'schema': 'ngas-a14-c1-r1-raw-manifest-v1',
        'protocol_sha256': protocol_sha256, 'files': dict(sorted(files.items())),
    })
    atomic_json(OUT / 'summary.json', summary)
    atomic_json(OUT / 'decision.json', decision)
    atomic_json(OUT / 'progress.json', {
        'schema': 'ngas-a14-c1-r1-progress-v1', 'status': 'COMPLETE',
        'completed_runs': len(rows), 'expected_runs': protocol['expected_runs'],
        'decision': decision['decision'], 'next_gate': decision['next_gate'],
        'protocol_sha256': protocol_sha256,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'production_variant': 'C1', 'R1_role': 'EXPLANATORY_ONLY',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    })
    print(json.dumps({'event': 'c1_r1_ablation_complete', **decision}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    protocol, config, protocol_sha256 = load_boundary()
    if args.finalize_only:
        finalize(protocol, config, protocol_sha256)
    else:
        run_all(args.device)


if __name__ == '__main__':
    main()
