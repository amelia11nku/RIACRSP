#!/usr/bin/env python3
"""Run and finalize the frozen six-way A1.4 R12 development subset."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.search.ngas_solver import (
    MODE_SETTINGS, search_config_from_dict, solve_ngas,
)

OUT = ROOT / 'outputs/ngas_a1/search_integration_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'


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
        raise RuntimeError('A1.4 protocol is not frozen')
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Formal source drift: {relative}')
    config_path = ROOT / protocol['config_path']
    if digest(config_path) != protocol['config_sha256']:
        raise RuntimeError('A1.4 config drift')
    config = json.loads(config_path.read_text())
    return protocol, config, digest(PROTOCOL)


def raw_path(mode: str, instance_id: str, seed: int) -> Path:
    return OUT / 'raw' / mode.lower() / instance_id / f'seed_{seed}.json'


def candidate_payload(candidate: Candidate) -> dict:
    return {
        'operation_order': list(candidate.operation_order),
        'island_assignment': list(candidate.island_assignment),
        'w_assignment': list(candidate.w_assignment),
        'f_assignment': list(candidate.f_assignment),
    }


def candidate_from_payload(value: dict) -> Candidate:
    return Candidate(*(tuple(value[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def serialize_result(result, protocol_sha256: str, config: dict, instance_row: dict,
                     seed: int, mode: str, time_limit: float) -> dict:
    return {
        'schema': 'ngas-a14-primary-run-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha256,
        'mode': mode, 'seed': seed,
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
    checkpoint = ROOT / config['representation_boundary']['checkpoint_path']
    critic = FrozenJointCritic(
        checkpoint, device,
        config['representation_boundary']['checkpoint_sha256'], 'C1')
    search_config = search_config_from_dict(config['search'])
    expected = protocol['expected_primary_runs']
    completed = 0
    for row in config['development_subset']:
        instance_path = ROOT / config['instance_source']['root'] / row['relative_path']
        if digest(instance_path) != row['sha256']:
            raise RuntimeError(f"Instance hash drift: {row['instance_id']}")
        instance = load_instance(instance_path)
        for seed in config['development_seeds']:
            for mode in config['ablation_modes']:
                path = raw_path(mode, instance.instance_id, seed)
                if path.exists():
                    validate_raw(path, protocol_sha256, config)
                    completed += 1
                    continue
                time_limit = config['budget']['multiplier'] * instance.num_operations
                print(json.dumps({
                    'event': 'run_started', 'mode': mode,
                    'instance_id': instance.instance_id, 'seed': seed,
                    'time_limit_seconds': time_limit,
                    'completed': completed, 'expected': expected,
                }), flush=True)
                result = solve_ngas(
                    instance, time_limit, seed, mode,
                    critic=critic if MODE_SETTINGS[mode]['neural'] else None,
                    config=search_config)
                payload = serialize_result(
                    result, protocol_sha256, config, row, seed, mode, time_limit)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as stream:
                    stream.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
                completed += 1
                atomic_json(OUT / 'progress.json', {
                    'schema': 'ngas-a14-primary-progress-v1', 'status': 'RUNNING',
                    'completed_runs': completed, 'expected_runs': expected,
                    'last_completed': str(path.relative_to(ROOT)),
                    'protocol_sha256': protocol_sha256,
                    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
                })
                print(json.dumps({
                    'event': 'run_completed', 'mode': mode,
                    'instance_id': instance.instance_id, 'seed': seed,
                    'best_makespan': result.best.makespan,
                    'iterations': result.iterations, 'completed': completed,
                }), flush=True)
    finalize(protocol, config, protocol_sha256)


def anytime_auc(row: dict) -> float:
    initial = row['convergence_trace'][0]['current_best_makespan']
    points = [(0., 1.)]
    for checkpoint in row['diagnostics']['telemetry']['budget_checkpoints']:
        if checkpoint['best_makespan'] is None:
            raise RuntimeError('No feasible incumbent at an A1.4 budget checkpoint')
        value = checkpoint['best_makespan'] / initial
        points.append((float(checkpoint['budget_fraction']), value))
    by_fraction = {fraction: value for fraction, value in points}
    ordered = sorted(by_fraction.items())
    return sum((right_x - left_x) * (left_y + right_y) / 2
               for (left_x, left_y), (right_x, right_y)
               in zip(ordered, ordered[1:]))


def validate_raw(path: Path, protocol_sha256: str, config: dict) -> dict:
    row = json.loads(path.read_text())
    if row.get('protocol_sha256') != protocol_sha256:
        raise RuntimeError(f'Protocol mismatch in {path}')
    instance_path = (ROOT / config['instance_source']['root']
                     / row['instance']['relative_path'])
    instance = load_instance(instance_path)
    replay = decode_candidate(instance, candidate_from_payload(row['best_candidate']))
    if not replay.feasible or replay.makespan != row['best_makespan']:
        raise RuntimeError(f'Final replay mismatch in {path}')
    row['_anytime_auc'] = anytime_auc(row)
    return row


def finalize(protocol: dict, config: dict, protocol_sha256: str) -> None:
    rows = []
    raw_manifest = {}
    for instance in config['development_subset']:
        for seed in config['development_seeds']:
            for mode in config['ablation_modes']:
                path = raw_path(mode, instance['instance_id'], seed)
                if not path.exists():
                    raise RuntimeError(f'Missing formal result: {path}')
                rows.append(validate_raw(path, protocol_sha256, config))
                raw_manifest[str(path.relative_to(ROOT))] = digest(path)
    expected = protocol['expected_primary_runs']
    if len(rows) != expected:
        raise RuntimeError('Unexpected formal run count')
    control = {(row['instance']['instance_id'], row['seed']): row for row in rows
               if row['mode'] == config['selection_gate']['internal_control']}
    summaries = {}
    for mode in config['ablation_modes']:
        selected = [row for row in rows if row['mode'] == mode]
        final_ratios = [row['best_makespan'] / row['convergence_trace'][0]['current_best_makespan']
                        for row in selected]
        influence = [row['diagnostics']['critic_influence_fraction'] for row in selected]
        guided = [row['diagnostics']['guided_iterations_per_critic_call'] for row in selected
                  if row['diagnostics']['guided_iterations_per_critic_call'] is not None]
        summaries[mode] = {
            'run_count': len(selected), 'feasible_replay_fraction': 1.,
            'mean_final_over_H1': statistics.fmean(final_ratios),
            'mean_anytime_auc_over_H1': statistics.fmean(row['_anytime_auc'] for row in selected),
            'mean_decoder_evaluations': statistics.fmean(row['decoder_evaluations'] for row in selected),
            'mean_critic_calls': statistics.fmean(
                row['diagnostics']['telemetry']['termination']['neural_calls'] for row in selected),
            'median_critic_influence_fraction': statistics.median(influence),
            'median_guided_iterations_per_critic_call': statistics.median(guided) if guided else None,
            'mean_effective_neural_overhead_seconds': statistics.fmean(
                row['diagnostics']['effective_neural_overhead_seconds'] for row in selected),
            'mean_acceptance_rate': statistics.fmean(
                row['diagnostics']['acceptance_rate'] for row in selected),
            'mean_current_improvement_rate': statistics.fmean(
                row['diagnostics']['current_improvement_rate'] for row in selected),
            'mean_new_best_rate': statistics.fmean(
                row['diagnostics']['new_best_rate'] for row in selected),
            'mean_unique_action_semantics': statistics.fmean(
                row['diagnostics']['unique_action_semantics'] for row in selected),
        }
        if mode != config['selection_gate']['internal_control']:
            paired = []
            for row in selected:
                base = control[row['instance']['instance_id'], row['seed']]
                paired.append({
                    'instance_id': row['instance']['instance_id'], 'seed': row['seed'],
                    'final_gain': (base['best_makespan'] - row['best_makespan']) / base['best_makespan'],
                    'anytime_auc_gain': (base['_anytime_auc'] - row['_anytime_auc']) / base['_anytime_auc'],
                })
            final_gains = [item['final_gain'] for item in paired]
            auc_gains = [item['anytime_auc_gain'] for item in paired]
            summaries[mode]['paired_control'] = {
                'pairs': paired,
                'mean_final_gain': statistics.fmean(final_gains),
                'mean_anytime_auc_gain': statistics.fmean(auc_gains),
                'wins': sum(value > 1e-12 for value in final_gains),
                'ties': sum(abs(value) <= 1e-12 for value in final_gains),
                'losses': sum(value < -1e-12 for value in final_gains),
            }
    gates = config['selection_gate']['hard_requirements']
    eligible = []
    eligibility = {}
    for mode in config['selection_gate']['production_candidates']:
        summary = summaries[mode]
        paired = summary['paired_control']
        hard = (
            summary['run_count'] == config['selection_gate']['required_run_count_per_mode']
            and summary['feasible_replay_fraction'] == gates['feasible_replay_fraction']
            and summary['median_critic_influence_fraction'] >= gates['minimum_median_critic_influence_fraction']
            and summary['median_guided_iterations_per_critic_call'] >= gates['minimum_median_guided_iterations_per_critic_call'])
        credible = (
            (paired['mean_final_gain'] > 0 and paired['wins'] >= paired['losses'])
            or (paired['mean_anytime_auc_gain'] >= .002
                and paired['mean_final_gain'] >= -.001))
        eligibility[mode] = {'hard_requirements_pass': hard, 'credible_quality_pass': credible}
        if hard and credible:
            eligible.append(mode)
    if eligible:
        selected_mode = max(eligible, key=lambda mode: (
            summaries[mode]['paired_control']['mean_final_gain'],
            summaries[mode]['paired_control']['mean_anytime_auc_gain'],
            -summaries[mode]['mean_effective_neural_overhead_seconds']))
        decision = 'NGAS_A1_4_MECHANISM_FROZEN'
        next_gate = 'MATCHED_C1_R1_EXPLANATORY_ABLATION'
    else:
        selected_mode = None
        decision = config['selection_gate']['no_eligible_candidate_decision']
        next_gate = 'STOP_NO_A1_5'
    summary_payload = {
        'schema': 'ngas-a14-primary-summary-v1',
        'protocol_sha256': protocol_sha256, 'run_count': len(rows),
        'method_summaries': summaries,
    }
    decision_payload = {
        'schema': 'ngas-a14-mechanism-decision-v1',
        'decision': decision, 'selected_mode': selected_mode,
        'eligible_modes': eligible, 'eligibility': eligibility,
        'next_gate': next_gate,
        'R1_role': 'EXPLANATORY_ABLATION_ONLY',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
        'protocol_sha256': protocol_sha256,
    }
    atomic_json(OUT / 'raw_manifest.json', {
        'schema': 'ngas-a14-raw-manifest-v1', 'protocol_sha256': protocol_sha256,
        'files': dict(sorted(raw_manifest.items())),
    })
    atomic_json(OUT / 'primary_summary.json', summary_payload)
    atomic_json(OUT / 'mechanism_decision.json', decision_payload)
    atomic_json(OUT / 'progress.json', {
        'schema': 'ngas-a14-primary-progress-v1', 'status': 'COMPLETE',
        'completed_runs': len(rows), 'expected_runs': expected,
        'decision': decision, 'selected_mode': selected_mode,
        'next_gate': next_gate, 'protocol_sha256': protocol_sha256,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    })
    print(json.dumps({'event': 'a14_primary_complete', **decision_payload}), flush=True)


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
