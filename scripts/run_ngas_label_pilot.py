#!/usr/bin/env python3
"""Frozen, resumable bounded joint-action pilot; never launches training."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_ngas.critic.dataset import (POLICY_VERSION, SAMPLING_RULES, balanced_actions,
                                      fallback_action, label_action, separated_pair, transition)
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_clgri.search.alns import REPAIR
from rcias_ngas.evaluation.bks import content_hash, write_immutable
from rcias_ngas.rng import RNGStreams
from scripts.audit_ngas_starting_state import digest

OUT = ROOT / 'outputs/ngas_a1/label_pilot'
PROTOCOL = OUT / 'protocol.json'
CONFIG = ROOT / 'configs/ngas_a1_label_pilot.json'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_boundary():
    boundary = json.loads((ROOT / 'outputs/ngas_a1/audit/starting_state.json').read_text())
    for path, expected in boundary['protected_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen evidence drift: {path}')
    return boundary


def freeze():
    boundary = verify_boundary()
    gate = json.loads((ROOT / 'outputs/ngas_a1/audit/infrastructure_bank_gate.json').read_text())
    if gate['status'] != 'PASS':
        raise ValueError('A1.0/A1.1 must pass before label pilot')
    config = json.loads(CONFIG.read_text())
    manifest = json.loads((ROOT / boundary['canonical_manifest_path']).read_text())
    chosen = [r for r in manifest['instances'] if r['instance_id'] in config['instances']]
    assert len(chosen) == 3
    sources = list((ROOT / 'rcias_ngas').rglob('*.py')) + [
        Path(__file__), ROOT / 'scripts/launch_ngas_label_pilot.py', CONFIG]
    # These shared helpers are analysis/decoder primitives, not copied comparators.
    sources += list((ROOT / 'rcias_clgri').rglob('*.py'))
    protocol = {
        'schema': 'ngas-label-pilot-freeze-v1', 'status': 'FROZEN_BEFORE_LABEL_OUTCOMES',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'starting_commit': boundary['starting_commit'],
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config': config, 'instances': chosen,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
        'gate_sha256': digest('outputs/ngas_a1/audit/infrastructure_bank_gate.json'),
        'canonical_manifest_sha256': boundary['canonical_manifest_sha256'],
        'max_states': 6, 'max_joint_actions': 450,
        'maximum_decoder_calls_in_paired_labels': 16200,
        'post_pilot': 'write gate; no automatic training or larger labeling',
    }
    write_immutable(PROTOCOL, protocol)
    print(json.dumps({'status': protocol['status'], 'protocol_sha256': digest(PROTOCOL)}))


def verify_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'FROZEN_BEFORE_LABEL_OUTCOMES':
        raise ValueError('Unfrozen protocol')
    for path, expected in protocol['source_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Pilot source/config drift: {path}')
    gate_path = ROOT / 'outputs/ngas_a1/audit/infrastructure_bank_gate.json'
    if digest(gate_path) != protocol['gate_sha256']:
        raise ValueError('Infrastructure gate drift')
    gate = json.loads(gate_path.read_text())
    for path, expected in gate['supplemental_checkpoint_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Supplemental checkpoint drift: {path}')
    assert protocol['config']['continuation_policy'] == POLICY_VERSION
    assert tuple(protocol['config']['sampling_rules']) == SAMPLING_RULES
    assert protocol['config']['size_fractions'] == SIZE_FRACTIONS
    assert tuple(protocol['config']['repairs']) == REPAIR
    return protocol


def persist_or_verify(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f'Resumed deterministic artifact mismatch: {path}')
    else:
        write_immutable(path, value)


def build_states(instance, seed):
    h1 = solve_dispatching(instance, 'H1')
    initial = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    current = initial
    rngs = RNGStreams(instance.instance_id, seed)
    for step in range(4):
        key = f'source_native:{step}'
        action = fallback_action(instance, current, rngs, key)
        current, _, _, _ = transition(instance, current, action, rngs, key, 2, .05 * initial.makespan * .995**step)
    return {'H1': initial, 'NATIVE4': current}


def interaction_diagnostics(rows):
    # Remove target main effect and repair main effect within each size. Residual
    # RMS is target x repair nonadditivity. Also compute target-rank reversals
    # across repair pairs; every compared cell is the same size/target/CRN design.
    target_repair_residuals, size_repair_residuals = [], []
    reversals = comparisons = material_repair_pairs = repair_pairs = 0
    for size in ('small', 'medium', 'large'):
        subset = [r for r in rows if r['action']['size'] == size]
        by_target, by_repair = defaultdict(list), defaultdict(list)
        for row in subset:
            by_target[row['action']['target']['target_id']].append(row['advantage_mean'])
            by_repair[row['action']['repair']].append(row['advantage_mean'])
        overall = statistics.fmean(r['advantage_mean'] for r in subset)
        for row in subset:
            target_repair_residuals.append(row['advantage_mean'] - statistics.fmean(by_target[row['action']['target']['target_id']])
                                           - statistics.fmean(by_repair[row['action']['repair']]) + overall)
        lookup = {(r['action']['target']['target_id'], r['action']['repair']): r for r in subset}
        for target in by_target:
            for r1, r2 in itertools.combinations(sorted(by_repair), 2):
                repair_pairs += 1
                material_repair_pairs += int(separated_pair(lookup[target, r1], lookup[target, r2]))
        for t1, t2 in itertools.combinations(sorted(by_target), 2):
            for r1, r2 in itertools.combinations(sorted(by_repair), 2):
                a, b = lookup[t1, r1], lookup[t2, r1]
                c, d = lookup[t1, r2], lookup[t2, r2]
                if separated_pair(a, b) and separated_pair(c, d):
                    comparisons += 1
                    reversals += int((a['advantage_mean'] - b['advantage_mean']) * (c['advantage_mean'] - d['advantage_mean']) < 0)
    cells = defaultdict(list)
    for row in rows:
        cells[row['action']['size'], row['action']['repair']].append(row['advantage_mean'])
    means = {key: statistics.fmean(values) for key, values in cells.items()}
    overall = statistics.fmean(means.values())
    for (size, repair), value in means.items():
        size_mean = statistics.fmean(v for (s, r), v in means.items() if s == size)
        repair_mean = statistics.fmean(v for (s, r), v in means.items() if r == repair)
        size_repair_residuals.append(value - size_mean - repair_mean + overall)
    return {
        'target_repair_residual_rms': statistics.fmean(x*x for x in target_repair_residuals)**.5,
        'size_repair_residual_rms': statistics.fmean(x*x for x in size_repair_residuals)**.5,
        'noise_separated_target_rank_reversals_across_repairs': reversals,
        'noise_separated_rank_comparisons': comparisons,
        'within_target_size_noise_separated_repair_pairs': material_repair_pairs,
        'within_target_size_total_repair_pairs': repair_pairs,
        'caveat': 'Descriptive pilot interactions, not significance tests; size-conditioned target sets differ',
    }


def summarize(all_rows, config):
    states = defaultdict(list)
    for row in all_rows:
        states[row['state_id']].append(row)
    summaries = []
    for state_id, rows in sorted(states.items()):
        pairs = list(itertools.combinations(rows, 2))
        separated = sum(separated_pair(a, b) for a, b in pairs)
        sizes = sorted({r['action']['size'] for r in rows})
        repairs = sorted({r['action']['repair'] for r in rows})
        families = sorted({f for r in rows for f in r['action']['target']['origin_families']})
        summaries.append({
            'state_id': state_id, 'action_count': len(rows), 'sizes': sizes, 'repairs': repairs,
            'families': families, 'coverage_pass': len(sizes) == 3 and len(repairs) == 5 and len(families) == 5,
            'noise_separated_pairs': separated, 'total_pairs': len(pairs),
            'informative_pair_fraction': separated / len(pairs),
            'mean_advantage_std': statistics.fmean(r['advantage_variance']**.5 for r in rows),
            'positive_mean_advantage_fraction': statistics.fmean(float(r['advantage_mean'] > 0) for r in rows),
            **interaction_diagnostics(rows),
        })
    gates = config['gates']
    checks = {
        'all_six_states': len(states) == 6,
        'all_feasible': all(r['feasible'] for r in all_rows),
        'balanced_coverage_each_state': all(s['coverage_pass'] for s in summaries),
        'noise_usable': sum(s['informative_pair_fraction'] >= gates['min_informative_pair_fraction'] for s in summaries) >= gates['min_informative_states'],
        'positive_opportunities': statistics.fmean(float(r['advantage_mean'] > 0) for r in all_rows) >= gates['min_positive_mean_advantage_fraction'],
        'cost_usable': statistics.fmean(r['runtime_seconds'] for r in all_rows) <= gates['max_mean_seconds_per_joint_action_with_three_replicates'],
    }
    return {'schema': 'ngas-label-pilot-gate-v1', 'checks': checks,
            'decision': 'NGAS_A1_LABEL_PILOT_PASS' if all(checks.values()) else 'NGAS_A1_REVISE_LABELS',
            'states': summaries, 'action_count': len(all_rows),
            'paired_replicates': sum(len(r['replicates']) for r in all_rows),
            'mean_seconds_per_joint_action': statistics.fmean(r['runtime_seconds'] for r in all_rows),
            'total_label_seconds': sum(r['runtime_seconds'] for r in all_rows),
            'training_started': False, 'r13': 'LOCKED', 'r14': 'LOCKED'}


def run():
    protocol = verify_protocol()
    verify_boundary()
    protocol_sha = digest(PROTOCOL)
    config = protocol['config']
    all_rows = []
    for spec in protocol['instances']:
        path = 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/' + spec['relative_path']
        if digest(path) != spec['sha256']:
            raise ValueError('Instance hash mismatch')
        instance = load_instance(ROOT / path)
        for source, current in build_states(instance, config['source_seed']).items():
            state_id = instance.instance_id + '__' + source
            directory = OUT / 'states' / state_id
            rngs = RNGStreams(instance.instance_id, config['source_seed'])
            actions, banks = balanced_actions(instance, current, state_id, rngs)
            state = {'state_id': state_id, 'source': source, 'instance_sha256': spec['sha256'],
                     'candidate': asdict(current.candidate), 'makespan': current.makespan,
                     'schedule': current.schedule.to_dict(), 'protocol_sha256': protocol_sha}
            persist_or_verify(directory / 'state.json', state)
            state_sha = digest(directory / 'state.json')
            persist_or_verify(directory / 'action_manifest.json', {
                'state_sha256': state_sha, 'protocol_sha256': protocol_sha,
                'actions': [a.metadata() for a in actions],
                'full_bank_counts': [{'size': b.size, 'rules': b.requested_count, 'unique_targets': len(b.targets)} for b in banks],
            })
            for action in actions:
                path = directory / 'labels' / (action.action_id + '.json')
                if path.exists():
                    row = json.loads(path.read_text())
                    stored_hash = row.pop('payload_sha256')
                    if content_hash(row) != stored_hash or row['protocol_sha256'] != protocol_sha or row['state_sha256'] != state_sha or row['action'] != json.loads(json.dumps(action.metadata())):
                        raise ValueError(f'Resume payload mismatch: {path}')
                    row['payload_sha256'] = stored_hash
                else:
                    start = time.perf_counter()
                    label = label_action(instance, current, action, state_id, config['crn_seeds'],
                                         config['continuation_steps'], config['repair_trials'])
                    row = {**label, 'state_id': state_id, 'source': source,
                           'instance_id': instance.instance_id, 'instance_sha256': spec['sha256'],
                           'state_sha256': state_sha, 'protocol_sha256': protocol_sha,
                           'runtime_seconds': time.perf_counter() - start}
                    row['payload_sha256'] = content_hash(row)
                    write_immutable(path, row)
                all_rows.append(row)
                progress = {
                    'status': 'RUNNING', 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'completed_actions': len(all_rows), 'maximum_actions': 450,
                    'state_id': state_id, 'protocol_sha256': protocol_sha,
                    'mean_action_seconds': statistics.fmean(r['runtime_seconds'] for r in all_rows),
                }
                atomic_json(OUT / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    result = summarize(all_rows, config)
    result['protocol_sha256'] = protocol_sha
    verify_boundary()
    result['frozen_evidence_unchanged'] = True
    persist_or_verify(OUT / 'gate.json', result)
    hash_manifest = {str(p.relative_to(ROOT)): digest(p) for p in sorted((OUT / 'states').rglob('*.json'))}
    persist_or_verify(OUT / 'result_hash_manifest.json', hash_manifest)
    report = ROOT / 'docs/reports/ngas_a1/06_joint_action_label_pilot.md'
    report.write_text(f'''# Joint-action label pilot

Decision: **{result['decision']}**. {len(all_rows)} joint actions, {result['paired_replicates']}
paired CRN replicates across six states. R12 DEVELOPMENT only; R13/R14 remain locked.
No training or expanded labeling was launched.

Action = (size, exact target, repair), with the encoded repair executed in every
label. Each first move uses two repair trials and simulated-annealing acceptance;
two subsequent native iterations use the same versioned policy in both branches.
Both branches retain the source state's incumbent. Advantage is
(fallback best - action best) / source makespan. Independent component namespaces
share the same per-step keys between branches. Feasibility, executed actions,
acceptance, immediate improvement, continuation mean/variance and beats-fallback
frequency are stored per action with three paired replicates.

Every size includes critical-sync priority, related variant, matched random control,
local perturbation and structured neighbor targets, crossed with all five repairs.
Exact target collisions are deduplicated. No model exists yet, so neural-priority
sampling is unavailable and no neural prediction is claimed. This short two-trial
pilot policy must not be relabeled as the eventual eight-trial online policy.

Noise-separated pairs require abs(mean paired difference) > max(.001, 2*paired SE).
Gate thresholds were frozen before outcomes in `configs/ngas_a1_label_pilot.json`.
Checks: `{json.dumps(result['checks'], sort_keys=True)}`.
Mean cost: {result['mean_seconds_per_joint_action']:.3f} s per joint action (three paired replicates).
State diagnostics and target/repair rank reversals are in `label_pilot/gate.json`.
Residual interaction RMS is descriptive, not a significance result. The diagonal
S/CF1, M/CF2, L/CF3 subset confounds scale and CF; no subgroup claims are valid.

Repair stays in the architecture regardless of this pilot's measured interaction.
If PASS, the next step is a separately frozen expanded development labeling/training
design with instance-level separation and multiple training seeds. If REVISE_LABELS,
stop before training and diagnose noise, opportunity coverage and policy horizon.
Raw rows and their hash manifest remain immutable and resumable.
''')
    atomic_json(OUT / 'progress.json', {**progress, 'status': 'COMPLETE', 'decision': result['decision']})
    atomic_json(ROOT / 'outputs/ngas_a1/final_decision.json', {
        'stage': 'A1.2', 'decision': result['decision'], 'status': 'PILOT_COMPLETE',
        'next_stage': 'A1.3_PREREGISTRATION' if all(result['checks'].values()) else 'REVISE_LABELS',
        'training_started': False, 'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    })
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.freeze:
            freeze()
        else:
            run()


if __name__ == '__main__':
    main()
