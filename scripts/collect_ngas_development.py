#!/usr/bin/env python3
"""Frozen all-R12 development collection; training is a separate gated action."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import Candidate, candidate_from_actions, decode_candidate
from rcias_ngas.csg.features import RULES, action_features, state_features
from rcias_ngas.critic.development import DEVELOPMENT_VERSION, sample_actions, source_state, state_plan
from rcias_ngas.critic.label_revision import aggregate, evaluate_fallback, evaluate_replicate, replicate_seeds
from rcias_ngas.critic.dataset import separated_pair
from rcias_ngas.evaluation.bks import write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.run_ngas_label_pilot import atomic_json
from scripts.run_ngas_label_pilot_v2 import load_record, save_record, verify_protocol as verify_v2

OUT = ROOT / 'outputs/ngas_a1/development_v1'
CONFIG = ROOT / 'configs/ngas_a1_development_v1.json'


def freeze():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Development protocol already frozen')
    verify_v2()
    audit_path = ROOT / 'outputs/ngas_a1/audit/v2_completion.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'PASS' and audit['decision'] == 'NGAS_A1_LABEL_PILOT_PASS'
    config = json.loads(CONFIG.read_text())
    assert (config['maximum_states'], config['replicates'], config['repair_trials'], config['continuation_steps']) == (72, 9, 8, 2)
    manifest_path = ROOT / 'outputs/frozen_2o_baselines/instance_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['instances']) == 18
    plan = state_plan(manifest['instances'], config['canonical_root_seeds'])
    assert len(plan) == config['maximum_states']
    with (OUT / 'regression.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(OUT / 'regression.xml')],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    suite = ET.parse(OUT / 'regression.xml').getroot().find('testsuite')
    sources = list((ROOT / 'rcias_ngas').rglob('*.py')) + list((ROOT / 'rcias_clgri').rglob('*.py'))
    sources += list((ROOT / 'scripts').glob('*ngas*.py')) + [CONFIG]
    protected = json.loads((ROOT / 'outputs/ngas_a1/label_pilot_v2/result_hash_manifest.json').read_text())
    for name in ('gate.json', 'protocol.json', 'progress.json', 'final_decision.json', 'result_hash_manifest.json'):
        path = ROOT / 'outputs/ngas_a1/label_pilot_v2' / name
        protected[str(path.relative_to(ROOT))] = digest(path)
    protected[str(audit_path.relative_to(ROOT))] = digest(audit_path)
    for path, expected in protected.items():
        assert digest(path) == expected
    write_immutable(OUT / 'protocol.json', {
        'schema': DEVELOPMENT_VERSION, 'status': 'FROZEN_BEFORE_EXPANDED_LABELS',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config': config, 'states': plan, 'instance_manifest_sha256': digest(manifest_path),
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))},
        'protected_v2_hashes': protected, 'full_regression_tests': int(suite.get('tests')),
        'regression_xml_sha256': digest(OUT / 'regression.xml'),
    })
    print(json.dumps({'status': 'FROZEN_BEFORE_EXPANDED_LABELS', 'tests': int(suite.get('tests')),
                      'states': len(plan), 'protocol_sha256': digest(OUT / 'protocol.json')}))


def verify_protocol():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['status'] == 'FROZEN_BEFORE_EXPANDED_LABELS'
    for field in ('source_hashes', 'protected_v2_hashes'):
        for path, expected in p[field].items():
            if digest(path) != expected:
                raise ValueError(f'Frozen development boundary changed: {path}')
    verify_v2()
    return p


def state_summary(spec, labels):
    from itertools import combinations
    count = len(labels) * (len(labels) - 1) // 2
    separated = sum(separated_pair(a, b) for a, b in combinations(labels, 2))
    return {'state_id': spec['state_id'], 'instance_id': spec['instance']['instance_id'],
            'fold': spec['fold'], 'source': spec['source'], 'actions': len(labels),
            'informative_pair_fraction': separated / count,
            'positive_action_fraction': statistics.fmean(float(r['advantage_mean'] > 0) for r in labels),
            'feasible': all(r['feasible'] for r in labels),
            'sizes': sorted({r['action']['size'] for r in labels}),
            'repairs': sorted({r['action']['repair'] for r in labels}),
            'families': sorted({f for r in labels for f in r['action']['target']['origin_families']}),
            'rules': sorted({f for r in labels for f in r['action']['target']['origin_rules']})}


def data_gate(summaries, config):
    gates = config['data_gates']
    folds = {str(f): statistics.fmean(float(s['informative_pair_fraction'] >= gates['informative_state_pair_fraction_min'])
                                     for s in summaries if s['fold'] == f) for f in range(3)}
    checks = {
        'all_states': len(summaries) == config['maximum_states'],
        'all_feasible': all(s['feasible'] for s in summaries),
        'balanced_coverage': all(len(s['sizes']) == 3 and len(s['repairs']) == 5 and len(s['families']) == 5 for s in summaries),
        'all_rules': set(RULES) <= {r for s in summaries for r in s['rules']},
        'informative_each_fold': all(v >= gates['each_fold_informative_state_fraction_min'] for v in folds.values()),
        'positive_opportunities': sum(s['positive_action_fraction'] * s['actions'] for s in summaries) / sum(s['actions'] for s in summaries) >= gates['positive_action_fraction_min'],
    }
    return {'checks': checks, 'informative_state_fraction_by_fold': folds,
            'decision': 'READY_FOR_JOINT_CRITIC_TRAINING' if all(checks.values()) else 'NGAS_A1_REVISE_DEVELOPMENT_LABELS'}


def run():
    protocol = verify_protocol()
    config = protocol['config']
    protocol_sha = digest(OUT / 'protocol.json')
    initial_by_instance = {}
    completed_actions = new_actions = evaluations = 0
    compute_seconds = 0.
    source_evaluations = 0
    summaries = []
    all_paths = set()
    for spec in protocol['states']:
        instance_id = spec['instance']['instance_id']
        if instance_id not in initial_by_instance:
            path = ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / spec['instance']['relative_path']
            assert digest(path) == spec['instance']['sha256']
            instance = load_instance(path)
            h1 = solve_dispatching(instance, 'H1')
            initial_by_instance[instance_id] = instance, decode_candidate(instance, candidate_from_actions(instance, h1.actions))
        instance, initial = initial_by_instance[instance_id]
        directory = OUT / 'states' / spec['state_id']
        context = {'protocol_sha256': protocol_sha, 'state_id': spec['state_id'], 'instance_sha256': spec['instance']['sha256']}
        state_path = directory / 'state.json'
        all_paths.add(state_path)
        if state_path.exists():
            record = load_record(state_path, context)
            current = decode_candidate(instance, Candidate(**{k: tuple(v) for k, v in record['candidate'].items()}))
            assert current.feasible and current.makespan == record['makespan']
        else:
            current, source_trace = source_state(instance, initial, spec)
            record = save_record(state_path, {'context': context, 'candidate': asdict(current.candidate),
                                              'makespan': current.makespan, 'source_trace': source_trace, 'fold': spec['fold'],
                                              'csg_features': state_features(instance, current, spec['state_id'])})
        source_evaluations += sum(r['decoder_evals'] for r in record['source_trace'])
        state_sha = digest(state_path)
        context = {**context, 'state_sha256': state_sha}
        actions, banks = sample_actions(instance, current, spec)
        assert len(actions) <= config['maximum_actions_per_state']
        input_data = {'context': context, 'actions': [a.metadata() for a in actions],
                      'full_banks': [asdict(b) for b in banks],
                      'action_features': action_features(record['csg_features'], actions)}
        action_path = directory / 'actions.json'
        all_paths.add(action_path)
        if action_path.exists():
            existing = load_record(action_path, context)
            assert all(existing[k] == json.loads(json.dumps(v)) for k, v in input_data.items())
        else:
            save_record(action_path, input_data)
        seeds = replicate_seeds(instance_id, spec['state_id'], config['canonical_root_seeds'], 3)
        fallbacks = []
        for index, seed in enumerate(seeds):
            path = directory / 'fallback' / f'{index:02d}.json'
            all_paths.add(path)
            key = {**context, 'seed': seed, 'trials': 8, 'steps': 2}
            if path.exists():
                fallback = load_record(path, key)
            else:
                started = time.perf_counter()
                trajectory = evaluate_fallback(instance, current, spec['state_id'], seed, 2, 8)
                fallback = save_record(path, {'context': key, 'trajectory': trajectory, 'seconds': time.perf_counter() - started})
            fallbacks.append(fallback)
            evaluations += 24
            compute_seconds += fallback['seconds']
        local = []
        for action in actions:
            path = directory / 'labels' / f'{action.action_id}.json'
            all_paths.add(path)
            key = {**context, 'action_id': action.action_id, 'fallback_hashes': [r['payload_sha256'] for r in fallbacks]}
            if path.exists():
                label = load_record(path, key)
            else:
                started = time.perf_counter()
                replicates = [evaluate_replicate(instance, current, action, spec['state_id'], seed, 2, 8, f['trajectory'])
                              for seed, f in zip(seeds, fallbacks)]
                payload = aggregate(action.metadata(), current.makespan, replicates, 8, 2)
                label = save_record(path, {'context': key, **payload, 'candidate_seconds': time.perf_counter() - started})
                new_actions += 1
            local.append(label)
            completed_actions += 1
            evaluations += 216
            compute_seconds += label['candidate_seconds']
            progress = {'status': 'RUNNING', 'pid': os.getpid(), 'protocol_sha256': protocol_sha,
                        'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'state_id': spec['state_id'],
                        'completed_states': len(summaries), 'total_states': 72,
                        'completed_actions': completed_actions, 'maximum_actions': 6480,
                        'new_actions_this_process': new_actions,
                        'retained_label_decoder_evals': evaluations, 'retained_label_compute_seconds': compute_seconds,
                        'source_search_decoder_evals': source_evaluations,
                        'seconds_per_decoder': compute_seconds / evaluations,
                        'remaining_label_decoder_upper_bound': max(0, config['maximum_label_decoder_calls'] - evaluations)}
            atomic_json(OUT / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        summaries.append(state_summary(spec, local))
    assert all_paths == set((OUT / 'states').rglob('*.json'))
    assert source_evaluations == config['source_search_decoder_calls']
    result = {'schema': DEVELOPMENT_VERSION, **data_gate(summaries, config),
              'states': summaries, 'action_count': completed_actions, 'label_decoder_evals': evaluations,
              'source_search_decoder_evals': source_evaluations, 'label_compute_seconds': compute_seconds,
              'protocol_sha256': protocol_sha, 'training_started': False, 'r13': 'LOCKED', 'r14': 'LOCKED'}
    verify_protocol()
    for path, value in ((OUT / 'data_gate.json', result),
                        (OUT / 'result_hash_manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(all_paths)})):
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write_immutable(path, value)
    atomic_json(OUT / 'progress.json', {**progress, 'status': 'COMPLETE', 'completed_states': 72, 'decision': result['decision']})
    (ROOT / 'docs/reports/ngas_a1/07b_development_collection_result.md').write_text(f'''# Expanded development collection

Decision: **{result['decision']}**. 72 planned R12 DEVELOPMENT states completed;
{completed_actions} deduplicated joint actions. Labels use nine paired CRN replicates,
eight repair trials, two continuation moves, and the frozen V2 primary semantics.
Retained label computation: {evaluations:,} decoder calls, {compute_seconds:.1f} seconds.
Native source generation: {source_evaluations:,} explicit decoder calls; H1 initialization,
source replays, CSG construction, storage and interrupted unpersisted work are additional.

Checks: `{json.dumps(result['checks'], sort_keys=True)}`. Complete per-state/fold
diagnostics and file hashes are under `outputs/ngas_a1/development_v1/`.
All cell replicates and source states stay within the same structural fold.
No training was launched automatically. On READY_FOR_JOINT_CRITIC_TRAINING, verify
the completed artifact contract and execute the preregistered 3-seed, 3-fold fixed-
epoch training. Otherwise diagnose the data gate before training. R13/R14 stay locked.
''')
    print(json.dumps({'status': 'COMPLETE', 'decision': result['decision']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze() if args.freeze else run()


if __name__ == '__main__':
    main()
