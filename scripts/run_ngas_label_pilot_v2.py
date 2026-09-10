#!/usr/bin/env python3
"""Pre-outcome frozen A1.2 revision with immutable per-replicate recovery."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.bank.provenance import Target
from rcias_ngas.critic.label_revision import (REVISION, aggregate, evaluate_fallback,
                                             evaluate_replicate, replicate_seeds)
from rcias_ngas.evaluation.bks import content_hash, write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.run_ngas_label_pilot import (atomic_json, summarize, verify_boundary,
                                         verify_protocol as verify_v1)

OUT = ROOT / 'outputs/ngas_a1/label_pilot_v2'
V1 = ROOT / 'outputs/ngas_a1/label_pilot'
CONFIG = ROOT / 'configs/ngas_a1_label_pilot_v2.json'


def save_record(path, payload):
    """Atomic publish without replacing any existing result, even after crashes."""
    record = {**payload, 'payload_sha256': content_hash(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return record


def load_record(path, context):
    row = json.loads(path.read_text())
    expected = row.pop('payload_sha256')
    if content_hash(row) != expected or row['context'] != context:
        raise ValueError(f'Corrupt or incompatible result: {path}')
    row['payload_sha256'] = expected
    return row


def freeze():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('V2 protocol already frozen')
    verify_v1()
    verify_boundary()
    audit_path = ROOT / 'outputs/ngas_a1/audit/label_pilot_completion.json'
    audit = json.loads(audit_path.read_text())
    assert audit['integrity'] == 'PASS' and audit['pilot_decision'] == 'NGAS_A1_REVISE_LABELS'
    with (OUT / 'regression.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(OUT / 'regression.xml')],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    suite = ET.parse(OUT / 'regression.xml').getroot().find('testsuite')
    protected = json.loads((V1 / 'result_hash_manifest.json').read_text())
    for name in ('protocol.json', 'gate.json', 'result_hash_manifest.json', 'progress.json'):
        path = V1 / name
        protected[str(path.relative_to(ROOT))] = digest(path)
    protected[str(audit_path.relative_to(ROOT))] = digest(audit_path)
    sources = list((ROOT / 'rcias_ngas').rglob('*.py')) + list((ROOT / 'rcias_clgri').rglob('*.py'))
    sources += list((ROOT / 'scripts').glob('*ngas*.py')) + [CONFIG]
    config = json.loads(CONFIG.read_text())
    assert config['revision'] == REVISION
    protocol = {
        'schema': 'ngas-a12-revision-freeze-v2', 'status': 'FROZEN_BEFORE_V2_OUTCOMES',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config': config,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))},
        'protected_v1_hashes': protected,
        'full_regression_tests': int(suite.get('tests')),
        'regression_xml_sha256': digest(OUT / 'regression.xml'),
        'r13': 'LOCKED', 'r14': 'LOCKED', 'training_started': False,
    }
    verify_v1()
    verify_boundary()
    for path, expected in protected.items():
        assert digest(path) == expected
    write_immutable(OUT / 'protocol.json', protocol)
    print(json.dumps({'status': protocol['status'], 'tests': protocol['full_regression_tests'],
                      'protocol_sha256': digest(OUT / 'protocol.json')}))


def verify_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if protocol['status'] != 'FROZEN_BEFORE_V2_OUTCOMES':
        raise ValueError('V2 protocol is not frozen')
    for field in ('source_hashes', 'protected_v1_hashes'):
        for path, expected in protocol[field].items():
            if digest(path) != expected:
                raise ValueError(f'Frozen boundary drift: {path}')
    verify_v1()
    verify_boundary()
    return protocol


def action_from_metadata(metadata):
    target = Target(**{k: tuple(v) if isinstance(v, list) else v for k, v in metadata['target'].items()})
    action = JointAction(metadata['size'], target, metadata['repair'])
    if action.action_id != metadata['action_id']:
        raise ValueError('Frozen action identity mismatch')
    return action


def final_summary(rows_by_variant, config):
    summaries = {name: summarize(rows, config) for name, rows in rows_by_variant.items()}
    primary = summaries[config['primary_variant']]
    first_three = {}
    blocks = {}
    for name, rows in rows_by_variant.items():
        def restricted(indices):
            selected = []
            for row in rows:
                label = aggregate(row['action'], row['initial_makespan'],
                                  [row['replicates'][i] for i in indices],
                                  row['repair_trials'], row['continuation_steps'])
                selected.append({**row, **label})
            return summarize(selected, config)
        first_three[name] = restricted(range(3))
        blocks[name] = [restricted(range(i, 9, 3)) for i in range(3)]
    sparsity = {}
    for name, rows in rows_by_variant.items():
        sparsity[name] = []
        for state_id in sorted({r['state_id'] for r in rows}):
            local = [r for r in rows if r['state_id'] == state_id]
            reps = [(r, rep) for r in local for rep in r['replicates']]
            sparsity[name].append({
                'state_id': state_id,
                'zero_advantage_fraction': statistics.fmean(float(rep['advantage'] == 0) for r, rep in reps),
                'candidate_no_new_best_fraction': statistics.fmean(float(rep['candidate']['best_makespan'] == r['initial_makespan']) for r, rep in reps),
                'mean_advantage_std': statistics.fmean(r['advantage_variance']**.5 for r in local),
            })
    # Matched seeds and action IDs make the repair-effort comparison explicit.
    control = {(r['state_id'], r['action']['action_id']): r for r in rows_by_variant['T2_R9']}
    primary_rows = rows_by_variant['T8_R9']
    matched = []
    for row in primary_rows:
        other = control[row['state_id'], row['action']['action_id']]
        assert [r['crn_seed'] for r in row['replicates']] == [r['crn_seed'] for r in other['replicates']]
        matched.append(row['advantage_mean'] - other['advantage_mean'])
    return {
        'schema': 'ngas-a12-v2-gate', 'decision': primary['decision'],
        'primary_variant': config['primary_variant'], 'no_control_rescue': True,
        'variants': summaries, 'first_three_replicates_diagnostic': first_three,
        'canonical_root_blocks_diagnostic': blocks,
        'sparsity_diagnostic': sparsity,
        'matched_mean_advantage_T8_minus_T2': statistics.fmean(matched),
        'primary_positive_difference_fraction': statistics.fmean(float(v > 0) for v in matched),
        'action_rows': sum(len(r) for r in rows_by_variant.values()),
        'paired_replicates': sum(len(r['replicates']) for rows in rows_by_variant.values() for r in rows),
        'training_started': False, 'r13': 'LOCKED', 'r14': 'LOCKED',
    }


def run():
    protocol = verify_protocol()
    config = protocol['config']
    protocol_sha = digest(OUT / 'protocol.json')
    v1_protocol = json.loads((V1 / 'protocol.json').read_text())
    rows_by_variant = {v['id']: [] for v in config['variants']}
    seen_evals = 0
    seen_seconds = 0.
    new_replicates = completed_replicates = 0
    expected_files = set()
    process_started = time.perf_counter()
    for state_path in sorted((V1 / 'states').glob('*/state.json')):
        state = json.loads(state_path.read_text())
        state_sha = digest(state_path)
        state_id = state['state_id']
        spec = next(r for r in v1_protocol['instances'] if r['instance_id'] == state_id.split('__')[0])
        instance = load_instance(ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / spec['relative_path'])
        current = decode_candidate(instance, Candidate(**{k: tuple(v) for k, v in state['candidate'].items()}))
        if not current.feasible or current.makespan != state['makespan']:
            raise ValueError('Source state replay failed')
        actions = json.loads((state_path.parent / 'action_manifest.json').read_text())['actions']
        assert len(actions) == config['actions_per_state']
        seeds = replicate_seeds(instance.instance_id, state_id, config['canonical_root_seeds'], config['independent_repeats_per_root'])
        for variant in config['variants']:
            trials, steps = variant['repair_trials'], config['continuation_steps']
            directory = OUT / 'states' / state_id / variant['id']
            fallback_rows = []
            for index, seed in enumerate(seeds):
                context = {'protocol_sha256': protocol_sha, 'state_sha256': state_sha,
                           'state_id': state_id, 'variant': variant['id'], 'crn_seed': seed,
                           'trials': trials, 'steps': steps}
                path = directory / 'fallback' / f'replicate_{index:02d}.json'
                expected_files.add(path)
                if path.exists():
                    record = load_record(path, context)
                else:
                    start = time.perf_counter()
                    trajectory = evaluate_fallback(instance, current, state_id, seed, steps, trials)
                    record = save_record(path, {'context': context, 'trajectory': trajectory,
                                                'runtime_seconds': time.perf_counter() - start})
                fallback_rows.append(record)
                seen_evals += trials * (steps + 1)
                seen_seconds += record['runtime_seconds']
            for metadata in actions:
                action = action_from_metadata(metadata)
                replicates, candidate_seconds = [], 0.
                for index, seed in enumerate(seeds):
                    fallback = fallback_rows[index]
                    context = {**fallback['context'], 'action_id': action.action_id,
                               'fallback_sha256': fallback['payload_sha256']}
                    path = directory / 'replicates' / action.action_id / f'replicate_{index:02d}.json'
                    expected_files.add(path)
                    if path.exists():
                        record = load_record(path, context)
                    else:
                        start = time.perf_counter()
                        label = evaluate_replicate(instance, current, action, state_id, seed, steps, trials, fallback['trajectory'])
                        record = save_record(path, {'context': context, 'replicate': label,
                                                    'runtime_seconds': time.perf_counter() - start})
                        new_replicates += 1
                    replicates.append(record['replicate'])
                    candidate_seconds += record['runtime_seconds']
                    seen_seconds += record['runtime_seconds']
                    seen_evals += trials * (steps + 1)
                    completed_replicates += 1
                    progress = {
                        'status': 'RUNNING', 'pid': os.getpid(),
                        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                        'protocol_sha256': protocol_sha, 'state_id': state_id, 'variant': variant['id'],
                        'completed_paired_replicates': completed_replicates, 'total_paired_replicates': 8100,
                        'new_replicates_this_process': new_replicates,
                        'completed_action_rows': sum(len(r) for r in rows_by_variant.values()),
                        'maximum_action_rows': 900, 'actual_label_decoder_evals': seen_evals,
                        'remaining_decoder_upper_bound': max(0, config['maximum_label_decoder_calls'] - seen_evals),
                        'observed_seconds_per_decoder': seen_seconds / seen_evals,
                        'process_elapsed_seconds': time.perf_counter() - process_started,
                    }
                    atomic_json(OUT / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                row = {
                    **aggregate(metadata, current.makespan, replicates, trials, steps),
                    'state_id': state_id, 'variant': variant['id'], 'state_sha256': state_sha,
                    'protocol_sha256': protocol_sha,
                    'candidate_runtime_seconds': candidate_seconds,
                    'amortized_fallback_seconds': sum(r['runtime_seconds'] for r in fallback_rows) / len(actions),
                    'runtime_seconds': candidate_seconds + sum(r['runtime_seconds'] for r in fallback_rows) / len(actions),
                }
                path = directory / 'labels' / (action.action_id + '.json')
                expected_files.add(path)
                if path.exists():
                    assert load_record(path, {'protocol_sha256': protocol_sha})['label'] == json.loads(json.dumps(row))
                else:
                    save_record(path, {'context': {'protocol_sha256': protocol_sha}, 'label': row})
                rows_by_variant[variant['id']].append(row)
    assert seen_evals == config['maximum_label_decoder_calls']
    assert expected_files == set((OUT / 'states').rglob('*.json'))
    result = final_summary(rows_by_variant, config)
    assert result['action_rows'] == 900 and result['paired_replicates'] == 8100
    result.update({'protocol_sha256': protocol_sha, 'actual_label_decoder_evals': seen_evals,
                   'equivalent_uncached_label_decoder_evals': config['candidate_decoder_calls'] * 2,
                   'actual_label_compute_seconds': seen_seconds, 'v1_evidence_unchanged': True})
    verify_protocol()
    def persist(path, value):
        if path.exists():
            assert json.loads(path.read_text()) == json.loads(json.dumps(value))
        else:
            write_immutable(path, value)
    persist(OUT / 'gate.json', result)
    persist(OUT / 'result_hash_manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(expected_files)})
    primary = result['variants'][config['primary_variant']]
    report = ROOT / 'docs/reports/ngas_a1/06b_label_pilot_v2_results.md'
    report.write_text(f'''# A1.2-v2 label revision results

Decision: **{result['decision']}**, determined only by preregistered T8_R9.
Control T2_R9 is diagnostic; there is no rescue or replacement of the failed V1 gate.
Completed 900 action/variant rows and 8,100 paired replicates on the same six states.
Actual label work: {seen_evals:,} decoder calls, {seen_seconds:.2f} compute seconds.
Equivalent uncached work: {result['equivalent_uncached_label_decoder_evals']:,} decoder calls.
Actual costs count fallback cache creation once, amortized equally over each state's
75 actions for the per-action cost gate. Source replay and file I/O are additional
worker overhead, not silently counted as decoder work.

Primary checks: `{json.dumps(primary['checks'], sort_keys=True)}`.
The .001 material gap, twice paired SE, >=10% informative pairs in at least three
states, feasibility, coverage, positive opportunities and 30-second cost cap remain.
Nine independent repeats are derived from three canonical root seeds; V1 samples
are not pooled. Both variants share the same new per-step CRN keys and joint actions.
Encoded repair identity is preserved in every label and checked before publishing.

Detailed first-three/all-nine, root-block, matched T8/T2 and repair-interaction
diagnostics are in `outputs/ngas_a1/label_pilot_v2/gate.json`. These reused-state
development diagnostics are not untouched-test evidence or subgroup performance.
V1 files and frozen comparator hashes were reverified after completion.

No model training or expanded labels were launched. On PASS, preregister expanded
development labeling and instance-separated critic training. Otherwise remain in
A1.2 label revision. R13/R14 remain locked; Gurobi was not run.
''')
    atomic_json(OUT / 'progress.json', {**progress, 'status': 'COMPLETE', 'completed_action_rows': 900,
                                       'decision': result['decision']})
    atomic_json(OUT / 'final_decision.json', {'stage': 'A1.2-v2', 'decision': result['decision'],
                                            'primary_variant': 'T8_R9', 'training_started': False,
                                            'v1_decision': 'NGAS_A1_REVISE_LABELS'})
    print(json.dumps({'decision': result['decision'], 'status': 'COMPLETE'}), flush=True)


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
