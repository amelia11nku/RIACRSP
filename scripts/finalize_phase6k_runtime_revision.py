#!/usr/bin/env python3
"""Audit and close Phase 6K at the preregistered runtime gate."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_phase6k_start import digest, write_once
from scripts.measure_phase6k_formal_latency import verify_equivalence

OUT = ROOT / 'outputs/phase6k_runtime_v1'
RUNTIME = OUT / 'runtime'


def audit_formal_stream(rows, result, expected_ids):
    expected_ids = set(expected_ids)
    by_state = defaultdict(list)
    for row in rows:
        by_state[row['state_id']].append(row)
        if row['scale'] not in ('S', 'M', 'L'):
            raise ValueError('unexpected scale in formal stream')
        if any(not math.isfinite(float(row[key])) or float(row[key]) < 0
               for key in ('neural_ms', 'neural_cuda_span_ms', 'full_live_ms')):
            raise ValueError('formal stream contains invalid timing')
        if row['full_live_ms'] < row['neural_ms']:
            raise ValueError('full-live timing cannot be shorter than its neural subset')
        decision = row.get('decision', {})
        if not decision.get('selected_target_set_id'):
            raise ValueError('formal row has no selected action')
        timing = row.get('preprocessing_timings', {})
        if not {'historical', 'source_features_ms', 'j1_same_decision_reuse_ms'} <= timing.keys():
            raise ValueError('formal row omits required fresh-live timing scope')
    if set(by_state) != expected_ids:
        raise ValueError('formal stream state coverage changed')
    expected_repetitions = list(range(-3, 5))
    if any(sorted(row['repetition'] for row in state_rows) != expected_repetitions
           for state_rows in by_state.values()):
        raise ValueError('each state requires exactly three warmups and five measurements')
    measured = [row for row in rows if row['repetition'] >= 0]
    warmups = [row for row in rows if row['repetition'] < 0]
    if len(measured) != 5 * len(expected_ids) or len(warmups) != 3 * len(expected_ids):
        raise ValueError('formal row counts changed')
    scales = Counter(row['scale'] for row in measured)
    if len(expected_ids) == 288 and scales != Counter({'S': 480, 'M': 480, 'L': 480}):
        raise ValueError('formal scale balance changed')
    summaries = {}
    for scale in ('overall', 'S', 'M', 'L'):
        subset = measured if scale == 'overall' else [row for row in measured if row['scale'] == scale]
        summaries[scale] = {
            key: {f'p{q}': float(np.percentile([row[key] for row in subset], q))
                  for q in (50, 90, 99)}
            for key in ('neural_ms', 'full_live_ms')
        }
    for scale, metrics in summaries.items():
        for metric, values in metrics.items():
            for percentile, value in values.items():
                if not math.isclose(value, result['summary'][scale][metric][percentile],
                                    rel_tol=0, abs_tol=1e-9):
                    raise ValueError('stored percentile does not match raw measurements')
    neural_pass = summaries['overall']['neural_ms']['p90'] <= 30
    live_pass = summaries['overall']['full_live_ms']['p90'] <= 100
    expected_status = 'PASS' if neural_pass and live_pass else 'MODEL_REVISION_RUNTIME'
    if (result['status'], result['neural_cap_pass'], result['full_live_cap_pass']) != (
            expected_status, neural_pass, live_pass):
        raise ValueError('formal gate summary contradicts raw measurements')
    return summaries, measured, warmups


def main():
    completion_path = RUNTIME / 'formal_latency_e4r/completion_audit.json'
    final_path = OUT / 'final/final_decision_v2.json'
    if completion_path.exists() or final_path.exists():
        if not (completion_path.exists() and final_path.exists()):
            raise RuntimeError('partial terminal finalization requires inspection')
        print(json.dumps({'status': 'ALREADY_COMPLETE',
                          'decision': json.loads(final_path.read_text())['decision']}))
        return

    config_path = ROOT / 'configs/phase6k_runtime_v1_e4r.json'
    config = json.loads(config_path.read_text())
    if config['terminal_rule'] != ('if E4R fails equivalence, memory stability, or either formal '
                                  'latency cap, close MODEL_REVISION_RUNTIME without another '
                                  'optimization candidate'):
        raise ValueError('E4R terminal rule changed')
    equivalence_path = RUNTIME / 'e4r_equivalence/result.json'
    equivalence = json.loads(equivalence_path.read_text())
    expected_ids = json.loads((RUNTIME / 'e1_equivalence/protocol.json').read_text())['state_ids']
    repeated_ids = json.loads((RUNTIME / 'e1_equivalence/protocol.json').read_text())['repeated_state_ids']
    verify_equivalence(equivalence, expected_ids, repeated_ids)
    if equivalence['variant'] != 'E4R' or equivalence['maximum_absolute_error'] > 1e-5:
        raise ValueError('E4R runtime equivalence failed')

    measurement_path = RUNTIME / 'formal_latency_e4r/measurements.jsonl'
    result_path = RUNTIME / 'formal_latency_e4r/result.json'
    result = json.loads(result_path.read_text())
    rows = [json.loads(line) for line in measurement_path.read_text().splitlines()]
    summaries, measured, warmups = audit_formal_stream(rows, result, expected_ids)
    if digest(measurement_path) != result['measurements_sha256']:
        raise ValueError('formal measurement hash changed')
    if len(expected_ids) != 288 or len(rows) != 2304:
        raise ValueError('formal R12 stream is incomplete')
    if result['status'] != 'MODEL_REVISION_RUNTIME':
        raise ValueError('this finalizer only closes the failed runtime gate')

    protected_path = OUT / 'audit/protected_evidence.json'
    protected = json.loads(protected_path.read_text())
    if any(digest(ROOT / path) != record['sha256'] for path, record in protected.items()):
        raise ValueError('protected predecessor evidence changed')
    access_paths = [str(path.relative_to(ROOT))
                    for path in (ROOT / 'outputs/phase6j_caur').rglob('*')
                    if path.is_file() and any(token in path.name.lower()
                                              for token in ('r13', 'r14'))]
    if access_paths:
        raise ValueError('R13/R14 outcome access artifact detected')
    if (OUT / 'final/csgni_v1_freeze.json').exists():
        raise ValueError('a failed runtime candidate cannot have a v1 freeze')

    progress_path = RUNTIME / 'formal_latency_e4r/progress.json'
    progress_before = json.loads(progress_path.read_text())
    if progress_before.get('completed_states') != 288:
        raise ValueError('progress marker does not reach all states')
    raw_artifacts = [
        config_path,
        ROOT / 'configs/phase6k_runtime_v1_amended.json',
        OUT / 'amendment/approval.json',
        OUT / 'diagnostics/preprocessing_v2/completion_audit.json',
        RUNTIME / 'device_boundary_v2/result.json',
        RUNTIME / 'e1_equivalence/result.json',
        RUNTIME / 'e2_equivalence/result.json',
        RUNTIME / 'e3_compile/result.json',
        RUNTIME / 'formal_latency/result.json',
        RUNTIME / 'e3_stability/result.json',
        RUNTIME / 'e4s_equivalence/result.json',
        equivalence_path,
        RUNTIME / 'selection_v5.json',
        RUNTIME / 'formal_latency_e4r/protocol.json',
        measurement_path,
        result_path,
        OUT / 'audit/boundary_and_baseline_audit.json',
        OUT / 'audit/regression_v2.json',
        ROOT / 'configs/lghga_2o_baseline.json',
        ROOT / 'docs/reports/phase6k_lghga_2o_budget_fidelity.md',
        Path(__file__),
    ]
    completion = {
        'status': 'COMPLETE',
        'decision': 'MODEL_REVISION_RUNTIME',
        'formal_stream': {
            'states': len(expected_ids),
            'warmup_rows': len(warmups),
            'measured_rows': len(measured),
            'repetitions_per_state': sorted(Counter(row['repetition'] for row in rows)),
            'scale_measured_rows': dict(sorted(Counter(row['scale'] for row in measured).items())),
            'all_timings_finite': True,
            'all_decisions_present': True,
            'stored_percentiles_recomputed_exactly': True,
            'measurement_sha256': digest(measurement_path),
        },
        'runtime_equivalence': {
            'status': 'PASS',
            'states': len(equivalence['rows']),
            'candidate_rows': equivalence['candidate_rows'],
            'repeated_realization_pairs': len(equivalence['robustness']),
            'maximum_absolute_error': equivalence['maximum_absolute_error'],
            'all_pair_checks_pass': all(row['pass'] for row in
                                        equivalence['rows'] + equivalence['robustness']),
        },
        'summary': summaries,
        'neural_cap_pass': True,
        'full_live_cap_pass': False,
        'protected_files_unchanged': len(protected),
        'r13_r14_access_paths': access_paths,
        'r13_accessed': False,
        'r14_accessed': False,
        'progress_before_finalization': progress_before,
        'progress_before_sha256': digest(progress_path),
        'artifact_sha256': {str(path.relative_to(ROOT)): digest(path)
                            for path in raw_artifacts},
    }
    write_once(completion_path, completion)

    final = {
        'schema': 'phase6k-runtime-terminal-v2',
        'decision': 'MODEL_REVISION_RUNTIME',
        'reason': ('runtime-equivalent E4R passes the 30 ms neural gate but misses '
                   'the frozen 100 ms full-live p90 gate'),
        'stop_boundary': 'BEFORE_DEPLOYABLE_BUNDLE_FREEZE_AND_R12_SOLVER_GATE',
        'phase6k_success': False,
        'runtime_equivalence': completion['runtime_equivalence'],
        'formal_runtime': {
            'status': result['status'],
            'variant': 'E4R',
            'states': result['states'],
            'warmup_rows': result['warmup_rows'],
            'measured_rows': result['measured_rows'],
            'neural_p90_ms': summaries['overall']['neural_ms']['p90'],
            'neural_cap_ms': 30,
            'neural_cap_pass': True,
            'full_live_p90_ms': summaries['overall']['full_live_ms']['p90'],
            'full_live_cap_ms': 100,
            'full_live_cap_pass': False,
        },
        'historical_preprocessing_contract_preserved': True,
        'csg_ni_v1_frozen': False,
        'deployable_bundle': 'NOT_CREATED_RUNTIME_GATE_FAILED',
        'r12_solver_sanity': 'NOT_RUN_RUNTIME_GATE_FAILED',
        'r13_confirmation': 'LOCKED_NOT_RUN',
        'r14_promotion': 'LOCKED_NOT_RUN',
        'r13_accessed': False,
        'r14_accessed': False,
        'protected_evidence_unchanged': True,
        'supersedes_scientific_status_of': 'outputs/phase6k_runtime_v1/final/final_decision.json',
        'preserves_prior_hold_evidence': True,
        'completion_audit_sha256': digest(completion_path),
        'artifact_sha256': completion['artifact_sha256'],
        'next_step': ('start a separately preregistered model/runtime revision; do not run the '
                      'Phase 6K solver or access R13/R14 with this failed candidate'),
    }
    write_once(final_path, final)

    progress = {
        'status': 'MODEL_REVISION_RUNTIME',
        'completed_states': 288,
        'total_states': 288,
        'elapsed_seconds': result['elapsed_seconds'],
        'result_sha256': digest(result_path),
        'final_decision_sha256': digest(final_path),
    }
    temporary = progress_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, progress_path)
    print(json.dumps({'status': completion['status'], 'decision': final['decision'],
                      'neural_p90_ms': final['formal_runtime']['neural_p90_ms'],
                      'full_live_p90_ms': final['formal_runtime']['full_live_p90_ms']}))


if __name__ == '__main__':
    main()
