#!/usr/bin/env python3
"""Correct duplicated proposal construction, then re-audit the frozen E3 path."""
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6k_runtime as audit
from scripts import phase6k_runtime_common as c
from rcias_clgri.ni.phase6k_runtime import eager_reconstruction, DeviceDecision
from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble


def update_progress(directory, record):
    temporary = directory / 'progress.tmp.json'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(directory / 'progress.json')
    print(json.dumps(record), flush=True)


def main():
    old_selection = c.r.load_json(c.OUT / 'selection.json')
    if old_selection['status'] != 'MODEL_REVISION_RUNTIME':
        raise RuntimeError('correction applies only to the retained failed diagnostic selection')
    directory = c.freeze_stage('live_integration_correction', {
        'reason': 'diagnostic live path generated the same proposal bank twice after historical preprocessing',
        'correction': 'capture the generated bank and neutral records returned inside the original historical call',
        'historical_preprocessing_implementation_changed': False,
        'old_selection_sha256': c.digest(c.OUT / 'selection.json'),
        'requirements': {'full_states': 288, 'candidate_rows': 6809,
                         'repeated_states': 5, 'repetitions_per_repeated_state': 30,
                         'atol': 1e-5, 'rtol': 1e-5},
    }, [__file__, Path(c.__file__), Path(audit.__file__),
        ROOT / 'rcias_clgri/ni/phase6k_runtime.py',
        ROOT / 'rcias_clgri/ni/phase6k_vectorized.py'])
    env = c.setup()
    eager = VectorizedEnsemble(eager_reconstruction(env['e0']))
    gate = DeviceDecision(env['protocol']).cuda().eval()
    options = c.r.load_json(c.ACTIVE)['runtime_audit']['compile']
    model = torch.compile(eager, mode=options['mode'], fullgraph=options['fullgraph'], dynamic=options['dynamic'])
    started = time.perf_counter()
    rows = []
    repeated_rows = []
    with torch.inference_mode():
        # Exercise every preregistered shape cell before warm diagnostic timing.
        for sid in c.cell_states(env):
            row = audit.check_pair(env, model, gate, c.context(env, sid), 0, 'E3')
            if not row['pass']:
                raise RuntimeError(f'corrected E3 warm shape failed parity: {sid}')
        update_progress(directory, {'stage': 'VARIABLE_SHAPE_WARMUP', 'completed': 9,
                                    'elapsed_seconds': time.perf_counter() - started})
        for sid in sorted(env['frames']):
            rows.append(audit.check_pair(env, model, gate, c.context(env, sid), 0, 'E3'))
            if len(rows) % 24 == 0:
                update_progress(directory, {'stage': 'FULL_PAIRED_EQUIVALENCE',
                    'states_completed': len(rows), 'failed_pairs': sum(not x['pass'] for x in rows),
                    'elapsed_seconds': time.perf_counter() - started})
        repeated_ids = c.r.load_json(
            ROOT / 'outputs/phase6k_runtime_v1/diagnostics/preprocessing_v2/protocol.json')['state_ids']
        for sid in repeated_ids:
            context = c.context(env, sid)
            for repetition in range(30):
                repeated_rows.append(audit.check_pair(env, model, gate, context, repetition, 'E3'))
            update_progress(directory, {'stage': 'REPEATED_REALIZATION',
                'pairs_completed': len(repeated_rows),
                'failed_pairs': sum(not x['pass'] for x in repeated_rows),
                'elapsed_seconds': time.perf_counter() - started})
        passed = all(x['pass'] for x in rows + repeated_rows)
        latency = audit.diagnostic_latency(env, model, gate) if passed else None
    from torch._dynamo.utils import counters
    counter_summary = {str(key): {str(a): int(b) for a, b in value.items()}
                       for key, value in counters.items()}
    result = {'status': 'PASS' if passed else 'FAIL', 'variant': 'E3',
              'rows': rows, 'robustness': repeated_rows,
              'candidate_rows': sum(x['candidate_rows'] for x in rows),
              'diagnostic_latency': latency, 'counters': counter_summary,
              'elapsed_seconds': time.perf_counter() - started,
              'r13_accessed': False, 'r14_accessed': False}
    c.write_once(directory / 'result.json', result)
    print(json.dumps({'status': result['status'], 'candidate_rows': result['candidate_rows'],
        'diagnostic_latency': {k: v for k, v in latency.items() if k != 'rows'} if latency else None,
        'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
