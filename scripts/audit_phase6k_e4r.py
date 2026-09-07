#!/usr/bin/env python3
"""Audit the preregistered same-decision tensor-reuse E4R candidate."""
import json
from pathlib import Path
import sys
import time

import torch
import torch._inductor.config as inductor_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6k_runtime as audit
from scripts import phase6k_runtime_common as c
from scripts.audit_phase6k_e3_stability import memory_record, update
from rcias_clgri.ni.phase6k_runtime import (
    eager_reconstruction, hoisted_qkv_reconstruction, DeviceDecision,
)
from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble

CONFIG = ROOT / 'configs/phase6k_runtime_v1_e4r.json'


def main():
    settings = c.r.load_json(CONFIG)
    if settings['status'] != 'E4R_FROZEN_BEFORE_IMPLEMENTATION':
        raise RuntimeError('invalid E4R amendment status')
    for key in ('e4s_config', 'e4s_result'):
        path = ROOT / settings['selection_trigger'][key]
        if c.digest(path) != settings['selection_trigger'][key + '_sha256']:
            raise RuntimeError(f'E4R trigger changed: {key}')
    state_ids = c.r.load_json(c.OUT / 'e1_equivalence/protocol.json')['state_ids']
    directory = c.freeze_stage('e4r_equivalence', {
        'e4r_config_sha256': c.digest(CONFIG),
        'state_ids': state_ids,
        'memory_sampling': 'after every paired R12 state in one process; no cache clearing',
        'candidate': settings['candidate'],
        'prepare_path': 'fresh historical preprocessing followed by same-decision tensor reuse',
    }, [__file__, Path(audit.__file__), Path(c.__file__),
        ROOT / 'rcias_clgri/ni/phase6k_runtime.py',
        ROOT / 'rcias_clgri/ni/phase6k_vectorized.py'])
    env = c.setup()
    if sorted(state_ids) != sorted(env['frames']):
        raise RuntimeError('R12 state boundary changed')
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    e2 = VectorizedEnsemble(eager_reconstruction(env['e0']))
    candidate = hoisted_qkv_reconstruction(e2)
    gate = DeviceDecision(env['protocol']).cuda().eval()
    runtime = settings['retained_runtime']
    model = torch.compile(candidate, mode=runtime['compile_mode'],
                          fullgraph=runtime['fullgraph'], dynamic=runtime['dynamic'])
    started = time.perf_counter()
    rows, repeated_rows, memory = [], [], []
    with torch.inference_mode():
        for sid in sorted(env['frames']):
            rows.append(audit.check_pair(env, model, gate, c.context(env, sid), 0, 'E4R',
                                         prepare_fn=audit.prepare_reused))
            memory.append({'state_id': sid, **memory_record()})
            if len(rows) % 24 == 0:
                update(directory, {'stage': 'FULL_PAIRED_MEMORY_STREAM',
                    'states_completed': len(rows), 'failed_pairs': sum(not row['pass'] for row in rows),
                    'memory': memory[-1], 'elapsed_seconds': time.perf_counter() - started})
        repeated_ids = c.r.load_json(
            ROOT / 'outputs/phase6k_runtime_v1/diagnostics/preprocessing_v2/protocol.json')['state_ids']
        for sid in repeated_ids:
            context = c.context(env, sid)
            for repetition in range(30):
                repeated_rows.append(audit.check_pair(
                    env, model, gate, context, repetition, 'E4R',
                    prepare_fn=audit.prepare_reused))
            update(directory, {'stage': 'REPEATED_REALIZATION',
                'pairs_completed': len(repeated_rows),
                'failed_pairs': sum(not row['pass'] for row in repeated_rows),
                'memory': memory_record(), 'elapsed_seconds': time.perf_counter() - started})
        passed = all(row['pass'] for row in rows + repeated_rows)
        latency = audit.diagnostic_latency(
            env, model, gate, prepare_fn=audit.prepare_reused) if passed else None
    from torch._dynamo.utils import counters
    counter_summary = {str(key): {str(a): int(b) for a, b in value.items()}
                       for key, value in counters.items()}
    result = {'status': 'PASS' if passed else 'FAIL', 'variant': 'E4R',
              'rows': rows, 'robustness': repeated_rows,
              'candidate_rows': sum(row['candidate_rows'] for row in rows),
              'memory_stream': memory,
              'maximum_reserved_bytes': max(row['reserved_bytes'] for row in memory),
              'minimum_driver_free_bytes': min(row['driver_free_bytes'] for row in memory),
              'diagnostic_latency': latency, 'counters': counter_summary,
              'maximum_absolute_error': max(max(row['max_abs_by_output'])
                                            for row in rows + repeated_rows),
              'elapsed_seconds': time.perf_counter() - started,
              'r13_accessed': False, 'r14_accessed': False}
    c.write_once(directory / 'result.json', result)
    print(json.dumps({'status': result['status'], 'candidate_rows': result['candidate_rows'],
        'maximum_absolute_error': result['maximum_absolute_error'],
        'maximum_reserved_bytes': result['maximum_reserved_bytes'],
        'diagnostic_latency': {key: value for key, value in latency.items() if key != 'rows'} if latency else None,
        'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
