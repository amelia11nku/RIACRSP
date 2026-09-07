#!/usr/bin/env python3
"""Formal all-R12 fresh-live timing for the terminal E4R candidate."""
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
from rcias_clgri.ni.phase6k_runtime import (
    eager_reconstruction, hoisted_qkv_reconstruction, DeviceDecision,
)
from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble

CONFIG = ROOT / 'configs/phase6k_runtime_v1_e4r.json'
EQUIVALENCE = c.OUT / 'e4r_equivalence/result.json'


def main():
    settings = c.r.load_json(CONFIG)
    result = c.r.load_json(EQUIVALENCE)
    if settings['status'] != 'E4R_FROZEN_BEFORE_IMPLEMENTATION' or result['status'] != 'PASS':
        raise RuntimeError('formal E4R timing requires the frozen passing equivalence result')
    if result['variant'] != 'E4R':
        raise RuntimeError('invalid E4R equivalence identity')
    expected_ids = c.r.load_json(c.OUT / 'e1_equivalence/protocol.json')['state_ids']
    repeated_ids = c.r.load_json(c.OUT / 'e1_equivalence/protocol.json')['repeated_state_ids']
    from scripts.measure_phase6k_formal_latency import verify_equivalence
    verify_equivalence(result, expected_ids, repeated_ids)
    selection = {
        'status': 'FORMAL_ASSESSMENT_FROZEN',
        'variant': 'E4R',
        'basis': ('terminal preregistered same-decision tensor-reuse candidate; '
                  'formal 30/100 ms gates are authoritative'),
        'e4r_config_sha256': c.digest(CONFIG),
        'e4r_equivalence_sha256': c.digest(EQUIVALENCE),
        'diagnostic_latency': result['diagnostic_latency'],
        'r13_unlocked': False,
        'r14_unlocked': False,
    }
    c.write_once(c.OUT / 'selection_v5.json', selection)
    env = c.setup()
    if sorted(expected_ids) != sorted(env['frames']):
        raise RuntimeError('R12 state boundary changed')
    runtime = settings['retained_runtime']
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    e2 = VectorizedEnsemble(eager_reconstruction(env['e0']))
    candidate = hoisted_qkv_reconstruction(e2)
    model = torch.compile(candidate, mode=runtime['compile_mode'],
                          fullgraph=runtime['fullgraph'], dynamic=runtime['dynamic'])
    gate = DeviceDecision(env['protocol']).cuda().eval()
    directory = c.freeze_stage('formal_latency_e4r', {
        'selection': selection,
        'state_ids': sorted(env['frames']),
        'warmups_per_state': 3,
        'measured_repetitions_per_state': 5,
        'all_live_preprocessing_fresh': True,
        'same_decision_tensor_reuse_only': True,
        'neural_p90_cap_ms': 30,
        'full_live_p90_cap_ms': 100,
        'no_chunking_cache_clearing_filtering_or_rerun': True,
    }, [__file__, Path(audit.__file__), Path(c.__file__),
        ROOT / 'rcias_clgri/ni/phase6k_runtime.py',
        ROOT / 'rcias_clgri/ni/phase6k_vectorized.py'])
    rows, warmup_rows = [], []
    started = time.perf_counter()
    with torch.inference_mode(), (directory / 'measurements.jsonl').open('x') as stream:
        for sid in sorted(env['frames']):
            ctx = c.context(env, sid)
            for repetition in range(-3, 5):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                packed, _, _, timings = audit.prepare_reused(env, ctx)
                torch.cuda.synchronize()
                neural_begin = time.perf_counter()
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                output = model(packed['batch'], **c.deploy.model_inputs(packed))
                decision = audit.extract_decision(
                    gate(output, packed['support_tensor'], packed['lexical_rank'],
                         packed['fallback_indices']),
                    packed['batch'].target_set_ids,
                )
                last.record()
                torch.cuda.synchronize()
                row = {
                    'state_id': sid,
                    'scale': str(env['frames'][sid].scale.iloc[0]),
                    'repetition': repetition,
                    'neural_ms': (time.perf_counter() - neural_begin) * 1000,
                    'neural_cuda_span_ms': first.elapsed_time(last),
                    'full_live_ms': (time.perf_counter() - begin) * 1000,
                    'decision': decision,
                    'preprocessing_timings': timings,
                }
                (rows if repetition >= 0 else warmup_rows).append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
            completed = len(rows) // 5
            if completed % 12 == 0:
                record = {'status': 'RUNNING', 'completed_states': completed,
                          'total_states': 288,
                          'elapsed_seconds': time.perf_counter() - started}
                temporary = directory / 'progress.tmp.json'
                temporary.write_text(json.dumps(record) + '\n')
                temporary.replace(directory / 'progress.json')
                print(json.dumps(record), flush=True)
    summaries = {}
    for scale in ('overall', 'S', 'M', 'L'):
        subset = rows if scale == 'overall' else [row for row in rows if row['scale'] == scale]
        summaries[scale] = {
            key: c.percentiles([row[key] for row in subset])
            for key in ('neural_ms', 'full_live_ms')
        }
    neural_pass = summaries['overall']['neural_ms']['p90'] <= 30
    live_pass = summaries['overall']['full_live_ms']['p90'] <= 100
    formal_result = {
        'status': 'PASS' if neural_pass and live_pass else 'MODEL_REVISION_RUNTIME',
        'variant': 'E4R',
        'measured_rows': len(rows),
        'warmup_rows': len(warmup_rows),
        'states': len(rows) // 5,
        'summary': summaries,
        'neural_cap_pass': neural_pass,
        'full_live_cap_pass': live_pass,
        'warmup_total_seconds': sum(row['full_live_ms'] for row in warmup_rows) / 1000,
        'elapsed_seconds': time.perf_counter() - started,
        'measurements_sha256': c.digest(directory / 'measurements.jsonl'),
        'r13_unlocked': False,
        'r14_unlocked': False,
    }
    c.write_once(directory / 'result.json', formal_result)
    print(json.dumps(formal_result), flush=True)


if __name__ == '__main__':
    main()
