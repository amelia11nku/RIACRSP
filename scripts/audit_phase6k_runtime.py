#!/usr/bin/env python3
"""Paired full R12 and repeated-realization eligibility; no outcome selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import phase6k_runtime_common as c
from scripts.diagnose_phase6k_preprocessing import input_fingerprint
from rcias_clgri.ni.phase6k_runtime import eager_reconstruction, prepare_topology, transfer_topology, DeviceDecision, extract_decision


def prepare(env, ctx):
    packed, sample, bank, timing = c.realize(env, ctx)
    prepare_topology(packed)
    c.transfer(packed)
    transfer_topology(packed, 'cuda')
    return packed, sample, bank, timing


def prepare_reused(env, ctx):
    packed, sample, bank, timing, historical_cuda_batch = c.realize_reused(env, ctx)
    prepare_topology(packed)
    c.transfer_reused(packed, historical_cuda_batch)
    transfer_topology(packed, 'cuda')
    return packed, sample, bank, timing


def check_pair(env, model, gate, ctx, repetition, variant, prepare_fn=prepare):
    packed, sample, bank, timing = prepare_fn(env, ctx)
    sid = ctx['state_id']
    archive = c.inference_inputs(env['samples'][sid], env['frames'][sid], env['transform'], 'cpu')
    before = input_fingerprint(packed['batch'])
    reference = env['e0'](packed['batch'], **c.deploy.model_inputs(packed))
    expected = asdict(c.deploy.deployment_decision(reference, packed, env['protocol']))
    expected.pop('lcb')
    output = model(packed['batch'], **c.deploy.model_inputs(packed))
    actual = extract_decision(gate(output, packed['support_tensor'], packed['lexical_rank'], packed['fallback_indices']), packed['batch'].target_set_ids)
    finite = all(bool(torch.isfinite(x).all()) for x in (*reference, *output))
    exact = all(torch.equal(a, b) for a, b in zip(reference, output))
    tolerance = all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(reference, output))
    errors = [float((a-b).abs().max()) for a,b in zip(reference, output)]
    ref_values = env['frames'][sid].set_index('target_set_id')
    scores = np.array([a.raw_score for a in bank.arms], dtype=np.float32)
    checks = {'finite': finite, 'raw_outputs': exact if variant == 'E1' else tolerance,
              'decision': actual == expected,
              'support': np.array_equal(packed['supported'], archive['supported']),
              'candidate_order': list(bank.arms[i].target_set_id for i in range(len(bank.arms))) == ctx['replay']['full_bank_target_ids'],
              'candidate_membership': {x.target_set_id: list(x.destroyed_operations) for x in bank.arms} == ctx['replay']['full_bank_target_operations'],
              'graph_and_membership_inputs': before == input_fingerprint(archive['batch']),
              'input_unchanged': before == input_fingerprint(packed['batch'])}
    return {'state_id': sid, 'repetition': repetition, 'candidate_rows': len(packed['frame']), 'checks': checks,
            'pass': all(checks.values()), 'raw_outputs_bit_exact': exact, 'max_abs_by_output': errors,
            'reference': expected, 'actual': actual, 'score_sha256': hashlib.sha256(scores.tobytes()).hexdigest(),
            'archive_score_match': all(x.raw_score == ref_values.loc[x.target_set_id, 'frozen_raw_score'] for x in bank.arms)}


def diagnostic_latency(env, model, gate, prepare_fn=prepare):
    rows = []
    for sid in c.cell_states(env):
        ctx = c.context(env, sid)
        for repetition in range(-3, 3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            packed, _, _, _ = prepare_fn(env, ctx)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            neural_started = time.perf_counter()
            output = model(packed['batch'], **c.deploy.model_inputs(packed))
            extract_decision(gate(output, packed['support_tensor'], packed['lexical_rank'], packed['fallback_indices']), packed['batch'].target_set_ids)
            end.record()
            torch.cuda.synchronize()
            neural = (time.perf_counter() - neural_started) * 1000
            live = (time.perf_counter() - started) * 1000
            if repetition >= 0:
                rows.append({'state_id': sid, 'repetition': repetition, 'neural_ms': neural, 'full_live_ms': live, 'neural_cuda_span_ms': start.elapsed_time(end)})
    return {'rows': rows, 'neural_ms': c.percentiles([x['neural_ms'] for x in rows]),
            'full_live_ms': c.percentiles([x['full_live_ms'] for x in rows])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['E1', 'E2'], required=True)
    args = parser.parse_args()
    env = c.setup()
    assert (c.OUT / 'e0_components/result.json').exists()
    model = eager_reconstruction(env['e0'])
    if args.variant == 'E2':
        from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble
        model = VectorizedEnsemble(model)
    gate = DeviceDecision(env['protocol']).cuda().eval()
    repeated = c.r.load_json(ROOT / 'outputs/phase6k_runtime_v1/diagnostics/preprocessing_v2/protocol.json')['state_ids']
    directory = c.freeze_stage(args.variant.lower() + '_equivalence', {'variant': args.variant,
        'state_ids': sorted(env['frames']), 'repeated_state_ids': repeated, 'repetitions': 30,
        'eager_bit_exact': args.variant == 'E1', 'atol': 1e-5, 'rtol': 1e-5},
        [__file__, ROOT / 'rcias_clgri/ni/phase6k_runtime.py', ROOT / 'rcias_clgri/ni/phase6k_vectorized.py'])
    started, rows = time.perf_counter(), []
    with torch.inference_mode():
        for sid in sorted(env['frames']):
            rows.append(check_pair(env, model, gate, c.context(env, sid), 0, args.variant))
            if len(rows) % 24 == 0:
                print(json.dumps({'variant': args.variant, 'full_states': len(rows), 'failures': sum(not x['pass'] for x in rows), 'elapsed_seconds': time.perf_counter()-started}), flush=True)
        robustness = []
        for sid in repeated:
            ctx = c.context(env, sid)
            for repetition in range(30):
                robustness.append(check_pair(env, model, gate, ctx, repetition, args.variant))
            print(json.dumps({'variant': args.variant, 'robustness_pairs': len(robustness), 'elapsed_seconds': time.perf_counter()-started}), flush=True)
        passed = all(x['pass'] for x in rows + robustness)
        latency = diagnostic_latency(env, model, gate) if passed else None
    result = {'status': 'PASS' if passed else 'FAIL', 'variant': args.variant, 'rows': rows, 'robustness': robustness,
              'candidate_rows': sum(x['candidate_rows'] for x in rows), 'diagnostic_latency': latency,
              'elapsed_seconds': time.perf_counter()-started, 'r13_accessed': False, 'r14_accessed': False}
    c.write_once(directory / 'result.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'failed_checks': {key: sum(not x['checks'][key] for x in rows+robustness) for key in rows[0]['checks']},
        'latency': {k:v for k,v in (latency or {}).items() if k != 'rows'}}), flush=True)


if __name__ == '__main__':
    main()
