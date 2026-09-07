#!/usr/bin/env python3
"""Non-invasive component profile before implementation of the E1/E2 ladder."""
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from pathlib import Path
import sys
import time
from unittest.mock import patch
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.ni import encoder, action_encoder
from rcias_clgri.ni.calibration import FrozenCalibrator
from rcias_clgri.analysis.phase6j_caur import choose_caur_action
from scripts import phase6k_runtime_common as c


class Components:
    def __init__(self):
        self.records = defaultdict(list)

    @contextmanager
    def scope(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        wall = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - wall) * 1000
            end.record()
            self.records[name].append((start, end, elapsed))

    def wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            with self.scope(name):
                return fn(*args, **kwargs)
        return wrapped

    def finish(self):
        torch.cuda.synchronize()
        return {key: {'calls': len(rows), 'cuda_span_ms': sum(a.elapsed_time(b) for a, b, _ in rows),
                      'host_enqueue_or_cpu_ms': sum(w for _, _, w in rows)} for key, rows in self.records.items()}


@contextmanager
def instrumentation(model, timer):
    # Scoped wrappers only; the frozen modules and arithmetic are not edited.
    with ExitStack() as stack:
        modules = {'shared_state_encoder': model.base.state_encoder, 'target_set_encoder': model.base.action_encoder,
                   **{f'input_projection.{k}': v for k, v in model.base.state_encoder.input_projection.items()},
                   **{f'relation_layer.{i}': v for i, v in enumerate(model.base.state_encoder.layers)},
                   **{f'seed_head.{i}': v for i, v in enumerate(model.heads)}}
        for name, module in modules.items():
            stack.enter_context(patch.object(module, 'forward', timer.wrap(name, module.forward)))
        for module, function, name in ((encoder, 'segment_softmax', 'relation_segment_softmax'),
                (action_encoder, 'segment_softmax', 'target_segment_softmax'),
                (encoder, 'segment_mean', 'graph_pooling'), (action_encoder, 'segment_mean', 'target_pooling'),
                (torch, 'unique', 'relation_count_unique'), (torch, 'bincount', 'segment_counts')):
            stack.enter_context(patch.object(module, function, timer.wrap(name, getattr(module, function))))
        yield


def profile_reference_decision(output, packed, protocol, timer):
    with timer.scope('output_cpu_transfer'):
        advantage, logits, immediate = [x.detach().float().cpu().numpy() for x in output]
    with timer.scope('seed_aggregation'):
        mean, std = advantage.mean(axis=0), advantage.std(axis=0, ddof=0)
        logits, immediate = logits.mean(axis=0), immediate.mean(axis=0)
    with timer.scope('calibration'):
        probability = FrozenCalibrator(**protocol['calibrator']).predict(logits)
    with timer.scope('gate'):
        rows = [{'target_set_id': str(row.target_set_id), 'continuation_advantage_mean': float(mean[i]),
                 'continuation_advantage_std': float(std[i]), 'beats_fallback_probability': float(probability[i]),
                 'supported': bool(packed['supported'][i]), 'immediate_utility_prediction': float(immediate[i])}
                for i, row in enumerate(packed['frame'].itertuples(index=False))]
        gate = protocol['gate']
        return choose_caur_action(rows, fallback_target_set_id=str(packed['frame'].loc[packed['frame'].is_fallback, 'target_set_id'].item()),
            **{k: gate[k] for k in ('p_min', 'lcb_lambda', 'delta_min')}, immediate_harm_floor=protocol['immediate_harm_floor'])


def main():
    env = c.setup()
    ids = c.cell_states(env)
    directory = c.freeze_stage('e0_components', {'state_ids': ids, 'warmups': 3, 'repetitions': 3,
        'instrumented_measurements_are_not_latency_qualification': True}, [__file__])
    rows, instrumented = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for sid in ids:
            ctx = c.context(env, sid)
            for rep in range(-3, 3):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                packed, sample, bank, timings = c.realize(env, ctx)
                t = time.perf_counter()
                c.transfer(packed)
                torch.cuda.synchronize()
                timings['j1_transfer_ms'] = (time.perf_counter() - t) * 1000
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
                t = time.perf_counter()
                output = env['e0'](packed['batch'], **c.deploy.model_inputs(packed))
                decision = c.deploy.deployment_decision(output, packed, env['protocol'])
                event_end.record()
                torch.cuda.synchronize()
                neural = (time.perf_counter() - t) * 1000
                live = (time.perf_counter() - begin) * 1000
                if rep >= 0:
                    rows.append({'state_id': sid, 'repetition': rep, 'scale': str(env['frames'][sid].scale.iloc[0]),
                                 'neural_ms': neural, 'neural_cuda_span_ms': event_start.elapsed_time(event_end),
                                 'full_live_ms': live, 'components': timings})
            timer = Components()
            with instrumentation(env['e0'], timer), timer.scope('cached_decision'):
                output = env['e0'](packed['batch'], **c.deploy.model_inputs(packed))
                actual = profile_reference_decision(output, packed, env['protocol'], timer)
            assert asdict(actual) == asdict(decision)
            instrumented.append({'state_id': sid, 'components': timer.finish()})
            print(json.dumps({'profiled_states': len(instrumented), 'elapsed_seconds': time.perf_counter() - started}), flush=True)
    result = {'status': 'COMPLETE', 'uninstumented_rows': rows, 'instrumented_components': instrumented,
              'neural_ms': c.percentiles([x['neural_ms'] for x in rows]), 'full_live_ms': c.percentiles([x['full_live_ms'] for x in rows]),
              'elapsed_seconds': time.perf_counter() - started, 'r13_accessed': False, 'r14_accessed': False,
              'component_interpretation': 'nested inclusive event spans; host values are enqueue/CPU time, not additive exclusive GPU kernel time'}
    c.write_once(directory / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'neural_ms', 'full_live_ms', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
