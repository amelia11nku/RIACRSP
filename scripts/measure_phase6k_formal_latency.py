#!/usr/bin/env python3
"""Freeze runtime-only selection, then freshly measure all R12 live decisions."""
import json
from pathlib import Path
import sys
import time

import torch
import torch._inductor.config as inductor_config
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import phase6k_runtime_common as c
from scripts import audit_phase6k_runtime as audit
from rcias_clgri.ni.phase6k_runtime import eager_reconstruction, DeviceDecision
from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble


def select_variant(results):
    for cap in (27, 30):
        for variant in ('E1', 'E2', 'E3'):
            result = results[variant]
            latency = result.get('diagnostic_latency')
            if (result['status'] == 'PASS' and latency and
                    latency['neural_ms']['p90'] <= cap and latency['full_live_ms']['p90'] <= 100):
                return variant
    return None


def verify_equivalence(result, expected_ids, repeated_ids):
    if result['status'] != 'PASS':
        return
    rows, repeated = result['rows'], result['robustness']
    if sorted(x['state_id'] for x in rows) != sorted(expected_ids):
        raise ValueError('full 288-state coverage is required')
    if sum(x['candidate_rows'] for x in rows) != 6809:
        raise ValueError('full deduplicated candidate coverage is required')
    expected_pairs = sorted((sid, rep) for sid in repeated_ids for rep in range(30))
    if sorted((x['state_id'], x['repetition']) for x in repeated) != expected_pairs:
        raise ValueError('every preregistered realization pair is required')
    if not all(x['pass'] and all(x['checks'].values()) for x in rows + repeated):
        raise ValueError('an individual failed pair cannot be summarized as PASS')


def main():
    paths = {v: c.OUT / name / 'result.json' for v, name in
             [('E1','e1_equivalence'), ('E2','e2_equivalence'),
              ('E3','e3_stability')]}
    results = {v: c.r.load_json(p) for v, p in paths.items()}
    expected_ids = c.r.load_json(c.OUT/'e1_equivalence/protocol.json')['state_ids']
    repeated_ids = c.r.load_json(c.OUT/'e1_equivalence/protocol.json')['repeated_state_ids']
    for result in results.values():
        verify_equivalence(result, expected_ids, repeated_ids)
    variant = select_variant(results)
    selection = {'status':'SELECTED' if variant else 'MODEL_REVISION_RUNTIME', 'variant':variant,
                 'basis':'fixed order E1/E2/E3, diagnostic caps 27/100 then 30/100; no outcome selection',
                 'evidence_hashes':{v:c.digest(p) for v,p in paths.items()},
                 'r13_unlocked':False, 'r14_unlocked':False}
    selection['variant'] = 'E3S' if selection['variant'] == 'E3' else selection['variant']
    selection['supersedes'] = ('runtime/selection_v2.json and the incomplete formal_latency stream; '
                               'E3S skips dynamic CUDA Graph retention')
    c.write_once(c.OUT/'selection_v4.json', selection)
    if variant is None:
        print(json.dumps(selection), flush=True)
        return
    env = c.setup()
    model = eager_reconstruction(env['e0'])
    if variant in ('E2','E3'):
        model = VectorizedEnsemble(model)
    if variant == 'E3':
        stability = c.r.load_json(ROOT/'configs/phase6k_runtime_v1_stability_amended.json')
        options = stability['compiler']
        inductor_config.triton.cudagraph_skip_dynamic_graphs = True
        model = torch.compile(model, mode=options['mode'], fullgraph=options['fullgraph'], dynamic=options['dynamic'])
    gate = DeviceDecision(env['protocol']).cuda().eval()
    directory = c.freeze_stage('formal_latency_e3s', {'selection':selection, 'state_ids':sorted(env['frames']),
        'warmups_per_state':3, 'measured_repetitions_per_state':5, 'all_live_preprocessing_fresh':True,
        'cuda_graph_or_compiler_cache_reuse':'same eligible configuration, no tuning or retry'},
        [__file__, Path(audit.__file__), Path(c.__file__),
         ROOT/'rcias_clgri/ni/phase6k_runtime.py', ROOT/'rcias_clgri/ni/phase6k_vectorized.py'])
    rows, warmup_rows = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        with (directory/'measurements.jsonl').open('x') as stream:
            for sid in sorted(env['frames']):
                ctx = c.context(env, sid)
                for repetition in range(-3, 5):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    packed, _, _, timings = audit.prepare(env, ctx)
                    torch.cuda.synchronize()
                    neural_begin = time.perf_counter()
                    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    first.record()
                    output = model(packed['batch'], **c.deploy.model_inputs(packed))
                    decision = audit.extract_decision(gate(output, packed['support_tensor'], packed['lexical_rank'], packed['fallback_indices']), packed['batch'].target_set_ids)
                    last.record()
                    torch.cuda.synchronize()
                    neural_ms = (time.perf_counter()-neural_begin)*1000
                    full_ms = (time.perf_counter()-begin)*1000
                    row = {'state_id':sid, 'scale':str(env['frames'][sid].scale.iloc[0]), 'repetition':repetition,
                           'neural_ms':neural_ms, 'neural_cuda_span_ms':first.elapsed_time(last), 'full_live_ms':full_ms,
                           'decision':decision, 'preprocessing_timings':timings}
                    (rows if repetition >= 0 else warmup_rows).append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                completed = len(rows)//5
                if completed % 12 == 0:
                    record = {'status':'RUNNING','completed_states':completed,'total_states':288,
                              'elapsed_seconds':time.perf_counter()-started}
                    temporary=directory/'progress.tmp.json'
                    temporary.write_text(json.dumps(record)+'\n')
                    temporary.replace(directory/'progress.json')
                    print(json.dumps(record),flush=True)
    summaries = {}
    for scale in ('overall','S','M','L'):
        subset = rows if scale == 'overall' else [r for r in rows if r['scale']==scale]
        summaries[scale] = {key:c.percentiles([row[key] for row in subset]) for key in ('neural_ms','full_live_ms')}
    passed = summaries['overall']['neural_ms']['p90'] <= 30 and summaries['overall']['full_live_ms']['p90'] <= 100
    result = {'status':'PASS' if passed else 'MODEL_REVISION_RUNTIME','variant':selection['variant'],'measured_rows':len(rows),
              'warmup_rows':len(warmup_rows),'states':len(rows)//5,'summary':summaries,
              'warmup_total_seconds':sum(r['full_live_ms'] for r in warmup_rows)/1000,
              'elapsed_seconds':time.perf_counter()-started,'r13_unlocked':False,'r14_unlocked':False,
              'measurements_sha256':c.digest(directory/'measurements.jsonl')}
    c.write_once(directory/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
