#!/usr/bin/env python3
"""Single bounded reduce-overhead/fullgraph/dynamic E3 feasibility attempt."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import phase6k_runtime_common as c
from scripts import audit_phase6k_runtime as audit
from rcias_clgri.ni.phase6k_runtime import eager_reconstruction, DeviceDecision
from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble

DIRECTORY = c.OUT / 'e3_compile'


def counters():
    from torch._dynamo.utils import counters as values
    return {str(k): {str(a): int(b) for a, b in v.items()} for k, v in values.items()}


def progress(record):
    temporary = DIRECTORY / 'progress.tmp.json'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(DIRECTORY / 'progress.json')
    print(json.dumps(record), flush=True)


def worker():
    started, rows = time.perf_counter(), []
    try:
        env = c.setup()
        eager = VectorizedEnsemble(eager_reconstruction(env['e0']))
        gate = DeviceDecision(env['protocol']).cuda().eval()
        options = c.r.load_json(c.ACTIVE)['runtime_audit']['compile']
        compiled = torch.compile(eager, mode=options['mode'], fullgraph=options['fullgraph'], dynamic=options['dynamic'])
        warm_rows = []
        with torch.inference_mode():
            for sid in c.cell_states(env):
                ctx = c.context(env, sid)
                before = counters()
                t = time.perf_counter()
                row = audit.check_pair(env, compiled, gate, ctx, 0, 'E3')
                torch.cuda.synchronize()
                cold = time.perf_counter() - t
                rows.append(row)
                if not row['pass']:
                    raise RuntimeError('E3 paired equivalence failure')
                packed, _, _, _ = audit.prepare(env, ctx)
                after_cold = counters()
                for repetition in range(-3, 3):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    out = compiled(packed['batch'], **c.deploy.model_inputs(packed))
                    audit.extract_decision(gate(out, packed['support_tensor'], packed['lexical_rank'], packed['fallback_indices']), packed['batch'].target_set_ids)
                    torch.cuda.synchronize()
                    if repetition >= 0:
                        warm_rows.append({'state_id': sid, 'repetition': repetition, 'neural_ms': (time.perf_counter()-t)*1000})
                row['cold_pair_seconds_including_preprocessing_and_e0'] = cold
                row['counters_before'] = before
                row['counters_after_cold'] = after_cold
                row['counters_after_warm'] = counters()
                progress({'stage': 'VARIABLE_SHAPE_FEASIBILITY', 'states_completed': len(rows), 'elapsed_seconds': time.perf_counter()-started, 'counters': counters()})
            feasibility = {'status': 'PASS', 'rows': rows, 'warm_rows': warm_rows, 'counters': counters(),
                           'elapsed_seconds': time.perf_counter()-started,
                           'compile_time_note': 'cold pair wall includes preprocessing and E0; compiler internal metrics are separately exported'}
            c.write_once(DIRECTORY / 'feasibility.json', feasibility)
            # Same compiled object and attempt; all additional shapes must also pass.
            full, robust = [], []
            for sid in sorted(env['frames']):
                full.append(audit.check_pair(env, compiled, gate, c.context(env, sid), 0, 'E3'))
                if len(full) % 24 == 0:
                    progress({'stage': 'FULL_PAIRED_EQUIVALENCE', 'states_completed': len(full), 'elapsed_seconds': time.perf_counter()-started, 'counters': counters()})
            repeated = c.r.load_json(ROOT / 'outputs/phase6k_runtime_v1/diagnostics/preprocessing_v2/protocol.json')['state_ids']
            for sid in repeated:
                ctx = c.context(env, sid)
                for repetition in range(30):
                    robust.append(audit.check_pair(env, compiled, gate, ctx, repetition, 'E3'))
                progress({'stage': 'REPEATED_REALIZATION', 'pairs_completed': len(robust), 'elapsed_seconds': time.perf_counter()-started, 'counters': counters()})
            passed = all(x['pass'] for x in full+robust)
            latency = audit.diagnostic_latency(env, compiled, gate) if passed else None
        from torch._dynamo.utils import compile_times
        result = {'status': 'PASS' if passed else 'INELIGIBLE_PARITY', 'variant': 'E3', 'rows': full, 'robustness': robust,
                  'diagnostic_latency': latency, 'counters': counters(), 'compiler_times': compile_times(),
                  'elapsed_seconds': time.perf_counter()-started, 'r13_accessed': False, 'r14_accessed': False}
    except Exception as error:
        from torch._dynamo.utils import compile_times
        result = {'status': 'INELIGIBLE_COMPILER', 'variant': 'E3', 'error': str(error), 'traceback': traceback.format_exc(),
                  'completed_feasibility_pairs': rows, 'counters': counters(), 'compiler_times': compile_times(),
                  'elapsed_seconds': time.perf_counter()-started, 'r13_accessed': False, 'r14_accessed': False}
    c.write_once(DIRECTORY / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    for name in ('e1_equivalence', 'e2_equivalence'):
        assert c.r.load_json(c.OUT / name / 'result.json')['status'] == 'PASS'
    options = c.r.load_json(c.ACTIVE)['runtime_audit']['compile']
    c.freeze_stage('e3_compile', {'options': options,
        'timeout_scope': '240 seconds for initial variable-shape compiler feasibility; same object continues full parity only if feasibility passes',
        'predecessor_hashes': {name: c.digest(c.OUT/name/'result.json') for name in ('e1_equivalence', 'e2_equivalence')}},
        [__file__, Path(audit.__file__), ROOT/'rcias_clgri/ni/phase6k_runtime.py', ROOT/'rcias_clgri/ni/phase6k_vectorized.py'])
    started = time.perf_counter()
    with (DIRECTORY/'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, '--worker'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        c.write_once(DIRECTORY/'launch.json', {'pid': child.pid, 'parent_pid': os.getpid(), 'started_unix': time.time(), 'options': options})
        while child.poll() is None:
            if not (DIRECTORY/'feasibility.json').exists() and time.perf_counter()-started > options['timeout_seconds']:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                if not (DIRECTORY/'result.json').exists():
                    c.write_once(DIRECTORY/'result.json', {'status':'INELIGIBLE_COMPILE_TIMEOUT', 'variant':'E3',
                        'elapsed_seconds':time.perf_counter()-started, 'timeout_seconds':options['timeout_seconds'],
                        'last_progress':c.r.load_json(DIRECTORY/'progress.json') if (DIRECTORY/'progress.json').exists() else None,
                        'r13_accessed':False, 'r14_accessed':False})
                break
            time.sleep(0.5)
    if not (DIRECTORY/'result.json').exists():
        c.write_once(DIRECTORY/'result.json', {'status':'INELIGIBLE_COMPILER_PROCESS', 'exit_code':child.returncode,
            'elapsed_seconds':time.perf_counter()-started, 'r13_accessed':False, 'r14_accessed':False})
    result = c.r.load_json(DIRECTORY/'result.json')
    print(json.dumps({k:result.get(k) for k in ('status','elapsed_seconds','error')}), flush=True)


if __name__ == '__main__':
    main()
