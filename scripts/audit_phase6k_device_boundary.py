#!/usr/bin/env python3
"""CUDA-profiler host-extraction audit; no timing qualification or tuning."""
from pathlib import Path
import sys
import json

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import phase6k_runtime_common as c
from scripts import audit_phase6k_runtime as audit
from rcias_clgri.ni.phase6k_runtime import eager_reconstruction, DeviceDecision
from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble


def summarize(profile):
    counts={event.key:event.count for event in profile.key_averages()}
    return {'scalar_reads':counts.get('aten::_local_scalar_dense',0),
            'dtoh_copies':sum(count for name,count in counts.items() if 'Memcpy DtoH' in name),
            'copy_events':{name:count for name,count in counts.items() if 'Memcpy' in name}}


def profile():
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])


def main():
    env=c.setup()
    e1=eager_reconstruction(env['e0'])
    e2=VectorizedEnsemble(e1)
    gate=DeviceDecision(env['protocol']).cuda().eval()
    directory=c.freeze_stage('device_boundary_v2',{'variants':['E1','E2'],'latency_measurement':False,
        'replaces':'device_boundary INVALID_INSTRUMENTATION; CUDA profiler includes inference tensors'},[__file__])
    rows=[]
    with torch.inference_mode():
        packed,_,_,_=audit.prepare(env,c.context(env,c.cell_states(env)[0]))
        for name,model in [('E1',e1),('E2',e2)]:
            torch.cuda.synchronize()
            with profile() as device:
                output=model(packed['batch'],**c.deploy.model_inputs(packed))
                record=gate(output,packed['support_tensor'],packed['lexical_rank'],packed['fallback_indices'])
            with profile() as extraction:
                audit.extract_decision(record,packed['batch'].target_set_ids)
            device.export_chrome_trace(str(directory/f'{name}_device_trace.json'))
            extraction.export_chrome_trace(str(directory/f'{name}_extraction_trace.json'))
            a,b=summarize(device),summarize(extraction)
            rows.append({'variant':name,'device':a,'final_extraction':b})
    passed=all(x['device']['scalar_reads']==x['device']['dtoh_copies']==0 and
               x['final_extraction']['scalar_reads']==0 and x['final_extraction']['dtoh_copies']==1 for x in rows)
    c.write_once(directory/'result.json',{'status':'PASS' if passed else 'FAIL','rows':rows,'r13_accessed':False,'r14_accessed':False})
    print(json.dumps(rows),flush=True)
    assert passed


if __name__=='__main__':
    main()
