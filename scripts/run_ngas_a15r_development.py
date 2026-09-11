#!/usr/bin/env python3
"""Development-only latency check for the shared A1.5R runtime."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import ProductionRefreshRuntime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/ngas_a15r_runtime_revision_v1.json'
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/development/development_latency.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate(raw):
    return Candidate(*(tuple(raw[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def stats(values):
    return {
        'count': len(values), 'mean': statistics.fmean(values),
        'p50': float(np.percentile(values, 50, method='linear')),
        'p90': float(np.percentile(values, 90, method='linear')),
        'p99': float(np.percentile(values, 99, method='linear')),
        'max': max(values),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('A1.5R development timing requires CUDA')
    config = json.loads(CONFIG.read_text())
    state_path = ROOT / config['historical_A1_5']['representative_states']
    if digest(state_path) != config['historical_A1_5']['representative_states_sha256']:
        raise RuntimeError('Historical A1.5 representative states changed')
    checkpoint = config['production_checkpoint']
    critic = FrozenJointCritic(
        ROOT / checkpoint['path'], 'cuda:0', checkpoint['sha256'], checkpoint['variant'])
    runtime = ProductionRefreshRuntime(critic)
    measurement = config['development_targets']
    results = {}
    pooled_samples = []
    for state in json.loads(state_path.read_text())['states']:
        instance = load_instance(
            ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'
            / state['relative_path'])
        current = decode_candidate(instance, candidate(state['candidate']))
        if not current.feasible or abs(current.makespan - state['replayed_makespan']) > 1e-9:
            raise RuntimeError(f"Representative replay mismatch: {state['label']}")
        runtime.prepare_instance(instance)
        streams = RNGStreams(instance.instance_id, 746101)
        state_id = f"ngas-a15r-development:{state['label']}"
        for repetition in range(measurement['warmup_repetitions']):
            runtime.refresh(instance, current, state_id, streams, sample_seed=repetition)
        samples = [
            runtime.refresh(instance, current, state_id, streams, sample_seed=repetition)
            for repetition in range(measurement['measured_repetitions'])
        ]
        pooled_samples.extend(
            sample.components_ms['complete_refresh'] for sample in samples)
        names = samples[0].components_ms
        results[state['label']] = {
            'complete_refresh_ms': stats([
                sample.components_ms['complete_refresh'] for sample in samples]),
            'component_ms': {
                name: stats([sample.components_ms[name] for sample in samples])
                for name in names
            },
            'joint_actions': len(samples[-1].actions),
            'workspace_capacities': samples[-1].workspace_capacities,
            'workspace_resize_events': samples[-1].workspace_resize_events,
            'all_static_context_hits': all(sample.static_context_hit for sample in samples),
        }
        print(state['label'], results[state['label']]['complete_refresh_ms'], flush=True)
    pooled = stats(pooled_samples)
    payload = {
        'schema': 'ngas-a15r-development-latency-v1',
        'development_only': True,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG),
        'historical_A1_5_immutable': True,
        'results': results,
        'pooled_complete_refresh_ms': pooled,
        'target_p90_ms': measurement['per_state_complete_refresh_p90_ms'],
        'ready_to_freeze_formal': all(
            value['complete_refresh_ms']['p90'] <= measurement['per_state_complete_refresh_p90_ms']
            for value in results.values()),
        'device': {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'cuda': torch.version.cuda},
        'locks': config['locks'],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUT)
    print(json.dumps({'output': str(OUT.relative_to(ROOT)),
                      'ready_to_freeze_formal': payload['ready_to_freeze_formal']}, indent=2))


if __name__ == '__main__':
    main()
