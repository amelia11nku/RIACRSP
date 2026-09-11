#!/usr/bin/env python3
"""Run the preregistered A1.5R representative-state latency gate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import ProductionRefreshRuntime


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
PROGRESS = OUT / 'progress.json'
INSTANCE_ROOT = ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def candidate(raw):
    return Candidate(*(tuple(raw[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def validate(protocol):
    if protocol['status'] != 'FROZEN_BEFORE_FORMAL_RESULTS':
        raise RuntimeError('A1.5R protocol is not frozen before results')
    if digest(ROOT / protocol['config_path']) != protocol['config_sha256']:
        raise RuntimeError('A1.5R config changed after freeze')
    if digest(ROOT / protocol['representative_states_path']) \
            != protocol['representative_states_sha256']:
        raise RuntimeError('Representative states changed after freeze')
    if digest(ROOT / protocol['checkpoint_path']) != protocol['checkpoint_sha256']:
        raise RuntimeError('Production checkpoint changed after freeze')
    if digest(ROOT / protocol['preformal_audit_path']) \
            != protocol['preformal_audit_sha256']:
        raise RuntimeError('Preformal audit changed after freeze')
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen A1.5R source changed: {relative}')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Formal A1.5R qualification requires CUDA')
    protocol = json.loads(PROTOCOL.read_text())
    validate(protocol)
    protocol_sha = digest(PROTOCOL)
    measurement = protocol['measurement']
    critic = FrozenJointCritic(
        ROOT / protocol['checkpoint_path'], measurement['device'],
        protocol['checkpoint_sha256'], 'C1')
    runtime = ProductionRefreshRuntime(critic)
    states = json.loads((ROOT / protocol['representative_states_path']).read_text())['states']
    completed = 0
    formal_started = datetime.now(timezone.utc).isoformat()
    for state in states:
        result_path = OUT / 'formal/raw' / f"{state['label']}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get('protocol_sha256') != protocol_sha:
                raise RuntimeError(f'Existing result crosses protocol: {result_path}')
            completed += 1
            continue
        instance = load_instance(INSTANCE_ROOT / state['relative_path'])
        current = decode_candidate(instance, candidate(state['candidate']))
        if not current.feasible or abs(current.makespan - state['replayed_makespan']) > 1e-9:
            raise RuntimeError(f"Representative replay mismatch: {state['label']}")
        runtime.prepare_instance(instance)
        streams = RNGStreams(instance.instance_id, measurement['sample_seed'])
        state_id = f"ngas-a15r-formal:{state['label']}"
        print(f"START {state['label']}", flush=True)
        for repetition in range(measurement['warmup_complete_refreshes']):
            runtime.refresh(instance, current, state_id, streams, sample_seed=repetition)
        samples = []
        for repetition in range(measurement['complete_refresh_repetitions_per_state']):
            result = runtime.refresh(
                instance, current, state_id, streams,
                sample_seed=measurement['sample_seed'] + repetition)
            samples.append(result.components_ms)
            if (repetition + 1) % 50 == 0:
                print(f"PROGRESS {state['label']} {repetition + 1}", flush=True)
        numeric = [value for row in samples for value in row.values()]
        payload = {
            'schema': 'ngas-a15r-formal-latency-raw-v1',
            'protocol_sha256': protocol_sha, 'formal_started_at_utc': formal_started,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'state': {key: state[key] for key in (
                'label', 'instance_id', 'instance_sha256', 'num_operations',
                'candidate_sha256', 'replayed_makespan')},
            'checkpoint_sha256': critic.sha256,
            'warmup_complete_refreshes': measurement['warmup_complete_refreshes'],
            'complete_refresh_repetitions': len(samples),
            'component_samples_ms': samples,
            'joint_actions': len(result.actions),
            'all_numeric_values_finite': all(math.isfinite(value) for value in numeric),
            'all_static_context_hits': result.static_context_hit,
            'workspace_capacities': result.workspace_capacities,
            'workspace_resize_events': result.workspace_resize_events,
            'normal_gc': True,
            'device': {'name': torch.cuda.get_device_name(0),
                       'torch': torch.__version__, 'cuda': torch.version.cuda},
            'locks': protocol['locks'],
        }
        atomic_json(result_path, payload)
        completed += 1
        atomic_json(PROGRESS, {
            'schema': 'ngas-a15r-progress-v1',
            'status': 'FORMAL_MEASUREMENT_COMPLETE' if completed == 4 else 'RUNNING',
            'completed_states': completed, 'expected_states': 4,
            'protocol_sha256': protocol_sha, 'last_completed': state['label'],
            'decision': 'PENDING_TRANSITION_TRACE_AND_FINAL_AUDIT',
            'A1_6': 'LOCKED', **protocol['locks'],
        })
        print(f"COMPLETE {state['label']}", flush=True)
    print(json.dumps({
        'status': 'FORMAL_MEASUREMENT_COMPLETE', 'completed_states': completed,
        'protocol_sha256': protocol_sha,
    }, indent=2))


if __name__ == '__main__':
    main()
