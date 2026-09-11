#!/usr/bin/env python3
"""Run the frozen NGAS A1.5 complete-refresh latency measurements."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_clgri.search.dabc_chdg import build_generalized_chdg
from rcias_ngas.csg.critical_mapping import map_critical_events
from rcias_ngas.csg.critical_sync import analyze_graph
from rcias_ngas.csg.revised_features import action_features, state_features_from_components
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.latency.live_refresh import LiveRefreshEngine, _joint_bank, _tensorize_cpu
from rcias_ngas.rng import RNGStreams

OUT = ROOT / 'outputs/ngas_a1/latency_qualification_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
PROGRESS = OUT / 'progress.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def candidate_from_payload(raw: dict) -> Candidate:
    return Candidate(*(tuple(raw[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def prepare_raw_batch(instance, current, state_id, streams, device):
    graph = build_csg_from_schedule(
        instance, current.schedule, state_id=state_id,
        search_progress=0., search_stage='0-20%', attach_hash=False)
    event_graph = build_generalized_chdg(instance, current)
    analysis = analyze_graph(event_graph)
    mapping = map_critical_events(event_graph, graph, analysis)
    state = state_features_from_components(
        instance, graph, mapping, include_hash=False, include_diagnostics=False)
    actions = _joint_bank(instance, current, state_id, streams, analysis)
    features = action_features(state, actions)
    cpu_batch = _tensorize_cpu(state, features)
    return actions, {name: value.to(device) for name, value in cpu_batch.items()}


def forward_models(critics, batch):
    encoded = [critic.model.encode_state(batch) for critic in critics]
    outputs = [critic.model.score_actions(nodes, pooled, batch)
               for critic, (nodes, pooled) in zip(critics, encoded)]
    advantage = torch.stack([row['advantage'] for row in outputs]).mean(0)
    probability = torch.sigmoid(torch.stack(
        [row['beats_fallback_logit'] for row in outputs])).mean(0)
    return advantage, probability


def raw_samples(critics, batch, warmups, repetitions, device):
    with torch.inference_mode():
        for _ in range(warmups):
            forward_models(critics, batch)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            advantage, probability = forward_models(critics, batch)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.)
    if not torch.isfinite(advantage).all() or not torch.isfinite(probability).all():
        raise FloatingPointError('Non-finite raw model output')
    return samples


def validate_boundary(protocol: dict) -> None:
    if digest(ROOT / protocol['config_path']) != protocol['config_sha256']:
        raise RuntimeError('Frozen A1.5 config hash mismatch')
    if digest(ROOT / protocol['representative_states_path']) \
            != protocol['representative_states_sha256']:
        raise RuntimeError('Frozen A1.5 representative-state hash mismatch')
    for path, expected in protocol['source_hashes'].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f'Frozen A1.5 source hash mismatch: {path}')
    for item in [protocol['production'], *protocol['development_ensemble']]:
        path_key = 'checkpoint_path' if 'checkpoint_path' in item else 'path'
        hash_key = 'checkpoint_sha256' if 'checkpoint_sha256' in item else 'sha256'
        if digest(ROOT / item[path_key]) != item[hash_key]:
            raise RuntimeError(f'Frozen checkpoint hash mismatch: {item[path_key]}')


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen A1.5 measurements require CUDA')
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha256 = digest(PROTOCOL)
    validate_boundary(protocol)
    config = json.loads((ROOT / protocol['config_path']).read_text())
    state_manifest = json.loads((ROOT / protocol['representative_states_path']).read_text())
    measurement = protocol['measurement']
    device = torch.device(measurement['device'])
    torch.cuda.reset_peak_memory_stats(device)
    production = FrozenJointCritic(
        ROOT / protocol['production']['checkpoint_path'], device,
        protocol['production']['checkpoint_sha256'], 'C1')
    ensemble = tuple(FrozenJointCritic(
        ROOT / item['path'], device, item['sha256'], 'C1')
        for item in protocol['development_ensemble'])
    modes = (
        ('SINGLE_C1', (production,), measurement['production_complete_repetitions'],
         measurement['production_raw_repetitions']),
        ('TEACHER_ENSEMBLE', ensemble, measurement['ensemble_complete_repetitions'],
         measurement['ensemble_raw_repetitions']),
    )
    completed = 0
    formal_started = datetime.now(timezone.utc).isoformat()
    for state in state_manifest['states']:
        instance = load_instance(ROOT / config['instance_root'] / state['relative_path'])
        current = decode_candidate(instance, candidate_from_payload(state['candidate']))
        if not current.feasible or abs(current.makespan - state['replayed_makespan']) > 1e-9:
            raise RuntimeError(f"Frozen representative replay mismatch: {state['label']}")
        state_id = f"ngas-a15-formal:{state['label']}"
        streams = RNGStreams(instance.instance_id, measurement['sample_seed'])
        for mode, critics, complete_repetitions, raw_repetitions in modes:
            result_path = OUT / 'raw' / state['label'] / f'{mode}.json'
            if result_path.exists():
                existing = json.loads(result_path.read_text())
                if existing.get('protocol_sha256') != protocol_sha256:
                    raise RuntimeError(f'Existing raw result crosses protocol boundary: {result_path}')
                completed += 1
                continue
            engine = LiveRefreshEngine(critics)
            print(f"START {state['label']} {mode} complete={complete_repetitions} raw={raw_repetitions}",
                  flush=True)
            for repetition in range(measurement['warmup_complete_refreshes']):
                engine.refresh(instance, current, state_id, streams,
                               sample_seed=measurement['sample_seed'] + repetition)
            component_samples = []
            for repetition in range(complete_repetitions):
                row = engine.refresh(
                    instance, current, state_id, streams,
                    sample_seed=measurement['sample_seed'] + repetition)
                component_samples.append(row.components_ms)
                if (repetition + 1) % 50 == 0:
                    print(f"PROGRESS {state['label']} {mode} {repetition + 1}/{complete_repetitions}",
                          flush=True)
            actions, batch = prepare_raw_batch(instance, current, state_id, streams, device)
            raw = raw_samples(
                critics, batch, measurement['warmup_raw_inferences'], raw_repetitions, device)
            if len(actions) != len(row.actions):
                raise RuntimeError('Prepared raw batch action count mismatch')
            result = {
                'schema': 'ngas-a15-latency-raw-v1',
                'protocol_sha256': protocol_sha256,
                'formal_started_at_utc': formal_started,
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'state': {key: state[key] for key in (
                    'label', 'instance_id', 'instance_sha256', 'num_operations',
                    'candidate_sha256', 'replayed_makespan')},
                'mode': mode,
                'critic_count': len(critics),
                'checkpoint_sha256': [critic.sha256 for critic in critics],
                'joint_actions': len(actions),
                'warmup_complete_refreshes': measurement['warmup_complete_refreshes'],
                'complete_refresh_repetitions': complete_repetitions,
                'component_samples_ms': component_samples,
                'warmup_raw_inferences': measurement['warmup_raw_inferences'],
                'raw_inference_repetitions': raw_repetitions,
                'raw_inference_samples_ms': raw,
                'all_numeric_values_finite': all(
                    math.isfinite(value) for sample in component_samples
                    for value in sample.values()) and all(math.isfinite(value) for value in raw),
                'device': {
                    'name': torch.cuda.get_device_name(device),
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                },
                'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False},
            }
            atomic_json(result_path, result)
            completed += 1
            atomic_json(PROGRESS, {
                'schema': 'ngas-a15-latency-progress-v1',
                'status': 'RUNNING' if completed < 8 else 'MEASUREMENT_COMPLETE',
                'completed_units': completed, 'expected_units': 8,
                'protocol_sha256': protocol_sha256,
                'last_completed': f"{state['label']}:{mode}",
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'decision': 'PENDING_COMPLETION_AUDIT',
                'next_gate': 'A1_5_COMPLETION_AUDIT',
                'A1_6': 'LOCKED', 'R13': 'LOCKED', 'R14': 'LOCKED',
                'gurobi_run': False,
            })
            print(f"COMPLETE {state['label']} {mode} -> {result_path.relative_to(ROOT)}", flush=True)
    print(json.dumps({'status': 'MEASUREMENT_COMPLETE', 'completed_units': completed,
                      'protocol_sha256': protocol_sha256}, indent=2), flush=True)


if __name__ == '__main__':
    main()
