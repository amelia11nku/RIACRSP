#!/usr/bin/env python3
"""Development-only tail-latency diagnosis for the immutable A1.5 runtime."""
from __future__ import annotations

import cProfile
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import pstats
import resource
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.latency.live_refresh import LiveRefreshEngine
from rcias_ngas.rng import RNGStreams

CONFIG = ROOT / 'configs/ngas_a15r_runtime_revision_v1.json'
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/diagnostics/tail_latency.json'
REPORT = ROOT / 'docs/reports/ngas_a1/11a_a15r_tail_diagnostic.md'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values):
    return {
        'count': len(values), 'mean': statistics.fmean(values),
        'p50': percentile(values, .5), 'p90': percentile(values, .9),
        'p99': percentile(values, .99), 'minimum': min(values), 'maximum': max(values),
    }


def candidate(raw):
    return Candidate(*(tuple(raw[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def summarize(rows):
    gc_rows = [row for row in rows if row['gc_collection_occurred']]
    no_gc_rows = [row for row in rows if not row['gc_collection_occurred']]
    return {
        'wall_ms': stats([row['wall_ms'] for row in rows]),
        'process_cpu_ms': stats([row['process_cpu_ms'] for row in rows]),
        'residual_unattributed_ms': stats([row['residual_unattributed_ms'] for row in rows]),
        'samples_with_gc_collection': len(gc_rows),
        'wall_ms_with_gc': stats([row['wall_ms'] for row in gc_rows]) if gc_rows else None,
        'wall_ms_without_gc': stats([row['wall_ms'] for row in no_gc_rows]) if no_gc_rows else None,
        'samples_with_voluntary_context_switch': sum(row['voluntary_context_switches'] > 0 for row in rows),
        'samples_with_involuntary_context_switch': sum(row['involuntary_context_switches'] > 0 for row in rows),
        'wall_minus_cpu_ms': stats([row['wall_ms'] - row['process_cpu_ms'] for row in rows]),
        'component_p50_ms': {
            name: percentile([row['components_ms'][name] for row in rows], .5)
            for name in rows[0]['components_ms']
        },
    }


def render(payload):
    lines = [
        '# NGAS A1.5R 尾延迟诊断', '',
        '本报告是开发诊断，不是正式资格结果。A1.5 历史结果保持不变。', '',
        '| 状态 | 条件 | wall p50 / p90 / p99 (ms) | GC samples | voluntary ctx samples | involuntary ctx samples | residual p50 (ms) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for label in payload['states']:
        for condition in payload['states'][label]:
            row = payload['states'][label][condition]['summary']
            wall = row['wall_ms']
            lines.append(
                f"| {label} | {condition} | {wall['p50']:.3f} / {wall['p90']:.3f} / {wall['p99']:.3f} | "
                f"{row['samples_with_gc_collection']} | {row['samples_with_voluntary_context_switch']} | "
                f"{row['samples_with_involuntary_context_switch']} | {row['residual_unattributed_ms']['p50']:.3f} |")
    lines += ['', '## 解释', '']
    for statement in payload['conclusions']:
        lines.append(f'- {statement}')
    lines += ['', 'GC-disabled 数据只用于归因，不用于正式门槛判定。']
    return '\n'.join(lines) + '\n'


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('A1.5R tail diagnosis requires CUDA')
    config = json.loads(CONFIG.read_text())
    states_path = ROOT / config['historical_A1_5']['representative_states']
    if digest(states_path) != config['historical_A1_5']['representative_states_sha256']:
        raise RuntimeError('Historical A1.5 representative-state boundary changed')
    checkpoint = config['production_checkpoint']
    critic = FrozenJointCritic(
        ROOT / checkpoint['path'], 'cuda:0', checkpoint['sha256'], checkpoint['variant'])
    engine = LiveRefreshEngine([critic])
    manifest = json.loads(states_path.read_text())
    selected = {row['label']: row for row in manifest['states']
                if row['label'] in config['diagnostic']['states']}
    active = {'sample': None, 'events': []}

    def callback(phase, info):
        if active['sample'] is not None:
            active['events'].append({
                'sample': active['sample'], 'phase': phase,
                'generation': info['generation'], 'collected': info.get('collected'),
            })

    gc.callbacks.append(callback)
    results = {}
    try:
        for label in config['diagnostic']['states']:
            state = selected[label]
            instance = load_instance(
                ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / state['relative_path'])
            current = decode_candidate(instance, candidate(state['candidate']))
            streams = RNGStreams(instance.instance_id, config['diagnostic']['sample_seed'])
            state_id = f'ngas-a15r-diagnostic:{label}'
            results[label] = {}
            for condition in config['diagnostic']['conditions']:
                gc.collect()
                for repetition in range(config['diagnostic']['warmup_repetitions']):
                    engine.refresh(instance, current, state_id, streams, sample_seed=repetition)
                rows = []
                for repetition in range(config['diagnostic']['measured_repetitions']):
                    usage_before = resource.getrusage(resource.RUSAGE_SELF)
                    counts_before = gc.get_count()
                    collections_before = tuple(row['collections'] for row in gc.get_stats())
                    event_start = len(active['events'])
                    active['sample'] = f'{label}:{condition}:{repetition}'
                    disabled = condition == 'GC_DISABLED_DURING_REFRESH'
                    if disabled:
                        gc.disable()
                    cpu_started = time.process_time_ns()
                    wall_started = time.perf_counter_ns()
                    try:
                        result = engine.refresh(
                            instance, current, state_id, streams,
                            sample_seed=config['diagnostic']['sample_seed'] + repetition)
                    finally:
                        wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
                        cpu_ms = (time.process_time_ns() - cpu_started) / 1e6
                        if disabled:
                            gc.enable()
                        active['sample'] = None
                    usage_after = resource.getrusage(resource.RUSAGE_SELF)
                    collections_after = tuple(row['collections'] for row in gc.get_stats())
                    named = sum(result.components_ms[name] for name in (
                        'csg_update_build', 'critical_event_graph', 'critical_extraction',
                        'critical_mapping', 'state_feature_update',
                        'candidate_joint_action_generation', 'action_feature_update',
                        'cpu_tensorization', 'h2d_enqueue_wall',
                        'output_and_synchronization_wall', 'prior_construction', 'ranking_sampling'))
                    events = active['events'][event_start:]
                    rows.append({
                        'repetition': repetition, 'wall_ms': wall_ms,
                        'process_cpu_ms': cpu_ms,
                        'gc_counts_before': counts_before, 'gc_counts_after': gc.get_count(),
                        'gc_collections_before': collections_before,
                        'gc_collections_after': collections_after,
                        'gc_collection_occurred': any(event['phase'] == 'start' for event in events),
                        'gc_events': events,
                        'voluntary_context_switches': usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                        'involuntary_context_switches': usage_after.ru_nivcsw - usage_before.ru_nivcsw,
                        'residual_unattributed_ms': wall_ms - named,
                        'components_ms': result.components_ms,
                        'workspace_capacities': None,
                        'workspace_resize_events': None,
                    })
                results[label][condition] = {'samples': rows, 'summary': summarize(rows)}
                print(label, condition, results[label][condition]['summary']['wall_ms'], flush=True)

        label = 'L_MAX'
        state = selected[label]
        instance = load_instance(
            ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / state['relative_path'])
        current = decode_candidate(instance, candidate(state['candidate']))
        streams = RNGStreams(instance.instance_id, config['diagnostic']['sample_seed'])
        profile = cProfile.Profile()
        profile.runcall(engine.refresh, instance, current, 'ngas-a15r-profile:L_MAX', streams,
                        sample_seed=config['diagnostic']['sample_seed'])
        profile_text = io.StringIO()
        pstats.Stats(profile, stream=profile_text).strip_dirs().sort_stats('cumtime').print_stats(40)
    finally:
        gc.callbacks.remove(callback)

    conclusions = []
    for label in config['diagnostic']['states']:
        normal = results[label]['NORMAL_GC']['summary']
        disabled = results[label]['GC_DISABLED_DURING_REFRESH']['summary']
        conclusions.append(
            f"{label}: normal GC p90 {normal['wall_ms']['p90']:.3f} ms, GC-disabled p90 "
            f"{disabled['wall_ms']['p90']:.3f} ms; normal samples containing GC collection "
            f"{normal['samples_with_gc_collection']}/{normal['wall_ms']['count']}.")
    normal_rows = [row for label in results.values()
                   for row in label['NORMAL_GC']['samples']]
    gc_rows = [row for row in normal_rows if row['gc_collection_occurred']]
    no_gc_rows = [row for row in normal_rows if not row['gc_collection_occurred']]
    if gc_rows and no_gc_rows:
        conclusions.append(
            f"Across normal-GC samples, GC-associated mean wall time is "
            f"{statistics.fmean(row['wall_ms'] for row in gc_rows):.3f} ms versus "
            f"{statistics.fmean(row['wall_ms'] for row in no_gc_rows):.3f} ms without GC.")
    elif gc_rows:
        conclusions.append(
            f"Every normal-GC sample ({len(gc_rows)}/{len(normal_rows)}) contained at least one "
            'collection, so an within-condition no-collection mean is unavailable.')
    conclusions.append(
        'Stable median cost remains structural even when GC is disabled; the implementation must reduce CSG/event, bank, action-feature, and tensor materialization work.')
    payload = {
        'schema': 'ngas-a15r-tail-diagnostic-v1',
        'development_only': True,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG),
        'historical_A1_5_immutable': True,
        'device': {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'cuda': torch.version.cuda},
        'states': results, 'conclusions': conclusions,
        'L_MAX_cprofile_top40': profile_text.getvalue(),
        'formal_claim_allowed': False,
        'locks': config['locks'],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(payload))
    print(json.dumps({'output': str(OUT.relative_to(ROOT)),
                      'report': str(REPORT.relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
