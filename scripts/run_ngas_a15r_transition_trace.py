#!/usr/bin/env python3
"""Distinct-state anti-cache transition trace for A1.5R."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS, execute_action
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.evaluation.bks import content_hash
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import ProductionRefreshRuntime
from scripts.audit_ngas_a15r_equivalence import action_payload, max_error, reference


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
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


def state_hash(current):
    value = current.candidate
    return content_hash([
        value.operation_order, value.island_assignment,
        value.w_assignment, value.f_assignment])


def stats(values):
    return {
        'count': len(values), 'mean': float(np.mean(values)),
        'p50': float(np.percentile(values, 50, method='linear')),
        'p90': float(np.percentile(values, 90, method='linear')),
        'p99': float(np.percentile(values, 99, method='linear')),
        'max': max(values),
    }


def accepted_transition(instance, current, actions, repair, state_id, streams):
    temperature = .05 * max(1., current.makespan)
    matching = [action for action in actions if action.repair == repair]
    for trial, action in enumerate(matching):
        proposal, _ = execute_action(
            instance, current, action, streams,
            f'{state_id}:candidate:{trial}', trials=1)
        if proposal.candidate == current.candidate:
            continue
        delta = proposal.makespan - current.makespan
        accepted = delta <= 0 or streams.stream(
            'acceptance', state_id, trial).random() < math.exp(
                -delta / max(temperature, 1e-12))
        if accepted:
            return proposal, action
    raise RuntimeError(f'No accepted trace transition: {state_id}')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('A1.5R transition trace requires CUDA')
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha = digest(PROTOCOL)
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen source changed: {relative}')
    critic = FrozenJointCritic(
        ROOT / protocol['checkpoint_path'], protocol['measurement']['device'],
        protocol['checkpoint_sha256'], 'C1')
    runtime = ProductionRefreshRuntime(critic)
    states = json.loads((ROOT / protocol['representative_states_path']).read_text())['states']
    tolerance = protocol['equivalence_tolerance']
    per_scale, all_rows = {}, []
    for frozen in states:
        instance = load_instance(INSTANCE_ROOT / frozen['relative_path'])
        current = decode_candidate(instance, candidate(frozen['candidate']))
        streams = RNGStreams(instance.instance_id, protocol['transition_trace']['seed'])
        runtime.prepare_instance(instance)
        rows, seen = [], set()
        previous_signature = None
        for step in range(protocol['transition_trace']['distinct_states_per_scale']):
            before = state_hash(current)
            if before in seen:
                raise RuntimeError(f"Repeated timed state in {frozen['label']} trace")
            seen.add(before)
            state_id = f"ngas-a15r-transition:{frozen['label']}:{step}"
            result = runtime.refresh(
                instance, current, state_id, streams,
                sample_seed=protocol['transition_trace']['seed'] + step)
            expected = reference(
                instance, current, state_id, streams, critic,
                protocol['transition_trace']['seed'] + step)
            inference_error = max(
                max_error(expected['advantage'], result.advantage),
                max_error(expected['logits'], result.beats_fallback_logit),
                max_error(expected['probability'], result.beats_fallback_probability),
                max_error(expected['prior'], result.prior))
            equivalent = all((
                tuple(map(action_payload, expected['actions']))
                    == tuple(map(action_payload, result.actions)),
                expected['ranking'] == result.ranking,
                expected['sampled'] == result.sampled_index,
                expected['critical_identity'] == (
                    result.critical_signature, result.dominant_bottleneck),
                inference_error <= tolerance,
            ))
            repair = NGAS_REPAIR_IDS[step % len(NGAS_REPAIR_IDS)]
            following, action = accepted_transition(
                instance, current, result.actions, repair, state_id, streams)
            after = state_hash(following)
            row = {
                'scale': frozen['label'], 'step': step,
                'state_hash': before, 'next_state_hash': after,
                'distinct_transition': before != after,
                'repair': repair, 'action_id': action.action_id,
                'complete_refresh_ms': result.components_ms['complete_refresh'],
                'static_context_hit': result.static_context_hit,
                'workspace_resize_events': result.workspace_resize_events,
                'critical_signature': result.critical_signature,
                'critical_structure_changed': previous_signature is not None
                    and previous_signature != result.critical_signature,
                'reference_equivalent': equivalent,
                'max_external_error': inference_error,
            }
            rows.append(row)
            previous_signature = result.critical_signature
            current = following
        per_scale[frozen['label']] = {
            'complete_refresh_ms': stats([row['complete_refresh_ms'] for row in rows]),
            'distinct_state_hashes': len({row['state_hash'] for row in rows}),
            'critical_structure_changes': sum(
                row['critical_structure_changed'] for row in rows),
            'all_reference_equivalent': all(row['reference_equivalent'] for row in rows),
            'all_static_context_hits': all(row['static_context_hit'] for row in rows),
            'workspace_resize_events': max(row['workspace_resize_events'] for row in rows),
        }
        all_rows.extend(rows)
        print(frozen['label'], per_scale[frozen['label']], flush=True)
    assertions = {
        'all_timed_states_distinct_within_scale': all(
            value['distinct_state_hashes']
                == protocol['transition_trace']['distinct_states_per_scale']
            for value in per_scale.values()),
        'every_transition_changes_state': all(
            row['distinct_transition'] for row in all_rows),
        'all_reference_equivalent': all(
            row['reference_equivalent'] for row in all_rows),
        'all_static_context_hits': all(row['static_context_hit'] for row in all_rows),
        'zero_workspace_resizes': all(
            value['workspace_resize_events'] == 0 for value in per_scale.values()),
        'all_repair_operators_covered': set(row['repair'] for row in all_rows)
            == set(NGAS_REPAIR_IDS),
    }
    payload = {
        'schema': 'ngas-a15r-transition-trace-v1',
        'protocol_sha256': protocol_sha,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'total_timed_distinct_states': len(all_rows),
        'distinct_state_hashes_global': len({row['state_hash'] for row in all_rows}),
        'per_scale': per_scale, 'rows': all_rows,
        'assertions': assertions, 'pass': all(assertions.values()),
        'historical_A1_5_immutable': True, 'locks': protocol['locks'],
    }
    path = OUT / 'transition_trace/transition_trace.json'
    atomic_json(path, payload)
    print(json.dumps({
        'output': str(path.relative_to(ROOT)),
        'total_timed_distinct_states': len(all_rows),
        'assertions': assertions, 'pass': payload['pass'],
    }, indent=2))


if __name__ == '__main__':
    main()
