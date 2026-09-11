#!/usr/bin/env python3
"""Strong reference-vs-production equivalence matrix for A1.5R."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import Candidate, candidate_from_actions, decode_candidate
from rcias_clgri.search.dabc_chdg import build_generalized_chdg
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS, execute_action
from rcias_ngas.csg.critical_mapping import map_critical_events
from rcias_ngas.csg.critical_sync import analyze_graph
from rcias_ngas.csg.revised_features import action_features, state_features_from_components
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.evaluation.bks import content_hash
from rcias_ngas.latency.live_refresh import _joint_bank, _tensorize_cpu
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import ProductionRefreshRuntime
from rcias_ngas.search.ngas_solver import _critical_identity
from rcias_ngas.search.persistent_prior import normalized_prior


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/ngas_a15r_runtime_revision_v1.json'
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/equivalence/equivalence_matrix.json'
INSTANCE_ROOT = ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate(raw):
    return Candidate(*(tuple(raw[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def state_hash(current):
    value = current.candidate
    return content_hash([
        value.operation_order, value.island_assignment,
        value.w_assignment, value.f_assignment,
    ])


def resource_order_hash(current):
    schedule = current.schedule
    return content_hash({
        'islands': schedule.island_timelines,
        'w': {key: [task.operation_id for task in value]
              for key, value in schedule.w_timelines.items()},
        'f': {key: [task.operation_id for task in value]
              for key, value in schedule.f_timelines.items()},
    })


def reference(instance, current, state_id, streams, critic, sample_seed):
    graph = build_csg_from_schedule(
        instance, current.schedule, state_id=state_id,
        search_progress=0., search_stage='0-20%', attach_hash=False)
    event_graph = build_generalized_chdg(instance, current)
    analysis = analyze_graph(event_graph)
    mapping = map_critical_events(event_graph, graph, analysis)
    state = state_features_from_components(
        instance, graph, mapping, include_hash=False, include_diagnostics=False)
    actions = _joint_bank(instance, current, state_id, streams, analysis)
    cpu_batch = _tensorize_cpu(state, action_features(state, actions))
    batch = {name: value.to(critic.device) for name, value in cpu_batch.items()}
    with torch.inference_mode():
        output = critic.model(batch)
    advantage = tuple(float(value) for value in output['advantage'].detach().cpu().tolist())
    logits = tuple(
        float(value) for value in output['beats_fallback_logit'].detach().cpu().tolist())
    probability = tuple(
        float(value) for value in torch.sigmoid(
            output['beats_fallback_logit']).detach().cpu().tolist())
    prior = normalized_prior(advantage, .01, .05)
    ranking = tuple(sorted(
        range(len(actions)), key=lambda index: (-prior[index], actions[index].action_id)))
    sampled = random.Random(sample_seed).choices(
        range(len(actions)), weights=prior, k=1)[0]
    return {
        'actions': actions, 'cpu_batch': cpu_batch, 'advantage': advantage,
        'logits': logits, 'probability': probability, 'prior': prior,
        'ranking': ranking, 'sampled': sampled,
        'critical_identity': _critical_identity(analysis),
    }


def action_payload(action):
    return (
        action.action_id, action.size, action.target.target_id,
        action.target.operations, action.target.origin_rules,
        action.target.origin_families, action.target.origin_operators,
        action.repair,
    )


def max_error(left, right):
    return max((abs(a - b) for a, b in zip(left, right)), default=0.)


def compare(instance, current, label, state_id, streams, critic, runtime, tolerance):
    sample_seed = streams.seed('diagnostics', state_id)
    expected = reference(
        instance, current, state_id, streams, critic, sample_seed)
    actual = runtime.refresh(
        instance, current, state_id, streams, sample_seed=sample_seed)
    tensor_errors = {}
    integer_exact = True
    compact = runtime._builders[instance.instance_id].build(current, state_id, streams)
    for name, expected_tensor in expected['cpu_batch'].items():
        actual_tensor = compact.cpu_batch[name]
        if expected_tensor.dtype.is_floating_point:
            tensor_errors[name] = float(torch.max(
                torch.abs(expected_tensor - actual_tensor)).item())
        else:
            integer_exact &= torch.equal(expected_tensor, actual_tensor)
    errors = {
        'feature_tensors': max(tensor_errors.values(), default=0.),
        'advantages': max_error(expected['advantage'], actual.advantage),
        'logits': max_error(expected['logits'], actual.beats_fallback_logit),
        'probabilities': max_error(expected['probability'], actual.beats_fallback_probability),
        'prior': max_error(expected['prior'], actual.prior),
    }
    assertions = {
        'action_payload_exact': tuple(map(action_payload, expected['actions']))
            == tuple(map(action_payload, actual.actions)),
        'candidate_count_exact': len(expected['actions']) == len(actual.actions),
        'integer_tensors_exact': integer_exact,
        'float_tensors_within_tolerance': errors['feature_tensors'] <= tolerance,
        'inference_within_tolerance': max(
            errors['advantages'], errors['logits'], errors['probabilities']) <= tolerance,
        'prior_within_tolerance': errors['prior'] <= tolerance,
        'ranking_exact': expected['ranking'] == actual.ranking,
        'fixed_seed_selection_exact': expected['sampled'] == actual.sampled_index,
        'critical_identity_exact': expected['critical_identity'] == (
            actual.critical_signature, actual.dominant_bottleneck),
        'workspace_no_resize': actual.workspace_resize_events == 0,
    }
    return {
        'label': label, 'state_id': state_id, 'state_hash': state_hash(current),
        'resource_order_hash': resource_order_hash(current),
        'num_operations': instance.num_operations,
        'joint_actions': len(actual.actions), 'errors': errors,
        'tensor_errors': tensor_errors, 'assertions': assertions,
        'pass': all(assertions.values()),
    }


def accepted_repair_transition(instance, current, repair, step, streams, runtime):
    state_id = f'{instance.instance_id}:accepted-repair-source:{step}:{repair}'
    refresh = runtime.refresh(instance, current, state_id, streams, sample_seed=step)
    choices = [action for action in refresh.actions if action.repair == repair]
    temperature = .05 * max(1., current.makespan)
    for trial, action in enumerate(choices):
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
    raise RuntimeError(f'No deterministic accepted {repair} transition for {instance.instance_id}')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('A1.5R equivalence audit requires CUDA')
    config = json.loads(CONFIG.read_text())
    state_path = ROOT / config['historical_A1_5']['representative_states']
    if digest(state_path) != config['historical_A1_5']['representative_states_sha256']:
        raise RuntimeError('Historical A1.5 representative-state boundary changed')
    checkpoint = config['production_checkpoint']
    critic = FrozenJointCritic(
        ROOT / checkpoint['path'], 'cuda:0', checkpoint['sha256'], checkpoint['variant'])
    runtime = ProductionRefreshRuntime(critic)
    tolerance = config['equivalence']['external_float_max_abs_error']
    rows, transition_records, alternating = [], [], []
    for frozen in json.loads(state_path.read_text())['states']:
        instance = load_instance(INSTANCE_ROOT / frozen['relative_path'])
        representative = decode_candidate(instance, candidate(frozen['candidate']))
        h1 = solve_dispatching(instance, 'H1')
        initial = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
        streams = RNGStreams(instance.instance_id, 746101)
        runtime.prepare_instance(instance)
        rows.append(compare(
            instance, representative, f"{frozen['label']}:representative",
            f"a15r-equivalence:{frozen['label']}:representative",
            streams, critic, runtime, tolerance))
        rows.append(compare(
            instance, initial, f"{frozen['label']}:H1",
            f"a15r-equivalence:{frozen['label']}:H1",
            streams, critic, runtime, tolerance))
        current = representative
        first_transition = None
        for step, repair in enumerate(NGAS_REPAIR_IDS):
            before_hash = state_hash(current)
            before_resource = resource_order_hash(current)
            current, action = accepted_repair_transition(
                instance, current, repair, step, streams, runtime)
            if first_transition is None:
                first_transition = current
            row = compare(
                instance, current, f"{frozen['label']}:accepted:{repair}",
                f"a15r-equivalence:{frozen['label']}:accepted:{step}:{repair}",
                streams, critic, runtime, tolerance)
            rows.append(row)
            transition_records.append({
                'scale': frozen['label'], 'step': step, 'repair': repair,
                'action_id': action.action_id, 'before_state_hash': before_hash,
                'after_state_hash': row['state_hash'],
                'state_changed': before_hash != row['state_hash'],
                'resource_order_changed': before_resource != row['resource_order_hash'],
                'equivalence_pass': row['pass'],
            })
        if first_transition is None:
            raise RuntimeError('Transition trace unexpectedly empty')
        alternating_rows = [
            compare(instance, representative, f"{frozen['label']}:ABA:A1",
                    f"a15r-equivalence:{frozen['label']}:ABA:A", streams,
                    critic, runtime, tolerance),
            compare(instance, first_transition, f"{frozen['label']}:ABA:B",
                    f"a15r-equivalence:{frozen['label']}:ABA:B", streams,
                    critic, runtime, tolerance),
            compare(instance, representative, f"{frozen['label']}:ABA:A2",
                    f"a15r-equivalence:{frozen['label']}:ABA:A", streams,
                    critic, runtime, tolerance),
        ]
        alternating.append({
            'scale': frozen['label'],
            'a_state_hash_restored': alternating_rows[0]['state_hash']
                == alternating_rows[2]['state_hash'],
            'a_actions_restored': alternating_rows[0]['joint_actions']
                == alternating_rows[2]['joint_actions'],
            'all_equivalent': all(row['pass'] for row in alternating_rows),
        })
        rows.extend(alternating_rows)
    payload = {
        'schema': 'ngas-a15r-equivalence-matrix-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG),
        'checkpoint_sha256': critic.sha256,
        'tolerance': tolerance,
        'coverage': {
            'frozen_representatives': 4, 'H1_states': 4,
            'accepted_repair_states': len(transition_records),
            'repair_operators': list(NGAS_REPAIR_IDS),
            'alternating_A_B_A_scales': len(alternating),
            'distinct_state_hashes': len({row['state_hash'] for row in rows}),
            'resource_order_changes': sum(
                row['resource_order_changed'] for row in transition_records),
        },
        'rows': rows, 'accepted_transitions': transition_records,
        'alternating_A_B_A': alternating,
        'all_equivalence_checks_pass': all(row['pass'] for row in rows),
        'all_transitions_distinct': all(
            row['state_changed'] for row in transition_records),
        'all_A_B_A_checks_pass': all(
            all(value for key, value in row.items() if key != 'scale')
            for row in alternating),
        'workspace_resize_events': sum(
            builder.workspace.resize_events for builder in runtime._builders.values()),
        'historical_A1_5_immutable': True,
        'locks': config['locks'],
    }
    payload['pass'] = all((
        payload['all_equivalence_checks_pass'],
        payload['all_transitions_distinct'], payload['all_A_B_A_checks_pass'],
        payload['workspace_resize_events'] == 0,
    ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUT)
    print(json.dumps({
        'output': str(OUT.relative_to(ROOT)), 'rows': len(rows),
        'coverage': payload['coverage'], 'pass': payload['pass'],
    }, indent=2))


if __name__ == '__main__':
    main()
