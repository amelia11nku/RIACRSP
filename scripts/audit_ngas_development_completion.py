#!/usr/bin/env python3
"""Independent integrity and contract audit before any NGAS critic training."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.bank.provenance import Target
from rcias_ngas.csg.features import RULES
from rcias_ngas.critic.label_revision import aggregate
from rcias_ngas.evaluation.bks import content_hash, write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.collect_ngas_development import OUT, data_gate, verify_protocol


def unseal(path):
    value = json.loads(path.read_text())
    expected = value.pop('payload_sha256')
    if content_hash(value) != expected:
        raise ValueError(f'Payload hash mismatch: {path}')
    value['payload_sha256'] = expected
    return value


def action_from(metadata):
    target = Target(**{key: tuple(value) if isinstance(value, list) else value
                       for key, value in metadata['target'].items()})
    action = JointAction(metadata['size'], target, metadata['repair'])
    if action.action_id != metadata['action_id']:
        raise ValueError('Joint action ID mismatch')
    return action


def main():
    protocol = verify_protocol()
    config = protocol['config']
    manifest = json.loads((OUT / 'result_hash_manifest.json').read_text())
    actual = {str(path.relative_to(ROOT)) for path in (OUT / 'states').rglob('*.json')}
    if actual != set(manifest):
        raise ValueError('Expanded result file set differs from manifest')
    for path, expected in manifest.items():
        if digest(path) != expected:
            raise ValueError(f'Expanded result file drift: {path}')

    summaries = []
    total_actions = total_replicates = label_evals = source_evals = 0
    full_label_replays = 0
    by_fold = defaultdict(set)
    for spec in protocol['states']:
        directory = OUT / 'states' / spec['state_id']
        state_path = directory / 'state.json'
        state = unseal(state_path)
        context = state['context']
        if context['protocol_sha256'] != digest(OUT / 'protocol.json'):
            raise ValueError('State protocol mismatch')
        instance_path = (ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'
                         / spec['instance']['relative_path'])
        if digest(instance_path) != spec['instance']['sha256']:
            raise ValueError('Instance hash drift')
        instance = load_instance(instance_path)
        current = decode_candidate(instance, Candidate(**{
            key: tuple(value) for key, value in state['candidate'].items()}))
        if not current.feasible or current.makespan != state['makespan']:
            raise ValueError('Source candidate replay failed')
        source_evals += sum(int(row['decoder_evals']) for row in state['source_trace'])
        by_fold[spec['fold']].add(instance.instance_id)

        action_record = unseal(directory / 'actions.json')
        if action_record['context']['state_sha256'] != digest(state_path):
            raise ValueError('Action/state hash mismatch')
        metadata = action_record['actions']
        features = action_record['action_features']
        count = len(metadata)
        if not (len(features['membership']) == len(features['provenance'])
                == len(features['sizes']) == len(features['repairs']) == count):
            raise ValueError('Action feature row mismatch')
        if sum(len(bank['proposals']) for bank in action_record['full_banks']) != 72:
            raise ValueError('Full-bank-first evidence missing')
        actions = {row['action_id']: action_from(row) for row in metadata}
        labels = []
        fallback_hashes = []
        for path in sorted((directory / 'fallback').glob('*.json')):
            row = unseal(path)
            fallback_hashes.append(row['payload_sha256'])
            label_evals += sum(step['decoder_evals'] for step in row['trajectory']['steps'])
        if len(fallback_hashes) != 9 or len(set(fallback_hashes)) != 9:
            raise ValueError('Fallback cache incomplete or duplicated')
        label_paths = sorted((directory / 'labels').glob('*.json'))
        if len(label_paths) != count:
            raise ValueError('Action/label cardinality mismatch')
        for label_index, path in enumerate(label_paths):
            row = unseal(path)
            action_id = row['action']['action_id']
            action = actions.pop(action_id)
            if row['action'] != json.loads(json.dumps(action.metadata())):
                raise ValueError('Label action metadata mismatch')
            if row['context']['fallback_hashes'] != fallback_hashes:
                raise ValueError('Label fallback cache mismatch')
            if len(row['replicates']) != 9:
                raise ValueError('Label replicate count mismatch')
            for replicate in row['replicates']:
                candidate = replicate['candidate']
                fallback = replicate['fallback']
                first = candidate['steps'][0]
                if (first['action_id'], first['repair'], first['destroyed_operations']) != (
                        action_id, action.repair, list(action.target.operations)):
                    raise ValueError('Executed first action differs from label identity')
                for candidate_step, fallback_step in zip(candidate['steps'], fallback['steps']):
                    if not candidate_step['feasible'] or not fallback_step['feasible']:
                        raise ValueError('Infeasible label trajectory')
                    if candidate_step['decoder_evals'] != 8 or fallback_step['decoder_evals'] != 8:
                        raise ValueError('Label decoder accounting drift')
                    if (candidate_step['neighbor_seed'], candidate_step['acceptance_seed']) != (
                            fallback_step['neighbor_seed'], fallback_step['acceptance_seed']):
                        raise ValueError('Paired CRN mismatch')
                expected_advantage = ((fallback['best_makespan'] - candidate['best_makespan'])
                                      / current.makespan)
                if replicate['advantage'] != expected_advantage:
                    raise ValueError('Raw advantage mismatch')
                label_evals += sum(step['decoder_evals'] for step in candidate['steps'])
                total_replicates += 1
            derived = aggregate(row['action'], current.makespan, row['replicates'], 8, 2)
            for key, value in derived.items():
                if row[key] != value:
                    raise ValueError(f'Aggregate label mismatch: {key}')
            labels.append(row)
            total_actions += 1
        if actions:
            raise ValueError('Unlabelled action remains')
        # Membership/provenance are recomputed independently by construction tests;
        # here verify exact membership cardinality and absence of any label field.
        for row, membership in zip(metadata, features['membership']):
            if sum(membership) != row['destroy_count']:
                raise ValueError('Target membership/cardinality mismatch')
        forbidden = ('advantage', 'reward', 'makespan', 'label')
        if any(any(word in key for word in forbidden)
               for key in action_record['action_features']):
            raise ValueError('Outcome leaked into online action features')
        from scripts.collect_ngas_development import state_summary
        summaries.append(state_summary(spec, labels))

    recomputed_gate = data_gate(summaries, config)
    saved_gate = json.loads((OUT / 'data_gate.json').read_text())
    if recomputed_gate['decision'] != saved_gate['decision'] or recomputed_gate['checks'] != saved_gate['checks']:
        raise ValueError('Data gate did not reproduce')
    if total_actions != 6465 or total_replicates != total_actions * 9:
        raise ValueError('Expanded label totals differ')
    if label_evals != saved_gate['label_decoder_evals'] or source_evals != 6912:
        raise ValueError('Expanded decoder totals differ')
    if any(len(instances) != 6 for instances in by_fold.values()):
        raise ValueError('Fold instance count drift')
    audit = {
        'schema': 'ngas-development-completion-audit-v1', 'status': 'PASS',
        'protocol_sha256': digest(OUT / 'protocol.json'),
        'result_manifest_sha256': digest(OUT / 'result_hash_manifest.json'),
        'data_gate_sha256': digest(OUT / 'data_gate.json'),
        'verified_files': len(manifest), 'states': len(summaries),
        'actions': total_actions, 'paired_replicates': total_replicates,
        'label_decoder_evals': label_evals, 'source_decoder_evals': source_evals,
        'source_candidate_replays': len(summaries),
        'gate_reproduced': True, 'decision': saved_gate['decision'],
        'instances_per_fold': {str(key): len(value) for key, value in by_fold.items()},
        'all_rules_covered': set(RULES) <= {rule for row in summaries for rule in row['rules']},
        'training_started': False, 'r13': 'LOCKED', 'r14': 'LOCKED',
    }
    verify_protocol()
    output = ROOT / 'outputs/ngas_a1/audit/development_completion.json'
    if output.exists():
        if json.loads(output.read_text()) != audit:
            raise FileExistsError('Existing development completion audit differs')
    else:
        write_immutable(output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
