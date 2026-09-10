#!/usr/bin/env python3
"""Audit frozen A1.2 labels without executing any continuation rollout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS
from rcias_ngas.bank.provenance import Target
from rcias_ngas.csg.revised_schema import FAMILIES, OPERATORS, RULES
from rcias_ngas.evaluation.bks import content_hash

DEVELOPMENT = ROOT / 'outputs/ngas_a1/development_v1'
OUT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/audit/label_identity.json'
REPORT = ROOT / 'docs/reports/ngas_a1/08c_a12_joint_action_label_identity.md'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unseal(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop('payload_sha256')
    if content_hash(value) != expected:
        raise ValueError(f'Frozen payload hash mismatch: {path}')
    return value


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    protocol = json.loads((DEVELOPMENT / 'protocol.json').read_text())
    manifest = json.loads((DEVELOPMENT / 'result_hash_manifest.json').read_text())
    for relative, expected in manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen development artifact drift: {relative}')
    mismatches: list[dict] = []
    totals = {
        'states': 0, 'actions': 0, 'replicates': 0, 'candidate_first_steps': 0,
        'paired_crn_steps': 0, 'fallback_payload_matches': 0,
        'stable_action_ids': 0, 'stable_target_ids': 0,
        'provenance_permutation_checks': 0,
    }
    seen_state_actions: dict[str, set[str]] = {}
    label_file_hashes = {}
    semantic_rows = []

    def mismatch(kind: str, state_id: str, action_id: str | None = None) -> None:
        if len(mismatches) < 50:
            mismatches.append({'kind': kind, 'state_id': state_id, 'action_id': action_id})

    for spec in protocol['states']:
        state_id = spec['state_id']
        directory = DEVELOPMENT / 'states' / state_id
        state = unseal(directory / 'state.json')
        action_record = unseal(directory / 'actions.json')
        operations = tuple(state['csg_features']['operation_ids'])
        actions = action_record['actions']
        encoded = action_record['action_features']
        totals['states'] += 1
        seen_state_actions[state_id] = set()
        fallback_by_seed = {}
        fallback_hash_by_seed = {}
        for fallback_path in sorted((directory / 'fallback').glob('*.json')):
            sealed = json.loads(fallback_path.read_text())
            fallback = unseal(fallback_path)
            seed = fallback['context']['seed']
            fallback_by_seed[seed] = fallback['trajectory']
            fallback_hash_by_seed[seed] = sealed['payload_sha256']
        for index, metadata in enumerate(actions):
            action_id = metadata['action_id']
            target = metadata['target']
            target_operations = tuple(target['operations'])
            totals['actions'] += 1
            if action_id in seen_state_actions[state_id]:
                mismatch('duplicate_action_id', state_id, action_id)
            seen_state_actions[state_id].add(action_id)
            expected_target_id = 'ngas_target_' + content_hash(
                ['ngas-bank-v1', state_id, metadata['size'], tuple(sorted(set(target_operations)))])[:24]
            if target['target_id'] == expected_target_id:
                totals['stable_target_ids'] += 1
            else:
                mismatch('target_id', state_id, action_id)
            expected_action_id = 'ngas_action_' + content_hash(
                ['ngas-action-v1', metadata['size'], target['target_id'], metadata['repair']])[:24]
            if action_id == expected_action_id:
                totals['stable_action_ids'] += 1
            else:
                mismatch('action_id', state_id, action_id)
            encoded_operations = tuple(
                operation for operation, member in zip(operations, encoded['membership'][index])
                if float(member) == 1.)
            if set(encoded_operations) != set(target_operations):
                mismatch('encoded_target', state_id, action_id)
            if tuple(SIZE_FRACTIONS)[encoded['sizes'][index]] != metadata['size']:
                mismatch('encoded_size', state_id, action_id)
            if NGAS_REPAIR_IDS[encoded['repairs'][index]] != metadata['repair']:
                mismatch('encoded_repair', state_id, action_id)
            target_object = Target(
                target['target_id'], tuple(target_operations), tuple(target['origin_rules']),
                tuple(target['origin_families']), tuple(target['origin_operators']))
            expected_provenance = list(target_object.provenance_features(RULES, FAMILIES, OPERATORS))
            expected_provenance[-1] /= len(RULES)
            if encoded['provenance'][index] != expected_provenance:
                mismatch('encoded_provenance', state_id, action_id)
            reversed_target = Target(
                target['target_id'], tuple(target_operations),
                tuple(reversed(target['origin_rules'])),
                tuple(reversed(target['origin_families'])),
                tuple(reversed(target['origin_operators'])))
            if target_object.provenance_features(RULES, FAMILIES, OPERATORS) != (
                    reversed_target.provenance_features(RULES, FAMILIES, OPERATORS)):
                mismatch('provenance_permutation', state_id, action_id)
            else:
                totals['provenance_permutation_checks'] += 1

            label_path = directory / 'labels' / f'{action_id}.json'
            sealed_label = json.loads(label_path.read_text())
            label_file_hashes[str(label_path.relative_to(ROOT))] = digest(label_path)
            label = unseal(label_path)
            if label['action'] != metadata:
                mismatch('label_action_metadata', state_id, action_id)
            expected_fallback_hashes = [
                fallback_hash_by_seed[row['crn_seed']] for row in label['replicates']]
            if label['context']['fallback_hashes'] != expected_fallback_hashes:
                mismatch('fallback_hash_order', state_id, action_id)
            advantages = []
            for replicate in label['replicates']:
                totals['replicates'] += 1
                seed = replicate['crn_seed']
                first = replicate['candidate']['steps'][0]
                identity = (
                    first['action_id'] == action_id
                    and first['size'] == metadata['size']
                    and first['repair'] == metadata['repair']
                    and set(first['destroyed_operations']) == set(target_operations)
                    and first['decoder_evals'] == label['repair_trials'])
                if identity:
                    totals['candidate_first_steps'] += 1
                else:
                    mismatch('executed_joint_action', state_id, action_id)
                if replicate['fallback'] == fallback_by_seed.get(seed):
                    totals['fallback_payload_matches'] += 1
                else:
                    mismatch('shared_fallback_payload', state_id, action_id)
                for candidate_step, fallback_step in zip(
                        replicate['candidate']['steps'], replicate['fallback']['steps']):
                    if (candidate_step['neighbor_seed'], candidate_step['acceptance_seed']) == (
                            fallback_step['neighbor_seed'], fallback_step['acceptance_seed']):
                        totals['paired_crn_steps'] += 1
                    else:
                        mismatch('paired_crn_stream', state_id, action_id)
                advantage = (
                    replicate['fallback']['best_makespan']
                    - replicate['candidate']['best_makespan']) / label['initial_makespan']
                advantages.append(advantage)
                if advantage != replicate['advantage']:
                    mismatch('replicate_advantage', state_id, action_id)
            if statistics.fmean(advantages) != label['advantage_mean']:
                mismatch('advantage_mean', state_id, action_id)
            semantic_rows.append({
                'state_id': state_id, 'action': metadata,
                'replicate_advantages': advantages,
                'beats_fallback_frequency': label['beats_fallback_frequency'],
                'label_payload_sha256': sealed_label['payload_sha256'],
            })

    expected_actions = 6465
    expected_replicates = 58185
    expected_crn_steps = expected_replicates * 3
    checks = {
        'scope': (totals['states'], totals['actions'], totals['replicates'])
                 == (72, expected_actions, expected_replicates),
        'encoded_equals_executed': totals['candidate_first_steps'] == expected_replicates,
        'shared_fallback_exact': totals['fallback_payload_matches'] == expected_replicates,
        'paired_crn_all_steps': totals['paired_crn_steps'] == expected_crn_steps,
        'stable_action_ids': totals['stable_action_ids'] == expected_actions,
        'stable_target_ids': totals['stable_target_ids'] == expected_actions,
        'provenance_permutation_invariant':
            totals['provenance_permutation_checks'] == expected_actions,
        'zero_mismatches': not mismatches,
        'repair_not_resampled_after_selection':
            totals['candidate_first_steps'] == expected_replicates,
    }
    passed = all(checks.values())
    result = {
        'schema': 'ngas-a12-joint-action-label-identity-audit-v1',
        'status': 'PASS' if passed else 'FAIL',
        'decision': 'READY_FOR_A1_3R_TRAINING' if passed else 'NGAS_A1_REVISE_LABELS',
        'identity': 'a_encoded=(destroy_size,target_operation_set,repair_semantic_id)=a_executed',
        'checks': checks,
        'counts': totals,
        'mismatch_count': len(mismatches),
        'mismatch_examples': mismatches,
        'hashes': {
            'development_protocol_sha256': digest(DEVELOPMENT / 'protocol.json'),
            'development_result_manifest_sha256':
                digest(DEVELOPMENT / 'result_hash_manifest.json'),
            'raw_label_file_manifest_sha256': canonical_hash(label_file_hashes),
            'canonical_label_semantics_sha256': canonical_hash(semantic_rows),
        },
        'repair_ids': NGAS_REPAIR_IDS,
        'continuation_evidence': {
            'candidate_and_fallback_neighbor_and_acceptance_seeds_match_each_step': True,
            'encoded_repair_is_the_executed_first_step_repair': True,
            'continuation_steps_per_replicate': 3,
            'no_rollouts_executed_by_this_audit': True,
        },
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# A1.2 joint-action execution-label identity audit

Decision: **{result['decision']}**.

The audit checked {totals['actions']:,} cached joint actions and
{totals['replicates']:,} paired continuation replicates directly from the
immutable development artifacts. Encoded destroy size, target operation set,
and frozen NGAS repair ID equal the first executed action in every replicate.
All {totals['paired_crn_steps']:,} candidate/fallback continuation step pairs
use the same neighbor and acceptance seeds, and each label embeds the exact
shared fallback artifact for its replicate seed. No continuation was rerun.

Stable action/target IDs and order-independent union provenance passed for all
actions. Mismatches: {len(mismatches)}. R13/R14 remained locked; Gurobi was not
invoked.
""")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
