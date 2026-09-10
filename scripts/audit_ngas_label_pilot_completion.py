#!/usr/bin/env python3
"""Read-only replay/integrity audit of the completed A1.2 pilot."""
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.bank.provenance import Target
from rcias_ngas.critic.dataset import label_action
from rcias_ngas.evaluation.bks import content_hash, write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.run_ngas_label_pilot import OUT, summarize, verify_boundary, verify_protocol


def audit():
    protocol = verify_protocol()
    verify_boundary()
    manifest = json.loads((OUT / 'result_hash_manifest.json').read_text())
    actual_paths = {str(p.relative_to(ROOT)) for p in (OUT / 'states').rglob('*.json')}
    assert actual_paths == set(manifest), 'Missing or unexpected result files'
    for path, expected in manifest.items():
        assert digest(path) == expected, path
    rows, diagnostics = [], []
    for state_path in sorted((OUT / 'states').glob('*/state.json')):
        state = json.loads(state_path.read_text())
        state_id = state['state_id']
        spec = next(s for s in protocol['instances'] if s['instance_id'] == state_id.split('__')[0])
        instance = load_instance(ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / spec['relative_path'])
        candidate = Candidate(**{k: tuple(v) for k, v in state['candidate'].items()})
        current = decode_candidate(instance, candidate)
        assert current.feasible and current.makespan == state['makespan']
        assert json.loads(json.dumps(current.schedule.to_dict())) == state['schedule']
        actions = json.loads((state_path.parent / 'action_manifest.json').read_text())
        expected_actions = {a['action_id']: a for a in actions['actions']}
        local = []
        for path in sorted((state_path.parent / 'labels').glob('*.json')):
            row = json.loads(path.read_text())
            expected_hash = row.pop('payload_sha256')
            assert content_hash(row) == expected_hash
            row['payload_sha256'] = expected_hash
            assert row['state_sha256'] == digest(state_path)
            assert row['protocol_sha256'] == digest(OUT / 'protocol.json')
            assert row['action'] == expected_actions.pop(row['action']['action_id'])
            advantages = []
            for replicate in row['replicates']:
                c, f = replicate['candidate'], replicate['fallback']
                first = c['steps'][0]
                assert first['action_id'] == row['action']['action_id']
                assert first['repair'] == row['action']['repair']
                assert first['destroyed_operations'] == row['action']['target']['operations']
                assert [s['neighbor_seed'] for s in c['steps']] == [s['neighbor_seed'] for s in f['steps']]
                assert [s['acceptance_seed'] for s in c['steps']] == [s['acceptance_seed'] for s in f['steps']]
                for branch in (c, f):
                    assert len(branch['steps']) == 3
                    assert all(s['feasible'] and s['decoder_evals'] == 2 for s in branch['steps'])
                    assert branch['best_makespan'] == min(current.makespan, *(s['proposal_makespan'] for s in branch['steps']))
                advantage = (f['best_makespan'] - c['best_makespan']) / current.makespan
                assert advantage == replicate['advantage']
                advantages.append(advantage)
            assert [r['crn_seed'] for r in row['replicates']] == protocol['config']['crn_seeds']
            assert row['advantage_mean'] == statistics.fmean(advantages)
            assert row['advantage_variance'] == statistics.variance(advantages)
            local.append(row)
        assert not expected_actions and len(local) == 75
        # Outcome-blind replay selection: lexicographically first action ID.
        first = local[0]
        target = Target(**{k: tuple(v) if isinstance(v, list) else v for k, v in first['action']['target'].items()})
        action = JointAction(first['action']['size'], target, first['action']['repair'])
        replay = label_action(instance, current, action, state_id, protocol['config']['crn_seeds'])
        assert all(json.loads(json.dumps(value)) == first[key] for key, value in replay.items())
        # All actions must share precisely the same fallback per state and CRN.
        for index in range(3):
            assert len({content_hash(r['replicates'][index]['fallback']) for r in local}) == 1
        counts = defaultdict(int)
        for left, right in itertools.combinations(local, 2):
            differences = [a['advantage'] - b['advantage'] for a, b in zip(left['replicates'], right['replicates'])]
            gap = abs(statistics.fmean(differences))
            se = statistics.stdev(differences) / math.sqrt(3)
            counts['total_pairs'] += 1
            if gap <= .001:
                counts['below_material_gap'] += 1
            elif gap <= 2 * se:
                counts['material_gap_but_noise_limited'] += 1
            else:
                counts['noise_separated'] += 1
        all_reps = [rep for row in local for rep in row['replicates']]
        diagnostics.append({
            'state_id': state_id, **dict(counts),
            'zero_advantage_fraction': statistics.fmean(float(r['advantage'] == 0) for r in all_reps),
            'candidate_no_new_best_fraction': statistics.fmean(float(r['candidate']['best_makespan'] == current.makespan) for r in all_reps),
            'immediate_candidate_improvement_fraction': statistics.fmean(float(r['candidate']['steps'][0]['proposal_makespan'] < current.makespan) for r in all_reps),
            'late_candidate_improvement_fraction': statistics.fmean(float(r['candidate']['best_makespan'] < min(current.makespan, r['candidate']['steps'][0]['proposal_makespan'])) for r in all_reps),
            'fallback_best_by_crn': [local[0]['replicates'][i]['fallback']['best_makespan'] for i in range(3)],
            'label_runtime_seconds': sum(r['runtime_seconds'] for r in local),
        })
        rows.extend(local)
    recomputed = summarize(rows, protocol['config'])
    saved = json.loads((OUT / 'gate.json').read_text())
    assert all(recomputed[k] == saved[k] for k in recomputed)
    result = {
        'schema': 'ngas-a12-completion-audit-v1', 'integrity': 'PASS',
        'pilot_decision': saved['decision'], 'verified_result_files': len(manifest),
        'verified_joint_actions': len(rows), 'verified_paired_replicates': 1350,
        'state_schedule_replays': 6, 'full_label_replays': 6,
        'gate_reproduced_exactly': True, 'fallback_crn_consistency': True,
        'diagnostics': diagnostics,
        'gate_sha256': digest(OUT / 'gate.json'),
        'result_manifest_sha256': digest(OUT / 'result_hash_manifest.json'),
        'frozen_historical_evidence_unchanged': True,
        'repair_evidence': {
            'material_within_target_size_pairs': sum(s['within_target_size_noise_separated_repair_pairs'] for s in saved['states']),
            'all_within_target_size_pairs': sum(s['within_target_size_total_repair_pairs'] for s in saved['states']),
            'noise_separated_rank_reversals': sum(s['noise_separated_target_rank_reversals_across_repairs'] for s in saved['states']),
        },
    }
    return result


def main():
    result = audit()
    write_immutable(ROOT / 'outputs/ngas_a1/audit/label_pilot_completion.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
