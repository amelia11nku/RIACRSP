from dataclasses import replace

from rcias_ngas.critic.dataset import (balanced_actions, fallback_action,
                                       label_action, separated_pair)
from rcias_ngas.rng import RNGStreams
from tests.ngas.test_bank import small


def test_pilot_balanced_coverage(small):
    instance, current = small
    actions, banks = balanced_actions(instance, current, 'state', RNGStreams(instance.instance_id, 1))
    assert len(actions) <= 75
    assert {a.size for a in actions} == {'small', 'medium', 'large'}
    assert len({a.repair for a in actions}) == 5
    assert len(set(f for a in actions for f in a.target.origin_families)) == 5
    assert all(b.requested_count == 24 for b in banks)


def test_label_executes_encoded_repair_and_matched_crn(small):
    instance, current = small
    actions, _ = balanced_actions(instance, current, 's', RNGStreams(instance.instance_id, 1))
    for repair in ('greedy', 'regret2', 'regret3', 'reconfiguration_aware', 'transport_aware'):
        action = replace(actions[0], repair=repair)
        result = label_action(instance, current, action, 's', (1, 2, 3), steps=1, trials=1)
        assert result['feasible']
        for row in result['replicates']:
            c, f = row['candidate']['steps'], row['fallback']['steps']
            assert c[0]['repair'] == repair
            assert c[0]['action_id'] == action.action_id
            assert [r['neighbor_seed'] for r in c] == [r['neighbor_seed'] for r in f]
            assert [r['acceptance_seed'] for r in c] == [r['acceptance_seed'] for r in f]


def test_identical_fallback_action_has_zero_advantage(small):
    instance, current = small
    base = RNGStreams(instance.instance_id, 7)
    rngs = RNGStreams(instance.instance_id, base.seed('continuation_crn', 's'))
    action = fallback_action(instance, current, rngs, 's:continuation:0')
    result = label_action(instance, current, action, 's', (7,))
    assert result['advantage_mean'] == 0
    assert result['replicates'][0]['candidate'] == result['replicates'][0]['fallback']


def test_noise_aware_pairing():
    def row(values):
        return {'replicates': [{'crn_seed': i, 'advantage': v} for i, v in enumerate(values)]}
    assert not separated_pair(row([.0001] * 3), row([0.] * 3))
    assert not separated_pair(row([.1, -.1, .1]), row([0.] * 3))
    assert separated_pair(row([.01] * 3), row([0.] * 3))
