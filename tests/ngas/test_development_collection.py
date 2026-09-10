from copy import deepcopy

from rcias_ngas.csg.features import RULES, state_features, action_features, tensorize
from rcias_ngas.critic.development import sample_actions
from scripts.collect_ngas_development import data_gate
from tests.ngas.test_bank import small


def test_sampled_targets_have_full_bank_and_input_alignment(small):
    instance, current = small
    spec = {'state_id': 'development:test', 'source_seed': 746101, 'ordinal': 7}
    actions, banks = sample_actions(instance, current, spec)
    assert len(actions) <= 90
    assert sum(b.requested_count for b in banks) == 72
    assert len({a.action_id for a in actions}) == len(actions)
    assert {f for a in actions for f in a.target.origin_families} == {
        'ORIGINAL_OPERATOR', 'RELATED_VARIANT', 'MATCHED_RANDOM', 'LOCAL_PERTURBATION', 'STRUCTURED_NEAR_NEIGHBOR'}
    state = state_features(instance, current, spec['state_id'])
    batch = tensorize(state, action_features(state, actions))
    assert batch['membership'].shape[0] == len(actions)
    assert batch['membership'].sum().item() == sum(len(a.target.operations) for a in actions)


def test_data_gate_requires_informative_labels_in_every_fold():
    config = {'maximum_states': 72, 'data_gates': {
        'informative_state_pair_fraction_min': .1, 'each_fold_informative_state_fraction_min': .5,
        'positive_action_fraction_min': .05}}
    rows = [{'fold': i % 3, 'informative_pair_fraction': .2, 'positive_action_fraction': .3,
             'actions': 90, 'feasible': True, 'sizes': list('sml'), 'repairs': list('abcde'),
             'families': list('ABCDE'), 'rules': list(RULES)} for i in range(72)]
    assert data_gate(rows, config)['decision'] == 'READY_FOR_JOINT_CRITIC_TRAINING'
    bad = deepcopy(rows)
    for row in bad:
        if row['fold'] == 2:
            row['informative_pair_fraction'] = .01
    assert data_gate(bad, config)['decision'] == 'NGAS_A1_REVISE_DEVELOPMENT_LABELS'
