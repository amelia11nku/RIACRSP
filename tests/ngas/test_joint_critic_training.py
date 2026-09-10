from copy import deepcopy
from dataclasses import replace

import torch

from rcias_ngas.csg.features import action_features, state_features
from rcias_ngas.critic.dataset import balanced_actions
from rcias_ngas.critic.train import fit, predict
from rcias_ngas.evaluation.ranking import material_pairs, state_metrics, summarize
from rcias_ngas.rng import RNGStreams
from tests.ngas.test_bank import small


def example_records(small):
    instance, current = small
    actions, _ = balanced_actions(instance, current, 'train_test', RNGStreams(instance.instance_id, 7))
    actions = [actions[0], replace(actions[0], repair='greedy'),
               replace(actions[0], repair='regret3'), actions[-1]]
    state = state_features(instance, current, 'train_test')
    features = action_features(state, actions)
    advantages = [[-.01] * 9, [0.] * 9, [.01] * 9, [.02] * 9]
    rows = []
    for index in range(4):
        rows.append({'state_id': f's{index}', 'instance_id': 'i0', 'fold': 0,
                     'scale': 'S', 'CF_level': 'CF1', 'source': 'TEST',
                     'state_features': state, 'action_features': features,
                     'actions': [action.metadata() for action in actions],
                     'replicate_advantages': advantages,
                     'beats_fallback_frequency': [0., 0., 1., 1.]})
    return rows


def test_fixed_epoch_training_is_deterministic(small):
    torch.set_num_threads(1)
    records = example_records(small)
    config = {'critic': {'hidden_dim': 16, 'message_passing_layers': 1},
              'training': {'epochs': 2, 'learning_rate': .0003,
                           'weight_decay': .0001, 'gradient_clip_norm': 1.}}
    first, first_history = fit(records, config, 13, 'cpu')
    second, second_history = fit(records, config, 13, 'cpu')
    assert first_history == second_history
    assert all(torch.equal(first.state_dict()[name], second.state_dict()[name])
               for name in first.state_dict())
    assert predict(first, records, 'cpu') == predict(second, records, 'cpu')


def test_noise_aware_metrics_and_gate_inputs(small):
    record = example_records(small)[0]
    assert len(material_pairs(record['replicate_advantages'])) == 6
    row = state_metrics(record, [-.01, 0., .01, .02], [0., .1, .9, 1.])
    assert row['spearman'] == 1.
    assert row['material_pair_accuracy'] == 1.
    assert row['top1_regret'] == 0.
    assert row['selected_advantage'] == .02
    summary = summarize([deepcopy(row), deepcopy(row)])
    assert summary['mean_state_spearman'] == 1.
    assert summary['material_pair_accuracy'] == 1.
