from copy import deepcopy
import json

import pytest

from rcias_ngas.critic.dataset import balanced_actions, label_action
from rcias_ngas.critic.label_revision import (aggregate, evaluate_fallback, evaluate_replicate,
                                             replicate_seeds)
from rcias_ngas.rng import RNGStreams
from scripts.run_ngas_label_pilot_v2 import final_summary, load_record, save_record
from tests.ngas.test_bank import small
from tests.ngas.test_pilot_runner import synthetic_rows


def test_new_crn_repeat_major_independence():
    seeds = replicate_seeds('i', 's', (746101, 746102, 746103), 3)
    assert len(seeds) == len(set(seeds)) == 9
    assert not set(seeds) & {746101, 746102, 746103}
    assert seeds == replicate_seeds('i', 's', (746101, 746102, 746103), 3)
    assert seeds[:3] == replicate_seeds('i', 's', (746101, 746102, 746103), 1)
    assert seeds[0] != replicate_seeds('i', 'different', (746101,), 1)[0]


def test_cached_fallback_matches_uncached_semantics(small):
    instance, current = small
    actions, _ = balanced_actions(instance, current, 's', RNGStreams(instance.instance_id, 1))
    seeds = replicate_seeds(instance.instance_id, 's', (746101, 746102, 746103), 1)
    for action in actions[:2]:
        for trials in (2, 8):
            fallbacks = [evaluate_fallback(instance, current, 's', seed, 2, trials) for seed in seeds]
            reps = [evaluate_replicate(instance, current, action, 's', seed, 2, trials, f) for seed, f in zip(seeds, fallbacks)]
            old = label_action(instance, current, action, 's', seeds, 2, trials)
            new = aggregate(action.metadata(), current.makespan, reps, trials, 2)
            assert new['replicates'] == old['replicates']
            assert new['advantage_mean'] == old['advantage_mean']
            assert new['advantage_variance'] == old['advantage_variance']


def test_more_trials_share_first_two_neighbor_draws(small):
    instance, current = small
    actions, _ = balanced_actions(instance, current, 's', RNGStreams(instance.instance_id, 1))
    seed = replicate_seeds(instance.instance_id, 's', (746101,), 1)[0]
    rows = []
    for trials in (2, 8):
        f = evaluate_fallback(instance, current, 's', seed, 2, trials)
        rows.append(evaluate_replicate(instance, current, actions[0], 's', seed, 2, trials, f))
    assert rows[1]['candidate']['steps'][0]['proposal_makespan'] <= rows[0]['candidate']['steps'][0]['proposal_makespan']
    assert rows[1]['candidate']['steps'][0]['neighbor_seed'] == rows[0]['candidate']['steps'][0]['neighbor_seed']


def test_cache_rejects_wrong_crn_and_trials(small):
    instance, current = small
    actions, _ = balanced_actions(instance, current, 's', RNGStreams(instance.instance_id, 1))
    fallback = evaluate_fallback(instance, current, 's', 1, 2, 2)
    with pytest.raises(ValueError, match='accounting'):
        evaluate_replicate(instance, current, actions[0], 's', 1, 2, 8, fallback)
    with pytest.raises(ValueError, match='CRN'):
        evaluate_replicate(instance, current, actions[0], 's', 2, 2, 2, fallback)


def test_atomic_immutable_resume_detects_corruption(tmp_path):
    path = tmp_path / 'label.json'
    context = {'protocol': 'p', 'seed': 1}
    record = save_record(path, {'context': context, 'advantage': .1})
    assert load_record(path, context) == record
    with pytest.raises(FileExistsError):
        save_record(path, {'context': context, 'advantage': .2})
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError):
        load_record(path, {'protocol': 'different', 'seed': 1})
    changed = json.loads(path.read_text())
    changed['advantage'] = .9
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        load_record(path, context)


def test_failed_primary_cannot_be_rescued_by_control():
    config = {'primary_variant': 'T8_R9', 'gates': {
        'min_informative_pair_fraction': .1, 'min_informative_states': 3,
        'min_positive_mean_advantage_fraction': .05,
        'max_mean_seconds_per_joint_action_with_three_replicates': 30}}
    rows = synthetic_rows()
    for row in rows:
        row['action']['action_id'] = ':'.join([row['action']['size'], row['action']['target']['target_id'], row['action']['repair']])
        row['replicates'] = [{**row['replicates'][i % 3], 'crn_seed': i,
                              'immediate_normalized_improvement': 0.,
                              'candidate': {'best_makespan': 100., 'feasible': True},
                              'fallback': {'best_makespan': 100., 'feasible': True}} for i in range(9)]
        row['initial_makespan'] = 100.
        row['repair_trials'] = 2
        row['continuation_steps'] = 2
    primary = deepcopy(rows)
    for row in primary:
        row['advantage_mean'] = 0.
        row['repair_trials'] = 8
        for rep in row['replicates']:
            rep['advantage'] = 0.
    result = final_summary({'T2_R9': rows, 'T8_R9': primary}, config)
    assert result['variants']['T2_R9']['decision'] == 'NGAS_A1_LABEL_PILOT_PASS'
    assert result['decision'] == 'NGAS_A1_REVISE_LABELS'
