from rcias_ngas.evaluation.a17ar import (
    clone_portfolio, outcome_class, ranking_metrics, replay_portfolio,
)


def test_ranking_metrics_use_tie_aware_realized_rank_and_full_bank_hits():
    result = ranking_metrics(
        [3., 2., 1.], [0., 5., 5.], ['a', 'b', 'c'])
    assert result['predicted_top1_action_id'] == 'a'
    assert result['top1_realized_rank'] == 3
    assert result['top1_normalized_rank'] == 1.
    assert result['best_action_hit_at_1'] is False
    assert result['best_action_hit_at_5'] is True
    assert result['best_set_recall_at_5'] == 1.
    assert result['top_k_utility_gap']['1'] == 5.


def test_portfolio_replay_and_clone_preserve_pre_state_factors_independently():
    rows = [{
        'iteration': index, 'action_size': 'small', 'repair': 'greedy',
        'origin_families': ['ORIGINAL_OPERATOR'],
        'outcome_class': 'new_global_best', 'relative_current_improvement': .01,
    } for index in range(1, 9)]
    original = replay_portfolio(
        rows, 8, segment_length=8, reaction=.25, strength=.7)
    cloned = clone_portfolio(original)
    key = 'repair:greedy'
    assert cloned.estimates[key] == original.estimates[key]
    cloned.estimates[key] = 99.
    assert cloned.estimates[key] != original.estimates[key]


def test_outcome_class_matches_frozen_solver_ordering():
    assert outcome_class(90., 100., 95., True)[0] == 'new_global_best'
    assert outcome_class(97., 100., 95., True)[0] == 'accepted_current_improvement'
    assert outcome_class(105., 100., 95., True)[0] == 'accepted_worse_or_neutral'
    assert outcome_class(105., 100., 95., False)[0] == 'rejected'
