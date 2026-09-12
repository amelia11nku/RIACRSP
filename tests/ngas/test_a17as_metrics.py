from rcias_ngas.evaluation.a17as import (
    UTILITY_PAIRS, deterministic_order, nonconstant, positive_best,
    state_alignment, state_support,
)


def action(action_id, u0, u1, u2, u3):
    return {
        'action_id': action_id,
        'critic_advantage': 0.,
        'utility': {
            'U0_immediate_best_gain': u0,
            'U1_short_horizon_best_gain': u1,
            'U2_horizon_gain_per_second': u2,
            'U3_mean_signed_improvement': u3,
        },
    }


def test_support_classification_separates_nonconstant_and_positive_best():
    actions = [action('b', 0., -2., 0., -1.),
               action('a', 0., -1., 0., 1.)]
    support = state_support(actions)
    assert support['U0_IMMEDIATE'] == {
        'nonconstant': False, 'positive_best': False}
    assert support['U1_SHORT_HORIZON'] == {
        'nonconstant': True, 'positive_best': False}
    assert support['U3_STOCHASTIC_ROBUSTNESS'] == {
        'nonconstant': True, 'positive_best': True}
    assert nonconstant([0., 1e-12]) is False
    assert positive_best([0., 1e-12]) is False


def test_alignment_uses_action_id_to_break_utility_ties():
    actions = [action('b', 0., 1., 2., 0.),
               action('a', 0., 1., 3., 1.),
               action('c', 0., 0., 1., 2.)]
    assert deterministic_order(actions, 'U1_SHORT_HORIZON')[0] == 1
    rows = state_alignment(actions)
    u1_u2 = next(row for row in rows
                 if (row['left'], row['right'])
                 == ('U1_SHORT_HORIZON', 'U2_COST_NORMALIZED'))
    assert u1_u2['same_top1_all_states'] is True
    assert u1_u2['tie_break'] == 'utility_desc_then_action_id_asc'


def test_alignment_emits_exactly_six_pairs_with_explicit_denominators():
    actions = [action('a', 1., 2., 3., 4.),
               action('b', 0., 1., 2., 3.),
               action('c', -1., 0., 1., 2.)]
    rows = state_alignment(actions)
    assert {(row['left'], row['right']) for row in rows} == set(UTILITY_PAIRS)
    assert all(row['top5_denominator'] == 3 for row in rows)
    assert all(row['top10_denominator'] == 3 for row in rows)
