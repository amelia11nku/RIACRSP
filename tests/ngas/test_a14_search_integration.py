import pytest

from rcias_clgri.data.loader import load_instance
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.bank.provenance import Target
from rcias_ngas.rng import RNGStreams
from rcias_ngas.search.a14_telemetry import A14Telemetry
from rcias_ngas.search.ngas_solver import MODE_SETTINGS, NGASSearchConfig, solve_ngas
from rcias_ngas.search.online_portfolio import OnlinePortfolio
from rcias_ngas.search.persistent_prior import (
    PriorCache, RefreshConfig, normalized_prior, refresh_reasons,
)
from rcias_ngas.search.telemetry import RunState


class FakeCritic:
    variant = 'C1'
    sha256 = 'fake-c1'

    def score(self, instance, current, state_id, actions):
        return ({
            'advantage': [index / 1000 for index in range(len(actions))],
            'beats_fallback_probability': [.5] * len(actions),
            'state_feature_hash': 'state', 'graph_hash': 'graph',
        }, {
            'state_feature_seconds': 0., 'action_feature_seconds': 0.,
            'tensor_transfer_seconds': 0., 'model_forward_seconds': 0.,
        })


def _action(size='small', repair='greedy'):
    target = Target('target', ('O1',), ('rule',), ('family',), ('operator',))
    return JointAction(size, target, repair)


def _cache(iteration=0):
    action = _action()
    return PriorCache('state', iteration, (action,), (1.,), (0.,), (.5,),
                      ('initial_state',), 'features', 'graph', 'critical', 'PRECEDENCE')


def test_prior_is_normalized_with_an_explicit_uniform_floor():
    probabilities = normalized_prior([-1., 0., 1.], scale=.1, uniform_mix=.06)
    assert sum(probabilities) == pytest.approx(1.)
    assert min(probabilities) >= .06 / 3
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_event_refresh_respects_spacing_and_reports_all_live_causes():
    config = RefreshConfig(minimum_spacing=5, maximum_age=10)
    common = dict(
        mode='event', cache=_cache(), new_global_best=True,
        meaningful_current_improvement=True, critical_structure_changed=True,
        bottleneck_changed=True, stagnation_age=20, high_prior_failures=4,
        stage_changed=True, config=config,
    )
    assert refresh_reasons(iteration=4, **common) == ()
    reasons = refresh_reasons(iteration=10, **common)
    assert set(reasons) == {
        'new_global_best', 'meaningful_current_improvement',
        'critical_structure_change', 'bottleneck_dominance_change',
        'stagnation', 'repeated_high_prior_failure', 'maximum_age',
        'search_stage_change',
    }


def test_online_portfolio_updates_only_at_segment_boundaries():
    portfolio = OnlinePortfolio(segment_length=2, reaction=.5)
    action = _action()
    before = portfolio.factor(action)
    portfolio.observe(action, 'rejected', 0.)
    assert portfolio.factor(action) == before
    portfolio.observe(action, 'new_global_best', .02)
    assert portfolio.segment_updates == 1
    assert portfolio.snapshot()['outcome_counts'] == {'new_global_best': 1, 'rejected': 1}


def test_final_checkpoint_is_the_true_termination_state():
    telemetry = A14Telemetry(10.)
    initial = RunState(1, 0, 20., 20.)
    final = RunState(3, 1, 18., 18., 1, 1, 1, 1)
    telemetry.observe(0., initial)
    telemetry.observe(6., final)
    payload = telemetry.finish(6.2)
    assert payload['budget_checkpoints'][0]['budget_fraction'] == .1
    assert payload['budget_checkpoints'][0]['decoder_evals'] == 1
    assert payload['budget_checkpoints'][-1]['budget_fraction'] == 1.
    assert payload['budget_checkpoints'][-1]['elapsed_time_sec'] == 6.2
    assert payload['budget_checkpoints'][-1]['decoder_evals'] == 3


def test_six_ablation_modes_share_decoder_and_preserve_role_boundaries():
    instance = load_instance('instances/tiny/tiny_01.json')
    config = NGASSearchConfig(candidate_trials=1, iteration_limit=2)
    results = {}
    for mode, settings in MODE_SETTINGS.items():
        result = solve_ngas(
            instance, 20., 746101, mode,
            critic=FakeCritic() if settings['neural'] else None,
            config=config,
        )
        results[mode] = result
        assert result.iterations == 2
        assert result.decoder_evaluations == 3
        assert result.diagnostics['final_replay']['feasible']
        assert result.diagnostics['telemetry']['budget_checkpoints'][-1]['budget_fraction'] == 1.
    assert results['ONLINE_PORTFOLIO_ONLY'].diagnostics['critic_variant'] is None
    assert results['ONLINE_PORTFOLIO_ONLY'].diagnostics['telemetry']['termination']['neural_calls'] == 0
    assert results['ONE_SHOT_TOP1'].diagnostics['guided_iterations'] == 1
    assert results['NEURAL_PRIOR_ONLY'].diagnostics['guided_iterations'] == 2


def test_new_rng_namespaces_are_independent_and_reproducible():
    streams = RNGStreams('instance', 7)
    first = streams.seed('neural_prior', 'state', 2)
    assert first == streams.seed('neural_prior', 'state', 2)
    assert first != streams.seed('online_exploration', 'state', 2)
    assert first != streams.seed('neighbor', 'state', 2)
