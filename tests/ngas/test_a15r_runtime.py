from types import SimpleNamespace

import torch

from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_clgri.search.dabc_chdg import build_generalized_chdg
from rcias_ngas.actions.repair import execute_action
from rcias_ngas.csg.critical_mapping import map_critical_events
from rcias_ngas.csg.critical_sync import analyze_graph, critical_sync
from rcias_ngas.csg.revised_features import action_features, state_features_from_components
from rcias_ngas.critic.revised_critic import RevisedJointCritic
from rcias_ngas.latency.live_refresh import LiveRefreshEngine, _joint_bank, _tensorize_cpu
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import CompactStateBuilder, ProductionRefreshRuntime
from rcias_ngas.search.ngas_solver import (
    NGASSearchConfig, _critical_identity, solve_ngas,
)


class TinyCritic:
    variant = 'C1'
    sha256 = 'test-c1'
    device = torch.device('cpu')

    def __init__(self):
        torch.manual_seed(103)
        self.model = RevisedJointCritic(
            'rt_hgt', hidden=16, layers=1, heads=4).eval()


def decoded_tiny():
    instance = load_instance('instances/tiny/tiny_01.json')
    h1 = solve_dispatching(instance, 'H1')
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    return instance, current


def reference_batch(instance, current, state_id, streams):
    graph = build_csg_from_schedule(
        instance, current.schedule, state_id=state_id,
        search_progress=0., search_stage='0-20%', attach_hash=False)
    event_graph = build_generalized_chdg(instance, current)
    analysis = analyze_graph(event_graph)
    mapping = map_critical_events(event_graph, graph, analysis)
    state = state_features_from_components(
        instance, graph, mapping, include_hash=False, include_diagnostics=False)
    actions = _joint_bank(instance, current, state_id, streams, analysis)
    return actions, _tensorize_cpu(state, action_features(state, actions)), analysis


def assert_compact_equals_reference(builder, instance, current, state_id, streams):
    actions, reference, analysis = reference_batch(
        instance, current, state_id, streams)
    compact = builder.build(current, state_id, streams)
    assert [action.action_id for action in compact.actions] == [
        action.action_id for action in actions]
    assert (compact.critical_signature, compact.dominant_bottleneck) == \
        _critical_identity(analysis)
    assert reference.keys() == compact.cpu_batch.keys()
    for name, expected in reference.items():
        actual = compact.cpu_batch[name]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        if expected.dtype.is_floating_point:
            assert torch.max(torch.abs(actual - expected)).item() <= 1e-7
        else:
            assert torch.equal(actual, expected)
    return compact


def test_compact_runtime_is_reference_equivalent_and_workspace_safe_for_a_b_a():
    instance, state_a = decoded_tiny()
    streams = RNGStreams(instance.instance_id, 746101)
    builder = CompactStateBuilder(instance)
    first_a = assert_compact_equals_reference(
        builder, instance, state_a, 'state-a', streams)
    state_b, _ = execute_action(
        instance, state_a, first_a.actions[-1], streams, 'transition-a-b', trials=1)
    assert state_b.candidate != state_a.candidate
    assert_compact_equals_reference(builder, instance, state_b, 'state-b', streams)
    repeated_a = assert_compact_equals_reference(
        builder, instance, state_a, 'state-a', streams)
    assert [action.action_id for action in repeated_a.actions] == [
        action.action_id for action in first_a.actions]
    assert builder.workspace.resize_events == 0


def test_production_refresh_matches_oracle_outputs_prior_ranking_and_selection():
    instance, current = decoded_tiny()
    critic = TinyCritic()
    streams = RNGStreams(instance.instance_id, 746101)
    state_id = 'production-equivalence'
    reference = LiveRefreshEngine([critic]).refresh(
        instance, current, state_id, streams, sample_seed=73)
    production = ProductionRefreshRuntime(critic).refresh(
        instance, current, state_id, streams, sample_seed=73)
    assert [action.action_id for action in production.actions] == [
        action.action_id for action in reference.actions]
    for actual, expected in (
        (production.advantage, reference.advantage),
        (production.beats_fallback_probability, reference.beats_fallback_probability),
        (production.prior, reference.prior),
    ):
        assert max(abs(left - right) for left, right in zip(actual, expected)) <= 1e-7
    assert production.ranking == reference.ranking
    assert production.sampled_index == reference.sampled_index


class ReferenceRuntime:
    """A1.5 oracle adapter used only for deterministic search comparison."""

    def __init__(self, critic):
        self.engine = LiveRefreshEngine([critic])

    def prepare_instance(self, instance):
        return False

    def refresh(self, instance, current, state_id, streams, sample_seed=0):
        result = self.engine.refresh(
            instance, current, state_id, streams, sample_seed=sample_seed)
        signature, bottleneck = _critical_identity(critical_sync(instance, current))
        return SimpleNamespace(
            actions=result.actions, advantage=result.advantage,
            beats_fallback_probability=result.beats_fallback_probability,
            prior=result.prior, state_feature_hash='', graph_hash='',
            critical_signature=signature, dominant_bottleneck=bottleneck,
            bank_summary={'joint_actions': len(result.actions)},
            components_ms=result.components_ms,
        )


def test_fixed_iteration_search_decisions_match_reference_runtime():
    instance, _ = decoded_tiny()
    critic = TinyCritic()
    config = NGASSearchConfig(candidate_trials=1, iteration_limit=5)
    optimized = solve_ngas(
        instance, 60., 746103, 'PERSISTENT_FIXED_REFRESH', critic, config,
        refresh_runtime=ProductionRefreshRuntime(critic))
    reference = solve_ngas(
        instance, 60., 746103, 'PERSISTENT_FIXED_REFRESH', critic, config,
        refresh_runtime=ReferenceRuntime(critic))
    fields = (
        'action_id', 'candidate_makespan', 'current_after', 'best_after',
        'accepted', 'outcome_class',
    )
    assert [tuple(row[field] for field in fields)
            for row in optimized.diagnostics['iterations']] == [
        tuple(row[field] for field in fields)
        for row in reference.diagnostics['iterations']]
    assert optimized.best.candidate == reference.best.candidate
    assert optimized.best.makespan == reference.best.makespan
    assert optimized.diagnostics['final_replay']['feasible']
