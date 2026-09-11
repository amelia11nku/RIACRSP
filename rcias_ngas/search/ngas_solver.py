"""A1.4 persistent-neural NGAS solver built on the frozen project decoder."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import time

from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import (
    SearchResult, TracePoint, candidate_from_actions, decode_candidate,
)
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS, construct_neighbor
from rcias_ngas.bank.ngas_bank_v1 import build_bank
from rcias_ngas.csg.critical_sync import critical_sync
from rcias_ngas.evaluation.bks import content_hash
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import ProductionRefreshRuntime
from rcias_ngas.search.a14_telemetry import A14Telemetry
from rcias_ngas.search.online_portfolio import OnlinePortfolio
from rcias_ngas.search.persistent_prior import (
    PriorCache, RefreshConfig, distribution_summary, normalized_prior,
    refresh_reasons,
)
from rcias_ngas.search.telemetry import RunState


MODE_SETTINGS = {
    'NEURAL_PRIOR_ONLY': {'neural': True, 'online': False, 'refresh': 'static'},
    'ONLINE_PORTFOLIO_ONLY': {'neural': False, 'online': True, 'refresh': 'fixed'},
    'NEURAL_X_PORTFOLIO': {'neural': True, 'online': True, 'refresh': 'static'},
    'ONE_SHOT_TOP1': {'neural': True, 'online': False, 'refresh': 'static'},
    'PERSISTENT_FIXED_REFRESH': {'neural': True, 'online': True, 'refresh': 'fixed'},
    'PERSISTENT_EVENT_REFRESH': {'neural': True, 'online': True, 'refresh': 'event'},
}


@dataclass(frozen=True)
class NGASSearchConfig:
    candidate_trials: int = 8
    iteration_limit: int | None = None
    initial_temperature_fraction: float = .05
    cooling_rate: float = .995
    prior_advantage_scale: float = .01
    prior_uniform_mix: float = .05
    online_exploration: float = .05
    portfolio_segment_length: int = 8
    portfolio_reaction: float = .25
    portfolio_strength: float = .7
    refresh: RefreshConfig = field(default_factory=RefreshConfig)


def search_config_from_dict(value: dict) -> NGASSearchConfig:
    fields = dict(value)
    fields['refresh'] = RefreshConfig(**fields['refresh'])
    return NGASSearchConfig(**fields)


def _critical_identity(analysis) -> tuple[str, str | None]:
    category_counts = Counter(
        edge['category'] for edge in analysis.edges if edge['critical'])
    dominant = min(category_counts, key=lambda name: (-category_counts[name], name)) \
        if category_counts else None
    payload = {
        'critical_operations': list(analysis.ranked_operations),
        'critical_edge_categories': dict(sorted(category_counts.items())),
    }
    return content_hash(payload), dominant


def _joint_bank(instance, current, state_id: str, streams: RNGStreams,
                analysis) -> tuple[tuple[JointAction, ...], dict]:
    banks = [build_bank(instance, current, state_id, size, streams, analysis=analysis)
             for size in SIZE_FRACTIONS]
    actions = tuple(JointAction(bank.size, target, repair)
                    for bank in banks for target in bank.targets
                    for repair in NGAS_REPAIR_IDS)
    return actions, {
        'requested_targets': sum(bank.requested_count for bank in banks),
        'unique_targets': sum(len(bank.targets) for bank in banks),
        'duplicate_targets': sum(bank.duplicate_count for bank in banks),
        'joint_actions': len(actions),
        'targets_by_size': {bank.size: len(bank.targets) for bank in banks},
    }


def _stage(elapsed: float, budget: float) -> str:
    fraction = elapsed / budget
    for ceiling, name in ((.2, '0-20%'), (.5, '20-50%'), (.8, '50-80%')):
        if fraction < ceiling:
            return name
    return '80-100%'


def _semantic_action(action: JointAction) -> tuple:
    return action.size, tuple(action.target.operations), action.repair


def _combine(probabilities: tuple[float, ...], actions: tuple,
             portfolio: OnlinePortfolio | None) -> tuple[float, ...]:
    weights = [probability * (portfolio.factor(action) if portfolio else 1.)
               for action, probability in zip(actions, probabilities)]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise FloatingPointError('Invalid combined action distribution')
    return tuple(value / total for value in weights)


def _select(actions: tuple, probabilities: tuple[float, ...], streams: RNGStreams,
            state_id: str, iteration: int, neural_live: bool,
            exploration: float, force_top1: bool) -> tuple[int, bool, float]:
    if force_top1:
        index = min(range(len(actions)), key=lambda i: (-probabilities[i], actions[i].action_id))
        return index, False, 1.
    online_rng = streams.stream('online_exploration', state_id, iteration)
    explored = online_rng.random() < exploration
    if explored:
        index = online_rng.randrange(len(actions))
    else:
        namespace = 'neural_prior' if neural_live else 'online_exploration'
        index = streams.stream(namespace, state_id, iteration).choices(
            range(len(actions)), weights=probabilities, k=1)[0]
    actual_probability = (exploration / len(actions)
                          + (1. - exploration) * probabilities[index])
    return index, explored, actual_probability


def solve_ngas(instance, time_limit: float, seed: int, mode: str, critic=None,
               config: NGASSearchConfig = NGASSearchConfig(),
               refresh_runtime: ProductionRefreshRuntime | None = None) -> SearchResult:
    """Run one causally instrumented A1.4 ablation under a wall-clock budget."""
    if mode not in MODE_SETTINGS or time_limit <= 0 or config.candidate_trials < 1:
        raise ValueError('Invalid NGAS A1.4 run configuration')
    settings = MODE_SETTINGS[mode]
    if settings['neural'] and critic is None:
        raise ValueError(f'{mode} requires a frozen critic')
    if settings['neural']:
        refresh_runtime = refresh_runtime or ProductionRefreshRuntime(
            critic,
            prior_advantage_scale=config.prior_advantage_scale,
            prior_uniform_mix=config.prior_uniform_mix,
        )
        preparation_started = time.perf_counter()
        refresh_runtime.prepare_instance(instance)
        preparation_seconds = time.perf_counter() - preparation_started
    else:
        preparation_seconds = 0.
    streams = RNGStreams(instance.instance_id, seed)
    portfolio = OnlinePortfolio(
        segment_length=config.portfolio_segment_length,
        reaction=config.portfolio_reaction,
        strength=config.portfolio_strength,
    ) if settings['online'] else None
    runtime = defaultdict(float)
    runtime['runtime_preparation_seconds'] = preparation_seconds
    started = time.perf_counter()
    h1_started = time.perf_counter()
    h1 = solve_dispatching(instance, 'H1')
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    runtime['initialization_seconds'] += time.perf_counter() - h1_started
    best = current
    best_time = time.perf_counter() - started
    evaluations = 1
    iterations = accepted_moves = improving_moves = new_best_moves = neural_calls = 0
    telemetry = A14Telemetry(time_limit)
    state = RunState(evaluations, neural_calls, current.makespan, best.makespan)
    telemetry.observe(best_time, state)
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    cache = None
    refresh_log, iteration_log = [], []
    selection_counts, outcome_counts = Counter(), Counter()
    selected_semantics = set()
    guided_iterations = high_prior_failures = 0
    last_best_iteration = last_refresh_anchor = 0
    pending_new_best = pending_meaningful = False
    pending_structure = pending_bottleneck = False
    pending_stage_change = False
    previous_stage = _stage(best_time, time_limit)

    while (time.perf_counter() - started < time_limit
           and (config.iteration_limit is None or iterations < config.iteration_limit)):
        iteration_started = time.perf_counter()
        elapsed = iteration_started - started
        stage = _stage(elapsed, time_limit)
        reasons = refresh_reasons(
            mode=settings['refresh'], iteration=iterations, cache=cache,
            new_global_best=pending_new_best,
            meaningful_current_improvement=pending_meaningful,
            critical_structure_changed=pending_structure,
            bottleneck_changed=pending_bottleneck,
            stagnation_age=iterations - max(last_best_iteration, last_refresh_anchor),
            high_prior_failures=high_prior_failures,
            stage_changed=pending_stage_change or stage != previous_stage,
            config=config.refresh,
        )
        if reasons:
            refresh_started = time.perf_counter()
            state_id = f'{instance.instance_id}:seed{seed}:iteration{iterations}'
            if settings['neural']:
                refresh = refresh_runtime.refresh(
                    instance, current, state_id, streams,
                    sample_seed=streams.seed('neural_prior', state_id, iterations))
                actions = refresh.actions
                bank_summary = refresh.bank_summary
                critical_signature = refresh.critical_signature
                bottleneck = refresh.dominant_bottleneck
                predictions = {
                    'advantage': refresh.advantage,
                    'beats_fallback_probability': refresh.beats_fallback_probability,
                    'state_feature_hash': refresh.state_feature_hash,
                    'graph_hash': refresh.graph_hash,
                }
                probabilities = refresh.prior
                for name, milliseconds in refresh.components_ms.items():
                    runtime['production_refresh_' + name + '_seconds'] += milliseconds / 1000.
                neural_calls += 1
            else:
                analysis_started = time.perf_counter()
                analysis = critical_sync(instance, current)
                runtime['critical_analysis_seconds'] += time.perf_counter() - analysis_started
                critical_signature, bottleneck = _critical_identity(analysis)
                bank_started = time.perf_counter()
                actions, bank_summary = _joint_bank(
                    instance, current, state_id, streams, analysis)
                runtime['bank_construction_seconds'] += time.perf_counter() - bank_started
                predictions = {
                    'advantage': [0.] * len(actions),
                    'beats_fallback_probability': [.5] * len(actions),
                    'state_feature_hash': '', 'graph_hash': '',
                }
                probabilities = tuple([1. / len(actions)] * len(actions))
            cache = PriorCache(
                state_id, iterations, actions, probabilities,
                tuple(predictions['advantage']),
                tuple(predictions['beats_fallback_probability']), reasons,
                predictions['state_feature_hash'], predictions['graph_hash'],
                critical_signature, bottleneck,
            )
            refresh_elapsed = time.perf_counter() - refresh_started
            runtime['refresh_total_seconds'] += refresh_elapsed
            refresh_log.append({
                'iteration': iterations, 'elapsed_time_sec': time.perf_counter() - started,
                'reasons': list(reasons), 'critic_called': settings['neural'],
                'critic_variant': critic.variant if settings['neural'] else None,
                'bank': bank_summary, 'critical_signature': critical_signature,
                'dominant_bottleneck': bottleneck,
                'distribution': distribution_summary(actions, probabilities),
                'refresh_seconds': refresh_elapsed,
            })
            last_refresh_anchor = iterations
            pending_new_best = pending_meaningful = False
            pending_structure = pending_bottleneck = False
            pending_stage_change = False
            high_prior_failures = 0
            state = RunState(evaluations, neural_calls, current.makespan, best.makespan,
                             iterations, accepted_moves, improving_moves, new_best_moves)
            telemetry.observe(time.perf_counter() - started, state)
            if time.perf_counter() - started >= time_limit:
                break

        assert cache is not None
        neural_live = settings['neural'] and not (
            mode == 'ONE_SHOT_TOP1' and iterations > 0)
        if neural_live:
            base = cache.probabilities
        else:
            base = tuple([1. / len(cache.actions)] * len(cache.actions))
        combined = _combine(base, cache.actions, portfolio)
        sampling_started = time.perf_counter()
        index, explored, actual_probability = _select(
            cache.actions, combined, streams, cache.state_id, iterations,
            neural_live, config.online_exploration,
            mode == 'ONE_SHOT_TOP1' and iterations == 0)
        runtime['action_sampling_seconds'] += time.perf_counter() - sampling_started
        action = cache.actions[index]
        if time.perf_counter() - started >= time_limit:
            break
        selected_semantics.add(_semantic_action(action))
        selection_counts[(action.size, action.repair)] += 1

        current_before, best_before = current, best
        candidates = []
        state_id = f'{instance.instance_id}:seed{seed}:iteration{iterations}'
        for trial in range(config.candidate_trials):
            if candidates and time.perf_counter() - started >= time_limit:
                break
            repair_started = time.perf_counter()
            neighbor = construct_neighbor(
                instance, current, action,
                streams.stream('neighbor', state_id, trial))
            runtime['repair_construction_seconds'] += time.perf_counter() - repair_started
            decoder_started = time.perf_counter()
            candidate = decode_candidate(instance, neighbor)
            runtime['decoder_seconds'] += time.perf_counter() - decoder_started
            if not candidate.feasible:
                raise RuntimeError('Frozen repair produced infeasible candidate')
            candidates.append(candidate)
            evaluations += 1
        candidate = min(candidates, key=lambda item: item.makespan)
        delta = candidate.makespan - current.makespan
        temperature = (config.initial_temperature_fraction * max(1., current.makespan)
                       * config.cooling_rate ** iterations)
        acceptance_started = time.perf_counter()
        accepted = delta <= 0 or streams.stream('acceptance', state_id).random() < \
            math.exp(-delta / max(temperature, 1e-12))
        runtime['acceptance_seconds'] += time.perf_counter() - acceptance_started
        if accepted:
            current = candidate
            accepted_moves += 1
        current_improvement = max(0., current_before.makespan - candidate.makespan)
        relative_improvement = current_improvement / max(1., current_before.makespan)
        new_best = candidate.makespan < best.makespan
        if new_best:
            best = candidate
            best_time = time.perf_counter() - started
            new_best_moves += 1
            last_best_iteration = iterations + 1
            trace.append(TracePoint(best_time, evaluations, best.makespan))
        if accepted and current_improvement > 0:
            improving_moves += 1
        if new_best:
            outcome = 'new_global_best'
        elif accepted and current_improvement > 0:
            outcome = 'accepted_current_improvement'
        elif accepted:
            outcome = 'accepted_worse_or_neutral'
        else:
            outcome = 'rejected'
        outcome_counts[outcome] += 1
        reward = portfolio.observe(action, outcome, relative_improvement) if portfolio else None
        if neural_live:
            guided_iterations += 1

        neural_order = sorted(range(len(cache.actions)),
                              key=lambda i: (-cache.probabilities[i], cache.actions[i].action_id))
        neural_rank = neural_order.index(index) + 1
        high_prior = neural_live and neural_rank <= max(1, math.ceil(.10 * len(cache.actions)))
        if high_prior and outcome in ('rejected', 'accepted_worse_or_neutral'):
            high_prior_failures += 1
        elif high_prior:
            high_prior_failures = 0
        pending_new_best = pending_new_best or new_best
        meaningful = accepted and relative_improvement >= config.refresh.meaningful_improvement_fraction
        pending_meaningful = pending_meaningful or meaningful
        pending_stage_change = pending_stage_change or stage != previous_stage

        monitor_seconds = 0.
        if (settings['refresh'] == 'event' and accepted
                and iterations + 1 - cache.refresh_iteration >= config.refresh.minimum_spacing):
            monitor_started = time.perf_counter()
            if settings['neural']:
                live_signature, live_bottleneck = refresh_runtime.structural_identity(
                    instance, current)
            else:
                live_analysis = critical_sync(instance, current)
                live_signature, live_bottleneck = _critical_identity(live_analysis)
            monitor_seconds = time.perf_counter() - monitor_started
            runtime['structural_monitor_seconds'] += monitor_seconds
            pending_structure = pending_structure or live_signature != cache.critical_signature
            pending_bottleneck = pending_bottleneck or live_bottleneck != cache.dominant_bottleneck

        iterations += 1
        elapsed = time.perf_counter() - started
        state = RunState(evaluations, neural_calls, current.makespan, best.makespan,
                         iterations, accepted_moves, improving_moves, new_best_moves)
        telemetry.observe(elapsed, state)
        iteration_log.append({
            'iteration': iterations, 'elapsed_time_sec': elapsed,
            'action_id': action.action_id, 'action_size': action.size,
            'target_id': action.target.target_id, 'target_operations': list(action.target.operations),
            'origin_families': list(action.target.origin_families), 'repair': action.repair,
            'prior_age_iterations': iterations - 1 - cache.refresh_iteration,
            'neural_live': neural_live, 'neural_rank': neural_rank if neural_live else None,
            'neural_probability': cache.probabilities[index] if neural_live else None,
            'combined_probability': combined[index],
            'actual_selection_probability': actual_probability, 'explored': explored,
            'candidate_trials': len(candidates), 'candidate_makespan': candidate.makespan,
            'current_before': current_before.makespan, 'current_after': current.makespan,
            'best_before': best_before.makespan, 'best_after': best.makespan,
            'accepted': accepted, 'outcome_class': outcome,
            'relative_current_improvement': relative_improvement,
            'portfolio_reward': reward, 'temperature': temperature,
            'structural_monitor_seconds': monitor_seconds,
        })
        previous_stage = stage

    if portfolio:
        portfolio.flush()
    elapsed = time.perf_counter() - started
    telemetry_payload = telemetry.finish(elapsed)
    replay = decode_candidate(instance, best.candidate)
    replay_audit = check_schedule(instance, replay.schedule)
    if not replay.feasible or not replay_audit['feasible'] or replay.makespan != best.makespan:
        raise RuntimeError('Final NGAS schedule replay failed')
    effective_neural = runtime['production_refresh_complete_refresh_seconds']
    diagnostics = {
        'schema': 'ngas-a14-run-diagnostics-v1',
        'mode': mode, 'seed': seed, 'time_limit_seconds': time_limit,
        'critic_variant': critic.variant if settings['neural'] else None,
        'critic_checkpoint_sha256': critic.sha256 if settings['neural'] else None,
        'refreshes': refresh_log, 'iterations': iteration_log,
        'runtime_components': dict(sorted(runtime.items())),
        'effective_neural_overhead_seconds': effective_neural,
        'guided_iterations': guided_iterations,
        'guided_iterations_per_critic_call': guided_iterations / neural_calls if neural_calls else None,
        'critic_influence_fraction': guided_iterations / iterations if iterations else 0.,
        'selection_counts_size_repair': {
            f'{size}|{repair}': count for (size, repair), count in sorted(selection_counts.items())},
        'unique_action_semantics': len(selected_semantics),
        'outcome_counts': dict(sorted(outcome_counts.items())),
        'acceptance_rate': accepted_moves / iterations if iterations else 0.,
        'current_improvement_rate': improving_moves / iterations if iterations else 0.,
        'new_best_rate': new_best_moves / iterations if iterations else 0.,
        'online_portfolio': portfolio.snapshot() if portfolio else None,
        'telemetry': telemetry_payload,
        'final_replay': {
            'feasible': True, 'makespan': replay.makespan,
            'violation_count': len(replay_audit['violations']),
        },
        'rng_namespaces': {
            'candidate_bank': 'target', 'neural_prior_sampling': 'neural_prior',
            'online_exploration': 'online_exploration', 'repair_construction': 'neighbor',
            'acceptance': 'acceptance', 'diagnostics': 'diagnostics',
        },
    }
    return SearchResult(
        f'NGAS-A1.4/{mode}', best, best_time, elapsed, evaluations,
        iterations, None, tuple(trace), diagnostics)
