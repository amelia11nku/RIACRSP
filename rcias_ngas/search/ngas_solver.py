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

BUDGET_ACCOUNTING_MODES = (
    'LEGACY_SEARCH_ONLY',
    'A16_INSTANCE_TOTAL',
)


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


@dataclass(frozen=True)
class NGASDiagnosticConfig:
    """Purely observational A1.6R search instrumentation."""

    enabled: bool = False
    capture_fractions: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if any(not 0. < value < 1. for value in self.capture_fractions):
            raise ValueError('diagnostic capture fractions must be inside (0, 1)')
        if tuple(sorted(set(self.capture_fractions))) != self.capture_fractions:
            raise ValueError('diagnostic capture fractions must be unique and sorted')


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


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -sum(value * math.log(max(value, 1e-300)) for value in probabilities)


def _candidate_payload(candidate) -> dict:
    return {
        'operation_order': list(candidate.operation_order),
        'island_assignment': list(candidate.island_assignment),
        'w_assignment': list(candidate.w_assignment),
        'f_assignment': list(candidate.f_assignment),
    }


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
               refresh_runtime: ProductionRefreshRuntime | None = None,
               budget_accounting: str = 'LEGACY_SEARCH_ONLY',
               diagnostic_config: NGASDiagnosticConfig | None = None) -> SearchResult:
    """Run NGAS with the selected wall-clock accounting boundary.

    ``A16_INSTANCE_TOTAL`` starts the solver clock immediately before the
    per-instance production-runtime preparation.  The legacy default retains
    the historical A1.4 boundary, which starts the clock after preparation.
    """
    if (mode not in MODE_SETTINGS or time_limit <= 0
            or config.candidate_trials < 1
            or budget_accounting not in BUDGET_ACCOUNTING_MODES):
        raise ValueError('Invalid NGAS A1.4 run configuration')
    settings = MODE_SETTINGS[mode]
    diagnostic_config = diagnostic_config or NGASDiagnosticConfig()
    a16_accounting = budget_accounting == 'A16_INSTANCE_TOTAL'
    if settings['neural'] and critic is None:
        raise ValueError(f'{mode} requires a frozen critic')
    started = time.perf_counter() if a16_accounting else None
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
    atomic_counts = Counter({'runtime_preparation': int(settings['neural'])})
    atomic_max_seconds = defaultdict(float)
    atomic_max_seconds['runtime_preparation'] = preparation_seconds

    def record_atomic(name: str, duration: float) -> None:
        atomic_counts[name] += 1
        atomic_max_seconds[name] = max(atomic_max_seconds[name], duration)

    if started is None:
        started = time.perf_counter()
    elif time.perf_counter() - started >= time_limit:
        raise RuntimeError('A1.6 runtime preparation exhausted the solver budget')
    h1_started = time.perf_counter()
    h1 = solve_dispatching(instance, 'H1')
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    initialization_seconds = time.perf_counter() - h1_started
    runtime['initialization_seconds'] += initialization_seconds
    record_atomic('h1_initialization', initialization_seconds)
    best = current
    best_time = time.perf_counter() - started
    evaluations = 1
    iterations = accepted_moves = improving_moves = new_best_moves = neural_calls = 0
    telemetry = A14Telemetry(time_limit)
    state = RunState(evaluations, neural_calls, current.makespan, best.makespan)
    telemetry.observe(best_time, state)
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    cache = None
    refresh_log, iteration_log, diagnostic_snapshots = [], [], []
    pending_capture_fractions = list(diagnostic_config.capture_fractions)
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
        if a16_accounting and iteration_started - started >= time_limit:
            break
        atomic_counts['iteration'] += 1
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
            if a16_accounting and refresh_started - started >= time_limit:
                break
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
            record_atomic('refresh', refresh_elapsed)
            runtime['refresh_total_seconds'] += refresh_elapsed
            refresh_record = {
                'iteration': iterations, 'elapsed_time_sec': time.perf_counter() - started,
                'reasons': list(reasons), 'critic_called': settings['neural'],
                'critic_variant': critic.variant if settings['neural'] else None,
                'components_ms': dict(refresh.components_ms) if settings['neural'] else None,
                'bank': bank_summary, 'critical_signature': critical_signature,
                'dominant_bottleneck': bottleneck,
                'distribution': distribution_summary(actions, probabilities),
                'refresh_seconds': refresh_elapsed,
            }
            if diagnostic_config.enabled:
                if refresh_log:
                    refresh_record['changed_since_previous_refresh'] = {
                        'critical_signature': (
                            critical_signature != refresh_log[-1]['critical_signature']),
                        'dominant_bottleneck': (
                            bottleneck != refresh_log[-1]['dominant_bottleneck']),
                    }
                else:
                    refresh_record['changed_since_previous_refresh'] = None
            refresh_log.append(refresh_record)
            if diagnostic_config.enabled and pending_capture_fractions:
                refresh_fraction = refresh_record['elapsed_time_sec'] / time_limit
                while (pending_capture_fractions
                       and refresh_fraction >= pending_capture_fractions[0]):
                    threshold = pending_capture_fractions.pop(0)
                    diagnostic_snapshots.append({
                        'schema': 'ngas-a16r-replayable-state-v1',
                        'capture_fraction': threshold,
                        'observed_budget_fraction': refresh_fraction,
                        'iteration': iterations,
                        'state_id': state_id,
                        'current_makespan': current.makespan,
                        'current_candidate': _candidate_payload(current.candidate),
                        'action_ids': [action.action_id for action in actions],
                        'prior': list(probabilities),
                        'advantage': list(predictions['advantage']),
                        'beats_fallback_probability': list(
                            predictions['beats_fallback_probability']),
                        'critical_signature': critical_signature,
                        'dominant_bottleneck': bottleneck,
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
        selected_portfolio_factor = (
            portfolio.factor(action) if diagnostic_config.enabled and portfolio else 1.)

        current_before, best_before = current, best
        candidates = []
        trial_diagnostics = []
        trial_best_makespan = current_before.makespan
        state_id = f'{instance.instance_id}:seed{seed}:iteration{iterations}'
        for trial in range(config.candidate_trials):
            repair_started = time.perf_counter()
            if ((a16_accounting or candidates)
                    and repair_started - started >= time_limit):
                break
            neighbor = construct_neighbor(
                instance, current, action,
                streams.stream('neighbor', state_id, trial))
            repair_seconds = time.perf_counter() - repair_started
            runtime['repair_construction_seconds'] += repair_seconds
            record_atomic('repair', repair_seconds)
            decoder_started = time.perf_counter()
            if a16_accounting and decoder_started - started >= time_limit:
                break
            candidate = decode_candidate(instance, neighbor)
            decoder_seconds = time.perf_counter() - decoder_started
            runtime['decoder_seconds'] += decoder_seconds
            record_atomic('decoder', decoder_seconds)
            if not candidate.feasible:
                raise RuntimeError('Frozen repair produced infeasible candidate')
            candidates.append(candidate)
            if diagnostic_config.enabled:
                previous_trial_best = trial_best_makespan
                trial_best_makespan = min(trial_best_makespan, candidate.makespan)
                trial_diagnostics.append({
                    'trial': trial + 1,
                    'repair_rng_seed': streams.seed('neighbor', state_id, trial),
                    'candidate_makespan': candidate.makespan,
                    'improved_current': candidate.makespan < current_before.makespan,
                    'improvement_from_current': max(
                        0., current_before.makespan - candidate.makespan),
                    'best_so_far_makespan': trial_best_makespan,
                    'marginal_best_gain': max(
                        0., previous_trial_best - trial_best_makespan),
                    'repair_seconds': repair_seconds,
                    'decoder_seconds': decoder_seconds,
                    'trial_seconds': repair_seconds + decoder_seconds,
                })
            evaluations += 1
        if not candidates:
            atomic_max_seconds['iteration'] = max(
                atomic_max_seconds['iteration'], time.perf_counter() - iteration_started)
            break
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
        if diagnostic_config.enabled:
            combined_order = sorted(
                range(len(cache.actions)),
                key=lambda i: (-combined[i], cache.actions[i].action_id))
            combined_rank = combined_order.index(index) + 1
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
        iteration_record = {
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
        }
        if diagnostic_config.enabled:
            final_trial = min(
                range(len(trial_diagnostics)),
                key=lambda value: (
                    trial_diagnostics[value]['candidate_makespan'], value)) + 1
            iteration_record['a16r_observation'] = {
                'budget_fraction': min(1., elapsed / time_limit),
                'prior_entropy': _entropy(base),
                'combined_entropy': _entropy(combined),
                'selected_advantage': cache.advantages[index] if neural_live else None,
                'selected_fallback_probability': (
                    cache.beats_fallback_probability[index] if neural_live else None),
                'combined_rank': combined_rank,
                'portfolio_factor': selected_portfolio_factor,
                'portfolio_rank_displacement': combined_rank - neural_rank,
                'portfolio_changed_top1': neural_order[0] != combined_order[0],
                'portfolio_top5_overlap': len(
                    set(neural_order[:5]) & set(combined_order[:5])),
                'critic_top1_action_id': cache.actions[neural_order[0]].action_id,
                'combined_top1_action_id': cache.actions[combined_order[0]].action_id,
                'target_origin_rules': list(action.target.origin_rules),
                'target_origin_operators': list(action.target.origin_operators),
                'final_best_trial': final_trial,
                'trials_improving_current': sum(
                    row['improved_current'] for row in trial_diagnostics),
                'best_of_trials_improvement': max(
                    0., current_before.makespan - candidate.makespan),
                'candidate_trials': trial_diagnostics,
            }
        iteration_log.append(iteration_record)
        atomic_max_seconds['iteration'] = max(
            atomic_max_seconds['iteration'], elapsed - (iteration_started - started))
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
        'schema': ('ngas-a16r-run-diagnostics-v1' if diagnostic_config.enabled
                   else 'ngas-a14-run-diagnostics-v1'),
        'mode': mode, 'seed': seed, 'time_limit_seconds': time_limit,
        'budget_accounting': budget_accounting,
        'solver_budget_elapsed_seconds': elapsed,
        'budget_overshoot_seconds': max(0., elapsed - time_limit),
        'atomic_budget_audit': {
            'operation_counts': dict(sorted(atomic_counts.items())),
            'maximum_operation_seconds': dict(sorted(atomic_max_seconds.items())),
            'started_after_deadline_count': 0,
            'rule': ('A16_STRICT_PRESTART_CHECKS' if a16_accounting
                     else 'LEGACY_HISTORICAL_BOUNDARY'),
        },
        'termination_reason': ('ITERATION_LIMIT' if config.iteration_limit is not None
                               and iterations >= config.iteration_limit else 'TIME_LIMIT'),
        'critic_variant': critic.variant if settings['neural'] else None,
        'critic_checkpoint_sha256': critic.sha256 if settings['neural'] else None,
        'refreshes': refresh_log, 'iterations': iteration_log,
        'a16r_replayable_states': diagnostic_snapshots,
        'a16r_instrumentation': {
            'enabled': diagnostic_config.enabled,
            'capture_fractions': list(diagnostic_config.capture_fractions),
            'unreached_capture_fractions': pending_capture_fractions,
            'observational_only': True,
        },
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
