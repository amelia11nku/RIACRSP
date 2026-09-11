"""Pure post hoc diagnostics for immutable A1.6R formal trajectories."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.stats import spearmanr


STAGE_BINS = (
    (0., .10, '0-10%'), (.10, .25, '10-25%'), (.25, .50, '25-50%'),
    (.50, .75, '50-75%'), (.75, 1.0000001, '75-100%'),
)
AGE_BINS = ((1, 1, '1'), (2, 5, '2-5'), (6, 10, '6-10'),
            (11, 15, '11-15'), (16, 20, '16-20'))


def stage_label(fraction: float) -> str:
    return next(name for lower, upper, name in STAGE_BINS
                if lower <= min(1., fraction) < upper)


def age_label(age: int) -> str:
    for lower, upper, name in AGE_BINS:
        if lower <= age <= upper:
            return name
    return '0' if age == 0 else '>20'


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return math.fsum(values) / len(values) if values else None


def _safe_spearman(left, right) -> dict:
    pairs = [(float(a), float(b)) for a, b in zip(left, right)
             if a is not None and b is not None]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 \
            or len({b for _, b in pairs}) < 2:
        return {'n': len(pairs), 'rho': None, 'p_value': None,
                'statistically_valid': False}
    result = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {'n': len(pairs), 'rho': float(result.statistic),
            'p_value': float(result.pvalue), 'statistically_valid': True}


def _iterations(payloads: list[dict]):
    for payload in payloads:
        for row in payload['search_diagnostics']['iterations']:
            observation = row.get('a16r_observation')
            if observation is None:
                raise RuntimeError('A1.6R iteration is missing observational diagnostics')
            yield payload, row, observation, stage_label(observation['budget_fraction'])


def search_stage_summary(payloads: list[dict]) -> dict:
    groups = defaultdict(list)
    refreshes = Counter()
    for payload in payloads:
        for refresh in payload['search_diagnostics']['refreshes']:
            refreshes[stage_label(refresh['elapsed_time_sec'] / payload['budget_seconds'])] += 1
    for payload, row, obs, stage in _iterations(payloads):
        groups[stage].append((payload, row, obs))
    rows = []
    for _, _, stage in STAGE_BINS:
        values = groups[stage]
        action_counts = Counter(
            f"{row['action_size']}|{row['repair']}" for _, row, _ in values)
        rule_counts = Counter(rule for _, _, obs in values
                              for rule in obs['target_origin_rules'])
        operator_counts = Counter(operator for _, _, obs in values
                                  for operator in obs['target_origin_operators'])
        repair_seconds = [trial['repair_seconds'] for _, _, obs in values
                          for trial in obs['candidate_trials']]
        decoder_seconds = [trial['decoder_seconds'] for _, _, obs in values
                           for trial in obs['candidate_trials']]
        rows.append({
            'stage': stage,
            'iterations': len(values),
            'decoder_evaluations': sum(row['candidate_trials'] for _, row, _ in values),
            'accepted_moves': sum(bool(row['accepted']) for _, row, _ in values),
            'improving_moves': sum(row['relative_current_improvement'] > 0
                                   and row['accepted'] for _, row, _ in values),
            'new_best_moves': sum(row['outcome_class'] == 'new_global_best'
                                  for _, row, _ in values),
            'refreshes': refreshes[stage],
            'guided_iterations': sum(bool(row['neural_live']) for _, row, _ in values),
            'mean_incumbent_makespan': _mean(row['best_after'] for _, row, _ in values),
            'mean_prior_entropy': _mean(obs['prior_entropy'] for _, _, obs in values),
            'mean_combined_entropy': _mean(obs['combined_entropy'] for _, _, obs in values),
            'candidate_trial_success_rate': _mean(
                obs['trials_improving_current'] > 0 for _, _, obs in values),
            'mean_best_of_8_improvement': _mean(
                obs['best_of_trials_improvement'] for _, _, obs in values),
            'mean_trials_improving_current': _mean(
                obs['trials_improving_current'] for _, _, obs in values),
            'repair_seconds': math.fsum(repair_seconds),
            'decoder_seconds': math.fsum(decoder_seconds),
            'action_size_repair_counts': dict(sorted(action_counts.items())),
            'target_origin_rule_counts': dict(sorted(rule_counts.items())),
            'target_origin_operator_counts': dict(sorted(operator_counts.items())),
        })
    return {'schema': 'ngas-a16r-search-stage-summary-v1', 'rows': rows}


def best_of_k_summary(payloads: list[dict]) -> dict:
    records = []
    first_best = Counter()
    for payload, _, obs, stage in _iterations(payloads):
        first_best[(payload['scale'], stage, obs['final_best_trial'])] += 1
        for trial in obs['candidate_trials']:
            records.append({
                'scale': payload['scale'], 'stage': stage, **trial,
            })
    rows = []
    for scale in ('ALL', 'S', 'M', 'L'):
        selected = records if scale == 'ALL' else [r for r in records if r['scale'] == scale]
        for trial in range(1, 9):
            values = [row for row in selected if row['trial'] == trial]
            gains = [row['marginal_best_gain'] for row in values]
            durations = [row['trial_seconds'] for row in values]
            total_ms = 1000. * math.fsum(durations)
            rows.append({
                'scale': scale, 'trial': trial, 'observations': len(values),
                'probability_improves_prior_best': _mean(value > 0 for value in gains),
                'mean_marginal_gain': _mean(gains),
                'mean_trial_milliseconds': 1000. * _mean(durations) if values else None,
                'aggregate_gain_per_millisecond': (
                    math.fsum(gains) / total_ms if total_ms else None),
            })
    first_rows = [
        {'scale': scale, 'stage': stage, 'trial': trial, 'count': count}
        for (scale, stage, trial), count in sorted(first_best.items())
    ]
    return {'schema': 'ngas-a16r-best-of-k-summary-v1', 'rows': rows,
            'final_best_first_discovery': first_rows}


def prior_staleness_summary(payloads: list[dict]) -> dict:
    groups = defaultdict(list)
    structural = Counter()
    for payload, row, obs, stage in _iterations(payloads):
        groups[(payload['scale'], stage, age_label(row['prior_age_iterations']))].append(
            (row, obs))
    for payload in payloads:
        for refresh in payload['search_diagnostics']['refreshes']:
            change = refresh.get('changed_since_previous_refresh')
            if change:
                structural['refresh_transitions'] += 1
                structural['critical_signature_changes'] += bool(change['critical_signature'])
                structural['dominant_bottleneck_changes'] += bool(change['dominant_bottleneck'])
    rows = []
    for (scale, stage, age), values in sorted(groups.items()):
        rows.append({
            'scale': scale, 'stage': stage, 'prior_age': age,
            'iterations': len(values),
            'mean_prior_entropy': _mean(obs['prior_entropy'] for _, obs in values),
            'mean_selected_neural_probability': _mean(
                row['neural_probability'] for row, _ in values),
            'mean_selected_neural_rank': _mean(row['neural_rank'] for row, _ in values),
            'mean_portfolio_factor': _mean(obs['portfolio_factor'] for _, obs in values),
            'mean_selected_combined_probability': _mean(
                row['combined_probability'] for row, _ in values),
            'accepted_rate': _mean(row['accepted'] for row, _ in values),
            'new_best_rate': _mean(
                row['outcome_class'] == 'new_global_best' for row, _ in values),
            'mean_realized_improvement': _mean(
                obs['best_of_trials_improvement'] for _, obs in values),
        })
    return {'schema': 'ngas-a16r-prior-staleness-summary-v1', 'rows': rows,
            'structural_changes_at_refresh_boundaries': dict(structural),
            'limitation': (
                'structural identities are observed only at frozen refreshes; '
                'no extra in-horizon structural computation was added')}


def critic_portfolio_summary(payloads: list[dict]) -> dict:
    records = []
    for payload, row, obs, stage in _iterations(payloads):
        records.append({
            'scale': payload['scale'], 'stage': stage,
            'changed_top1': obs['portfolio_changed_top1'],
            'top5_overlap': obs['portfolio_top5_overlap'],
            'displacement': obs['portfolio_rank_displacement'],
            'entropy_change': obs['combined_entropy'] - obs['prior_entropy'],
            'realized_improvement': obs['best_of_trials_improvement'],
            'accepted': row['accepted'],
            'new_best': row['outcome_class'] == 'new_global_best',
        })
    classes = {
        'promoted': [row for row in records if row['displacement'] < 0],
        'unchanged': [row for row in records if row['displacement'] == 0],
        'demoted': [row for row in records if row['displacement'] > 0],
    }
    return {
        'schema': 'ngas-a16r-critic-portfolio-summary-v1',
        'iterations': len(records),
        'top1_change_rate': _mean(row['changed_top1'] for row in records),
        'mean_top5_overlap': _mean(row['top5_overlap'] for row in records),
        'mean_rank_displacement': _mean(row['displacement'] for row in records),
        'mean_entropy_change': _mean(row['entropy_change'] for row in records),
        'displacement_outcomes': {
            name: {
                'iterations': len(rows),
                'mean_realized_improvement': _mean(
                    row['realized_improvement'] for row in rows),
                'accepted_rate': _mean(row['accepted'] for row in rows),
                'new_best_rate': _mean(row['new_best'] for row in rows),
            } for name, rows in classes.items()
        },
    }


def passive_critic_summary(payloads: list[dict]) -> dict:
    records = []
    for payload, row, obs, stage in _iterations(payloads):
        records.append({
            'scale': payload['scale'], 'stage': stage,
            'score': obs['selected_advantage'],
            'fallback_probability': obs['selected_fallback_probability'],
            'improvement': obs['best_of_trials_improvement'],
            'accepted': bool(row['accepted']),
            'new_best': row['outcome_class'] == 'new_global_best',
        })
    correlations = []
    for scale in ('ALL', 'S', 'M', 'L'):
        for stage in ('ALL', *(name for _, _, name in STAGE_BINS)):
            rows = [row for row in records
                    if (scale == 'ALL' or row['scale'] == scale)
                    and (stage == 'ALL' or row['stage'] == stage)]
            if not rows:
                continue
            correlations.append({
                'scale': scale, 'stage': stage,
                'advantage_vs_improvement': _safe_spearman(
                    [row['score'] for row in rows],
                    [row['improvement'] for row in rows]),
                'fallback_probability_vs_improvement': _safe_spearman(
                    [row['fallback_probability'] for row in rows],
                    [row['improvement'] for row in rows]),
            })
    finite = [row for row in records if row['score'] is not None]
    scores = np.asarray([row['score'] for row in finite], dtype=float)
    quantile_rows = []
    if len(scores):
        boundaries = np.quantile(scores, np.linspace(0., 1., 6))
        for index in range(5):
            lower, upper = float(boundaries[index]), float(boundaries[index + 1])
            rows = [row for row in finite if row['score'] >= lower
                    and (row['score'] <= upper if index == 4 else row['score'] < upper)]
            quantile_rows.append({
                'quantile': index + 1, 'score_lower': lower, 'score_upper': upper,
                'iterations': len(rows),
                'mean_realized_improvement': _mean(row['improvement'] for row in rows),
                'positive_improvement_rate': _mean(row['improvement'] > 0 for row in rows),
                'accepted_rate': _mean(row['accepted'] for row in rows),
                'new_best_rate': _mean(row['new_best'] for row in rows),
            })
    return {
        'schema': 'ngas-a16r-passive-critic-summary-v1',
        'interpretation': 'selected-action association only; not counterfactual ranking',
        'correlations': correlations,
        'advantage_quantiles': quantile_rows,
    }


def build_all(payloads: list[dict]) -> dict[str, dict]:
    return {
        'search_stage': search_stage_summary(payloads),
        'best_of_k': best_of_k_summary(payloads),
        'prior_staleness': prior_staleness_summary(payloads),
        'critic_portfolio': critic_portfolio_summary(payloads),
        'passive_critic': passive_critic_summary(payloads),
    }
