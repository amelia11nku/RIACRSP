"""Pure metric and replay helpers for the frozen A1.7A-R diagnostic audit."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy.stats import spearmanr

from rcias_ngas.search.online_portfolio import OnlinePortfolio


@dataclass(frozen=True)
class PortfolioAction:
    size: str
    repair: str
    target: object


@dataclass(frozen=True)
class PortfolioTarget:
    origin_families: tuple[str, ...]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_spearman(left, right) -> dict:
    pairs = [(float(a), float(b)) for a, b in zip(left, right)
             if a is not None and b is not None
             and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 \
            or len({b for _, b in pairs}) < 2:
        return {'n': len(pairs), 'rho': None, 'p_value': None, 'valid': False}
    result = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {
        'n': len(pairs), 'rho': float(result.statistic),
        'p_value': float(result.pvalue), 'valid': True,
    }


def ndcg(order: list[int], gains: list[float]) -> float | None:
    positive = [max(0., float(value)) for value in gains]
    ideal = sorted(positive, reverse=True)
    if not ideal or ideal[0] <= 0.:
        return None
    discounts = [math.log2(index + 2.) for index in range(len(positive))]
    actual = math.fsum(positive[index] / discounts[rank]
                       for rank, index in enumerate(order))
    maximum = math.fsum(value / discounts[rank]
                        for rank, value in enumerate(ideal))
    return actual / maximum


def ranking_metrics(scores: list[float], utility: list[float],
                    action_ids: list[str]) -> dict:
    if not scores or len(scores) != len(utility) or len(scores) != len(action_ids):
        raise ValueError('ranking inputs must be non-empty and aligned')
    predicted = sorted(range(len(scores)), key=lambda i: (-scores[i], action_ids[i]))
    best = max(utility)
    best_set = {index for index, value in enumerate(utility)
                if math.isclose(value, best, rel_tol=0., abs_tol=1e-12)}
    top1 = predicted[0]
    top1_rank = 1 + sum(value > utility[top1] + 1e-12 for value in utility)
    gaps = {
        str(k): best - max(utility[index] for index in predicted[:k])
        for k in (1, 5, 10)
    }
    return {
        'predicted_top1_action_id': action_ids[top1],
        'top1_realized_rank': top1_rank,
        'top1_normalized_rank': (
            (top1_rank - 1.) / (len(scores) - 1.) if len(scores) > 1 else 0.),
        'top1_utility': utility[top1], 'best_utility': best,
        'top1_regret': best - utility[top1],
        'best_action_hit_at_1': bool(best_set & set(predicted[:1])),
        'best_action_hit_at_5': bool(best_set & set(predicted[:5])),
        'best_action_hit_at_10': bool(best_set & set(predicted[:10])),
        'best_set_recall_at_1': len(best_set & set(predicted[:1])) / len(best_set),
        'best_set_recall_at_5': len(best_set & set(predicted[:5])) / len(best_set),
        'best_set_recall_at_10': len(best_set & set(predicted[:10])) / len(best_set),
        'ndcg_full': ndcg(predicted, utility),
        'spearman': safe_spearman(scores, utility),
        'top_k_utility_gap': gaps,
    }


def utility_agreement(actions: list[dict]) -> list[dict]:
    definitions = {
        'U0_IMMEDIATE': [row['utility']['U0_immediate_best_gain'] for row in actions],
        'U1_SHORT_HORIZON': [row['utility']['U1_short_horizon_best_gain'] for row in actions],
        'U2_COST_NORMALIZED': [
            row['utility']['U2_horizon_gain_per_second'] for row in actions],
        'U3_STOCHASTIC_ROBUSTNESS': [
            row['utility']['U3_mean_signed_improvement'] for row in actions],
    }
    rows = []
    names = list(definitions)
    action_ids = [row['action_id'] for row in actions]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            left_values, right_values = definitions[left], definitions[right]
            left_order = sorted(
                range(len(actions)), key=lambda i: (-left_values[i], action_ids[i]))
            right_order = sorted(
                range(len(actions)), key=lambda i: (-right_values[i], action_ids[i]))
            rows.append({
                'left': left, 'right': right,
                'spearman': safe_spearman(left_values, right_values),
                'same_top1': left_order[0] == right_order[0],
                'top5_overlap': len(set(left_order[:5]) & set(right_order[:5])),
                'top10_overlap': len(set(left_order[:10]) & set(right_order[:10])),
            })
    return rows


def portfolio_action(size: str, repair: str,
                     origin_families: list[str] | tuple[str, ...]) -> PortfolioAction:
    return PortfolioAction(
        size, repair, PortfolioTarget(tuple(origin_families) or ('unknown',)))


def replay_portfolio(iterations: list[dict], through_iteration: int, *,
                     segment_length: int, reaction: float,
                     strength: float) -> OnlinePortfolio:
    portfolio = OnlinePortfolio(
        segment_length=segment_length, reaction=reaction, strength=strength)
    selected = [row for row in iterations if int(row['iteration']) <= through_iteration]
    for row in selected:
        portfolio.observe(
            portfolio_action(
                row['action_size'], row['repair'], row.get('origin_families', [])),
            row['outcome_class'], float(row['relative_current_improvement']))
    return portfolio


def clone_portfolio(source: OnlinePortfolio) -> OnlinePortfolio:
    clone = OnlinePortfolio(
        segment_length=source.segment_length, reaction=source.reaction,
        strength=source.strength, factor_floor=source.factor_floor,
        factor_ceiling=source.factor_ceiling)
    clone.estimates.update(source.estimates)
    clone.pending = defaultdict(list, {
        key: list(values) for key, values in source.pending.items()})
    clone.outcome_counts.update(source.outcome_counts)
    clone.observations = source.observations
    clone.segment_updates = source.segment_updates
    return clone


def outcome_class(candidate_makespan: float, current_makespan: float,
                  best_makespan: float, accepted: bool) -> tuple[str, float]:
    improvement = max(0., current_makespan - candidate_makespan)
    relative = improvement / max(1., current_makespan)
    if candidate_makespan < best_makespan:
        return 'new_global_best', relative
    if accepted and improvement > 0.:
        return 'accepted_current_improvement', relative
    if accepted:
        return 'accepted_worse_or_neutral', relative
    return 'rejected', relative


def aggregate(values) -> dict:
    data = np.asarray(list(values), dtype=float)
    if not len(data):
        raise ValueError('cannot aggregate empty values')
    return {
        'count': int(len(data)), 'mean': float(np.mean(data)),
        'standard_deviation': float(np.std(data, ddof=0)),
        'minimum': float(np.min(data)), 'maximum': float(np.max(data)),
    }
