"""Single-source utility support and alignment metrics for A1.7A-S."""
from __future__ import annotations

import math

from rcias_ngas.evaluation.a17ar import ranking_metrics, safe_spearman


UTILITY_FIELDS = {
    'U0_IMMEDIATE': 'U0_immediate_best_gain',
    'U1_SHORT_HORIZON': 'U1_short_horizon_best_gain',
    'U2_COST_NORMALIZED': 'U2_horizon_gain_per_second',
    'U3_STOCHASTIC_ROBUSTNESS': 'U3_mean_signed_improvement',
}
UTILITY_PAIRS = (
    ('U0_IMMEDIATE', 'U1_SHORT_HORIZON'),
    ('U0_IMMEDIATE', 'U2_COST_NORMALIZED'),
    ('U0_IMMEDIATE', 'U3_STOCHASTIC_ROBUSTNESS'),
    ('U1_SHORT_HORIZON', 'U2_COST_NORMALIZED'),
    ('U1_SHORT_HORIZON', 'U3_STOCHASTIC_ROBUSTNESS'),
    ('U2_COST_NORMALIZED', 'U3_STOCHASTIC_ROBUSTNESS'),
)
UTILITY_TOLERANCE = 1e-12


def finite_values(values) -> list[float]:
    return [float(value) for value in values
            if value is not None and math.isfinite(float(value))]


def nonconstant(values, tolerance: float = UTILITY_TOLERANCE) -> bool:
    data = finite_values(values)
    return len(data) >= 2 and max(data) - min(data) > tolerance


def positive_best(values, tolerance: float = UTILITY_TOLERANCE) -> bool:
    data = finite_values(values)
    return bool(data) and max(data) > tolerance


def action_utility_values(actions: list[dict], utility: str) -> list[float]:
    field = UTILITY_FIELDS[utility]
    values = [float(action['utility'][field]) for action in actions]
    if len(values) != len(actions) or len(values) < 1 or not all(map(math.isfinite, values)):
        raise ValueError(f'invalid action utility values for {utility}')
    return values


def deterministic_order(actions: list[dict], utility: str) -> list[int]:
    values = action_utility_values(actions, utility)
    return sorted(range(len(actions)),
                  key=lambda index: (-values[index], actions[index]['action_id']))


def state_support(actions: list[dict]) -> dict[str, dict[str, bool]]:
    return {
        utility: {
            'nonconstant': nonconstant(action_utility_values(actions, utility)),
            'positive_best': positive_best(action_utility_values(actions, utility)),
        }
        for utility in UTILITY_FIELDS
    }


def state_rankings(actions: list[dict]) -> dict[str, dict]:
    scores = [float(action['critic_advantage']) for action in actions]
    action_ids = [action['action_id'] for action in actions]
    return {
        utility: ranking_metrics(
            scores, action_utility_values(actions, utility), action_ids)
        for utility in UTILITY_FIELDS
    }


def state_alignment(actions: list[dict]) -> list[dict]:
    rows = []
    for left, right in UTILITY_PAIRS:
        left_values = action_utility_values(actions, left)
        right_values = action_utility_values(actions, right)
        left_order = deterministic_order(actions, left)
        right_order = deterministic_order(actions, right)
        spearman = safe_spearman(left_values, right_values)
        rows.append({
            'left': left,
            'right': right,
            'left_nonconstant': nonconstant(left_values),
            'right_nonconstant': nonconstant(right_values),
            'pair_informative': spearman['valid'],
            'spearman': spearman['rho'],
            'same_top1_all_states': (
                actions[left_order[0]]['action_id']
                == actions[right_order[0]]['action_id']),
            'top5_overlap_count': len(set(left_order[:5]) & set(right_order[:5])),
            'top5_denominator': min(5, len(actions)),
            'top10_overlap_count': len(set(left_order[:10]) & set(right_order[:10])),
            'top10_denominator': min(10, len(actions)),
            'tie_break': 'utility_desc_then_action_id_asc',
        })
    return rows
