"""Frozen metric and decision rules for the NGAS A1.6 solver comparison."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def paired_summary(
    ngas: Mapping[str, float],
    comparator: Mapping[str, float],
    tie_tolerance: float,
) -> dict:
    """Compare matched instance means; positive gain means NGAS is better."""
    if set(ngas) != set(comparator) or not ngas or tie_tolerance < 0:
        raise ValueError('Paired summaries require identical non-empty instance sets')
    rows = []
    for instance_id in sorted(ngas):
        ngas_value = float(ngas[instance_id])
        comparator_value = float(comparator[instance_id])
        if not all(map(math.isfinite, (ngas_value, comparator_value))) \
                or comparator_value <= 0:
            raise ValueError('Paired makespans must be finite and positive')
        difference = comparator_value - ngas_value
        outcome = ('WIN' if difference > tie_tolerance else
                   'LOSS' if difference < -tie_tolerance else 'TIE')
        rows.append({
            'instance_id': instance_id,
            'ngas_mean_makespan': ngas_value,
            'comparator_mean_makespan': comparator_value,
            'difference_comparator_minus_ngas': difference,
            'relative_gain_percent': 100. * difference / comparator_value,
            'outcome': outcome,
        })
    gains = [row['relative_gain_percent'] for row in rows]
    return {
        'instance_count': len(rows),
        'mean_relative_gain_percent': sum(gains) / len(gains),
        'median_relative_gain_percent': _median(gains),
        'wins': sum(row['outcome'] == 'WIN' for row in rows),
        'ties': sum(row['outcome'] == 'TIE' for row in rows),
        'losses': sum(row['outcome'] == 'LOSS' for row in rows),
        'rows': rows,
    }


def terminal_decision(
    integrity_pass: bool,
    phase6n: Mapping[str, object],
    lg_hga: Mapping[str, object],
    phase6n_by_scale: Mapping[str, Mapping[str, object]],
) -> tuple[str, dict]:
    """Apply the preregistered conjunctive A1.6 development gate."""
    primary = (float(phase6n['mean_relative_gain_percent']) > 0.
               and int(phase6n['wins']) >= int(phase6n['losses']))
    strong = float(lg_hga['mean_relative_gain_percent']) >= -1.
    scale = {
        name: float(phase6n_by_scale[name]['mean_relative_gain_percent']) >= -1.
        for name in ('S', 'M', 'L')
    }
    gates = {
        'hard_integrity': bool(integrity_pass),
        'primary_vs_phase6n_top1': primary,
        'strong_non_regression_vs_lg_hga_2o': strong,
        'no_material_scale_collapse': scale,
    }
    if not integrity_pass:
        return 'NGAS_A1_6_INVALID_COMPARISON', gates
    if primary and strong and all(scale.values()):
        return 'NGAS_A1_PASS_DEVELOPMENT', gates
    return 'NGAS_A1_REVISE_QUALITY', gates


def incumbent_at(trace: Sequence[Mapping[str, float]], elapsed_seconds: float) -> dict | None:
    """Return the last causally available incumbent without interpolation."""
    available = [row for row in trace if float(row['elapsed_time']) <= elapsed_seconds]
    return dict(available[-1]) if available else None


def right_continuous_auc(
    trace: Sequence[Mapping[str, float]],
    budget_seconds: float,
    h1_makespan: float,
) -> dict | None:
    """Integrate the incumbent step curve over its causally available horizon."""
    if budget_seconds <= 0 or h1_makespan <= 0 or not trace:
        raise ValueError('AUC inputs must be positive and non-empty')
    events = [
        (max(0., float(row['elapsed_time']) / budget_seconds),
         float(row['current_best_makespan']) / h1_makespan)
        for row in trace if float(row['elapsed_time']) <= budget_seconds
    ]
    if not events:
        return None
    if any(right[0] < left[0] or right[1] > left[1] + 1e-12
           for left, right in zip(events, events[1:])):
        raise ValueError('Incumbent trace is not monotone')
    area = 0.
    for (left_x, left_y), (right_x, _) in zip(events, events[1:]):
        area += (right_x - left_x) * left_y
    area += (1. - events[-1][0]) * events[-1][1]
    coverage = 1. - events[0][0]
    return {
        'auc_available_horizon': area,
        'available_budget_fraction': coverage,
        'mean_normalized_incumbent_over_available_horizon': (
            area / coverage if coverage > 0 else None),
        'first_incumbent_budget_fraction': events[0][0],
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in input order."""
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values):
        raise ValueError('p-values must be finite values in [0, 1]')
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.] * len(ordered)
    running = 0.
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1., (count - rank) * value))
        adjusted[index] = running
    return adjusted


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return .5 * (ordered[middle - 1] + ordered[middle])
