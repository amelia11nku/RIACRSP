"""Noise-aware state-ranking diagnostics for the NGAS joint critic."""
from __future__ import annotations

import math
import statistics


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(actual: list[float], predicted: list[float]) -> float:
    left, right = average_ranks(actual), average_ranks(predicted)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left)
                            * sum((b - right_mean) ** 2 for b in right))
    return numerator / denominator if denominator else 0.


def material_pairs(replicates: list[list[float]], minimum_gap=.001, se_multiplier=2.) -> list[tuple[int, int, int]]:
    pairs = []
    for left in range(len(replicates)):
        for right in range(left + 1, len(replicates)):
            differences = [a - b for a, b in zip(replicates[left], replicates[right])]
            mean = statistics.fmean(differences)
            se = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else math.inf
            if abs(mean) > max(minimum_gap, se_multiplier * se):
                pairs.append((left, right, 1 if mean > 0 else -1))
    return pairs


def state_metrics(record: dict, predicted: list[float], probabilities: list[float]) -> dict:
    replicates = record['replicate_advantages']
    actual = [statistics.fmean(values) for values in replicates]
    pairs = material_pairs(replicates)
    correct = 0.
    for left, right, direction in pairs:
        predicted_direction = (predicted[left] > predicted[right]) - (predicted[left] < predicted[right])
        correct += 1. if predicted_direction == direction else .5 if predicted_direction == 0 else 0.
    order = sorted(range(len(actual)), key=lambda index: (-predicted[index], index))
    selected = order[0]
    best = max(actual)
    k = min(5, len(actual))
    relevance = [value - min(actual) for value in actual]

    def dcg(indices: list[int]) -> float:
        return sum(relevance[index] / math.log2(rank + 2) for rank, index in enumerate(indices[:k]))

    ideal = sorted(range(len(actual)), key=lambda index: (-actual[index], index))
    ideal_dcg = dcg(ideal)
    brier = statistics.fmean((probability - observed) ** 2 for probability, observed in zip(
        probabilities, record['beats_fallback_frequency']))
    return {
        'state_id': record['state_id'], 'instance_id': record['instance_id'],
        'fold': record['fold'], 'scale': record['scale'], 'CF_level': record['CF_level'],
        'actions': len(actual), 'spearman': spearman(actual, predicted),
        'material_pairs': len(pairs), 'material_pair_correct': correct,
        'material_pair_accuracy': correct / len(pairs) if pairs else None,
        'ndcg_at_5': dcg(order) / ideal_dcg if ideal_dcg else 1.,
        'top5_beating_opportunity': any(actual[index] > 0 for index in order[:k]),
        'selected_action_id': record['actions'][selected]['action_id'],
        'selected_advantage': actual[selected],
        'best_advantage': best,
        'top1_regret': best - actual[selected],
        'uniform_expected_regret': best - statistics.fmean(actual),
        'continuation_lift': actual[selected] - statistics.fmean(actual),
        'brier': brier,
    }


def summarize(state_rows: list[dict]) -> dict:
    pair_count = sum(row['material_pairs'] for row in state_rows)
    pair_correct = sum(row['material_pair_correct'] for row in state_rows)

    def mean(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in state_rows)

    return {
        'states': len(state_rows),
        'mean_state_spearman': mean('spearman'),
        'material_pairs': pair_count,
        'material_pair_accuracy': pair_correct / pair_count if pair_count else 0.,
        'mean_state_material_pair_accuracy': statistics.fmean(
            row['material_pair_accuracy'] for row in state_rows if row['material_pair_accuracy'] is not None),
        'mean_ndcg_at_5': mean('ndcg_at_5'),
        'top5_beating_opportunity_fraction': mean('top5_beating_opportunity'),
        'mean_selected_advantage': mean('selected_advantage'),
        'mean_top1_regret': mean('top1_regret'),
        'mean_uniform_expected_regret': mean('uniform_expected_regret'),
        'mean_continuation_lift': mean('continuation_lift'),
        'beats_fallback_brier': mean('brier'),
    }


def subgroup_summaries(state_rows: list[dict]) -> dict:
    result = {}
    for field in ('fold', 'scale', 'CF_level'):
        result[field] = {
            str(value): summarize([row for row in state_rows if row[field] == value])
            for value in sorted({row[field] for row in state_rows}, key=str)}
    return result
