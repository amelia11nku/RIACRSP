"""Extended A1.3R OOF ranking and fallback diagnostics."""
from __future__ import annotations

import math
import statistics

from rcias_ngas.evaluation.ranking import material_pairs, spearman


def state_metrics(record: dict, predicted: list[float],
                  probabilities: list[float]) -> dict:
    replicates = record['replicate_advantages']
    actual = [statistics.fmean(values) for values in replicates]
    pairs = material_pairs(replicates)
    correct = 0.
    for left, right, direction in pairs:
        predicted_direction = (
            (predicted[left] > predicted[right])
            - (predicted[left] < predicted[right]))
        correct += 1. if predicted_direction == direction else .5 if predicted_direction == 0 else 0.
    order = sorted(range(len(actual)), key=lambda index: (-predicted[index], index))
    ideal = sorted(range(len(actual)), key=lambda index: (-actual[index], index))
    relevance = [value - min(actual) for value in actual]

    def ndcg(cutoff: int) -> float:
        def dcg(indices: list[int]) -> float:
            return sum(relevance[index] / math.log2(rank + 2)
                       for rank, index in enumerate(indices[:cutoff]))
        denominator = dcg(ideal)
        return dcg(order) / denominator if denominator else 1.

    selected = order[0]
    best = max(actual)
    observed = record['beats_fallback_frequency']
    predicted_class = [probability >= .5 for probability in probabilities]
    observed_class = [frequency > 0 for frequency in observed]
    return {
        'state_id': record['state_id'], 'instance_id': record['instance_id'],
        'fold': record['fold'], 'scale': record['scale'], 'CF_level': record['CF_level'],
        'actions': len(actual), 'spearman': spearman(actual, predicted),
        'material_pairs': len(pairs), 'material_pair_correct': correct,
        'material_pair_accuracy': correct / len(pairs) if pairs else None,
        'ndcg_at_1': ndcg(1), 'ndcg_at_3': ndcg(min(3, len(actual))),
        'ndcg_at_5': ndcg(min(5, len(actual))),
        'selected_action_id': record['actions'][selected]['action_id'],
        'selected_advantage': actual[selected],
        'best_advantage': best,
        'top1_regret': best - actual[selected],
        'uniform_expected_regret': best - statistics.fmean(actual),
        'continuation_lift': actual[selected] - statistics.fmean(actual),
        'beats_fallback_brier': statistics.fmean(
            (probability - frequency) ** 2
            for probability, frequency in zip(probabilities, observed)),
        'beats_fallback_accuracy': statistics.fmean(
            float(prediction == truth)
            for prediction, truth in zip(predicted_class, observed_class)),
        'selected_beats_fallback_probability': probabilities[selected],
        'selected_beats_fallback_frequency': observed[selected],
    }


def summarize(rows: list[dict]) -> dict:
    pairs = sum(row['material_pairs'] for row in rows)
    correct = sum(row['material_pair_correct'] for row in rows)

    def mean(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    return {
        'states': len(rows),
        'mean_state_spearman': mean('spearman'),
        'material_pairs': pairs,
        'material_pair_accuracy': correct / pairs if pairs else 0.,
        'mean_state_material_pair_accuracy': statistics.fmean(
            row['material_pair_accuracy'] for row in rows
            if row['material_pair_accuracy'] is not None),
        'mean_ndcg_at_1': mean('ndcg_at_1'),
        'mean_ndcg_at_3': mean('ndcg_at_3'),
        'mean_ndcg_at_5': mean('ndcg_at_5'),
        'mean_selected_advantage': mean('selected_advantage'),
        'mean_top1_regret': mean('top1_regret'),
        'mean_uniform_expected_regret': mean('uniform_expected_regret'),
        'mean_continuation_lift': mean('continuation_lift'),
        'beats_fallback_brier': mean('beats_fallback_brier'),
        'beats_fallback_accuracy': mean('beats_fallback_accuracy'),
        'mean_selected_beats_fallback_probability':
            mean('selected_beats_fallback_probability'),
        'mean_selected_beats_fallback_frequency':
            mean('selected_beats_fallback_frequency'),
    }


def subgroup_summaries(rows: list[dict]) -> dict:
    return {
        field: {
            str(value): summarize([row for row in rows if row[field] == value])
            for value in sorted({row[field] for row in rows}, key=str)
        }
        for field in ('fold', 'scale', 'CF_level')
    }
