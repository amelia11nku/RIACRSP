from __future__ import annotations

import pandas as pd
import pytest

from rcias_clgri.ni.baselines import (
    TABULAR_NUMERIC,
    Phase6CTabularDiagnostic,
    choose_best_fixed_original,
    fixed_original_selection,
    random_selection_expectation,
)
from rcias_clgri.ni.metrics import evaluate_action_scores, selected_action_table
from scripts.evaluate_phase6e_supervised_ni import benjamini_hochberg


def _scored_frame() -> pd.DataFrame:
    rows = []
    for state_index in range(2):
        for action_index, (operator, utility) in enumerate(
            (("related", 0.2), ("critical", 0.05), ("random", -0.1))
        ):
            rows.append({
                "state_id": f"state-{state_index}",
                "target_set_id": f"target-{state_index}-{action_index}",
                "training_split": "TRAIN_VALIDATION",
                "arm_family": "ORIGINAL_OPERATOR",
                "origin_destroy_operator": operator,
                "origin_rules": f'["operator_{operator}"]',
                "mean_relative_improvement": utility,
                "rank_percentile": 1.0 - action_index / 2,
                "rank_within_state": action_index + 1,
                "regret_to_best": 0.2 - utility,
                "top1": action_index == 0,
                "top3": True,
                "score": utility,
            })
    return pd.DataFrame(rows)


def test_perfect_ranking_and_selected_action_metrics():
    frame = _scored_frame()
    metrics = evaluate_action_scores(frame)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["within_state_spearman"] == pytest.approx(1.0)
    assert metrics["ndcg"] == pytest.approx(1.0)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top3_recall"] == 1.0
    assert metrics["mean_selected_utility"] == pytest.approx(0.2)
    assert metrics["median_selected_utility"] == pytest.approx(0.2)
    assert metrics["selected_top3_fraction"] == 1.0
    assert metrics["mean_selected_rank_percentile"] == 1.0
    assert metrics["mean_selected_regret"] == 0.0
    selected = selected_action_table(frame, model_name="perfect")
    assert len(selected) == 2
    assert selected["selected_positive"].all()


def test_exact_random_related_and_validation_frozen_operator_baselines():
    frame = _scored_frame()
    random = random_selection_expectation(frame)
    assert random["mean_selected_utility"].tolist() == pytest.approx([0.05, 0.05])
    related = fixed_original_selection(frame, "related")
    assert len(related) == 2
    assert related["mean_relative_improvement"].eq(0.2).all()
    best, summary = choose_best_fixed_original(
        frame, operators=("related", "critical", "random")
    )
    assert best == "related"
    assert summary.iloc[0]["origin_destroy_operator"] == "related"


def test_phase6c_tabular_baseline_uses_only_declared_legal_features():
    frame = _scored_frame()
    frame["training_split"] = "TRAIN"
    frame["scale"] = "S"
    frame["CF_level"] = "CF1"
    frame["RI_level"] = "RI1"
    frame["TI_level"] = "TI1"
    frame["search_stage"] = "0-20%"
    frame["bottleneck_proxy"] = "MIXED_OR_UNCERTAIN"
    frame["positive_under_2_of_3"] = frame["mean_relative_improvement"].gt(0)
    for index, column in enumerate(TABULAR_NUMERIC):
        frame[column] = float(index + 1)
    model = Phase6CTabularDiagnostic(sample_limit=100).fit(frame)
    scores = model.score(frame)
    assert scores.shape == (len(frame),)
    assert set(model.feature_columns).isdisjoint({
        "state_id", "target_set_id", "origin_destroy_operator", "regret_to_best", "top1"
    })
    forbidden = {"counterfactual_makespan", "mean_relative_improvement", "repair_seed"}
    assert set(model.feature_columns).isdisjoint(forbidden)


def test_phase6c_tabular_baseline_requires_frozen_robust_label():
    frame = _scored_frame()
    frame["training_split"] = "TRAIN"
    frame["scale"] = "S"
    frame["CF_level"] = "CF1"
    frame["RI_level"] = "RI1"
    frame["TI_level"] = "TI1"
    frame["search_stage"] = "0-20%"
    frame["bottleneck_proxy"] = "MIXED_OR_UNCERTAIN"
    for index, column in enumerate(TABULAR_NUMERIC):
        frame[column] = float(index + 1)
    with pytest.raises(ValueError, match="positive_under_2_of_3"):
        Phase6CTabularDiagnostic(sample_limit=100).fit(frame)


def test_fixed_original_selection_uses_deduplicated_origin_rules():
    frame = _scored_frame()
    mask = frame["origin_destroy_operator"].eq("related") & frame["state_id"].eq("state-0")
    frame.loc[mask, "origin_destroy_operator"] = "f_bottleneck"
    frame.loc[mask, "origin_rules"] = '["operator_f_bottleneck","operator_related"]'
    related = fixed_original_selection(frame, "related")
    assert related["state_id"].nunique() == 2
    assert len(related) == 2


def test_metrics_reject_missing_contract_columns():
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_action_scores(pd.DataFrame({"state_id": ["x"], "score": [0.0]}))


def test_benjamini_hochberg_is_monotone_in_rank_and_bounded():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])
    assert adjusted == pytest.approx([0.04, 0.05333333333333334, 0.05333333333333334, 0.8])
    assert all(0 <= value <= 1 for value in adjusted)
