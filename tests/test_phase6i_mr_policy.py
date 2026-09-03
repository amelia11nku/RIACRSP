import numpy as np
import pandas as pd
import pytest

from rcias_clgri.ni.phase6i_policy import (
    cross_fit_probability_calibration,
    cross_fit_utility_calibration,
    ensemble_oof_predictions,
    fit_support_bounds,
    select_immediate_actions,
    state_ranking_metrics,
    summarize_ranking_metrics,
    support_mask,
)


def _oof_frame():
    rows = []
    for seed, shift in [(11, 0.0), (12, 0.2), (13, 0.4)]:
        for state, fold in [("s0", "f0"), ("s1", "f1")]:
            for action, truth in [("a", -0.1), ("b", 0.2)]:
                rows.append({
                    "state_id": state,
                    "target_set_id": action,
                    "training_seed": seed,
                    "held_fold": fold,
                    "oof_fold": fold,
                    "decoded_immediate_utility": truth,
                    "positive_label": truth > 0,
                    "fallback_target_set_id": "a",
                    "fallback_decoded_utility": -0.1,
                    "scale": "S",
                    "prediction": truth + shift,
                    "score": truth - shift,
                    "instance_id": state,
                })
    return pd.DataFrame(rows)


def test_ensemble_oof_predictions_validates_and_averages_seeds():
    frame = _oof_frame()
    result = ensemble_oof_predictions(
        frame,
        value_column="prediction",
        score_column="score",
        expected_training_seeds=[11, 12, 13],
    )
    row = result[(result.state_id == "s0") & (result.target_set_id == "b")].iloc[0]
    assert row.ensemble_raw_value == pytest.approx(0.4)
    assert row.ensemble_raw_score == pytest.approx(0.0)
    assert len(result) == 4


def test_ensemble_oof_predictions_rejects_fold_leakage():
    frame = _oof_frame()
    frame.loc[0, "held_fold"] = "wrong"
    with pytest.raises(ValueError, match="held fold"):
        ensemble_oof_predictions(
            frame,
            value_column="prediction",
            score_column="score",
            expected_training_seeds=[11, 12, 13],
        )


def test_cross_fitted_calibrators_are_finite_and_finalized():
    values = np.array([-2.0, -1.0, -0.2, 0.2, 1.0, 2.0])
    folds = np.array(["a", "a", "b", "b", "c", "c"])
    probability = cross_fit_probability_calibration(
        values, np.array([0, 0, 0, 1, 1, 1]), folds
    )
    utility = cross_fit_utility_calibration(values, values / 10.0, folds)
    assert probability.calibrator.method == "PLATT"
    assert utility.calibrator.method == "ISOTONIC"
    assert np.isfinite(probability.predictions).all()
    assert np.isfinite(utility.predictions).all()


def test_support_bounds_use_literal_finite_range():
    train = pd.DataFrame({"x": [0.0, 1.0], "y": [-2.0, 3.0]})
    bounds = fit_support_bounds(train, ["x", "y"])
    probe = pd.DataFrame({"x": [0.5, 2.0, np.nan], "y": [0.0, 0.0, 0.0]})
    assert support_mask(probe, bounds).tolist() == [True, False, False]


def test_policy_filters_then_ranks_and_does_not_try_second_choice():
    frame = pd.DataFrame([
        {
            "state_id": "s", "instance_id": "i", "scale": "S",
            "target_set_id": "a", "calibrated_probability": 0.9,
            "calibrated_utility": -0.01, "ensemble_raw_value": 2.0,
            "ensemble_raw_score": 1.0, "supported": True,
            "decoded_immediate_utility": 0.3,
            "fallback_target_set_id": "f", "fallback_decoded_utility": 0.0,
        },
        {
            "state_id": "s", "instance_id": "i", "scale": "S",
            "target_set_id": "b", "calibrated_probability": 0.8,
            "calibrated_utility": 0.2, "ensemble_raw_value": 1.0,
            "ensemble_raw_score": 2.0, "supported": True,
            "decoded_immediate_utility": 0.2,
            "fallback_target_set_id": "f", "fallback_decoded_utility": 0.0,
        },
    ])
    selected = select_immediate_actions(
        frame, probability_threshold=0.1, utility_threshold=0.0
    ).iloc[0]
    assert not selected.intervened
    assert selected.selected_target_set_id == "f"
    assert selected.best_forced_utility == pytest.approx(0.3)


def test_ranking_summary_uses_equal_weight_instance_means():
    frame = pd.DataFrame([
        {"state_id": "s1", "instance_id": "i1", "scale": "S", "CF_level": "CF1",
         "search_stage": "EARLY", "candidate_role": "ALNS_RELATED_FALLBACK",
         "target_set_id": "a", "ensemble_raw_value": 0.0,
         "decoded_immediate_utility": 0.0},
        {"state_id": "s1", "instance_id": "i1", "scale": "S", "CF_level": "CF1",
         "search_stage": "EARLY", "candidate_role": "FROZEN_NEURAL_TOP1",
         "target_set_id": "b", "ensemble_raw_value": 1.0,
         "decoded_immediate_utility": 1.0},
        {"state_id": "s2", "instance_id": "i2", "scale": "M", "CF_level": "CF2",
         "search_stage": "LATE", "candidate_role": "ALNS_RELATED_FALLBACK",
         "target_set_id": "a", "ensemble_raw_value": 1.0,
         "decoded_immediate_utility": 0.0},
        {"state_id": "s2", "instance_id": "i2", "scale": "M", "CF_level": "CF2",
         "search_stage": "LATE", "candidate_role": "FROZEN_NEURAL_TOP1",
         "target_set_id": "b", "ensemble_raw_value": 0.0,
         "decoded_immediate_utility": 1.0},
    ])
    states = state_ranking_metrics(frame)
    summary = summarize_ranking_metrics(states)
    assert states.set_index("state_id").loc["s1", "spearman"] == pytest.approx(1.0)
    assert states.set_index("state_id").loc["s2", "spearman"] == pytest.approx(-1.0)
    assert summary["overall_per_instance_spearman"] == pytest.approx(0.0)
