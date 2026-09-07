import numpy as np
import pandas as pd
import torch

from rcias_clgri.ni.phase6m_selective_risk import (
    FORBIDDEN_ONLINE_COLUMNS,
    SelectiveRiskMLP,
    build_selector_feature_frame,
    fit_selector_transform,
    fit_support_transform,
    selective_risk_loss,
    support_features,
    transform_selector_features,
)
from scripts.train_phase6m_selective_confidence import inner_cross_fit_pairs


def source_frame():
    rows = []
    for state, scale in (("s1", "S"), ("s2", "M")):
        for index in range(3):
            rows.append({
                "state_id": state, "target_set_id": f"{state}-t{index}", "instance_id": state,
                "oof_fold": 0, "CF_level": "CF1", "search_stage": "0-20%", "scale": scale,
                "primary_origin_rule": "operator_related" if index == 0 else "near_low_slack",
                "origin_destroy_operator": "related", "origin_family": "ORIGINAL_OPERATOR",
                "origin_rule_count": 1, "origin_family_count": 1,
                "destroy_target_cardinality": index + 1, "destroy_target_fraction": (index + 1) / 10,
                "fallback_overlap_fraction": 1.0 if index == 0 else 0.5,
                "fallback_jaccard": 1.0 if index == 0 else 0.5,
                "critical_overlap_fraction": index / 3, "bottleneck_overlap_fraction": index / 4,
                "normalized_diversity_rank": index / 2, "is_fallback": index == 0,
                "target_progress": 0.5, "search_progress": 0.25, "bottleneck_proxy": 0.2,
                "fallback_target_set_id": f"{state}-t0", "requested_bank_count": 24,
                "full_bank_unique_count": 3,
            })
    return pd.DataFrame(rows)


def ranker_predictions():
    source = source_frame()
    rows = []
    for seed, shift in ((1, 0.0), (2, 0.1), (3, -0.1)):
        current = source.copy()
        current["training_seed"] = seed
        current["predicted_continuation_advantage"] = np.tile([0.0, 0.5, 0.25], 2) + shift
        current["predicted_beats_fallback_logit"] = current.predicted_continuation_advantage
        current["predicted_beats_fallback_probability_raw"] = 1 / (
            1 + np.exp(-current.predicted_beats_fallback_logit)
        )
        current["predicted_immediate_utility"] = 0.01 + shift
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def test_support_binary_fallback_is_not_a_continuous_outlier():
    frame = source_frame()
    fit = frame[~frame.is_fallback]
    transform = fit_support_transform(fit)
    held = fit.iloc[[0]].copy()
    without_flag = support_features(held, transform)
    held["is_fallback"] = True
    with_flag = support_features(held, transform)
    pd.testing.assert_frame_equal(without_flag, with_flag)


def test_fine_oov_is_allowed_but_high_level_oov_fails_closed():
    frame = source_frame()
    transform = fit_support_transform(frame)
    held = frame.iloc[:2].copy()
    held.loc[held.index[0], "primary_origin_rule"] = "new_fine_rule"
    held.loc[held.index[1], "origin_family"] = "NEW_FAMILY"
    result = support_features(held, transform)
    assert result.iloc[0].fine_rule_oov == 1
    assert bool(result.iloc[0].hard_supported)
    assert not bool(result.iloc[1].hard_supported)


def test_selector_features_have_exact_margins_and_no_outcome_fields():
    predictions = ranker_predictions()
    transform = fit_support_transform(source_frame())
    features = build_selector_feature_frame(predictions, transform)
    winner = features[features.target_set_id.eq("s1-t1")].iloc[0]
    assert np.isclose(winner.candidate_fallback_advantage_margin, 0.5)
    assert np.isclose(winner.candidate_best_other_advantage_margin, 0.25)
    assert np.isclose(winner.seed_top1_vote_fraction, 1.0)
    assert not (set(FORBIDDEN_ONLINE_COLUMNS) & set(features))
    selector_transform = fit_selector_transform(features)
    matrix = transform_selector_features(features, selector_transform)
    assert matrix.shape == (6, len(selector_transform.feature_names))
    assert np.isfinite(matrix).all()
    repeated = build_selector_feature_frame(predictions, transform)
    pd.testing.assert_frame_equal(features, repeated)


def test_selective_risk_model_and_state_balanced_loss_are_finite():
    model = SelectiveRiskMLP(7, dropout=0.0)
    output = model(torch.zeros((5, 7)))
    outcomes = torch.tensor([[0.1, 0.2], [-0.1, 0.0], [0.2, 0.3], [0.0, 0.1], [-0.2, -0.1]])
    loss = selective_risk_loss(output, outcomes, torch.tensor([0, 2, 5]))
    assert set(loss) == {"loss", "gaussian_nll", "seed_positive_bce", "candidate_mean_huber"}
    assert all(torch.isfinite(value) for value in loss.values())
    loss["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_inner_cross_fit_excludes_outer_and_prediction_fold():
    for held in range(3):
        pairs = inner_cross_fit_pairs(held)
        assert len(pairs) == 2
        assert {prediction for _, prediction in pairs} == ({0, 1, 2} - {held})
        for training, prediction in pairs:
            assert held not in (training, prediction)
            assert training != prediction


def test_full_bank_and_canonical_fallback_fail_closed():
    predictions = ranker_predictions()
    transform = fit_support_transform(source_frame())
    damaged = predictions.copy()
    damaged.loc[damaged.state_id.eq("s1"), "requested_bank_count"] = 8
    try:
        build_selector_feature_frame(damaged, transform)
    except ValueError as error:
        assert "24-rule" in str(error)
    else:
        raise AssertionError("reduced bank was accepted")
    damaged = predictions.copy()
    damaged.loc[damaged.state_id.eq("s1"), "fallback_target_set_id"] = "wrong"
    try:
        build_selector_feature_frame(damaged, transform)
    except ValueError as error:
        assert "fallback identity" in str(error)
    else:
        raise AssertionError("fallback drift was accepted")
