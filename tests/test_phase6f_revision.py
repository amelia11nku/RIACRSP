import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from rcias_clgri.ni.calibration import (
    calibration_metrics,
    fit_probability_calibrator,
    fit_utility_calibrator,
)
from rcias_clgri.ni.selective_policy import threshold_study
from rcias_clgri.ni.utility_losses import Phase6FLossConfig, phase6f_loss
from scripts.generate_phase6f_revision_holdout import priority
from scripts.generate_phase6f_revision_labels import phase6c_protocol
from scripts.run_phase6c_dataset import shard_directory


ROOT = Path(__file__).resolve().parents[1]


def test_phase6f_protocol_is_fresh_offline_and_preregistered():
    config = json.loads((ROOT / "configs/phase6f_revision.json").read_text())
    holdout = config["revision_holdout"]
    assert holdout["instance_count"] == 81
    assert holdout["structural_cell_count"] == 81
    assert holdout["states_per_structural_cell"] == 100
    assert holdout["state_count"] == 8100
    assert config["frozen_boundaries"]["phase6e_internal_holdout_selection_use"] == "FORBIDDEN"
    assert len(config["objective_candidates"]) == 3
    assert len(config["compact_model_candidates"]) == 3
    assert config["success_gates"]["model_decision_p90_ms_hard"] == 150.0
    assert config["success_gates"]["minimum_intervention_coverage"] >= 0.2


def test_phase6f_label_protocol_reuses_phase6c_semantics_with_new_namespaces():
    config = json.loads((ROOT / "configs/phase6f_revision.json").read_text())
    protocol = phase6c_protocol(config, {"freeze_hash": "state-freeze"})
    assert protocol["label_generation_version"] == "phase6c-v1"
    assert protocol["destroy_fraction"] == 0.15
    assert protocol["repair_operator"] == "transport_aware"
    assert protocol["repair_seed_count"] == 3
    assert protocol["production_freeze_hash"] == "state-freeze"
    assert protocol["seed_namespaces"] == {
        "arm_generation": 669300000,
        "repair": 669400000,
    }


def test_revision_split_path_and_state_priority_are_deterministic():
    root = Path("/tmp/phase6f-label-test")
    assert shard_directory(root, "REVISION_HOLDOUT", "example") == (
        root / "revision_holdout" / "example"
    )
    assert priority("state-1", 669200000) == priority("state-1", 669200000)
    assert priority("state-1", 669200000) != priority("state-2", 669200000)


def test_regret_weighted_multitask_loss_is_finite_and_differentiable():
    scores = torch.tensor([0.2, -0.1, 0.3, -0.2], requires_grad=True)
    predicted_utility = torch.tensor([0.1, -0.1, 0.2, -0.2], requires_grad=True)
    batch = SimpleNamespace(
        rank_better_index=torch.tensor([0, 2]),
        rank_worse_index=torch.tensor([1, 3]),
        utility=torch.tensor([0.3, -0.2, 0.1, -0.4]),
        positive=torch.tensor([True, False, True, False]),
        action_ptr=torch.tensor([0, 2, 4]),
    )
    config = Phase6FLossConfig(
        objective="O3_UTILITY_AWARE_MULTITASK",
        utility_weight=0.25,
        rank_gap_scale=0.1,
        utility_clip=0.4,
    )
    losses = phase6f_loss(scores, predicted_utility, batch, config)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["utility_loss"] > 0
    assert scores.grad is not None and predicted_utility.grad is not None


def test_calibration_and_selective_policy_are_validation_deterministic():
    score = np.array([-2.0, -1.0, 1.0, 2.0])
    positive = np.array([0, 0, 1, 1])
    probability_model = fit_probability_calibrator(score, positive, "PLATT")
    probability = probability_model.predict(score)
    assert np.all(np.diff(probability) > 0)
    assert calibration_metrics(probability, positive)["brier_score"] < 0.25
    utility_model = fit_utility_calibrator(score, [-0.2, -0.1, 0.1, 0.2])
    assert np.all(np.diff(utility_model.predict(score)) >= 0)

    rows = []
    for state, neural_utility, fallback_utility in (("s1", 0.3, 0.0), ("s2", 0.2, -0.1)):
        rows.extend([
            {
                "state_id": state, "target_set_id": f"{state}-neural",
                "score": 2.0, "calibrated_probability": 0.9,
                "calibrated_utility": neural_utility,
                "mean_relative_improvement": neural_utility, "regret_to_best": 0.0,
                "arm_family": "MATCHED_RANDOM", "origin_destroy_operator": "random",
                "origin_rules": '["matched_random_1"]',
            },
            {
                "state_id": state, "target_set_id": f"{state}-related",
                "score": 0.0, "calibrated_probability": 0.5,
                "calibrated_utility": fallback_utility,
                "mean_relative_improvement": fallback_utility, "regret_to_best": 0.2,
                "arm_family": "ORIGINAL_OPERATOR", "origin_destroy_operator": "related",
                "origin_rules": '["operator_related"]',
            },
        ])
    study, selected = threshold_study(pd.DataFrame(rows), minimum_coverage=0.2)
    chosen = study[study["selected"]].iloc[0]
    assert selected.confidence <= 0.9
    assert chosen["coverage"] >= 0.2
    assert chosen["incremental_utility_vs_fallback"] > 0
