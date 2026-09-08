import pandas as pd
import torch

from rcias_clgri.ni.phase6o_top_utility import TopUtilityCSGModel, top_utility_loss
from scripts import train_phase6o_top_utility as training


def test_phase6o_top_utility_loss_filters_near_tie_pairs_and_is_finite():
    prediction = torch.tensor([0.4, 0.1, -0.2, 0.3, 0.0], requires_grad=True)
    result = top_utility_loss(
        prediction,
        torch.zeros(5),
        torch.tensor([0.020, 0.018, -0.010, 0.006, -0.020]),
        torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]),
        torch.tensor([0, 3, 5]),
        torch.tensor([0.5, 1.5]),
        noise_margin=0.005,
        policy_temperature=0.01,
        utility_scale=0.02,
        regret_scale=0.01,
        huber_delta=0.01,
    )
    assert torch.isfinite(result["loss"])
    assert int(result["near_count"]) == 3
    assert int(result["pair_count"]) == 3
    result["loss"].backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_phase6o_top_set_term_is_symmetric_inside_near_set():
    common = dict(
        beats_fallback_logit=torch.zeros(3),
        advantage_target=torch.tensor([0.020, 0.020, -0.010]),
        beats_fallback_target=torch.tensor([1.0, 1.0, 0.0]),
        action_ptr=torch.tensor([0, 3]),
        state_weights=torch.ones(1),
        noise_margin=0.005,
        policy_temperature=0.01,
        utility_scale=0.02,
        regret_scale=0.01,
        huber_delta=0.01,
    )
    left = top_utility_loss(torch.tensor([0.7, 0.2, -0.1]), **common)
    right = top_utility_loss(torch.tensor([0.2, 0.7, -0.1]), **common)
    assert torch.equal(left["top_set_cross_entropy"], right["top_set_cross_entropy"])
    assert torch.equal(left["regret_logistic"], right["regret_logistic"])


def test_phase6o_model_freezes_the_complete_phase6f_encoder():
    checkpoint = torch.load(
        training.ROOT
        / "outputs/phase6f/training/final_seeds/seed_660301/checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    base = training.phase6n.CSGTargetSetScorer(
        training.phase6n.CSGTensorizer(),
        training.phase6n.NIModelConfig(**checkpoint["model_config"]),
    )
    base.load_state_dict(checkpoint["model_state"])
    model = TopUtilityCSGModel(base, (25, 8, 6)).train()
    assert not model.state_encoder.training
    assert not any(parameter.requires_grad for parameter in model.state_encoder.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("state_encoder.")
    )


def test_phase6o_fold_contract_balances_sources_and_filters_noise_pairs():
    frame = training.attach_candidate_noise(
        pd.read_parquet(training.SOURCE),
        pd.read_parquet(training.RAW),
    )
    contract = training.fit_fold_contract(frame[frame.oof_fold.ne(0)])
    assert contract.state_weights.keys() == set(frame.loc[frame.oof_fold.ne(0), "state_id"])
    assert all(abs(value - 0.5) < 1e-12 for value in contract.source_weight_totals.values())
    assert contract.noise_margin >= 0.0025
    assert contract.retained_pair_count > 0
