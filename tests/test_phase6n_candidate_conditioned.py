import inspect
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import pandas as pd
import torch

from rcias_clgri.ni.phase6n_candidate_conditioned import (
    BOUNDARY_FAMILIES,
    FAMILY,
    CandidateConditionedCSGModel,
    _unique_boundary_values,
)
from scripts import train_phase6n_candidate_conditioned as training


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
)
CACHE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/tensor_cache"


@cache
def _fixture():
    frame = pd.read_parquet(SOURCE)
    samples = training.load_samples()
    frames = training.state_frames(frame)
    state_id = sorted(samples)[0]
    transform = training.fit_feature_transform(frame)
    packed = training.build_batch(
        [state_id], samples, frames, transform, torch.device("cpu")
    )
    protocol_stub = {
        "base_checkpoint": {
            "path": "outputs/phase6f/training/final_seeds/seed_660301/checkpoint_best.pt",
            "sha256": training.digest(
                ROOT / "outputs/phase6f/training/final_seeds/seed_660301/checkpoint_best.pt"
            ),
        },
        "model": {
            "total_parameters": 5766462,
            "trainable_parameters": 3060926,
        },
    }
    model = training.initialize_model(
        726101, transform, protocol_stub, torch.device("cpu")
    )
    return frame, samples, frames, packed, model


def test_phase6n_combined_tensor_cache_is_complete_and_locked():
    integrity = json.loads((CACHE / "tensor_cache_integrity.json").read_text())
    assert integrity["status"] == "PASS"
    assert all(integrity["checks"].values())
    assert integrity["states"] == 864
    assert integrity["actions"] == 20441
    assert integrity["historical_score_calls"] == 0
    assert integrity["r13_accessed"] is False
    assert integrity["r14_accessed"] is False


def test_phase6n_model_has_exact_frozen_and_trainable_boundary():
    _, _, _, _, model = _fixture()
    assert isinstance(model, CandidateConditionedCSGModel)
    assert model.family == FAMILY
    assert model.parameter_counts() == (5766462, 3060926)
    assert not any(
        parameter.requires_grad
        for parameter in model.state_encoder.input_projection.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.state_encoder.layers[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.state_encoder.layers[1].parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.state_encoder.graph_projection.parameters()
    )
    assert not hasattr(model, "immediate_utility_head")
    assert not hasattr(model, "scale_head")
    assert not hasattr(model, "selector")


def test_phase6n_candidate_pooling_is_finite_and_deterministic():
    _, _, _, packed, model = _fixture()
    model.eval()
    with torch.inference_mode():
        first = training.forward_model(model, packed)
        second = training.forward_model(model, packed)
    assert first.advantage.shape == (packed["batch"].action_count,)
    assert first.candidate_embeddings.shape == (packed["batch"].action_count, 128)
    assert torch.isfinite(first.advantage).all()
    assert torch.equal(first.advantage, second.advantage)
    assert torch.equal(first.beats_fallback_logit, second.beats_fallback_logit)
    assert torch.equal(first.candidate_embeddings, second.candidate_embeddings)


def test_phase6n_relation_boundary_memberships_are_unique_and_candidate_indexed():
    _, _, _, packed, model = _fixture()
    batch = packed["batch"]
    offset = 0
    node_hidden = {}
    for node_type, features in batch.node_features.items():
        identities = torch.arange(offset, offset + len(features), dtype=torch.float32)
        node_hidden[node_type] = identities[:, None].repeat(1, 128)
        offset += len(features)
    membership = torch.zeros(
        (batch.action_count, len(node_hidden["OP"])), dtype=torch.bool
    )
    membership[batch.target_action_index, batch.target_operation_indices] = True
    values, actions = _unique_boundary_values(
        node_hidden,
        batch,
        membership,
        BOUNDARY_FAMILIES["precedence"],
        "both",
        require_binding=False,
    )
    repeated_values, repeated_actions = _unique_boundary_values(
        node_hidden,
        batch,
        membership,
        BOUNDARY_FAMILIES["precedence"],
        "both",
        require_binding=False,
    )
    assert len(values) > 0
    assert len(values) == len(actions)
    assert actions.min() >= 0 and actions.max() < batch.action_count
    pairs = list(zip(actions.tolist(), values[:, 0].tolist()))
    assert len(pairs) == len(set(pairs))
    assert torch.equal(values, repeated_values)
    assert torch.equal(actions, repeated_actions)


def test_phase6n_feature_and_forward_contract_has_no_outcome_or_legacy_score_input():
    signature = inspect.signature(CandidateConditionedCSGModel.forward)
    assert set(signature.parameters) == {
        "self",
        "batch",
        "fallback_action_indices",
        "categorical",
        "numeric",
        "critical_operation_mask",
        "bottleneck_operation_mask",
    }
    assert tuple(training.NUMERIC_COLUMNS) == (
        "origin_rule_count",
        "origin_family_count",
        "destroy_target_cardinality",
        "destroy_target_fraction",
        "fallback_overlap_fraction",
        "fallback_jaccard",
        "critical_overlap_fraction",
        "bottleneck_overlap_fraction",
        "normalized_diversity_rank",
        "is_fallback",
    )
    source = Path(inspect.getsourcefile(CandidateConditionedCSGModel)).read_text()
    assert "FrozenLiveInference" not in source
    assert "phase6l_legacy_score" not in source


def test_phase6n_predictions_do_not_read_outcome_tensors_from_batch():
    _, _, _, packed, model = _fixture()
    model.eval()
    with torch.inference_mode():
        reference = training.forward_model(model, packed)
        perturbed = dict(packed)
        perturbed["batch"] = replace(
            packed["batch"],
            utility=torch.randn_like(packed["batch"].utility) * 1000,
            positive=1 - packed["batch"].positive,
            rank_within_state=torch.randn_like(packed["batch"].rank_within_state),
            rank_percentile=torch.randn_like(packed["batch"].rank_percentile),
            regret_to_best=torch.randn_like(packed["batch"].regret_to_best),
            top1=~packed["batch"].top1,
            top3=~packed["batch"].top3,
        )
        observed = training.forward_model(model, perturbed)
    assert torch.equal(reference.advantage, observed.advantage)
    assert torch.equal(reference.beats_fallback_logit, observed.beats_fallback_logit)


def test_phase6n_objective_scales_and_whole_instance_roles_are_reproducible():
    frame = pd.read_parquet(SOURCE)
    first = training.fit_objective_scales(frame[frame.oof_fold.eq(0)])
    second = training.fit_objective_scales(frame[frame.oof_fold.eq(0)])
    assert first == second
    assert first.pair_gap_scale > 0
    assert 0.005 <= first.huber_delta <= 0.05
    assert [training.nested_fold_roles(fold) for fold in range(3)] == [
        (2, 1),
        (0, 2),
        (1, 0),
    ]
    for held_fold in range(3):
        held = set(frame.loc[frame.oof_fold.eq(held_fold), "instance_id"])
        train = set(frame.loc[~frame.oof_fold.eq(held_fold), "instance_id"])
        assert held.isdisjoint(train)
        assert len(held) == 6
        assert len(train) == 12
