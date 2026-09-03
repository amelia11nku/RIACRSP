import torch

from rcias_clgri.ni.phase6i_heads import (
    Phase6IObjective,
    build_phase6i_head,
    parameter_count,
    phase6i_state_loss,
)


def test_phase6i_head_shapes_and_parameter_caps():
    embeddings = torch.randn(3, 4, 128)
    context = torch.randn(3, 4, 19)
    u1 = build_phase6i_head("U1")
    u2 = build_phase6i_head("U2")
    assert u1(embeddings).shape == (3, 4)
    assert u2(embeddings, context).shape == (3, 4)
    assert parameter_count(u1) == 16_641
    assert parameter_count(u2) == 25_729


def test_phase6i_loss_prefers_correct_ranking_and_handles_padding():
    raw = torch.tensor([[0.03, 0.01, -0.02], [0.02, -0.01, 0.0]])
    normalized = raw / 0.1
    positive = raw.gt(0)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    objective = Phase6IObjective(
        pair_gap_scale=0.01,
        listwise_temperature=0.01,
        positive_class_weight=2.0,
    )
    correct = phase6i_state_loss(
        normalized, raw, normalized, positive, mask, objective
    )
    reversed_loss = phase6i_state_loss(
        -normalized, raw, normalized, positive, mask, objective
    )
    assert torch.isfinite(correct["loss"])
    assert correct["pair_count"].item() == 4
    assert correct["loss"] < reversed_loss["loss"]


def test_u2_requires_context():
    model = build_phase6i_head("U2")
    try:
        model(torch.randn(2, 128))
    except ValueError as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("U2 accepted a missing context tensor")
