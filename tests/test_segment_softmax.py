from __future__ import annotations

import torch

from rcias_clgri.nn.rt_hgt import RTHGTLayer


def _reference(scores, targets):
    result = torch.empty_like(scores)
    for target in torch.unique(targets):
        mask = targets == target
        result[mask] = torch.softmax(scores[mask], dim=0)
    return result


def test_segment_softmax_matches_reference():
    torch.manual_seed(3)
    scores = torch.randn(17, 4, dtype=torch.float64)
    targets = torch.tensor([0, 2, 0, 1, 2, 4, 4, 1, 3, 2, 0, 3, 1, 4, 3, 2, 0])
    actual = RTHGTLayer._segment_softmax(scores, targets)
    assert torch.allclose(actual, _reference(scores, targets), atol=1e-12, rtol=1e-12)


def test_segment_softmax_sum_to_one():
    scores = torch.randn(12, 3)
    targets = torch.tensor([0, 0, 1, 2, 1, 2, 2, 3, 3, 3, 1, 0])
    weights = RTHGTLayer._segment_softmax(scores, targets)
    for target in torch.unique(targets):
        assert torch.allclose(weights[targets == target].sum(dim=0), torch.ones(3))


def test_segment_softmax_gradient():
    scores = torch.randn(9, 2, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3])
    weights = RTHGTLayer._segment_softmax(scores, targets)
    loss = (weights * torch.arange(18, dtype=torch.float64).reshape(9, 2)).sum()
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0.0


def test_segment_softmax_no_nan():
    scores = torch.tensor([[1e20, -1e20], [1e20, -1e20], [-1e20, 1e20]])
    targets = torch.tensor([0, 0, 1])
    weights = RTHGTLayer._segment_softmax(scores, targets)
    assert torch.isfinite(weights).all()
    assert torch.equal(weights[2], torch.ones(2))
