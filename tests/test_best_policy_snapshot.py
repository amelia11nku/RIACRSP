import torch
from torch import nn

from rcias_clgri.learning.best_policy import BestPolicyReference


def test_best_policy_snapshot_updates_only_for_better_score():
    model = nn.Linear(2, 1, bias=False)
    reference = BestPolicyReference(model, score=1.0)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not reference.update_if_better(model, 1.1)
    reference.restore(model)
    torch.testing.assert_close(model.weight, original)
    with torch.no_grad():
        model.weight.add_(2.0)
    improved = model.weight.detach().clone()
    assert reference.update_if_better(model, 0.9)
    with torch.no_grad():
        model.weight.zero_()
    reference.restore(model)
    torch.testing.assert_close(model.weight, improved)

