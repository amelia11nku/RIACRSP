import torch
from torch import nn

from rcias_clgri.learning.best_policy import BestPolicyReference, RollbackController


class _Trainer:
    def __init__(self):
        self.calls = []

    def reduce_learning_rates(self, factor, *, reset_optimizer_state):
        self.calls.append((factor, reset_optimizer_state))


def test_rollback_restores_parameters_after_three_regressions():
    model = nn.Linear(2, 1, bias=False)
    reference = BestPolicyReference(model, score=1.0)
    original = model.weight.detach().clone()
    controller = RollbackController(patience=3, relative_regression=0.04)
    assert not controller.observe(1.05, reference.score)
    assert not controller.observe(1.06, reference.score)
    assert controller.observe(1.07, reference.score)
    with torch.no_grad():
        model.weight.add_(3.0)
    trainer = _Trainer()
    assert controller.rollback(reference, model, trainer)
    torch.testing.assert_close(model.weight, original)
    assert controller.rollback_count == 1


def test_lr_reduction_after_rollback_resets_optimizer_state():
    model = nn.Linear(2, 1)
    reference = BestPolicyReference(model, score=1.0)
    controller = RollbackController(learning_rate_factor=0.5)
    trainer = _Trainer()
    assert controller.rollback(reference, model, trainer)
    assert trainer.calls == [(0.5, True)]

