"""Best-policy snapshots and bounded rollback control for Phase 5B."""

from __future__ import annotations

from dataclasses import dataclass
import copy

import torch
from torch import nn


class BestPolicyReference:
    def __init__(self, model: nn.Module, score: float) -> None:
        self.score = float(score)
        self._state = self._clone_state(model)

    @staticmethod
    def _clone_state(model: nn.Module):
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    def update_if_better(self, model: nn.Module, score: float) -> bool:
        if float(score) >= self.score:
            return False
        self.score = float(score)
        self._state = self._clone_state(model)
        return True

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(copy.deepcopy(self._state))


@dataclass
class RollbackController:
    patience: int = 3
    relative_regression: float = 0.04
    learning_rate_factor: float = 0.5
    maximum_rollbacks: int = 2
    consecutive_regressions: int = 0
    rollback_count: int = 0

    def observe(self, score: float, best_score: float) -> bool:
        threshold = float(best_score) * (1.0 + self.relative_regression)
        self.consecutive_regressions = (
            self.consecutive_regressions + 1 if float(score) > threshold else 0
        )
        return self.consecutive_regressions >= self.patience

    def rollback(self, reference: BestPolicyReference, model: nn.Module, trainer) -> bool:
        if self.rollback_count >= self.maximum_rollbacks:
            return False
        reference.restore(model)
        trainer.reduce_learning_rates(self.learning_rate_factor, reset_optimizer_state=True)
        self.rollback_count += 1
        self.consecutive_regressions = 0
        return True

