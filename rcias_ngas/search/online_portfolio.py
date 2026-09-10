"""Independent segmented online portfolio for NGAS joint actions."""
from __future__ import annotations

from collections import defaultdict
import math


class OnlinePortfolio:
    def __init__(self, segment_length: int = 8, reaction: float = .25,
                 strength: float = .7, factor_floor: float = .2,
                 factor_ceiling: float = 5.) -> None:
        if segment_length < 1 or not 0 < reaction <= 1:
            raise ValueError('Invalid online portfolio configuration')
        self.segment_length = segment_length
        self.reaction = reaction
        self.strength = strength
        self.factor_floor = factor_floor
        self.factor_ceiling = factor_ceiling
        self.estimates = defaultdict(lambda: 1.)
        self.pending = defaultdict(list)
        self.outcome_counts = defaultdict(int)
        self.observations = 0
        self.segment_updates = 0

    @staticmethod
    def keys(action) -> tuple[str, ...]:
        families = action.target.origin_families or ('unknown',)
        return tuple(sorted({
            f'size:{action.size}', f'repair:{action.repair}',
            f'size_repair:{action.size}|{action.repair}',
            *(f'family:{family}' for family in families),
            *(f'family_repair:{family}|{action.repair}' for family in families),
        }))

    def factor(self, action) -> float:
        log_factor = sum(self.strength * (self.estimates[key] - 1.)
                         for key in self.keys(action)) / len(self.keys(action))
        return max(self.factor_floor, min(self.factor_ceiling, math.exp(log_factor)))

    def observe(self, action, outcome: str, relative_improvement: float) -> float:
        if outcome == 'rejected':
            reward = 0.
        elif outcome == 'accepted_worse_or_neutral':
            reward = .25
        elif outcome == 'accepted_current_improvement':
            reward = 1.5 + min(2., max(0., relative_improvement) / .01)
        elif outcome == 'new_global_best':
            reward = 3. + min(2., max(0., relative_improvement) / .01)
        else:
            raise ValueError(f'Unknown portfolio outcome: {outcome}')
        for key in self.keys(action):
            self.pending[key].append(reward)
        self.outcome_counts[outcome] += 1
        self.observations += 1
        if self.observations % self.segment_length == 0:
            self.flush()
        return reward

    def flush(self) -> None:
        if not self.pending:
            return
        for key, values in self.pending.items():
            mean = sum(values) / len(values)
            self.estimates[key] = ((1. - self.reaction) * self.estimates[key]
                                   + self.reaction * mean)
        self.pending.clear()
        self.segment_updates += 1

    def snapshot(self) -> dict:
        return {
            'observations': self.observations,
            'segment_updates': self.segment_updates,
            'outcome_counts': dict(sorted(self.outcome_counts.items())),
            'estimates': dict(sorted(self.estimates.items())),
            'pending_observations': {key: len(value) for key, value in sorted(self.pending.items())},
        }
