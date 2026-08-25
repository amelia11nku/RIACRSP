"""Validation-gated curriculum with replay of earlier levels."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Mapping


@dataclass
class CurriculumState:
    current_level_index: int = 0
    validation_history: dict[str, list[float]] = field(
        default_factory=lambda: {"S": [], "M": [], "L": []}
    )
    updates_at_level: int = 0


class CurriculumManager:
    """Promote only after feasible, non-collapsed validation has stabilized."""

    LEVELS = ("S", "M", "L")

    def __init__(
        self,
        *,
        plateau_window: int = 3,
        plateau_relative_improvement: float = 0.03,
        minimum_updates: int = 3,
        current_level_probability: float = 0.75,
        minimum_normalized_entropy: float = 0.02,
        state: CurriculumState | None = None,
    ) -> None:
        if plateau_window < 2:
            raise ValueError("plateau_window must be at least two")
        if not 0.5 <= current_level_probability <= 1.0:
            raise ValueError("current_level_probability must be in [0.5, 1]")
        self.plateau_window = plateau_window
        self.plateau_relative_improvement = plateau_relative_improvement
        self.minimum_updates = minimum_updates
        self.current_level_probability = current_level_probability
        self.minimum_normalized_entropy = minimum_normalized_entropy
        self.state = state or CurriculumState()

    @property
    def current_level(self) -> str:
        return self.LEVELS[self.state.current_level_index]

    def sampling_probabilities(self) -> Mapping[str, float]:
        index = self.state.current_level_index
        if index == 0:
            return {"S": 1.0}
        probabilities = {self.current_level: self.current_level_probability}
        replay = (1.0 - self.current_level_probability) / index
        for old_level in self.LEVELS[:index]:
            probabilities[old_level] = replay
        return probabilities

    def sample_level(self, rng: random.Random) -> str:
        probabilities = self.sampling_probabilities()
        draw = rng.random()
        cumulative = 0.0
        for level, probability in probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return level
        return self.current_level

    def record_validation(
        self,
        makespan: float,
        *,
        feasibility_rate: float,
        normalized_entropy: float,
    ) -> bool:
        level = self.current_level
        history = self.state.validation_history[level]
        history.append(float(makespan))
        self.state.updates_at_level += 1
        if level == "L" or len(history) < self.plateau_window:
            return False
        window = history[-self.plateau_window:]
        scale = max(abs(window[0]), 1e-12)
        relative_improvement = (window[0] - min(window)) / scale
        stable = relative_improvement <= self.plateau_relative_improvement
        eligible = (
            self.state.updates_at_level >= self.minimum_updates
            and feasibility_rate == 1.0
            and normalized_entropy >= self.minimum_normalized_entropy
            and stable
        )
        if eligible:
            self.state.current_level_index += 1
            self.state.updates_at_level = 0
        return eligible

    def to_dict(self) -> dict[str, object]:
        return {
            "current_level": self.current_level,
            "current_level_index": self.state.current_level_index,
            "updates_at_level": self.state.updates_at_level,
            "validation_history": self.state.validation_history,
            "sampling_probabilities": dict(self.sampling_probabilities()),
        }
