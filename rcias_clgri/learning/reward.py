"""Objective-consistent dense reward for constructive scheduling."""

from __future__ import annotations

from rcias_clgri.data.instance import Instance


def horizon_scale(instance: Instance) -> float:
    """Return the frozen deterministic normalization scale for one instance."""

    return float(max(instance.horizon, 1.0))


def telescoping_makespan_reward(
    previous_partial_makespan: float,
    next_partial_makespan: float,
    scale: float,
) -> float:
    if scale <= 0:
        raise ValueError("reward normalization scale must be positive")
    if next_partial_makespan + 1e-12 < previous_partial_makespan:
        raise ValueError("partial makespan cannot decrease during construction")
    return -(
        float(next_partial_makespan) - float(previous_partial_makespan)
    ) / float(scale)
