from __future__ import annotations

import pytest

from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.learning.reward import horizon_scale, telescoping_makespan_reward


def test_reward_telescopes_to_final_makespan(automotive_instance):
    expert = solve_dispatching(automotive_instance, "H1")
    env = RCIASConstructionEnv(automotive_instance)
    scale = horizon_scale(automotive_instance)
    previous = 0.0
    rewards = []
    for action in expert.actions:
        env.step(action)
        current = env.objective().makespan
        rewards.append(telescoping_makespan_reward(previous, current, scale))
        previous = current
    assert sum(rewards) == pytest.approx(-env.objective().makespan / scale, abs=1e-12)
