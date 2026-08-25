from __future__ import annotations

from pathlib import Path

import pytest

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.rcias_env import RCIASConstructionEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def automotive_instance():
    return load_instance(ROOT / "instances" / "tiny" / "tiny_01.json")


@pytest.fixture()
def fjsp_instance():
    return load_instance(ROOT / "instances" / "tiny" / "tiny_02.json")


@pytest.fixture()
def controlled_env(automotive_instance):
    """Schedule containing config changes, same-config, same/cross-island W cases."""

    env = RCIASConstructionEnv(automotive_instance)
    decisions = [
        ("o11", "M1"),
        ("o12", "M1"),
        ("o13", "M3"),
        ("o21", "M2"),
        ("o22", "M2"),
        ("o23", "M1"),
    ]
    for op_id, island_id in decisions:
        w_agv_id = env.get_feasible_w_agvs(op_id, island_id)[0]
        env.step(Action(op_id, island_id, w_agv_id, "F1"))
    return env
