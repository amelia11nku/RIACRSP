from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rcias_clgri.data.loader import load_instance_dict
from rcias_clgri.data.validator import InstanceValidationError
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.rcias_env import RCIASConstructionEnv

ROOT = Path(__file__).resolve().parents[1]


def test_nonlinear_dag_and_ready_set(automotive_instance):
    product = automotive_instance.product_data["J1"]
    assert ("o11", "o12") in product.precedence
    assert ("o11", "o13") in product.precedence
    assert "o12" not in automotive_instance.transitive_predecessors["o13"]
    env = RCIASConstructionEnv(automotive_instance)
    assert set(env.get_ready_operations()) == {"o11", "o21", "o22"}
    env.step(Action("o11", "M1", "W1", "F1"))
    assert {"o12", "o13"} <= set(env.get_ready_operations())


def test_nonready_action_is_hard_rejected(automotive_instance):
    env = RCIASConstructionEnv(automotive_instance)
    with pytest.raises(ValueError, match="not topologically ready"):
        env.step(Action("o12", "M1", "W1", "F1"))


def test_cycle_is_rejected():
    raw = json.loads((ROOT / "instances/tiny/tiny_01.json").read_text(encoding="utf-8"))
    broken = copy.deepcopy(raw)
    broken["products"]["J1"]["precedence"].append(["o12", "o11"])
    with pytest.raises(InstanceValidationError, match="DAG"):
        load_instance_dict(broken)


def test_realized_order_is_topological(controlled_env):
    instance = controlled_env.instance
    for product_id, sequence in controlled_env.schedule.product_sequences.items():
        position = {op_id: index for index, op_id in enumerate(sequence)}
        assert all(position[source] < position[target] for source, target in instance.product_data[product_id].precedence)
