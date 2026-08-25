from __future__ import annotations

import json
from pathlib import Path

from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact
from rcias_clgri.learning.demonstrations import replay_demonstration
from rcias_clgri.nn import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "bc_validation" / "run_1"


def test_exact_demonstration_records_every_pre_action_graph(automotive_instance):
    exact = solve_tiny_exact(automotive_instance, time_limit_seconds=30.0)
    episode = replay_demonstration(automotive_instance, "EXACT", exact.actions)
    assert episode.feasible
    assert len(episode.steps) == len(automotive_instance.operations) == 6
    for step in episode.steps:
        action = step.action
        assert step.graph.operation_mask[action.operation_id]
        assert step.graph.island_masks[action.operation_id][action.island_id]
        assert action.w_agv_id in step.graph.w_masks[(action.operation_id, action.island_id)]
        assert action.f_agv_id in step.graph.f_masks[(action.operation_id, action.island_id)]


def test_bc_artifacts_meet_acceptance_criteria():
    info = json.loads((RUN / "final_info.json").read_text(encoding="utf-8"))
    demonstrations = json.loads((RUN / "demonstrations.json").read_text(encoding="utf-8"))
    assert info["bc_validated"]
    assert info["expert_action_accuracy"]["joint"] == 1.0
    assert info["rollout_feasibility"] == 1.0
    assert info["rollout"]["action_sequence_equal"]
    assert info["rollout"]["makespan"] == info["rollout"]["exact_makespan"] == 157.0
    assert {episode["source"] for episode in demonstrations["episodes"]} == {
        "EXACT", "H1", "H2", "H3"
    }
    assert len(demonstrations["episodes"]) == 8
    first_step = demonstrations["episodes"][0]["steps"][0]
    assert {"node_features", "edges", "masks", "candidate_features"} <= set(
        first_step["graph_state"]
    )
    assert (RUN / "Figure_1_bc_loss.png").stat().st_size > 30_000
    assert (RUN / "Figure_2_bc_accuracy.png").stat().st_size > 30_000


def test_default_neural_configuration_matches_phase_specification():
    config = ModelConfig()
    assert config.embedding_dim == 128
    assert config.heads == 4
    assert config.layers in {2, 3}
