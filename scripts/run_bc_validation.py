#!/usr/bin/env python3
"""Overfit the autoregressive policy to a proven-optimal Tiny trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
from time import perf_counter
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.learning.demonstrations import episode_record, replay_demonstration
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel

TINY_PATHS = (
    ROOT / "instances" / "tiny" / "tiny_01.json",
    ROOT / "instances" / "tiny" / "tiny_02.json",
)


def _accuracy(model, tensors, actions) -> dict[str, float]:
    correct = {key: 0 for key in ("operation", "island", "w", "f", "joint")}
    model.eval()
    with torch.no_grad():
        for graph, expert in zip(tensors, actions):
            hidden = model.encode(graph)
            op_id = model.policy.operation_distribution(graph, hidden).argmax()
            island_id = model.policy.island_distribution(graph, hidden, op_id).argmax()
            w_id = model.policy.w_distribution(graph, hidden, op_id, island_id).argmax()
            f_id = model.policy.f_distribution(graph, hidden, op_id, island_id, w_id).argmax()
            predicted = (op_id, island_id, w_id, f_id)
            actual = (expert.operation_id, expert.island_id, expert.w_agv_id, expert.f_agv_id)
            for index, key in enumerate(("operation", "island", "w", "f")):
                correct[key] += int(predicted[index] == actual[index])
            correct["joint"] += int(predicted == actual)
    return {key: value / len(actions) for key, value in correct.items()}


def _rollout(model, tensorizer, instance):
    env = RCIASConstructionEnv(instance)
    actions = []
    while not env.done:
        graph = build_graph_state(instance, env.schedule)
        tensor = tensorizer.tensorize(graph)
        with torch.no_grad():
            action = model.greedy_action(tensor)
        actions.append(action)
        env.step(action)
    return env, tuple(actions), check_schedule(instance, env.schedule)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bc_validation/run_1"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument(
        "--target-loss", type=float, default=0.12,
        help="symmetry-aware mean CE threshold for the ID-invariant Tiny policy",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print("This run tests whether RT-HGT can exactly imitate a proven-optimal Tiny action trajectory.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    instances = [load_instance(path) for path in TINY_PATHS]
    episodes = []
    exact_by_instance = {}
    for instance in instances:
        exact = solve_tiny_exact(instance, time_limit_seconds=60.0)
        if exact.status != "OPTIMAL":
            raise RuntimeError(f"exact expert unavailable for {instance.instance_id}: {exact.status}")
        exact_by_instance[instance.instance_id] = exact
        episodes.append(replay_demonstration(instance, "EXACT", exact.actions))
        for method in ("H1", "H2", "H3"):
            heuristic = solve_dispatching(instance, method)
            episodes.append(replay_demonstration(instance, method, heuristic.actions))
    write_json({
        "source_priority": ["EXACT", "best_heuristic", "H1", "H2", "H3"],
        "training_selection": {"instance": "tiny_01", "source": "EXACT"},
        "episodes": [episode_record(episode) for episode in episodes],
    }, args.out_dir / "demonstrations.json")

    train_episode = next(
        episode for episode in episodes
        if episode.instance_id == "tiny_01" and episode.source == "EXACT"
    )
    tensorizer = GraphTensorizer(train_episode.steps[0].graph)
    tensors = [tensorizer.tensorize(step.graph) for step in train_episode.steps]
    actions = [step.action for step in train_episode.steps]
    config = ModelConfig(
        embedding_dim=args.embedding_dim, heads=args.heads, layers=args.layers, dropout=0.0
    )
    model = RCIASNeuralModel(tensorizer, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history = []
    started = perf_counter()
    consecutive_success = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {key: torch.zeros(()) for key in ("operation", "island", "w", "f", "total")}
        for graph, action in zip(tensors, actions):
            sample_losses = model.action_losses(graph, action)
            for key in losses:
                losses[key] = losses[key] + sample_losses[key]
        for key in losses:
            losses[key] = losses[key] / len(actions)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        accuracy = _accuracy(model, tensors, actions)
        record = {
            "epoch": epoch,
            "losses": {key: float(value.detach()) for key, value in losses.items()},
            "accuracy": accuracy,
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} loss={record['losses']['total']:.6f} "
                f"joint={accuracy['joint']:.3f} operation={accuracy['operation']:.3f} "
                f"island={accuracy['island']:.3f}"
            )
        success = accuracy["joint"] == 1.0 and record["losses"]["total"] <= args.target_loss
        consecutive_success = consecutive_success + 1 if success else 0
        if consecutive_success >= 5:
            break
    runtime = perf_counter() - started

    model.eval()
    final_accuracy = _accuracy(model, tensors, actions)
    rollout_env, rollout_actions, rollout_audit = _rollout(model, tensorizer, instances[0])
    exact = exact_by_instance["tiny_01"]
    action_sequence_equal = rollout_actions == exact.actions
    makespan_equal = rollout_env.objective().makespan == exact.objective.makespan
    bc_validated = (
        history[-1]["losses"]["total"] <= args.target_loss
        and final_accuracy["joint"] == 1.0
        and rollout_audit["feasible"]
        and action_sequence_equal
        and makespan_equal
    )
    history_path = args.out_dir / "training_history.json"
    write_json({"epochs": history}, history_path)
    rollout_record = {
        "actions": [asdict(action) for action in rollout_actions],
        "feasible": bool(rollout_audit["feasible"]),
        "violations": rollout_audit["violations"],
        "makespan": rollout_env.objective().makespan,
        "exact_makespan": exact.objective.makespan,
        "action_sequence_equal": action_sequence_equal,
        "makespan_equal": makespan_equal,
    }
    if not rollout_audit["feasible"]:
        write_json({
            "instance": instances[0].instance_id,
            "seed": args.seed,
            "last_state": episode_record(train_episode)["steps"][-1]["graph_state"],
            "actions": rollout_record["actions"],
            "failure_constraint": rollout_audit["violations"],
        }, args.out_dir / "failure_case.json")
    final_info = {
        "experiment": "tiny_exact_behavior_cloning",
        "seed": args.seed,
        "model_config": config.to_dict(),
        "default_model_config": ModelConfig().to_dict(),
        "training_instance": "tiny_01",
        "expert_source": "EXACT",
        "expert_steps": len(actions),
        "epochs_completed": len(history),
        "runtime_seconds": runtime,
        "final_losses": history[-1]["losses"],
        "expert_action_accuracy": final_accuracy,
        "rollout": rollout_record,
        "rollout_feasibility": float(bool(rollout_audit["feasible"])),
        "prohibited_components_used": {
            "ppo": False, "multiobjective_preference_policy": False,
            "critical_synchronization_graph": False, "neural_destroy_repair": False,
        },
        "bc_validated": bc_validated,
    }
    write_json(final_info, args.out_dir / "final_info.json")
    notes = (
        "# Behavior Cloning validation\n\n"
        "The exact tiny_01 trajectory was selected over heuristic demonstrations.\n\n"
        f"- Epochs: {len(history)}\n"
        f"- Final loss: {history[-1]['losses']['total']:.8f}\n"
        f"- Joint expert reproduction: {final_accuracy['joint']:.1%}\n"
        f"- Rollout feasibility: {float(bool(rollout_audit['feasible'])):.1%}\n"
        f"- Policy/exact makespan: {rollout_env.objective().makespan} / {exact.objective.makespan}\n"
        f"- Exact action sequence reproduced: {action_sequence_equal}\n"
    )
    (args.out_dir.parent / "notes.txt").write_text(notes, encoding="utf-8")
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "plot_bc_results.py"),
            str(history_path), "--output-dir", str(args.out_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        f"BC_VALIDATED = {str(bc_validated).upper()} | loss={history[-1]['losses']['total']:.6f} "
        f"joint={final_accuracy['joint']:.3f} feasible={rollout_audit['feasible']} "
        f"makespan={rollout_env.objective().makespan}"
    )
    if not bc_validated:
        raise RuntimeError("Behavior Cloning validation did not meet all acceptance criteria")


if __name__ == "__main__":
    main()
