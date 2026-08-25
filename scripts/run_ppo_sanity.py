#!/usr/bin/env python3
"""Tiny PPO sanity: exact warm start, real PPO updates, and deterministic replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance
from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact
from rcias_clgri.learning.buffer import RolloutBuffer
from rcias_clgri.learning.demonstrations import replay_demonstration
from rcias_clgri.learning.evaluator import save_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, resolve_device, seed_everything
from rcias_clgri.learning.rollout import collect_episode
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer
from rcias_clgri.nn import BatchGraphTensor, GraphTensorizer, ModelConfig, RCIASNeuralModel


def _train_one(instance_path, config, device, seed, updates, out_dir):
    instance = load_instance(instance_path)
    exact = solve_tiny_exact(instance, time_limit_seconds=60.0)
    if exact.status != "OPTIMAL":
        raise RuntimeError(f"Tiny expert is not optimal: {instance.instance_id}")
    demonstration = replay_demonstration(instance, "EXACT", exact.actions)
    tensorizer = GraphTensorizer(demonstration.steps[0].graph)
    graphs = [tensorizer.tensorize(step.graph) for step in demonstration.steps]
    actions = [step.action for step in demonstration.steps]
    seed_everything(seed)
    model = RCIASNeuralModel(tensorizer, ModelConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for epoch in range(1, 201):
        batch = BatchGraphTensor.from_graphs(graphs).to(device)
        hidden_batch = model.encode_batch(batch)
        loss = torch.stack([
            model.policy.action_losses(graph, hidden, action)["total"]
            for graph, hidden, action in zip(batch.graphs, hidden_batch, actions)
        ]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct = sum(
                int(model.policy.greedy_action(graph, hidden) == action)
                for graph, hidden, action in zip(batch.graphs, hidden_batch, actions)
            )
        if correct == len(actions) and float(loss.detach()) < 0.05:
            break
    bc_episode = collect_episode(
        model, tensorizer, instance, device=device,
        deterministic=True, store_transitions=False,
    )
    ppo_config = PPOConfig(
        **{
            **PPOConfig.from_mapping(config["ppo"]).__dict__,
            "update_epochs": 2,
            "minibatch_size": 32,
            "rollout_transitions": 48,
            "target_kl": 0.1,
        }
    )
    trainer = PPOTrainer(model, tensorizer, ppo_config, device=device)
    initial_parameters = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    update_records = []
    best_makespan = bc_episode.makespan
    feasibility = []
    for update in range(1, updates + 1):
        buffer = RolloutBuffer()
        episodes = []
        while len(buffer) < ppo_config.rollout_transitions:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed * 1000 + update * 100 + len(episodes))
            episode = collect_episode(
                model, tensorizer, instance, device=device,
                deterministic=False, generator=generator, store_transitions=True,
            )
            episodes.append(episode)
            feasibility.append(float(episode.feasible))
            for transition in episode.transitions:
                buffer.add(transition)
        buffer.compute_advantages(gamma=1.0, gae_lambda=0.95)
        metrics = trainer.update(buffer, seed=seed + update)
        deterministic = collect_episode(
            model, tensorizer, instance, device=device,
            deterministic=True, store_transitions=False,
        )
        best_makespan = min(best_makespan, deterministic.makespan)
        update_records.append({
            "update": update,
            "deterministic_makespan": deterministic.makespan,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "approx_kl": metrics["approx_kl"],
            "gradient_norm": metrics["gradient_norm_before"],
        })
    changed = any(
        not torch.equal(before, after.detach().cpu())
        for before, after in zip(initial_parameters, model.parameters())
    )
    final_episode = collect_episode(
        model, tensorizer, instance, device=device,
        deterministic=True, store_transitions=False,
    )
    save_checkpoint(
        out_dir / f"{instance.instance_id}_final.pt", model, tensorizer,
        metadata={"phase": "tiny_ppo_sanity", "instance": instance.instance_id, "seed": seed},
    )
    return {
        "instance": instance.instance_id,
        "exact_makespan": exact.objective.makespan,
        "bc_makespan": bc_episode.makespan,
        "best_makespan": best_makespan,
        "final_makespan": final_episode.makespan,
        "feasibility_rate": sum(feasibility) / len(feasibility),
        "ppo_parameters_changed": changed,
        "finite_updates": all(
            all(torch.isfinite(torch.tensor(float(row[key]))) for key in (
                "policy_loss", "value_loss", "approx_kl", "gradient_norm"
            )) for row in update_records
        ),
        "updates": update_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase3/tiny_sanity"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=4)
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "This experiment checks feasible Tiny rollouts, exact-quality behavior, finite PPO "
        "updates, and real parameter changes before curriculum training."
    )
    results = [
        _train_one(
            ROOT / "instances" / "tiny" / name,
            config, device, int(config["global_seed"]) + index,
            args.updates, args.out_dir,
        )
        for index, name in enumerate(("tiny_01.json", "tiny_03.json"), start=1)
    ]
    validated = all(
        row["best_makespan"] == row["exact_makespan"]
        and row["feasibility_rate"] == 1.0
        and row["ppo_parameters_changed"]
        and row["finite_updates"]
        for row in results
    )
    write_json({
        "device": str(device), "torch_version": torch.__version__,
        "results": results, "tiny_ppo_sanity_validated": validated,
    }, args.out_dir / "final_info.json")
    print(f"TINY_PPO_SANITY_VALIDATED = {str(validated).upper()}")
    if not validated:
        raise RuntimeError("Tiny PPO sanity acceptance criteria failed")


if __name__ == "__main__":
    main()
