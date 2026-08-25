#!/usr/bin/env python3
"""Pretrain RT-HGT on best-of-H1/H2/H3 synthetic demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from time import perf_counter
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.heuristic import solve_dispatching
from rcias_clgri.learning.demonstrations import replay_demonstration
from rcias_clgri.learning.evaluator import save_checkpoint
from rcias_clgri.learning.experiment import (
    initialize_model,
    load_phase3_config,
    make_factory,
    resolve_device,
    seed_everything,
    validate_policy,
)
from rcias_clgri.nn import BatchGraphTensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase3/bc_pretrain"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    bc_config = config["bc_warm_start"]
    epochs = args.epochs or int(bc_config["epochs"])
    seed = int(config["global_seed"])
    device = resolve_device(args.device)
    seed_everything(seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "This experiment tests whether best-of-H1/H2/H3 demonstrations from the "
        "independent synthetic distribution provide a useful BC initialization."
    )
    print(f"device={device} torch={torch.__version__}")
    factory = make_factory(ROOT, config)
    model, tensorizer = initialize_model(factory, config, seed=seed, device=device)
    demonstration_count = int(bc_config["demonstration_episodes"])
    levels = tuple(bc_config["levels"])
    episode_seed_start = int(config["training_seed_policy"]["episode_seed_start"])
    demonstrations = []
    summary = []
    for index in range(demonstration_count):
        level = levels[index % len(levels)]
        episode_seed = episode_seed_start + index
        instance = factory.sample(episode_seed, level)
        candidates = [solve_dispatching(instance, method) for method in ("H1", "H2", "H3")]
        expert = min(candidates, key=lambda result: (result.objective.makespan, result.method))
        episode = replay_demonstration(instance, expert.method, expert.actions)
        demonstrations.append(episode)
        summary.append({
            "instance_id": instance.instance_id,
            "seed": episode_seed,
            "level": level,
            "expert": expert.method,
            "makespan": expert.objective.makespan,
            "steps": len(episode.steps),
            "feasible": episode.feasible,
            "candidate_makespans": {
                result.method: result.objective.makespan for result in candidates
            },
        })
        print(
            f"demonstration={index + 1}/{demonstration_count} level={level} "
            f"steps={len(episode.steps)} expert={expert.method} "
            f"makespan={expert.objective.makespan:.1f}"
        )
    training_items = [
        (tensorizer.tensorize(step.graph), step.action)
        for episode in demonstrations for step in episode.steps
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(bc_config["learning_rate"]),
        weight_decay=float(bc_config["weight_decay"]),
    )
    batch_size = int(bc_config["batch_size"])
    initial_validation = validate_policy(
        model, tensorizer, factory, config["validation_seeds"],
        levels=("S",), device=device,
    )
    best_score = float("inf")
    best_epoch = 0
    history = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(training_items)))
        random.Random(seed + epoch).shuffle(indices)
        losses = []
        correct = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            pairs = [training_items[index] for index in selected]
            batch = BatchGraphTensor.from_graphs([graph for graph, _ in pairs]).to(device)
            hidden_batch = model.encode_batch(batch)
            sample_losses = [
                model.policy.action_losses(graph, hidden, action)["total"]
                for graph, hidden, (_, action) in zip(batch.graphs, hidden_batch, pairs)
            ]
            loss = torch.stack(sample_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                correct += sum(
                    int(model.policy.greedy_action(graph, hidden) == action)
                    for graph, hidden, (_, action) in zip(batch.graphs, hidden_batch, pairs)
                )
        validation = validate_policy(
            model, tensorizer, factory, config["validation_seeds"],
            levels=("S",), device=device,
        )
        record = {
            "epoch": epoch,
            "mean_loss": sum(losses) / len(losses),
            "training_joint_accuracy": correct / len(training_items),
            "validation_mean_makespan": validation["mean_makespan"],
            "validation_mean_normalized_makespan": validation["mean_normalized_makespan"],
            "validation_feasibility_rate": validation["feasibility_rate"],
        }
        history.append(record)
        score = float(validation["mean_normalized_makespan"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                args.out_dir / "best.pt", model, tensorizer,
                metadata={
                    "phase": "synthetic_bc_pretrain", "epoch": epoch,
                    "training_seed": seed, "validation_score": score,
                },
            )
        print(
            f"epoch={epoch:03d} loss={record['mean_loss']:.5f} "
            f"joint={record['training_joint_accuracy']:.3f} "
            f"val_norm={score:.5f} feasible={validation['feasibility_rate']:.3f}"
        )
    runtime = perf_counter() - started
    save_checkpoint(
        args.out_dir / "final.pt", model, tensorizer,
        metadata={
            "phase": "synthetic_bc_pretrain", "epoch": epochs,
            "training_seed": seed, "validation_score": history[-1]["validation_mean_normalized_makespan"],
        },
    )
    final_info = {
        "experiment": "phase3_synthetic_bc_pretrain",
        "device": str(device),
        "torch_version": torch.__version__,
        "training_seed": seed,
        "training_instances": demonstration_count,
        "training_steps": len(training_items),
        "canonical_instances_used": 0,
        "initial_validation": initial_validation,
        "final_validation": validate_policy(
            model, tensorizer, factory, config["validation_seeds"],
            levels=("S",), device=device,
        ),
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_validation_normalized_makespan": best_score,
        "runtime_seconds": runtime,
        "history": history,
    }
    write_json(summary, args.out_dir / "demonstrations_summary.json")
    write_json({
        **config,
        "run_metadata": {
            "training_seed": seed,
            "epochs": epochs,
            "output_directory": args.out_dir.as_posix(),
        },
    }, args.out_dir / "config.json")
    write_json(final_info, args.out_dir / "metrics.json")
    write_json({"epochs": history}, args.out_dir / "training_history.json")
    (args.out_dir / "notes.txt").write_text(
        "# Synthetic BC warm start\n\n"
        "Training uses only independently generated S/M instances and best-of-H1/H2/H3 "
        "actions. Canonical public instances are excluded.\n",
        encoding="utf-8",
    )
    print(
        "BC_PRETRAIN_COMPLETE = TRUE | "
        f"best_epoch={best_epoch} best_val_norm={best_score:.6f} "
        f"runtime={runtime:.1f}s"
    )


if __name__ == "__main__":
    main()
