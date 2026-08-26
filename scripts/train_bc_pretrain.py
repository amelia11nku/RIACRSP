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
    run_metadata,
    seed_everything,
    validate_policy,
)
from rcias_clgri.nn import BatchGraphTensor


def _demonstration_plan(
    bc_config: dict[str, object], episode_seed_start: int,
) -> list[tuple[str, int]]:
    """Build the deterministic, level-stratified synthetic demonstration plan."""
    configured_counts = bc_config.get("demonstration_instances")
    if configured_counts is not None:
        plan = [
            (level, episode_seed_start + offset)
            for offset, level in enumerate(
                level
                for level in ("S", "M", "L")
                for _ in range(int(configured_counts.get(level, 0)))
            )
        ]
        if not plan:
            raise ValueError("demonstration_instances must request at least one instance")
        return plan
    count = int(bc_config["demonstration_episodes"])
    levels = tuple(str(level) for level in bc_config["levels"])
    return [
        (levels[index % len(levels)], episode_seed_start + index)
        for index in range(count)
    ]


def _configure_capacity_study(
    config: dict[str, object], capacity_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    candidates = {
        str(candidate["name"]): candidate
        for candidate in config["model_capacity_study"]
    }
    if capacity_name not in candidates:
        raise ValueError(f"unknown capacity candidate: {capacity_name}")
    candidate = candidates[capacity_name]
    model_config = dict(config["model"])
    model_config.update({
        "embedding_dim": int(candidate["embedding_dim"]),
        "layers": int(candidate["layers"]),
    })
    bc_config = dict(config["bc_warm_start"])
    settings = config["capacity_study_settings"]
    bc_config["demonstration_instances"] = dict(settings["demonstration_instances"])
    bc_config["epochs"] = int(settings["epochs"])
    config["model"] = model_config
    config["bc_warm_start"] = bc_config
    return model_config, bc_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--capacity", choices=("D32-L2", "D64-L2", "D64-L3", "D128-L3"))
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    if args.capacity is not None:
        _, bc_config = _configure_capacity_study(config, args.capacity)
    else:
        bc_config = config["bc_warm_start"]
    output_root = Path(str(config.get("output_root", "outputs/phase3")))
    out_dir = args.out_dir or (
        output_root / "capacity_study" / args.capacity
        if args.capacity is not None
        else output_root / (
            "bc_large" if "demonstration_instances" in bc_config else "bc_pretrain"
        )
    )
    epochs = args.epochs or int(bc_config["epochs"])
    seed = int(config["global_seed"])
    device = resolve_device(args.device)
    seed_everything(seed)
    hardware_metadata = run_metadata(args.config, device=device, training_seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "This experiment tests whether best-of-H1/H2/H3 demonstrations from the "
        "independent synthetic distribution provide a useful BC initialization."
    )
    print(f"device={device} torch={torch.__version__}")
    factory = make_factory(ROOT, config)
    model, tensorizer = initialize_model(factory, config, seed=seed, device=device)
    episode_seed_start = int(config["training_seed_policy"]["episode_seed_start"])
    plan = _demonstration_plan(bc_config, episode_seed_start)
    demonstration_count = len(plan)
    validation_levels = (
        tuple(level for level in ("S", "M", "L") if any(item[0] == level for item in plan))
        if "demonstration_instances" in bc_config else ("S",)
    )
    training_items = []
    summary = []
    for index, (level, episode_seed) in enumerate(plan):
        instance = factory.sample(episode_seed, level)
        candidates = [solve_dispatching(instance, method) for method in ("H1", "H2", "H3")]
        expert = min(candidates, key=lambda result: (result.objective.makespan, result.method))
        episode = replay_demonstration(instance, expert.method, expert.actions)
        training_items.extend(
            (tensorizer.tensorize(step.graph), step.action) for step in episode.steps
        )
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(bc_config["learning_rate"]),
        weight_decay=float(bc_config["weight_decay"]),
    )
    batch_size = int(bc_config["batch_size"])
    initial_validation = validate_policy(
        model, tensorizer, factory, config["validation_seeds"],
        levels=validation_levels, device=device,
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
            levels=validation_levels, device=device,
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
                out_dir / "best.pt", model, tensorizer,
                metadata={
                    "phase": "synthetic_bc_pretrain", "epoch": epoch,
                    "training_seed": seed, "validation_score": score,
                    "run_metadata": hardware_metadata,
                },
            )
        print(
            f"epoch={epoch:03d} loss={record['mean_loss']:.5f} "
            f"joint={record['training_joint_accuracy']:.3f} "
            f"val_norm={score:.5f} feasible={validation['feasibility_rate']:.3f}"
        )
    runtime = perf_counter() - started
    save_checkpoint(
        out_dir / "final.pt", model, tensorizer,
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
        "run_metadata": hardware_metadata,
        "training_instances": demonstration_count,
        "training_steps": len(training_items),
        "capacity_candidate": args.capacity,
        "canonical_instances_used": 0,
        "initial_validation": initial_validation,
        "final_validation": validate_policy(
            model, tensorizer, factory, config["validation_seeds"],
            levels=validation_levels, device=device,
        ),
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_validation_normalized_makespan": best_score,
        "runtime_seconds": runtime,
        "history": history,
    }
    write_json(summary, out_dir / "demonstrations_summary.json")
    write_json({
        **config,
        "run_metadata": {
            "training_seed": seed,
            "epochs": epochs,
            "output_directory": out_dir.as_posix(),
            "capacity_candidate": args.capacity,
            "hardware": hardware_metadata,
        },
    }, out_dir / "config.json")
    write_json(final_info, out_dir / "metrics.json")
    write_json({"epochs": history}, out_dir / "training_history.json")
    (out_dir / "notes.txt").write_text(
        "# Synthetic BC warm start\n\n"
        "Training uses only independently generated S/M/L instances and best-of-H1/H2/H3 "
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
