#!/usr/bin/env python3
"""Train one validation-gated constructive PPO seed on synthetic instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import load_checkpoint, save_checkpoint
from rcias_clgri.learning.experiment import (
    initialize_model,
    load_phase3_config,
    make_factory,
    resolve_device,
    seed_everything,
    validate_policy,
)
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer
from rcias_clgri.training.curriculum import CurriculumManager


def _memory(device: torch.device) -> dict[str, int]:
    try:
        import psutil
        rss = int(psutil.Process().memory_info().rss)
    except ImportError:
        rss = 0
    return {
        "cpu_rss_bytes": rss,
        "gpu_allocated_bytes": (
            int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gpu_reserved_bytes": (
            int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }


def _level_subset(validation: dict[str, object], level: str) -> dict[str, object]:
    records = [row for row in validation["records"] if row["level"] == level]
    return {
        "records": records,
        "mean_makespan": sum(float(row["makespan"]) for row in records) / len(records),
        "mean_normalized_makespan": (
            sum(float(row["normalized_makespan"]) for row in records) / len(records)
        ),
        "feasibility_rate": sum(float(bool(row["feasible"])) for row in records) / len(records),
        "mean_normalized_entropy": (
            sum(float(row["normalized_entropy"]) for row in records) / len(records)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init", default="outputs/phase3/bc_pretrain/best.pt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int)
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    allowed_seeds = set(int(value) for value in config["training_seed_policy"]["independent_training_seeds"])
    if args.seed not in allowed_seeds:
        raise ValueError(f"seed {args.seed} is not one of the frozen training seeds")
    validation_seed_set = {
        int(seed) for values in config["validation_seeds"].values() for seed in values
    }
    if args.seed in validation_seed_set:
        raise ValueError("training and validation seeds overlap")
    out_dir = args.out_dir or Path(f"outputs/phase3/ppo_seed_{sorted(allowed_seeds).index(args.seed) + 1}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    factory = make_factory(ROOT, config)
    if args.init.lower() == "random":
        model, tensorizer = initialize_model(
            factory, config, seed=args.seed, device=device
        )
        initialization = "random"
    else:
        model, tensorizer, _ = load_checkpoint(args.init, device=device)
        initialization = str(Path(args.init))
    print(
        "This experiment tests whether joint-log-probability PPO improves an RT-HGT "
        "constructive policy on held-out synthetic validation without canonical gradients."
    )
    print(
        f"seed={args.seed} device={device} torch={torch.__version__} init={initialization}"
    )
    ppo_config = PPOConfig.from_mapping(config["ppo"])
    trainer = PPOTrainer(model, tensorizer, ppo_config, device=device)
    curriculum = CurriculumManager(
        plateau_window=int(config["ppo"]["plateau_window"]),
        plateau_relative_improvement=float(
            config["ppo"]["plateau_relative_improvement"]
        ),
        minimum_updates=int(config["ppo"]["minimum_updates_per_level"]),
        current_level_probability=float(config["ppo"]["current_level_probability"]),
        minimum_normalized_entropy=float(config["ppo"]["minimum_normalized_entropy"]),
    )
    total_updates = args.updates or int(config["ppo"]["total_updates"])
    seed_rng = random.Random(int(config["global_seed"]) + args.seed)
    initial_validation = validate_policy(
        model, tensorizer, factory, config["validation_seeds"],
        levels=("S", "M"), device=device,
    )
    best_score = float(initial_validation["mean_normalized_makespan"])
    best_update = 0
    save_checkpoint(
        out_dir / "best.pt", model, tensorizer,
        metadata={
            "phase": "constructive_ppo", "seed": args.seed, "update": 0,
            "initialization": initialization, "validation_score": best_score,
        },
    )
    history = []
    cumulative_steps = 0
    cumulative_episodes = 0
    wall_started = perf_counter()
    for update in range(1, total_updates + 1):
        level_before = curriculum.current_level
        buffer, episodes, rollout_time = trainer.collect(
            factory,
            curriculum,
            seed_rng=seed_rng,
            forbidden_seeds=validation_seed_set,
        )
        rollout = trainer.rollout_metrics(episodes)
        update_metrics = trainer.update(buffer, seed=args.seed + update * 1000)
        cumulative_steps += int(rollout["environment_steps"])
        cumulative_episodes += int(rollout["episodes"])
        selection_levels = ("S", "M") if level_before != "L" else ("S", "M", "L")
        selection_validation = validate_policy(
            model, tensorizer, factory, config["validation_seeds"],
            levels=selection_levels, device=device,
        )
        current_validation = _level_subset(selection_validation, level_before)
        promoted = curriculum.record_validation(
            float(current_validation["mean_makespan"]),
            feasibility_rate=float(current_validation["feasibility_rate"]),
            normalized_entropy=float(current_validation["mean_normalized_entropy"]),
        )
        selection_score = float(
            selection_validation["mean_normalized_makespan"]
            if level_before != "L"
            else sum(
                float(row["normalized_makespan"])
                for row in selection_validation["records"]
                if row["level"] in {"S", "M"}
            ) / sum(
                1 for row in selection_validation["records"]
                if row["level"] in {"S", "M"}
            )
        )
        if selection_score < best_score:
            best_score = selection_score
            best_update = update
            save_checkpoint(
                out_dir / "best.pt", model, tensorizer,
                metadata={
                    "phase": "constructive_ppo", "seed": args.seed,
                    "update": update, "initialization": initialization,
                    "validation_score": selection_score,
                },
            )
        timing = rollout["timing"]
        record = {
            "update": update,
            "environment_steps": cumulative_steps,
            "episodes": cumulative_episodes,
            "curriculum_level": level_before,
            "promoted_after_update": promoted,
            "mean_episode_makespan": rollout["mean_episode_makespan"],
            "normalized_return": rollout["mean_normalized_return"],
            "validation_makespan": current_validation["mean_makespan"],
            "validation_normalized_makespan": current_validation["mean_normalized_makespan"],
            "selection_validation_normalized_makespan": selection_score,
            "feasibility_rate": rollout["feasibility_rate"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "total_loss": update_metrics["total_loss"],
            "entropy": update_metrics["entropy"],
            "normalized_entropy": update_metrics["normalized_entropy"],
            "operation_entropy": update_metrics["operation_entropy"],
            "island_entropy": update_metrics["island_entropy"],
            "w_entropy": update_metrics["w_entropy"],
            "f_entropy": update_metrics["f_entropy"],
            "approx_kl": update_metrics["approx_kl"],
            "clip_fraction": update_metrics["clip_fraction"],
            "explained_variance": update_metrics["explained_variance"],
            "gradient_norm": update_metrics["gradient_norm_before"],
            "gradient_norm_after": update_metrics["gradient_norm_after"],
            "learning_rate": update_metrics["learning_rate"],
            "rollout_time": rollout_time,
            "graph_build_time": timing["graph_build"],
            "forward_time": timing["forward"],
            "policy_scoring_time": timing["policy_scoring"],
            "decoder_time": timing["decoder"],
            "update_time": update_metrics["update_time"],
            "action_statistics": rollout["action_statistics"],
            "memory": _memory(device),
        }
        if not all(
            torch.isfinite(torch.tensor(float(record[key])))
            for key in (
                "policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"
            )
        ):
            raise FloatingPointError("non-finite training log metric")
        history.append(record)
        print(
            f"update={update:03d}/{total_updates} level={level_before} "
            f"steps={cumulative_steps} return={record['normalized_return']:.5f} "
            f"val_norm={selection_score:.5f} kl={record['approx_kl']:.5f} "
            f"entropy={record['normalized_entropy']:.3f} feasible={record['feasibility_rate']:.3f} "
            f"promoted={promoted}"
        )
        buffer.clear()
    wall_time = perf_counter() - wall_started
    # The last update already evaluated the unchanged final weights on S/M.
    # Reuse that exact result instead of repeating an expensive deterministic rollout.
    final_validation = selection_validation
    save_checkpoint(
        out_dir / "final.pt", model, tensorizer,
        metadata={
            "phase": "constructive_ppo", "seed": args.seed,
            "update": total_updates, "initialization": initialization,
            "validation_score": history[-1]["selection_validation_normalized_makespan"],
        },
    )
    final_info = {
        "experiment": "phase3_constructive_ppo",
        "training_seed": args.seed,
        "initialization": initialization,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "updates": total_updates,
        "environment_steps": cumulative_steps,
        "episodes": cumulative_episodes,
        "wall_clock_seconds": wall_time,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "large_validation_deferred_to_frozen_evaluation": True,
        "best_update": best_update,
        "best_validation_normalized_makespan": best_score,
        "curriculum": curriculum.to_dict(),
        "training_validation_seed_overlap": False,
        "canonical_gradient_instances": 0,
        "feasibility_100": all(row["feasibility_rate"] == 1.0 for row in history),
        "numerically_stable": True,
        "history": history,
    }
    write_json({
        **config,
        "run_metadata": {
            "training_seed": args.seed,
            "initialization": initialization,
            "requested_updates": total_updates,
            "output_directory": out_dir.as_posix(),
        },
    }, out_dir / "config.json")
    write_json({"updates": history}, out_dir / "training_history.json")
    write_json(final_info, out_dir / "metrics.json")
    (out_dir / "notes.txt").write_text(
        "# Constructive PPO run\n\n"
        f"Training seed: {args.seed}. Initialization: {initialization}. Canonical gradient "
        "instances: zero. Best checkpoint is selected only on fixed synthetic validation. "
        f"Output directory: {out_dir.as_posix()}.\n",
        encoding="utf-8",
    )
    print(
        "PPO_TRAINING_COMPLETE = TRUE | "
        f"seed={args.seed} best_update={best_update} best_val_norm={best_score:.6f} "
        f"steps={cumulative_steps} wall={wall_time:.1f}s"
    )


if __name__ == "__main__":
    main()
