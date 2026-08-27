#!/usr/bin/env python3
"""Train one validation-gated constructive PPO seed on synthetic instances."""

from __future__ import annotations

import argparse
import math
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
    run_metadata,
    seed_everything,
    validate_policy,
)
from rcias_clgri.learning.teacher_anchor import freeze_teacher
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer
from rcias_clgri.training.curriculum import CurriculumManager, MixedScaleCurriculum


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


def _weighted_validation_score(
    validation: dict[str, object],
    levels: tuple[str, ...],
    weights: dict[str, object],
) -> float:
    """Return a level-balanced score with weights normalized over active levels."""
    unknown = set(levels) - set(weights)
    if unknown:
        raise ValueError(f"missing checkpoint-selection weights for {sorted(unknown)}")
    active_weights = {level: float(weights[level]) for level in levels}
    if any(weight < 0.0 for weight in active_weights.values()):
        raise ValueError("checkpoint-selection weights must be non-negative")
    weight_sum = sum(active_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("active checkpoint-selection weights must sum to a positive value")
    level_scores = {}
    for level in levels:
        records = [row for row in validation["records"] if row["level"] == level]
        if not records:
            raise ValueError(f"validation has no records for active level {level}")
        level_scores[level] = sum(
            float(row["normalized_makespan"]) for row in records
        ) / len(records)
    return sum(
        active_weights[level] * level_scores[level] for level in levels
    ) / weight_sum


def _robust_validation_score(
    validation: dict[str, object], levels: tuple[str, ...],
    weights: dict[str, object], robust_lambda: float,
) -> tuple[float, float, float]:
    mean_score = _weighted_validation_score(validation, levels, weights)
    weight_sum = sum(float(weights[level]) for level in levels)
    variance = 0.0
    for level in levels:
        values = [
            float(row["normalized_makespan"])
            for row in validation["records"] if row["level"] == level
        ]
        level_weight = float(weights[level]) / weight_sum
        variance += level_weight * sum(
            (value - mean_score) ** 2 for value in values
        ) / len(values)
    standard_deviation = math.sqrt(variance)
    return mean_score, standard_deviation, mean_score + robust_lambda * standard_deviation


def _legacy_validation_score(
    validation: dict[str, object], levels: tuple[str, ...],
) -> float:
    """Preserve the frozen Phase 3 checkpoint-selection behavior."""
    selected = [
        row for row in validation["records"]
        if row["level"] in ({"S", "M"} if "L" in levels else set(levels))
    ]
    return sum(float(row["normalized_makespan"]) for row in selected) / len(selected)


def _reward_statistics(episodes: list[object]) -> dict[str, float]:
    """Summarize the unchanged dense R0 rewards collected for one PPO update."""
    rewards = [
        float(transition.reward)
        for episode in episodes
        for transition in episode.transitions
    ]
    if not rewards:
        raise ValueError("reward statistics require at least one transition")
    mean = sum(rewards) / len(rewards)
    return {
        "count": float(len(rewards)),
        "zero_fraction": sum(reward == 0.0 for reward in rewards) / len(rewards),
        "mean": mean,
        "std": math.sqrt(
            sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        ),
    }


def _relative_parameter_drift(model, teacher, prefix: str | None = None) -> float:
    current = dict(model.named_parameters())
    reference = dict(teacher.named_parameters())
    names = [name for name in current if prefix is None or name.startswith(prefix)]
    numerator = sum(
        float((current[name].detach() - reference[name].detach()).pow(2).sum().cpu())
        for name in names
    )
    denominator = sum(
        float(reference[name].detach().pow(2).sum().cpu()) for name in names
    )
    return math.sqrt(numerator / max(denominator, 1e-24))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--screening-stage-a", choices=("A0", "A1", "A2"))
    parser.add_argument("--anchor-profile", choices=("B0", "B1", "B2"))
    parser.add_argument("--critic-profile", choices=("C0", "C1"))
    parser.add_argument("--curriculum-profile", choices=("D0", "D1"))
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    if args.screening_stage_a is not None:
        config["ppo"].update(
            config["screening"]["stage_a"][args.screening_stage_a]
        )
        config["teacher_anchor"] = {
            **config.get("teacher_anchor", {}), "enabled": False
        }
    if args.anchor_profile is not None:
        profiles = {
            "B0": {"enabled": False},
            "B1": {"enabled": True, "beta_initial": 0.05, "beta_minimum": 0.005},
            "B2": {"enabled": True, "beta_initial": 0.20, "beta_minimum": 0.020},
        }
        config["teacher_anchor"] = {
            **config.get("teacher_anchor", {}), **profiles[args.anchor_profile]
        }
    if args.critic_profile is not None:
        config["ppo"]["critic_stop_gradient"] = args.critic_profile == "C1"
    if args.curriculum_profile == "D0":
        config["curriculum"] = {"mode": "staged"}
    elif args.curriculum_profile == "D1":
        config["curriculum"] = {
            "mode": "mixed", "weights": {"S": 0.25, "M": 0.35, "L": 0.40},
            "window_size": 8, "require_all_levels_per_window": True,
        }
    allowed_seeds = set(int(value) for value in config["training_seed_policy"]["independent_training_seeds"])
    if args.seed not in allowed_seeds:
        raise ValueError(f"seed {args.seed} is not one of the frozen training seeds")
    validation_seed_set = {
        int(seed) for values in config["validation_seeds"].values() for seed in values
    }
    if args.seed in validation_seed_set:
        raise ValueError("training and validation seeds overlap")
    output_root = Path(str(config.get("output_root", "outputs/phase3")))
    out_dir = args.out_dir or output_root / f"ppo_seed_{sorted(allowed_seeds).index(args.seed) + 1}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    hardware_metadata = run_metadata(
        args.config, device=device, training_seed=args.seed
    )
    factory = make_factory(ROOT, config)
    initialization_arg = args.init or str(
        config.get("ppo_initialization", "outputs/phase3/bc_pretrain/best.pt")
    )
    if initialization_arg.lower() == "random":
        model, tensorizer = initialize_model(
            factory, config, seed=args.seed, device=device
        )
        initialization = "random"
    else:
        model, tensorizer, _ = load_checkpoint(initialization_arg, device=device)
        initialization = str(Path(initialization_arg))
    print(
        "This experiment tests whether joint-log-probability PPO improves an RT-HGT "
        "constructive policy on held-out synthetic validation without canonical gradients."
    )
    print(
        f"seed={args.seed} device={device} torch={torch.__version__} init={initialization}"
    )
    ppo_config = PPOConfig.from_mapping(config["ppo"])
    curriculum_config = config.get("curriculum", {})
    mixed_mode = curriculum_config.get("mode") == "mixed"
    phase5_mode = "development_seeds" in config
    if mixed_mode:
        curriculum = MixedScaleCurriculum(
            curriculum_config["weights"],
            window_size=int(curriculum_config["window_size"]),
        )
    else:
        curriculum = CurriculumManager(
            plateau_window=int(config["ppo"]["plateau_window"]),
            plateau_relative_improvement=float(config["ppo"]["plateau_relative_improvement"]),
            minimum_updates=int(config["ppo"]["minimum_updates_per_level"]),
            current_level_probability=float(config["ppo"]["current_level_probability"]),
            minimum_normalized_entropy=float(config["ppo"]["minimum_normalized_entropy"]),
        )
    teacher_model = None
    teacher_anchor_config = config.get("teacher_anchor", {"enabled": False})
    if "teacher_checkpoint" in config:
        teacher_model, _, _ = load_checkpoint(config["teacher_checkpoint"], device=device)
        freeze_teacher(teacher_model)
    trainer = PPOTrainer(
        model, tensorizer, ppo_config, device=device,
        teacher_model=teacher_model,
        teacher_anchor_config=teacher_anchor_config,
    )
    total_updates = args.updates or int(config["ppo"]["total_updates"])
    selection_config = config.get("checkpoint_selection")
    selection_weights = None if selection_config is None else selection_config["weights"]
    robust_lambda = (
        float(selection_config.get("robust_lambda", 0.0))
        if selection_config is not None else 0.0
    )
    seed_rng = random.Random(int(config["global_seed"]) + args.seed)
    initial_levels = ("S", "M", "L") if phase5_mode else ("S", "M")
    initial_validation = validate_policy(
        model, tensorizer, factory, config["validation_seeds"],
        levels=initial_levels, device=device,
    )
    best_score = (
        _legacy_validation_score(initial_validation, initial_levels)
        if selection_weights is None
        else _weighted_validation_score(initial_validation, initial_levels, selection_weights)
    )
    best_update = 0
    best_robust_update = 0
    best_robust_score = (
        _robust_validation_score(
            initial_validation, initial_levels, selection_weights, robust_lambda
        )[2] if selection_weights is not None else best_score
    )
    save_checkpoint(
        out_dir / "best.pt", model, tensorizer,
        metadata={
            "phase": "constructive_ppo", "seed": args.seed, "update": 0,
            "initialization": initialization, "validation_score": best_score,
            "run_metadata": hardware_metadata,
        },
    )
    if robust_lambda > 0.0:
        save_checkpoint(
            out_dir / "best_mean.pt", model, tensorizer,
            metadata={"phase": "constructive_ppo", "seed": args.seed, "update": 0, "selection": "mean"},
        )
        save_checkpoint(
            out_dir / "best_robust.pt", model, tensorizer,
            metadata={"phase": "constructive_ppo", "seed": args.seed, "update": 0, "selection": "mean_plus_std"},
        )
    history = []
    stopped_early = False
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
        reward_statistics = _reward_statistics(episodes)
        update_metrics = trainer.update(
            buffer, seed=args.seed + update * 1000, update_number=update
        )
        cumulative_steps += int(rollout["environment_steps"])
        cumulative_episodes += int(rollout["episodes"])
        selection_levels = (
            ("S", "M", "L") if phase5_mode
            else (("S", "M") if level_before != "L" else ("S", "M", "L"))
        )
        selection_validation = validate_policy(
            model, tensorizer, factory, config["validation_seeds"],
            levels=selection_levels, device=device,
        )
        current_validation = (
            selection_validation if mixed_mode
            else _level_subset(selection_validation, level_before)
        )
        promoted = curriculum.record_validation(
            float(current_validation["mean_makespan"]),
            feasibility_rate=float(current_validation["feasibility_rate"]),
            normalized_entropy=float(current_validation["mean_normalized_entropy"]),
        )
        selection_score = (
            _legacy_validation_score(selection_validation, selection_levels)
            if selection_weights is None
            else _weighted_validation_score(
                selection_validation, selection_levels, selection_weights
            )
        )
        robust_score = (
            _robust_validation_score(
                selection_validation, selection_levels, selection_weights, robust_lambda
            )[2] if selection_weights is not None else selection_score
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
                    "run_metadata": hardware_metadata,
                },
            )
            if robust_lambda > 0.0:
                save_checkpoint(
                    out_dir / "best_mean.pt", model, tensorizer,
                    metadata={"phase": "constructive_ppo", "seed": args.seed, "update": update, "selection": "mean", "validation_score": selection_score},
                )
        if robust_score < best_robust_score:
            best_robust_score = robust_score
            best_robust_update = update
            save_checkpoint(
                out_dir / "best_robust.pt", model, tensorizer,
                metadata={"phase": "constructive_ppo", "seed": args.seed, "update": update, "selection": "mean_plus_std", "validation_score": robust_score},
            )
        timing = rollout["timing"]
        record = {
            "update": update,
            "environment_steps": cumulative_steps,
            "episodes": cumulative_episodes,
            "episodes_this_update": rollout["episodes"],
            "unique_instances_this_update": rollout["unique_instances"],
            "curriculum_level": level_before,
            "promoted_after_update": promoted,
            "mean_episode_makespan": rollout["mean_episode_makespan"],
            "normalized_return": rollout["mean_normalized_return"],
            "reward_statistics": reward_statistics,
            "validation_makespan": current_validation["mean_makespan"],
            "validation_normalized_makespan": current_validation["mean_normalized_makespan"],
            "selection_validation_normalized_makespan": selection_score,
            "robust_selection_score": robust_score,
            "feasibility_rate": rollout["feasibility_rate"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "total_loss": update_metrics["total_loss"],
            "entropy": update_metrics["entropy"],
            "normalized_entropy": update_metrics["normalized_entropy"],
            "normalized_operation_entropy": update_metrics["normalized_operation_entropy"],
            "normalized_island_entropy": update_metrics["normalized_island_entropy"],
            "normalized_w_entropy": update_metrics["normalized_w_entropy"],
            "normalized_f_entropy": update_metrics["normalized_f_entropy"],
            "operation_entropy": update_metrics["operation_entropy"],
            "island_entropy": update_metrics["island_entropy"],
            "w_entropy": update_metrics["w_entropy"],
            "f_entropy": update_metrics["f_entropy"],
            "approx_kl": update_metrics["approx_kl"],
            "max_kl": update_metrics["max_kl"],
            "p95_kl": update_metrics["p95_kl"],
            "teacher_beta": update_metrics["teacher_beta"],
            "teacher_kl": update_metrics["teacher_kl"],
            "teacher_kl_operation": update_metrics["teacher_kl_operation"],
            "teacher_kl_island": update_metrics["teacher_kl_island"],
            "teacher_kl_w": update_metrics["teacher_kl_w"],
            "teacher_kl_f": update_metrics["teacher_kl_f"],
            "policy_parameter_drift_from_bc": (
                _relative_parameter_drift(model, teacher_model) if teacher_model else 0.0
            ),
            "encoder_parameter_drift_from_bc": (
                _relative_parameter_drift(model, teacher_model, "encoder.")
                if teacher_model else 0.0
            ),
            "advantage_mean": update_metrics["advantage_mean"],
            "advantage_std": update_metrics["advantage_std"],
            "return_variance": update_metrics["return_variance"],
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
        patience = int(selection_config.get("early_stopping_patience", 0)) if selection_config else 0
        minimum_updates = int(config["ppo"].get("minimum_updates", total_updates))
        if patience > 0 and update >= minimum_updates and update - best_update >= patience:
            kl_window = [row["teacher_kl"] for row in history[-patience:]]
            if kl_window[-1] > kl_window[0]:
                stopped_early = True
                print(
                    f"early_stop update={update} patience={patience} "
                    f"teacher_kl={kl_window[0]:.6f}->{kl_window[-1]:.6f}"
                )
                break
    wall_time = perf_counter() - wall_started
    completed_updates = len(history)
    # The last update already evaluated the unchanged final weights on S/M.
    # Reuse that exact result instead of repeating an expensive deterministic rollout.
    final_validation = selection_validation
    save_checkpoint(
        out_dir / "final.pt", model, tensorizer,
        metadata={
            "phase": "constructive_ppo", "seed": args.seed,
            "update": completed_updates, "initialization": initialization,
            "validation_score": history[-1]["selection_validation_normalized_makespan"],
        },
    )
    final_info = {
        "experiment": "phase3_constructive_ppo",
        "training_seed": args.seed,
        "initialization": initialization,
        "device": str(device),
        "torch_version": torch.__version__,
        "run_metadata": hardware_metadata,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "updates": completed_updates,
        "requested_updates": total_updates,
        "stopped_early": stopped_early,
        "environment_steps": cumulative_steps,
        "episodes": cumulative_episodes,
        "wall_clock_seconds": wall_time,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "large_validation_deferred_to_frozen_evaluation": True,
        "best_update": best_update,
        "best_validation_normalized_makespan": best_score,
        "best_robust_update": best_robust_update,
        "best_robust_score": best_robust_score,
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
            "hardware": hardware_metadata,
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
