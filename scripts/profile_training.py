#!/usr/bin/env python3
"""Profile Phase 3 rollout/update throughput and check CPU/GPU memory stability."""

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
from rcias_clgri.learning.evaluator import load_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device
from rcias_clgri.learning.rollout import collect_episode
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer
from rcias_clgri.training.curriculum import CurriculumManager, CurriculumState


def _memory(device):
    try:
        import psutil
        rss = int(psutil.Process().memory_info().rss)
    except ImportError:
        rss = 0
    return {
        "cpu_rss_bytes": rss,
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0,
    }


def _monotonic(values):
    return all(right >= left for left, right in zip(values, values[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase3/ppo_seed_1/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase3/profiling"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    model, tensorizer, _ = load_checkpoint(args.checkpoint, device=device)
    factory = make_factory(ROOT, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "This experiment measures rollout components, one PPO update, throughput, and "
        "memory stability over hundreds of detached episodes."
    )
    timing = {"graph_build": 0.0, "forward": 0.0, "policy_scoring": 0.0, "decoder": 0.0}
    memory_samples = []
    total_steps = 0
    started = perf_counter()
    for index in range(args.episodes):
        instance = factory.sample(600000 + index, "S")
        episode = collect_episode(
            model, tensorizer, instance, device=device,
            deterministic=False,
            generator=torch.Generator(device=device).manual_seed(700000 + index),
            store_transitions=False,
        )
        total_steps += len(episode.actions)
        for key in timing:
            timing[key] += episode.timing[key]
        if index % 25 == 0 or index + 1 == args.episodes:
            memory_samples.append({"episode": index + 1, **_memory(device)})
    rollout_wall = perf_counter() - started
    ppo_config = PPOConfig.from_mapping(config["ppo"])
    trainer = PPOTrainer(model, tensorizer, ppo_config, device=device)
    medium_state = CurriculumState(current_level_index=1)
    curriculum = CurriculumManager(state=medium_state)
    buffer, episodes, collection_time = trainer.collect(
        factory, curriculum,
        seed_rng=random.Random(808080),
        forbidden_seeds={
            int(seed) for values in config["validation_seeds"].values() for seed in values
        },
    )
    update_metrics = trainer.update(buffer, seed=909090)
    timing["ppo_update"] = float(update_metrics["update_time"])
    measured_total = sum(timing.values())
    percentages = {
        key: 100.0 * value / max(measured_total, 1e-12) for key, value in timing.items()
    }
    tail = memory_samples[max(1, len(memory_samples) // 5):]
    cpu_values = [row["cpu_rss_bytes"] for row in tail if row["cpu_rss_bytes"] > 0]
    gpu_values = [row["gpu_reserved_bytes"] for row in tail]
    cpu_growth = 0 if not cpu_values else cpu_values[-1] - cpu_values[0]
    gpu_growth = 0 if not gpu_values else gpu_values[-1] - gpu_values[0]
    cpu_stable = not cpu_values or not _monotonic(cpu_values) or cpu_growth <= 0.05 * cpu_values[0]
    gpu_stable = not _monotonic(gpu_values) or gpu_growth <= 0.05 * max(gpu_values[0], 1)
    final_info = {
        "checkpoint": args.checkpoint.as_posix(),
        "device": str(device),
        "episodes": args.episodes,
        "environment_steps": total_steps,
        "rollout_wall_seconds": rollout_wall,
        "environment_steps_per_second": total_steps / rollout_wall,
        "episodes_per_second": args.episodes / rollout_wall,
        "timing_seconds": timing,
        "timing_percent": percentages,
        "ppo_collection_seconds": collection_time,
        "ppo_update_metrics": update_metrics,
        "memory_samples": memory_samples,
        "cpu_tail_growth_bytes": cpu_growth,
        "gpu_tail_growth_bytes": gpu_growth,
        "cpu_memory_stable": cpu_stable,
        "gpu_memory_stable": gpu_stable,
        "memory_stable": cpu_stable and gpu_stable,
    }
    write_json(final_info, args.out_dir / "final_info.json")
    print(
        "TRAINING_PROFILE_COMPLETE = TRUE | "
        f"steps_per_second={final_info['environment_steps_per_second']:.2f} "
        f"episodes_per_second={final_info['episodes_per_second']:.2f} "
        f"memory_stable={final_info['memory_stable']}"
    )
    if not final_info["memory_stable"]:
        raise RuntimeError("training memory stability gate failed")


if __name__ == "__main__":
    main()
