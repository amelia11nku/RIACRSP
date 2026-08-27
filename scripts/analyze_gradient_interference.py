#!/usr/bin/env python3
"""Measure actor/critic encoder-gradient cosine on S/M/L BC rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.buffer import RolloutBuffer
from rcias_clgri.learning.evaluator import load_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device
from rcias_clgri.learning.rollout import collect_episode


def gradient_cosine(actor_gradients, critic_gradients) -> float:
    pairs = [(a.reshape(-1), c.reshape(-1)) for a, c in zip(actor_gradients, critic_gradients) if a is not None and c is not None]
    actor = torch.cat([pair[0] for pair in pairs])
    critic = torch.cat([pair[1] for pair in pairs])
    return float(torch.dot(actor, critic) / (actor.norm() * critic.norm()).clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5a_training.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5a/gradient_diagnosis"))
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    model, tensorizer, _ = load_checkpoint(config["teacher_checkpoint"], device=device)
    factory = make_factory(ROOT, config)
    parameters = tuple(model.encoder.parameters())
    records = []
    for level in ("S", "M", "L"):
        for sample_index, seed in enumerate(config["development_seeds"][level][:4]):
            instance = factory.sample(int(seed), level)
            episode = collect_episode(model, tensorizer, instance, device=device, deterministic=False, generator=torch.Generator(device=device).manual_seed(int(seed)), store_transitions=True)
            buffer = RolloutBuffer()
            for transition in episode.transitions:
                buffer.add(transition)
            buffer.compute_advantages(gamma=1.0, gae_lambda=.95)
            indices = list(range(min(64, len(buffer))))
            log_probs, values = [], []
            for index in indices:
                transition = buffer.transitions[index]
                graph = transition.graph.to(device)
                hidden = model.encode(graph)
                log_probs.append(model.policy.evaluate_action(graph, hidden, transition.action).joint_log_prob)
                values.append(model.value(hidden))
            advantages = buffer.advantages[indices].to(device)
            returns = buffer.returns[indices].to(device)
            actor_loss = -(torch.stack(log_probs) * advantages).mean()
            critic_loss = F.smooth_l1_loss(torch.stack(values), returns)
            actor_gradients = torch.autograd.grad(actor_loss, parameters, retain_graph=True, allow_unused=True)
            critic_gradients = torch.autograd.grad(critic_loss, parameters, allow_unused=True)
            records.append({"level": level, "seed": int(seed), "cosine": gradient_cosine(actor_gradients, critic_gradients)})
    summary = {}
    for level in ("S", "M", "L"):
        values = [row["cosine"] for row in records if row["level"] == level]
        summary[level] = {"mean": statistics.mean(values), "median": statistics.median(values), "fraction_negative": sum(x < 0 for x in values) / len(values), "fraction_below_minus_0_2": sum(x < -.2 for x in values) / len(values)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"canonical_instances": 0, "summary": summary, "records": records}, args.out_dir / "final_info.json")
    print("ACTOR_CRITIC_INTERFERENCE_DIAGNOSED = TRUE")


if __name__ == "__main__":
    main()
