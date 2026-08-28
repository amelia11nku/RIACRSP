#!/usr/bin/env python3
"""Train Phase 5B PPO on M/W/F under a frozen greedy BC operation branch."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.best_policy import BestPolicyReference, RollbackController
from rcias_clgri.learning.evaluator import load_checkpoint, save_operation_anchored_checkpoint
from rcias_clgri.learning.experiment import (
    load_phase3_config,
    make_factory,
    resolve_device,
    run_metadata,
    seed_everything,
    validate_policy,
)
from rcias_clgri.learning.teacher_anchor import freeze_teacher, teacher_stage_kl
from rcias_clgri.learning.trainer import PPOConfig, PPOTrainer
from rcias_clgri.nn.hierarchical_policy import OperationAnchoredModel
from rcias_clgri.training.curriculum import MixedScaleCurriculum
from scripts.train_ppo import _reward_statistics, _weighted_validation_score


def _save(path, model, tensorizer, frozen_path, *, seed, update, score, selection):
    save_operation_anchored_checkpoint(
        path,
        model,
        tensorizer,
        frozen_operation_checkpoint=frozen_path,
        metadata={
            "phase": "phase5b_downstream_ppo",
            "training_seed": seed,
            "update": update,
            "validation_score": score,
            "selection": selection,
            "canonical_gradient_instances": 0,
        },
    )


def _best_policy_kl(model, reference, transitions, device):
    reference_model = copy.deepcopy(model.downstream)
    reference.restore(reference_model)
    freeze_teacher(reference_model)
    values = {stage: [] for stage in model.trainable_stages}
    for transition in transitions[:32]:
        graph = transition.graph.to(device)
        with torch.no_grad():
            hidden = model.downstream.encode(graph)
            stage_values = teacher_stage_kl(
                model.downstream, reference_model, graph, hidden, transition.action
            )
        for stage in values:
            values[stage].append(float(stage_values[stage].cpu()))
    del reference_model
    return {
        **{stage: mean(stage_values) if stage_values else 0.0 for stage, stage_values in values.items()},
        "mean": mean([value for stage_values in values.values() for value in stage_values]) if any(values.values()) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b_training.json"))
    parser.add_argument("--seed", type=int, default=520101)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5b/downstream_pilot"))
    parser.add_argument("--freeze-prefix-stages", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    allowed = {int(seed) for seed in config["training_seed_policy"]["independent_training_seeds"]}
    if args.seed not in allowed:
        raise ValueError("pilot seed is not one of the frozen Phase 5B training seeds")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = str(config["teacher_checkpoint"])
    frozen_bc, tensorizer, _ = load_checkpoint(frozen_path, device=device)
    downstream, downstream_tensorizer, _ = load_checkpoint(frozen_path, device=device)
    if tensorizer.to_schema() != downstream_tensorizer.to_schema():
        raise RuntimeError("frozen and downstream tensorizer schemas differ")
    model = OperationAnchoredModel(
        frozen_bc, downstream, frozen_prefix_stages=args.freeze_prefix_stages
    ).to(device)
    ppo_config = PPOConfig.from_mapping(config["ppo"])
    trainer = PPOTrainer(
        model,
        tensorizer,
        ppo_config,
        device=device,
        teacher_model=None,
        teacher_anchor_config={"enabled": False},
    )
    curriculum_config = config["curriculum"]
    curriculum = MixedScaleCurriculum(
        curriculum_config["weights"], window_size=int(curriculum_config["window_size"])
    )
    factory = make_factory(ROOT, config)
    levels = ("S", "M", "L")
    weights = config["checkpoint_selection"]["weights"]
    initial_validation = validate_policy(
        model, tensorizer, factory, config["development_seeds"], levels=levels, device=device
    )
    best_score = _weighted_validation_score(initial_validation, levels, weights)
    best_update = 0
    reference = BestPolicyReference(model.downstream, best_score)
    rollback_config = config["downstream_ppo"]
    controller = RollbackController(
        patience=int(rollback_config["rollback_patience"]),
        relative_regression=float(rollback_config["rollback_relative_regression"]),
        learning_rate_factor=float(rollback_config["rollback_learning_rate_factor"]),
        maximum_rollbacks=int(rollback_config["maximum_rollbacks"]),
    )
    _save(args.out_dir / "best.pt", model, tensorizer, frozen_path, seed=args.seed, update=0, score=best_score, selection="best")
    forbidden = {
        int(seed)
        for key in ("development_seeds", "historical_validation_seeds", "phase5b_holdout_seeds")
        for seeds in config[key].values()
        for seed in seeds
    }
    forbidden.update(int(seed) for seed in config["structural_scenarios"]["seeds"])
    forbidden.update(int(seed) for seed in config["phase5b_structural_scenarios"]["seeds"])
    total_updates = args.updates or int(rollback_config["pilot_updates"])
    seed_rng = random.Random(int(config["global_seed"]) + args.seed)
    history = []
    rollback_events = []
    cumulative_steps = 0
    cumulative_episodes = 0
    started = perf_counter()
    stopped_early = False
    for update in range(1, total_updates + 1):
        buffer, episodes, rollout_time = trainer.collect(
            factory, curriculum, seed_rng=seed_rng, forbidden_seeds=forbidden
        )
        rollout = trainer.rollout_metrics(episodes)
        update_metrics = trainer.update(buffer, seed=args.seed + 1000 * update, update_number=update)
        cumulative_steps += int(rollout["environment_steps"])
        cumulative_episodes += int(rollout["episodes"])
        validation = validate_policy(
            model, tensorizer, factory, config["development_seeds"], levels=levels, device=device
        )
        score = _weighted_validation_score(validation, levels, weights)
        improved = reference.update_if_better(model.downstream, score)
        if improved:
            best_score = score
            best_update = update
            _save(args.out_dir / "best.pt", model, tensorizer, frozen_path, seed=args.seed, update=update, score=score, selection="best")
        best_kl = _best_policy_kl(model, reference, buffer.transitions, device)
        rollback_triggered = controller.observe(score, reference.score)
        rollback_applied = False
        if rollback_triggered:
            rollback_applied = controller.rollback(reference, model.downstream, trainer)
            rollback_events.append({
                "update": update,
                "score_before_rollback": score,
                "best_reference_score": reference.score,
                "applied": rollback_applied,
                "new_learning_rates": [float(group["lr"]) for group in trainer.optimizer.param_groups],
            })
            if not rollback_applied:
                stopped_early = True
        record = {
            "update": update,
            "environment_steps": cumulative_steps,
            "episodes": cumulative_episodes,
            "episodes_this_update": rollout["episodes"],
            "unique_instances_this_update": rollout["unique_instances"],
            "validation_normalized_makespan": score,
            "best_reference_score": reference.score,
            "improved_best_reference": improved,
            "rollback_triggered": rollback_triggered,
            "rollback_applied": rollback_applied,
            "rollback_count": controller.rollback_count,
            "best_policy_kl_mwf": best_kl,
            "feasibility_rate": rollout["feasibility_rate"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "approx_kl": update_metrics["approx_kl"],
            "normalized_entropy_mwf": update_metrics["normalized_entropy"],
            "normalized_operation_entropy_measurement": update_metrics["normalized_operation_entropy"],
            "normalized_island_entropy": update_metrics["normalized_island_entropy"],
            "normalized_w_entropy": update_metrics["normalized_w_entropy"],
            "normalized_f_entropy": update_metrics["normalized_f_entropy"],
            "reward_statistics": _reward_statistics(episodes),
            "rollout_time": rollout_time,
            "update_time": update_metrics["update_time"],
            "learning_rates": [float(group["lr"]) for group in trainer.optimizer.param_groups],
        }
        history.append(record)
        print(
            f"update={update:03d}/{total_updates} score={score:.6f} best={reference.score:.6f} "
            f"kl_mwf={best_kl['mean']:.6f} rollback={rollback_applied}"
        )
        buffer.clear()
        if stopped_early:
            break
    wall_time = perf_counter() - started
    _save(args.out_dir / "last.pt", model, tensorizer, frozen_path, seed=args.seed, update=len(history), score=history[-1]["validation_normalized_makespan"], selection="last")
    reference.restore(model.downstream)
    selected_validation = validate_policy(
        model, tensorizer, factory, config["development_seeds"], levels=levels, device=device
    )
    selected_score = _weighted_validation_score(selected_validation, levels, weights)
    _save(args.out_dir / "selected_best.pt", model, tensorizer, frozen_path, seed=args.seed, update=best_update, score=selected_score, selection="selected_best")
    memory = {
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }
    write_json({
        "experiment": "phase5b_frozen_operation_downstream_ppo",
        "training_seed": args.seed,
        "updates": len(history),
        "requested_updates": total_updates,
        "stopped_early": stopped_early,
        "best_update": best_update,
        "best_validation_normalized_makespan": best_score,
        "selected_best_validation": selected_validation,
        "initial_validation": initial_validation,
        "rollback_events": rollback_events,
        "rollback_count": controller.rollback_count,
        "environment_steps": cumulative_steps,
        "episodes": cumulative_episodes,
        "wall_clock_seconds": wall_time,
        "memory": memory,
        "all_feasible": all(record["feasibility_rate"] == 1.0 for record in history),
        "frozen_operation_checkpoint": frozen_path,
        "frozen_prefix_stages": args.freeze_prefix_stages,
        "canonical_gradient_instances": 0,
        "history": history,
        "run_metadata": run_metadata(args.config, device=device, training_seed=args.seed),
    }, args.out_dir / "metrics.json")
    write_json({"updates": history}, args.out_dir / "training_history.json")
    write_json(config, args.out_dir / "config.json")
    print(f"PHASE5B_DOWNSTREAM_PPO_COMPLETE best_update={best_update} best_score={best_score:.6f} canonical=0")


if __name__ == "__main__":
    main()
