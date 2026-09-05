#!/usr/bin/env python3
"""Freeze and run the conditional J3 family without altering J1/J2 evidence."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.phase6j_relational import RelationalCAURModel  # noqa: E402
from scripts import train_phase6j_caur as regular  # noqa: E402
from scripts.audit_phase6j_caur_training import OUTPUT as REGULAR_AUDIT  # noqa: E402

FAMILY = "J3_CONT_RELATIONAL"
OUT = regular.OUT / "j3_relational"
PROTOCOL_PATH = ROOT / "outputs/phase6j_caur/frozen/r12_j3_training_protocol.json"
CODE_FILES = (
    "rcias_clgri/ni/phase6j_relational.py",
    "rcias_clgri/ni/encoder.py",
    "rcias_clgri/ni/action_encoder.py",
    "rcias_clgri/ni/batching.py",
    "rcias_clgri/ni/calibration.py",
    "rcias_clgri/ni/scorer.py",
    "rcias_clgri/ni/tensorize.py",
    "rcias_clgri/ni/dataset.py",
    "rcias_clgri/ni/cache.py",
    "scripts/run_phase6j_caur_j3.py",
    "scripts/audit_phase6j_caur_training.py",
)


def freeze_protocol() -> dict:
    if PROTOCOL_PATH.exists():
        return validate_protocol()
    parent = regular.validate_protocol()
    audit = regular.load_json(REGULAR_AUDIT)
    if not (audit["status"] == "PASS" and audit["j3_required"]
            and all(audit["checks"].values())):
        raise RuntimeError("J3 requires the complete regular-family audit and activation")
    for relative, expected in audit["artifact_sha256"].items():
        if regular.digest(ROOT / relative) != expected:
            raise RuntimeError(f"audited regular artifact changed: {relative}")
    config = regular.load_json(regular.CONFIG_PATH)
    model = RelationalCAURModel(regular.load_base_model(parent), (25, 8, 6))
    total, trainable = model.parameter_counts()
    limits = config["model_families"][FAMILY]
    if total > limits["total_parameter_cap"] or trainable > limits["trainable_parameter_cap"]:
        raise RuntimeError("J3 exceeds frozen parameter caps")
    protocol = copy.deepcopy(parent)
    protocol.update({
        "schema": "phase6j-caur-j3-training-protocol-v1",
        "status": "FROZEN_BEFORE_J3_OPTIMIZER_STEP",
        "parent_protocol_sha256": regular.digest(regular.PROTOCOL_PATH),
        "regular_training_audit_sha256": regular.digest(REGULAR_AUDIT),
        "j3_activation_sha256": regular.digest(regular.OUT / "j3_activation_decision.json"),
        "code_hashes": {**parent["code_hashes"], **{
            relative: regular.digest(ROOT / relative) for relative in CODE_FILES
        }},
        "families": {FAMILY: {
            "total_parameters": total, "trainable_parameters": trainable,
            "total_parameter_cap": limits["total_parameter_cap"],
            "trainable_parameter_cap": limits["trainable_parameter_cap"],
        }},
        "architecture": {
            "base": "same J2 last-block and action-projection trainability",
            "interaction_rank": 8,
            "added_parameters": sum(p.numel() for p in model.interaction.parameters()),
            "interaction": "W_out[tanh(W_action(a-f))*tanh(W_state(s)+W_origin(o))]",
            "placement": "residual added to candidate embedding before unchanged heads",
            "biases": False, "fallback_residual": "exactly zero when a=f",
            "candidate_fallback_projection_shared": True,
            "origin": "shared three categorical embeddings (12 coordinates)",
            "design_scope": "single implementation of conditionally registered J3; no architecture sweep",
        },
        "epoch_selection": "same nested folds, patience and fixed outer refit as J1/J2",
        "expected_runs": 9,
        "latency_protocol": {
            "device": "cuda", "state_scope": "all 288 R12 cached full banks",
            "batch_size": 1, "ensemble_seeds": config["rng"]["model_seeds"],
            "models_per_decision": 3, "warmup_decisions": 10,
            "measured_repetitions": 3, "synchronize_cuda": True,
            "neural_scope": "prepacked on-device inputs; all three models",
            "cached_total_scope": "CPU batch, transfer, three models, output, calibration and gate",
            "live_total_decision_requires_separate_graph_and_proposal_measurement": True,
        },
        "r13_accessed": False, "r14_accessed": False,
    })
    # The parent commit belongs to the parent protocol; this addendum hashes its own code.
    protocol["parent_freeze_implementation_commit"] = protocol.pop("freeze_implementation_commit")
    protocol["training"]["learning_rate"] = {FAMILY: config["training"]["learning_rate"][FAMILY]}
    regular.atomic_json(protocol, PROTOCOL_PATH)
    return protocol


def validate_protocol() -> dict:
    regular.validate_protocol()
    protocol = regular.load_json(PROTOCOL_PATH)
    if not (
        protocol["schema"] == "phase6j-caur-j3-training-protocol-v1"
        and protocol["status"] == "FROZEN_BEFORE_J3_OPTIMIZER_STEP"
        and protocol["parent_protocol_sha256"] == regular.digest(regular.PROTOCOL_PATH)
        and protocol["regular_training_audit_sha256"] == regular.digest(REGULAR_AUDIT)
        and protocol["j3_activation_sha256"] == regular.digest(regular.OUT / "j3_activation_decision.json")
    ):
        raise RuntimeError("J3 training authorization changed")
    for relative, expected in protocol["code_hashes"].items():
        if regular.digest(ROOT / relative) != expected:
            raise RuntimeError(f"J3 frozen code changed: {relative}")
    manifest = pd.read_csv(regular.CACHE / "tensor_manifest.csv")
    if regular.digest(regular.CACHE / "tensor_manifest.csv") != protocol["input_hashes"]["tensor_manifest"]:
        raise RuntimeError("J3 tensor manifest changed")
    for row in manifest.itertuples(index=False):
        if regular.digest(Path(row.cache_path)) != row.cache_sha256:
            raise RuntimeError(f"J3 tensor shard changed: {row.cache_path}")
    return protocol


def initialize(seed, transform, protocol, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sizes = tuple(len(transform.vocabularies[name]) + 1 for name in regular.CATEGORICAL_COLUMNS)
    model = RelationalCAURModel(regular.load_base_model(protocol), sizes)
    for section in (model.heads, model.interaction):
        for module in section.modules():
            if isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[0].zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    expected = protocol["families"][FAMILY]
    if model.parameter_counts() != (expected["total_parameters"], expected["trainable_parameters"]):
        raise RuntimeError("J3 parameter boundary changed")
    return model.to(device)


def checkpoint_state(model):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable(model, state):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise RuntimeError("J3 checkpoint has missing or unexpected trainable parameters")
    for value in state.values():
        if not torch.isfinite(value).all():
            raise RuntimeError("non-finite J3 checkpoint tensor")
    model.load_state_dict(state, strict=False)


def optimize(seed, fold, train_ids, validation_ids, samples, frames, transform,
             protocol, device, epochs, on_epoch):
    model = initialize(seed, transform, protocol, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    training = protocol["training"]
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"][FAMILY],
                                 weight_decay=training["weight_decay"])
    best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
    history = []
    width = training["state_groups_per_batch"]
    inner = validation_ids is not None
    salt = fold * 10_007 + (101 if inner else 202)
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1_000_003 + salt)
        ids = [train_ids[index] for index in rng.permutation(len(train_ids))]
        losses_by_name = {}
        for start in range(0, len(ids), width):
            packed = regular.build_batch(ids[start:start + width], samples, frames, transform, device)
            optimizer.zero_grad(set_to_none=True)
            losses = regular.loss_for_batch(model, packed, protocol)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("non-finite J3 loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, training["gradient_norm_clip"], error_if_nonfinite=True)
            optimizer.step()
            for name, value in losses.items():
                if name != "pair_count":
                    losses_by_name.setdefault(name, []).append(float(value.detach()))
        row = {"epoch": epoch, **{name: float(np.mean(values))
                                for name, values in losses_by_name.items()}}
        if inner:
            predictions = regular.predict(model, validation_ids, samples, frames, transform, protocol, device)
            score, metrics = regular.validation_score(predictions, protocol)
            row.update({f"validation_{name}": value for name, value in metrics.items()})
        else:
            score = float(epoch)
        history.append(row)
        if best_state is None or score > best_score + training["minimum_lcb_improvement"]:
            best_score, best_epoch, best_state, stale = score, epoch, checkpoint_state(model), 0
        else:
            stale += 1
        event = {"event": "phase6j_j3_epoch", "family": FAMILY, "seed": seed,
                 "held_fold": fold, "mode": "INNER_EPOCH_SELECTION" if inner else "OUTER_FINAL_FIT",
                 "best_epoch": best_epoch, "stale_epochs": stale, **row}
        print(json.dumps(event), flush=True)
        on_epoch(event)
        if inner and stale >= training["patience"]:
            break
    if best_state is None:
        raise RuntimeError("J3 produced no checkpoint")
    restore_trainable(model, best_state)
    return model, history, best_epoch


def train_fold(seed, fold, samples, frames, source, protocol, device, smoke, on_epoch):
    outer_ids = sorted(key for key, value in frames.items() if int(value.oof_fold.iloc[0]) != fold)
    held_ids = sorted(set(frames) - set(outer_ids))
    fit_fold, validation_fold = regular.nested_fold_roles(fold)
    fit_ids = sorted(key for key in outer_ids if int(frames[key].oof_fold.iloc[0]) == fit_fold)
    validation_ids = sorted(set(outer_ids) - set(fit_ids))
    transform = regular.fit_feature_transform(source[source.state_id.isin(fit_ids)])
    model, inner_history, selected_epoch = optimize(
        seed, fold, fit_ids, validation_ids, samples, frames, transform, protocol, device,
        2 if smoke else protocol["training"]["maximum_epochs"], on_epoch,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    transform = regular.fit_feature_transform(source[source.state_id.isin(outer_ids)])
    model, outer_history, last_epoch = optimize(
        seed, fold, outer_ids, None, samples, frames, transform, protocol, device,
        selected_epoch, on_epoch,
    )
    if last_epoch != selected_epoch:
        raise RuntimeError("J3 outer epoch count differs from inner selection")
    predictions = regular.predict(model, held_ids, samples, frames, transform, protocol, device)
    return model, transform, predictions, {
        "inner_training_fold": fit_fold, "inner_validation_fold": validation_fold,
        "inner_epoch_selection": inner_history, "outer_final_fit": outer_history,
    }, selected_epoch


def completed_runs(protocol_sha, root=OUT):
    protocol = regular.load_json(PROTOCOL_PATH)
    result = []
    for seed in protocol["training"]["seeds"]:
        for fold in range(3):
            paths = regular.run_paths(FAMILY, seed, fold, root=root)
            if regular.valid_run(paths, protocol_sha):
                result.append((seed, fold, paths))
            elif paths[2].exists():
                raise RuntimeError(f"existing J3 run record is invalid: {paths[2]}")
    return result


def summarize(protocol):
    paths = completed_runs(regular.digest(PROTOCOL_PATH))
    if len(paths) != 9:
        raise RuntimeError("J3 requires all nine OOF runs")
    frames = [pd.read_parquet(row[2][1]) for row in paths]
    combined = pd.concat(frames, ignore_index=True)
    source = pd.read_parquet(regular.SOURCE_PATH)
    lookup = source.set_index(["state_id", "target_set_id"]).sort_index()
    for seed, group in combined.groupby("training_seed", sort=True):
        actual = group.set_index(["state_id", "target_set_id"]).sort_index()
        pd.testing.assert_index_equal(lookup.index, actual.index)
        for column in source.columns.difference(["state_id", "target_set_id"]):
            pd.testing.assert_series_equal(lookup[column], actual[column])
        if not np.isfinite(group.filter(regex="^predicted_").to_numpy(float)).all():
            raise RuntimeError(f"non-finite J3 predictions for seed {seed}")
    result = regular.ensemble_family(combined, FAMILY, protocol)
    regular.atomic_parquet(combined, OUT / "oof_predictions.parquet")
    for name, field in (("ensemble_oof", "ensemble"), ("state_metrics", "states"),
                        ("selected_winners", "selected_winners")):
        regular.atomic_parquet(result[field], OUT / f"{name}.parquet")
    regular.atomic_csv(result["calibration_table"], OUT / "calibration_metrics.csv")
    regular.atomic_csv(result["gate_table"], OUT / "gate_grid.csv")
    stability = []
    for seed, frame in combined.groupby("training_seed", sort=True):
        state = regular.ranking_state_metrics(frame, "predicted_continuation_advantage")
        stability.append({"model_family": FAMILY, "training_seed": int(seed), **{
            name: float(state[name].mean()) for name in
            ("spearman", "pairwise_accuracy", "ndcg_at_1", "selected_lift", "selection_regret")
        }})
    regular.atomic_csv(pd.DataFrame(stability), OUT / "three_seed_stability.csv")
    summary = {"schema": "phase6j-caur-j3-oof-summary-v1", "status": "COMPLETE_J3_OOF",
               "model_family": FAMILY, "training_protocol_sha256": regular.digest(PROTOCOL_PATH),
               "metrics": result["metrics"], "calibration": result["calibration"],
               "selected_gate": result["selected_gate"], "r13_accessed": False, "r14_accessed": False,
               "r13_eligible": False, "remaining": ["completion audit", "latency", "eligibility and deployable bundle"]}
    regular.atomic_json(summary, OUT / "oof_summary.json")
    return summary


def run_training(*, smoke=False, max_new_runs=None, device_name="cuda"):
    OUT.mkdir(parents=True, exist_ok=True)
    # The worker itself owns the lock through every optimizer step and summary write.
    with (OUT / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a J3 training worker already owns the lock") from exc
        return train_locked(smoke, max_new_runs, device_name)


def train_locked(smoke, max_new_runs, device_name):
    protocol = validate_protocol()
    protocol_sha = regular.digest(PROTOCOL_PATH)
    root = OUT / "smoke" if smoke else OUT
    prior = [] if smoke else completed_runs(protocol_sha)
    done = {(seed, fold) for seed, fold, _ in prior}
    started = time.perf_counter()
    progress = {
        "schema": "phase6j-caur-j3-training-progress-v1", "status": "RUNNING",
        "worker_pid": os.getpid(), "completed_runs": len(done), "expected_runs": 1 if smoke else 9,
        "new_runs": 0, "training_protocol_sha256": protocol_sha,
        "r13_accessed": False, "r14_accessed": False,
    }

    def update(event=None):
        if event:
            progress["current"] = {key: event[key] for key in
                                   ("seed", "held_fold", "mode", "epoch", "best_epoch")}
        progress["elapsed_seconds"] = time.perf_counter() - started
        progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        regular.atomic_json(progress, root / "progress.json")

    update()
    print(json.dumps({"event": "PHASE6J_J3_START", **progress}), flush=True)
    status_path = root / "worker_status.json"
    regular.atomic_json({**progress, "exit_code": None}, status_path)
    try:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("J3 CUDA runtime unavailable")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(4)
        samples = regular.load_samples()
        source = pd.read_parquet(regular.SOURCE_PATH)
        source["oof_fold"] = [regular.grouped_oof_fold(str(s), str(c))
                              for s, c in zip(source.scale, source.CF_level)]
        frames = regular.state_frames(source)
        tasks = [(seed, fold) for seed in protocol["training"]["seeds"] for fold in range(3)
                 if (seed, fold) not in done]
        if smoke:
            tasks = tasks[:1]
        elif max_new_runs is not None:
            tasks = tasks[:max_new_runs]
        for seed, fold in tasks:
            begin = time.perf_counter()
            model, transform, held, history, best_epoch = train_fold(
                seed, fold, samples, frames, source, protocol, device, smoke, update
            )
            held["model_family"], held["training_seed"] = FAMILY, seed
            held["held_fold"], held["best_epoch"] = fold, best_epoch
            paths = regular.run_paths(FAMILY, seed, fold, root=root)
            regular.save_checkpoint({
                "schema": "phase6j-caur-j3-oof-checkpoint-v1", "model_family": FAMILY,
                "training_seed": seed, "held_fold": fold, "training_protocol_sha256": protocol_sha,
                "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"],
                "feature_transform": transform.to_dict(), "trainable_model_state": checkpoint_state(model),
            }, paths[0])
            regular.atomic_parquet(held, paths[1])
            record = {
                "schema": "phase6j-caur-j3-oof-run-v1", "status": "COMPLETE",
                "model_family": FAMILY, "training_seed": seed, "held_fold": fold,
                "best_epoch": best_epoch, "inner_epochs_run": len(history["inner_epoch_selection"]),
                "outer_final_epochs_run": len(history["outer_final_fit"]), "history": history,
                "runtime_seconds": time.perf_counter() - begin,
                "training_protocol_sha256": protocol_sha, "checkpoint_sha256": regular.digest(paths[0]),
                "predictions_sha256": regular.digest(paths[1]), "r13_accessed": False, "r14_accessed": False,
            }
            regular.atomic_json(record, paths[2])
            progress["completed_runs"] += 1
            progress["new_runs"] += 1
            update()
            print(json.dumps({"event": "phase6j_j3_run_complete", "seed": seed, "held_fold": fold,
                              "best_epoch": best_epoch, "runtime_seconds": record["runtime_seconds"]}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if smoke:
            progress["status"] = "SMOKE_COMPLETE"
        elif progress["completed_runs"] == 9:
            progress["status"] = "SUMMARIZING"
            update()
            summarize(protocol)
            progress["status"] = "COMPLETE_J3_OOF"
        else:
            progress["status"] = "PAUSED_AT_RUN_BOUNDARY"
        update()
    except BaseException as exc:
        progress.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        update()
        regular.atomic_json({**progress, "exit_code": 1}, status_path)
        raise
    regular.atomic_json({**progress, "exit_code": 0}, status_path)
    print(json.dumps(progress, indent=2), flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "smoke", "train"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs <= 0:
        parser.error("--max-new-runs must be positive")
    if args.mode == "freeze":
        protocol = freeze_protocol()
        print(json.dumps({"status": protocol["status"], "sha256": regular.digest(PROTOCOL_PATH),
                          "families": protocol["families"], "architecture": protocol["architecture"]}, indent=2))
    else:
        run_training(smoke=args.mode == "smoke", max_new_runs=args.max_new_runs, device_name=args.device)


if __name__ == "__main__":
    main()
