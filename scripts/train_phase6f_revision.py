#!/usr/bin/env python3
"""Train one leakage-safe Phase 6F objective/model candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.revision_trainer import Phase6FTrainer  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.ni.trainer import NITrainingConfig  # noqa: E402
from rcias_clgri.ni.utility_losses import Phase6FLossConfig  # noqa: E402


CONFIG = ROOT / "configs" / "phase6f_revision.json"
CONSTANTS = ROOT / "outputs" / "phase6f" / "audit" / "training_constants.json"
PHASE6E_FREEZE = ROOT / "outputs" / "phase6e" / "audit" / "experiment_freeze.json"
TEACHER_MANIFEST = (
    ROOT / "outputs" / "phase6e" / "training" / "final_seeds" / "checkpoint_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_teachers(tensorizer: CSGTensorizer) -> tuple[list[CSGTargetSetScorer], str]:
    manifest = read_json(TEACHER_MANIFEST)
    if manifest.get("status") != "COMPLETE" or len(manifest.get("checkpoints", [])) != 3:
        raise ValueError("frozen Phase 6E teacher manifest is incomplete")
    models = []
    for record in manifest["checkpoints"]:
        path = Path(record["checkpoint_path"])
        if sha256_file(path) != record["checkpoint_sha256"]:
            raise ValueError(f"teacher checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = CSGTargetSetScorer(
            tensorizer, NIModelConfig(**checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model_state"])
        models.append(model)
    return models, sha256_file(TEACHER_MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=660250)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-shards", type=int)
    parser.add_argument("--max-validation-shards", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cache-manifest", type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6f/development/runs"
    )
    parser.add_argument("--run-name")
    args = parser.parse_args()

    raw = read_json(CONFIG)
    constants = read_json(CONSTANTS)
    phase6e_freeze = read_json(PHASE6E_FREEZE)
    if constants.get("status") != "FROZEN_TRAIN_ONLY":
        raise ValueError("Phase 6F TRAIN-only constants are not frozen")
    try:
        objective = raw["objective_candidates"][args.objective]
        compact = raw["compact_model_candidates"][args.model]
    except KeyError as error:
        raise ValueError(f"unknown frozen Phase 6F candidate: {error}") from error
    if args.distillation_weight not in raw["distillation"]["candidate_weights"]:
        raise ValueError("distillation weight is outside the frozen candidate set")

    model_config = NIModelConfig(
        **compact,
        use_edge_features=True,
        relation_mode="FULL_CSG",
        message_passing=True,
        utility_head=args.objective == "O3_UTILITY_AWARE_MULTITASK",
    )
    selected_training = dict(phase6e_freeze["selected_training"])
    selected_training["epochs"] = int(
        args.epochs or raw["development_protocol"]["objective_study_epochs"]
    )
    training_config = NITrainingConfig(**selected_training)

    manifest = pd.read_csv(args.cache_manifest)
    complete = manifest[manifest["status"].eq("COMPLETE")].copy()
    train = complete[complete["training_split"].eq("TRAIN")].sort_values("instance_id")
    validation = complete[
        complete["training_split"].eq("TRAIN_VALIDATION")
    ].sort_values("instance_id")
    if args.max_train_shards is not None:
        train = train.head(args.max_train_shards)
    if args.max_validation_shards is not None:
        validation = validation.head(args.max_validation_shards)
    if train.empty or validation.empty:
        raise ValueError("TRAIN and TRAIN_VALIDATION caches are both required")
    if set(train["training_split"]) != {"TRAIN"} or set(validation["training_split"]) != {"TRAIN_VALIDATION"}:
        raise ValueError("Phase 6F development split leakage")
    if any("revision_holdout" in str(path) for path in pd.concat([train, validation])["cache_path"]):
        raise ValueError("R06 cache access is forbidden during model development")

    loss_config = Phase6FLossConfig(
        objective=args.objective,
        rank_weight=float(objective["rank_weight"]),
        classification_weight=float(objective["classification_weight"]),
        utility_weight=float(objective["utility_weight"]),
        positive_weight=float(constants["positive_weight"]),
        rank_gap_scale=float(constants["rank_gap_scale_train_median_nonzero_pairwise"]),
        rank_weight_min=float(constants["rank_weight_min"]),
        rank_weight_max=float(constants["rank_weight_max"]),
        utility_clip=float(constants["utility_clip_absolute_train_p99"]),
        distillation_weight=args.distillation_weight,
    )
    tensorizer = CSGTensorizer()
    model = CSGTargetSetScorer(tensorizer, model_config)
    teachers: list[CSGTargetSetScorer] = []
    teacher_manifest_hash = None
    if args.distillation_weight:
        teachers, teacher_manifest_hash = load_teachers(tensorizer)

    run_name = args.run_name or (
        f"{args.objective}__{args.model}__D{args.distillation_weight:g}__seed{args.seed}"
    )
    output = args.output_root / run_name
    output.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema": "phase6f-training-launch-v1",
        "status": "RUNNING",
        "objective": args.objective,
        "model_candidate": args.model,
        "distillation_weight": args.distillation_weight,
        "seed": args.seed,
        "device": args.device,
        "train_shards": len(train),
        "validation_shards": len(validation),
        "train_states": int(train["state_count"].sum()),
        "validation_states": int(validation["state_count"].sum()),
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "training": asdict(training_config),
        "parameter_count": model.parameter_count(),
        "phase6f_config_sha256": sha256_file(CONFIG),
        "training_constants_sha256": sha256_file(CONSTANTS),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "teacher_manifest_sha256": teacher_manifest_hash,
        "gradient_split": "TRAIN",
        "selection_split": "TRAIN_VALIDATION",
        "train_internal_holdout_accessed": False,
        "revision_holdout_accessed": False,
    }
    (output / "launch_record.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "launch", **launch}), flush=True)
    trainer = Phase6FTrainer(
        model,
        model_config=model_config,
        loss_config=loss_config,
        training_config=training_config,
        teacher_models=teachers,
        device=torch.device(args.device),
        seed=args.seed,
        output_directory=output,
    )
    summary = trainer.fit(train, validation, resume_path=args.resume)
    print(json.dumps({"event": "training_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
