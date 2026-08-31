#!/usr/bin/env python3
"""Train/resume one Phase 6E supervised Neural Improvement run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.losses import NILossConfig  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.ni.trainer import NITrainer, NITrainingConfig  # noqa: E402


def merged(base: dict, override: dict) -> dict:
    return {**base, **override}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="H96_L2_LR3E4")
    parser.add_argument("--seed", type=int, default=660201)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--variant",
        choices=("FULL_CSG", "FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES"),
        default="FULL_CSG",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-shards", type=int)
    parser.add_argument("--max-validation-shards", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase6e_training.json"
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/training"
    )
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        candidate = raw["development_candidates"][args.candidate]
    except KeyError as error:
        raise ValueError(f"unknown development candidate: {args.candidate}") from error
    model_config = NIModelConfig(**candidate["model"])
    if args.variant == "FLAT_SET":
        model_config = replace(model_config, message_passing=False)
    elif args.variant == "STATIC_CSG":
        model_config = replace(model_config, relation_mode="STATIC_CSG")
    elif args.variant == "NO_EDGE_FEATURES":
        model_config = replace(model_config, use_edge_features=False)
    training_values = merged(raw["training_defaults"], candidate.get("training", {}))
    if args.epochs is not None:
        training_values["epochs"] = args.epochs
    training_config = NITrainingConfig(**training_values)

    manifest = pd.read_csv(args.cache_manifest)
    manifest = manifest[manifest["status"] == "COMPLETE"].copy()
    train = manifest[manifest["training_split"] == "TRAIN"]
    validation = manifest[manifest["training_split"] == "TRAIN_VALIDATION"]
    if args.max_train_shards is not None:
        train = train.sort_values("instance_id").head(args.max_train_shards)
    if args.max_validation_shards is not None:
        validation = validation.sort_values("instance_id").head(args.max_validation_shards)
    if train.empty or validation.empty:
        raise ValueError("cache manifest must include both TRAIN and TRAIN_VALIDATION shards")
    positive = float(train["positive_count"].sum())
    actions = float(train["action_count"].sum())
    loss_config = NILossConfig(
        **raw["loss"], positive_weight=max((actions - positive) / max(positive, 1), 1.0)
    )
    run_name = args.run_name or f"{args.candidate}_seed{args.seed}"
    output = args.output_root / run_name
    tensorizer = CSGTensorizer()
    model = CSGTargetSetScorer(tensorizer, model_config)
    trainer = NITrainer(
        model,
        model_config=model_config,
        loss_config=loss_config,
        training_config=training_config,
        device=torch.device(args.device),
        seed=args.seed,
        output_directory=output,
    )
    launch = {
        "schema": "phase6e-training-launch-v1",
        "candidate": args.candidate,
        "variant": args.variant,
        "seed": args.seed,
        "device": args.device,
        "train_shards": len(train),
        "validation_shards": len(validation),
        "train_states": int(train["state_count"].sum()),
        "validation_states": int(validation["state_count"].sum()),
        "train_actions": int(train["action_count"].sum()),
        "validation_actions": int(validation["action_count"].sum()),
        "positive_weight": loss_config.positive_weight,
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "training": asdict(training_config),
        "parameter_count": model.parameter_count(),
        "internal_holdout_shards_loaded": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch_record.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "launch", **launch}), flush=True)
    summary = trainer.fit(train, validation, resume_path=args.resume)
    print(json.dumps({"event": "training_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
