#!/usr/bin/env python3
"""Run the preregistered grouped-OOF pairwise representation probes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_phase6i_mr_training_data import atomic_json, digest, load_json  # noqa: E402
from scripts.train_phase6i_mr_heads import load_source  # noqa: E402


PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
OUT = ROOT / "outputs/phase6i_mr/model_training/representation_probe.json"


def pair_table(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    folds = []
    for _, group in frame.groupby("state_id", sort=True):
        ordered = group.sort_values("target_set_id", kind="stable")
        values = ordered.decoded_immediate_utility.to_numpy(dtype=float)
        matrix = ordered[columns].to_numpy(dtype=np.float32)
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                if values[first] == values[second]:
                    continue
                difference = matrix[first] - matrix[second]
                label = values[first] > values[second]
                features.extend([difference, -difference])
                labels.extend([label, not label])
                folds.extend([ordered.oof_fold.iloc[0], ordered.oof_fold.iloc[0]])
    return np.asarray(features), np.asarray(labels, dtype=int), np.asarray(folds)


def grouped_oof_accuracy(
    frame: pd.DataFrame, columns: list[str], fold_names: list[str]
) -> tuple[float, dict[str, float], int]:
    features, labels, folds = pair_table(frame, columns)
    fold_accuracy = {}
    correct = 0
    count = 0
    for fold in fold_names:
        train = folds != fold
        test = folds == fold
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=688001,
                solver="liblinear",
            ),
        )
        model.fit(features[train], labels[train])
        predictions = model.predict(features[test])
        hits = int(np.sum(predictions == labels[test]))
        fold_accuracy[fold] = hits / int(np.sum(test))
        correct += hits
        count += int(np.sum(test))
    return correct / count, fold_accuracy, count


def main() -> None:
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("r10_accessed") is not False or protocol.get("r11_accessed") is not False:
        raise RuntimeError("representation probes must precede protected-split access")
    frame = load_source("r09")
    embeddings = [f"embedding_{index:03d}" for index in range(128)]
    contexts = protocol["context_columns"]
    folds = list(protocol["grouped_oof_folds"])
    embedding_accuracy, embedding_folds, pair_count = grouped_oof_accuracy(
        frame, embeddings, folds
    )
    context_accuracy, context_folds, context_pair_count = grouped_oof_accuracy(
        frame, contexts, folds
    )
    rule = protocol["contrastive_activation_rule"]
    activate = embedding_accuracy < 0.55 and context_accuracy >= 0.60
    payload = {
        "schema": "phase6i-mr-representation-probe-v1.2",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_construction": "all non-tied within-state pairs in both orientations",
        "grouping": "the frozen three scale-CF R09 folds",
        "probe": "StandardScaler plus L2 logistic regression C=1.0",
        "frozen_embedding": {
            "dimension": 128,
            "pair_count": pair_count,
            "overall_accuracy": embedding_accuracy,
            "fold_accuracy": embedding_folds,
        },
        "normalized_context": {
            "dimension": len(contexts),
            "pair_count": context_pair_count,
            "overall_accuracy": context_accuracy,
            "fold_accuracy": context_folds,
            "interpretation": "state context is intentionally identical across actions in one state, so it cannot alone identify the better member of a pair",
        },
        "activation_rule": rule["activation"],
        "contrastive_activated": activate,
        "decision": "ACTIVATE_U3_CONTRASTIVE" if activate else "DO_NOT_ACTIVATE_U3_CONTRASTIVE",
        "source_hashes": {
            "training_protocol": digest(PROTOCOL_PATH),
            "r09_embedding_manifest": protocol["input_hashes"]["embedding_cache_manifest"],
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(payload, OUT)
    print(payload)


if __name__ == "__main__":
    main()
