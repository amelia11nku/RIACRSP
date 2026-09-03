"""Offline DTR training and online strict gate for LG_HGA-RIACRSP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

from .lghga_neighborhoods import NEIGHBORHOODS


class Regressor(Protocol):
    def predict(self, values): ...


@dataclass(frozen=True)
class DTRBundle:
    models: Mapping[str, Regressor]
    model_hashes: Mapping[str, str]
    knowledge_manifest_hash: str


def improvement_rate_pct(
    reference_makespans: Sequence[float],
    neighbor_makespans: Sequence[float],
) -> tuple[int, int, float]:
    if len(reference_makespans) != len(neighbor_makespans):
        raise ValueError("every LG_HGA neighbor requires one reference makespan")
    generated = len(neighbor_makespans)
    better = sum(
        neighbor < reference
        for reference, neighbor in zip(reference_makespans, neighbor_makespans)
    )
    return better, generated, 100.0 * better / generated if generated else 0.0


def _feature(generation_index: int, max_generations: int) -> list[list[float]]:
    if max_generations <= 0:
        raise ValueError("max_generations must be positive")
    return [[generation_index / max_generations]]


def train_dtr_bundle(
    rows: Sequence[Mapping[str, object]],
    *,
    random_state: int,
    knowledge_manifest_hash: str,
) -> DTRBundle:
    """Train one source-minimal DTR per neighborhood on normalized generation."""

    models: dict[str, DecisionTreeRegressor] = {}
    for index, neighborhood in enumerate(NEIGHBORHOODS):
        selected = [row for row in rows if row["neighborhood_id"] == neighborhood]
        if not selected:
            raise ValueError(f"no knowledge rows for {neighborhood}")
        x = np.asarray(
            [[float(row["normalized_generation_index"])] for row in selected], dtype=float
        )
        y = np.asarray([float(row["R_pct"]) for row in selected], dtype=float)
        model = DecisionTreeRegressor(random_state=random_state + index)
        model.fit(x, y)
        models[neighborhood] = model
    return DTRBundle(models, {}, knowledge_manifest_hash)


def predict_rates(
    bundle: DTRBundle,
    generation_index: int,
    max_generations: int,
) -> dict[str, float]:
    feature = _feature(generation_index, max_generations)
    return {
        neighborhood: float(bundle.models[neighborhood].predict(feature)[0])
        for neighborhood in NEIGHBORHOODS
    }


def select_neighborhood(
    rates: Mapping[str, float],
    threshold_pct: float,
) -> tuple[str, bool]:
    missing = set(NEIGHBORHOODS) - set(rates)
    if missing:
        raise ValueError(f"missing DTR predictions: {sorted(missing)}")
    selected = min(NEIGHBORHOODS, key=lambda item: (-float(rates[item]), item))
    return selected, float(rates[selected]) > threshold_pct


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dtr_bundle(bundle: DTRBundle, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for neighborhood in NEIGHBORHOODS:
        path = output_dir / f"{neighborhood}.joblib"
        joblib.dump(bundle.models[neighborhood], path)
        hashes[neighborhood] = _sha256(path)
    manifest = {
        "schema": "lghga-dtr-bundle-v1",
        "features": ["normalized_generation_index"],
        "neighborhoods": list(NEIGHBORHOODS),
        "model_hashes": hashes,
        "knowledge_manifest_hash": bundle.knowledge_manifest_hash,
        "estimator": "sklearn.tree.DecisionTreeRegressor",
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "effective_parameters": {
            neighborhood: bundle.models[neighborhood].get_params()
            for neighborhood in NEIGHBORHOODS
        },
    }
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_dtr_bundle(model_dir: Path) -> DTRBundle:
    manifest = json.loads((model_dir / "model_manifest.json").read_text())
    models = {}
    for neighborhood in NEIGHBORHOODS:
        path = model_dir / f"{neighborhood}.joblib"
        actual = _sha256(path)
        expected = manifest["model_hashes"][neighborhood]
        if actual != expected:
            raise ValueError(f"DTR model hash mismatch for {neighborhood}")
        models[neighborhood] = joblib.load(path)
    return DTRBundle(
        models,
        dict(manifest["model_hashes"]),
        str(manifest["knowledge_manifest_hash"]),
    )
