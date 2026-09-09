"""Frozen Phase 6N OOF critic inference for the Phase 6P live candidate bank."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from rcias_clgri.analysis.phase6j_caur import (
    critical_and_bottleneck_operations,
    grouped_oof_fold,
)
from rcias_clgri.analysis.phase6l_legacy_score import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_score_free_candidate_source_features,
    select_score_free_fallback,
)
from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.instance import Instance
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.encoder import NIModelConfig
from rcias_clgri.ni.phase6n_candidate_conditioned import (
    FAMILY,
    CandidateConditionedCSGModel,
)
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank
from rcias_clgri.ni.scorer import CSGTargetSetScorer
from rcias_clgri.ni.tensorize import CSGTensorizer
from rcias_clgri.search.alns import DESTROY
from rcias_clgri.search.common import DecodedCandidate
from rcias_clgri.search.phase6c import ArmGenerationResult, Phase6CTargetArm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/phase6p_adaptive_portfolio_v1/preregistration/phase6n_checkpoint_manifest.json"
)
PHASE6N_PROTOCOL = (
    ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training/training_protocol.json"
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def origin_destroy_operators(
    generated: ArmGenerationResult,
) -> dict[str, tuple[str, ...]]:
    """Map each deduplicated target to all unique destroy-operator origins."""
    result: dict[str, tuple[str, ...]] = {}
    for arm in generated.arms:
        origins = tuple(dict.fromkeys(
            proposal.origin_destroy_operator
            for proposal in generated.proposals
            if proposal.destroyed_operations == arm.destroyed_operations
        ))
        if not origins or not set(origins) <= set(DESTROY):
            raise RuntimeError(f"invalid destroy provenance for {arm.target_set_id}")
        result[arm.target_set_id] = origins
    if set(result) != {arm.target_set_id for arm in generated.arms}:
        raise RuntimeError("incomplete Phase 6P target provenance")
    return result


@dataclass(frozen=True)
class FeatureTransform:
    vocabularies: Mapping[str, tuple[str, ...]]
    medians: Mapping[str, float]
    iqrs: Mapping[str, float]

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> "FeatureTransform":
        if tuple(payload["categorical_columns"]) != CATEGORICAL_COLUMNS:
            raise RuntimeError("Phase 6N categorical feature schema drifted")
        if tuple(payload["numeric_columns"]) != NUMERIC_COLUMNS:
            raise RuntimeError("Phase 6N numeric feature schema drifted")
        if int(payload["unknown_category_index"]) != 0:
            raise RuntimeError("Phase 6N unknown-category semantics drifted")
        if [float(value) for value in payload["numeric_clip"]] != [-8.0, 8.0]:
            raise RuntimeError("Phase 6N numeric clipping semantics drifted")
        return cls(
            {
                column: tuple(str(value) for value in payload["vocabularies"][column])
                for column in CATEGORICAL_COLUMNS
            },
            {column: float(payload["medians"][column]) for column in NUMERIC_COLUMNS},
            {column: float(payload["iqrs"][column]) for column in NUMERIC_COLUMNS},
        )

    def transform(
        self, rows: Sequence[Mapping[str, object]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        categorical = np.empty((len(rows), len(CATEGORICAL_COLUMNS)), dtype=np.int64)
        numeric = np.empty((len(rows), len(NUMERIC_COLUMNS)), dtype=np.float32)
        supported = np.ones(len(rows), dtype=bool)
        for column_index, column in enumerate(CATEGORICAL_COLUMNS):
            mapping = {
                value: index + 1
                for index, value in enumerate(self.vocabularies[column])
            }
            values = np.asarray(
                [mapping.get(str(row[column]), 0) for row in rows], dtype=np.int64
            )
            categorical[:, column_index] = values
            supported &= values > 0
        for column_index, column in enumerate(NUMERIC_COLUMNS):
            values = np.asarray([float(row[column]) for row in rows], dtype=float)
            robust = (values - self.medians[column]) / self.iqrs[column]
            supported &= np.isfinite(robust) & (robust >= -8.0) & (robust <= 8.0)
            numeric[:, column_index] = np.clip(robust, -8.0, 8.0).astype(np.float32)
        return categorical, numeric, supported


@dataclass(frozen=True)
class LoadedCritic:
    training_seed: int
    held_fold: int
    checkpoint_sha256: str
    transform: FeatureTransform
    model: CandidateConditionedCSGModel


@dataclass(frozen=True)
class Phase6PBankScores:
    state_id: str
    generated: ArmGenerationResult
    fallback: Phase6CTargetArm
    target_scores: Mapping[str, float]
    per_seed_scores: Mapping[int, Mapping[str, float]]
    ranked_target_ids: tuple[str, ...]
    top_target_ids: tuple[str, ...]
    origin_destroy_operators: Mapping[str, tuple[str, ...]]
    supported_by_seed: Mapping[int, tuple[bool, ...]]
    graph_hash: str
    held_fold: int
    timings_ms: Mapping[str, float]


class FrozenPhase6NCriticEnsemble:
    """Load and apply the nine frozen OOF models under whole-instance routing."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        manifest_path: Path = DEFAULT_MANIFEST,
        shortlist_k: int = 6,
    ) -> None:
        if shortlist_k != 6:
            raise ValueError("Phase 6P preregisters shortlist_k=6")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Phase 6P critic supports CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        self.shortlist_k = shortlist_k
        self.tensorizer = CSGTensorizer(dtype=torch.float32)
        self.manifest_path = Path(manifest_path)
        self.manifest = _load_json(self.manifest_path)
        self.protocol = _load_json(PHASE6N_PROTOCOL)
        self._validate_manifest()
        self.models_by_fold = self._load_models()

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema") != "phase6p-phase6n-checkpoint-manifest-v1"
            or manifest.get("status") != "FROZEN_BEFORE_PHASE6P_SOLVER_OUTCOMES"
            or manifest.get("family") != FAMILY
            or manifest.get("precision") != "FP32_ONLY"
            or manifest.get("historical_score_calls") != 0
            or len(manifest.get("checkpoints", ())) != 9
        ):
            raise RuntimeError("invalid frozen Phase 6P checkpoint manifest")
        if _digest(PHASE6N_PROTOCOL) != manifest["phase6n_training_protocol_sha256"]:
            raise RuntimeError("Phase 6N training protocol changed after Phase 6P freeze")
        expected_route = {
            f"{scale}_{cf}": grouped_oof_fold(scale, cf)
            for scale in ("S", "M", "L")
            for cf in ("CF1", "CF2", "CF3")
        }
        if manifest.get("fold_route") != expected_route:
            raise RuntimeError("Phase 6N whole-instance fold route drifted")

    def _base_model(self) -> CSGTargetSetScorer:
        record = self.protocol["base_checkpoint"]
        path = ROOT / record["path"]
        if _digest(path) != record["sha256"]:
            raise RuntimeError("Phase 6N base checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = CSGTargetSetScorer(
            CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model_state"])
        return model

    def _load_models(self) -> dict[int, tuple[LoadedCritic, ...]]:
        by_fold: dict[int, list[LoadedCritic]] = {0: [], 1: [], 2: []}
        seen = set()
        for record in self.manifest["checkpoints"]:
            held_fold = int(record["held_fold"])
            training_seed = int(record["training_seed"])
            key = (training_seed, held_fold)
            if key in seen:
                raise RuntimeError(f"duplicate Phase 6N checkpoint route: {key}")
            seen.add(key)
            path = ROOT / record["path"]
            if _digest(path) != record["sha256"]:
                raise RuntimeError(f"Phase 6N checkpoint changed: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            transform_payload = checkpoint["feature_transform"]
            transform_hash = hashlib.sha256(json.dumps(
                transform_payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            if (
                checkpoint.get("schema") != "phase6n-oof-checkpoint-v1"
                or checkpoint.get("model_family") != FAMILY
                or int(checkpoint.get("held_fold")) != held_fold
                or int(checkpoint.get("training_seed")) != training_seed
                or checkpoint.get("base_checkpoint_sha256")
                != self.protocol["base_checkpoint"]["sha256"]
                or transform_hash != record["feature_transform_sha256"]
            ):
                raise RuntimeError(f"Phase 6N checkpoint contract drifted: {path}")
            transform = FeatureTransform.from_checkpoint(transform_payload)
            categorical_sizes = tuple(
                len(transform.vocabularies[column]) + 1
                for column in CATEGORICAL_COLUMNS
            )
            model = CandidateConditionedCSGModel(
                self._base_model(), categorical_sizes, family=FAMILY
            )
            incompatible = model.load_state_dict(
                checkpoint["trainable_model_state"], strict=False
            )
            expected_missing = {
                name for name in model.state_dict()
                if name.startswith("state_encoder.")
                and not name.startswith("state_encoder.layers.1")
            }
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise RuntimeError(f"incomplete Phase 6N model restore: {path}")
            model.to(self.device).eval()
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise RuntimeError("Phase 6P J1 inference must remain FP32")
            by_fold[held_fold].append(LoadedCritic(
                training_seed, held_fold, str(record["sha256"]), transform, model
            ))
        if seen != {(seed, fold) for seed in (726101, 726102, 726103) for fold in range(3)}:
            raise RuntimeError("Phase 6P requires the exact three-seed by three-fold ensemble")
        result = {
            fold: tuple(sorted(models, key=lambda item: item.training_seed))
            for fold, models in by_fold.items()
        }
        if any(len(models) != 3 for models in result.values()):
            raise RuntimeError("Phase 6P fold ensemble must contain three seeds")
        return result

    def prepare_instance(self, instance: Instance, schedule) -> None:
        del schedule
        self._route(instance)

    @staticmethod
    def _route(instance: Instance) -> int:
        scale = str(instance.metadata.get("scale", ""))
        cf_level = str(instance.metadata.get("CF_level", ""))
        return grouped_oof_fold(scale, cf_level)

    def score_bank(
        self,
        instance: Instance,
        current: DecodedCandidate,
        *,
        state_id: str,
        destroy_count: int,
        search_progress: float,
        search_stage: str,
        proposal_seed_namespace: int,
    ) -> Phase6PBankScores:
        started = time.perf_counter()
        generated, action_records = build_live_proposal_bank(
            instance,
            current,
            state_id=state_id,
            destroy_count=destroy_count,
            seed_namespace=proposal_seed_namespace,
        )
        bank_finished = time.perf_counter()
        if generated.requested_arm_count != 24:
            raise RuntimeError("Phase 6P requires the complete 24-rule bank")
        fallback = select_score_free_fallback(generated)
        critical, bottleneck, bottleneck_proxy = critical_and_bottleneck_operations(
            instance, current
        )
        feature_rows = build_score_free_candidate_source_features(
            generated,
            state_id=state_id,
            operation_count=instance.num_operations,
            critical_operations=critical,
            bottleneck_operations=bottleneck,
        )
        graph = build_csg_from_schedule(
            instance,
            current.schedule,
            state_id=state_id,
            search_progress=search_progress,
            search_stage=search_stage,
            natural_bottleneck_proxy=bottleneck_proxy,
            diagnostic_metadata={
                "scale": instance.metadata.get("scale"),
                "CF_level": instance.metadata.get("CF_level"),
                "RI_level": instance.metadata.get("RI_level"),
                "TI_level": instance.metadata.get("TI_level"),
            },
        )
        actions = tensorize_action_records(graph, action_records)
        tensor_graph = self.tensorizer.tensorize(graph)
        if actions.target_set_ids != tuple(row["target_set_id"] for row in feature_rows):
            raise RuntimeError("Phase 6P action and feature order diverged")
        sample = NIStateSample(
            tensor_graph,
            actions,
            {
                "scale": str(instance.metadata.get("scale", "")),
                "CF_level": str(instance.metadata.get("CF_level", "")),
                "search_stage": search_stage,
                "bottleneck_proxy": bottleneck_proxy,
            },
        )
        batch = batch_state_samples([sample]).to(self.device)
        operation_keys = tensor_graph.node_keys["OP"]
        critical_set, bottleneck_set = set(critical), set(bottleneck)
        critical_mask = torch.tensor(
            [operation in critical_set for operation in operation_keys],
            dtype=torch.bool,
            device=self.device,
        )
        bottleneck_mask = torch.tensor(
            [operation in bottleneck_set for operation in operation_keys],
            dtype=torch.bool,
            device=self.device,
        )
        fallback_matches = [
            index for index, row in enumerate(feature_rows) if bool(row["is_fallback"])
        ]
        if len(fallback_matches) != 1:
            raise RuntimeError("Phase 6P bank must contain one canonical fallback")
        fallback_indices = torch.tensor(
            fallback_matches, dtype=torch.long, device=self.device
        )
        tensor_finished = time.perf_counter()

        held_fold = self._route(instance)
        target_ids = actions.target_set_ids
        per_seed: dict[int, dict[str, float]] = {}
        support: dict[int, tuple[bool, ...]] = {}
        with torch.inference_mode():
            for loaded in self.models_by_fold[held_fold]:
                categorical, numeric, supported = loaded.transform.transform(feature_rows)
                output = loaded.model(
                    batch,
                    fallback_action_indices=fallback_indices,
                    categorical=torch.as_tensor(
                        categorical, dtype=torch.long, device=self.device
                    ),
                    numeric=torch.as_tensor(
                        numeric, dtype=torch.float32, device=self.device
                    ),
                    critical_operation_mask=critical_mask,
                    bottleneck_operation_mask=bottleneck_mask,
                )
                values = output.advantage.float().cpu().numpy()
                if len(values) != len(target_ids) or not np.isfinite(values).all():
                    raise RuntimeError("Phase 6N critic returned incomplete/non-finite scores")
                per_seed[loaded.training_seed] = {
                    target_id: float(value)
                    for target_id, value in zip(target_ids, values)
                }
                support[loaded.training_seed] = tuple(bool(value) for value in supported)
        inference_finished = time.perf_counter()
        scores = {
            target_id: math.fsum(
                per_seed[seed][target_id] for seed in sorted(per_seed)
            ) / 3.0
            for target_id in target_ids
        }
        if set(scores) != set(target_ids) or not all(math.isfinite(x) for x in scores.values()):
            raise RuntimeError("Phase 6P ensemble score coverage failed")
        ranked = tuple(sorted(target_ids, key=lambda item: (-scores[item], item)))
        finished = time.perf_counter()
        return Phase6PBankScores(
            state_id=state_id,
            generated=generated,
            fallback=fallback,
            target_scores=scores,
            per_seed_scores=per_seed,
            ranked_target_ids=ranked,
            top_target_ids=ranked[: self.shortlist_k],
            origin_destroy_operators=origin_destroy_operators(generated),
            supported_by_seed=support,
            graph_hash=graph.graph_hash,
            held_fold=held_fold,
            timings_ms={
                "candidate_bank": (bank_finished - started) * 1000.0,
                "graph_features_tensorization": (tensor_finished - bank_finished) * 1000.0,
                "three_seed_inference": (inference_finished - tensor_finished) * 1000.0,
                "ranking": (finished - inference_finished) * 1000.0,
                "total": (finished - started) * 1000.0,
            },
        )
