"""Frozen single-model inference for live CSG-NI decisions."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.instance import Instance
from rcias_clgri.search.common import DecodedCandidate

from .batching import batch_state_samples
from .calibration import FrozenCalibrator
from .dataset import NIStateSample, tensorize_action_records
from .encoder import NIModelConfig
from .live_policy import InterventionDecision
from .proposal_bank import build_live_proposal_bank
from .scorer import CSGTargetSetScorer
from .tensorize import CSGTensorizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenLiveInference:
    """Load the Phase 6F artifact once and keep it resident on the chosen device."""

    def __init__(
        self,
        experiment_freeze: Path,
        *,
        device: str = "cuda",
        proposal_seed_namespace: int = 670102,
    ) -> None:
        freeze = json.loads(experiment_freeze.read_text(encoding="utf-8"))
        checkpoint_path = Path(str(freeze["selected_checkpoint_path"]))
        expected_hash = str(freeze["selected_checkpoint_sha256"])
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != expected_hash:
            raise ValueError("frozen Phase 6F checkpoint hash mismatch")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Phase 6G live inference but is unavailable")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.tensorizer = CSGTensorizer()
        self.model = CSGTargetSetScorer(
            self.tensorizer, NIModelConfig(**checkpoint["model_config"])
        )
        if checkpoint["tensor_schema_hash"] != self.model.tensor_schema_hash:
            raise ValueError("checkpoint/tensor schema mismatch")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        if self.model.utility_head is None:
            raise ValueError("frozen Phase 6F checkpoint has no utility head")
        self.probability = FrozenCalibrator(**freeze["probability_calibrator"])
        self.utility = FrozenCalibrator(**freeze["utility_calibrator"])
        self.thresholds = dict(freeze["selective_intervention_thresholds"])
        self.proposal_seed_namespace = int(proposal_seed_namespace)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = actual_hash

    def decide(
        self,
        instance: Instance,
        current: DecodedCandidate,
        *,
        state_id: str,
        destroy_count: int,
        search_progress: float,
        search_stage: str,
    ) -> InterventionDecision:
        started = time.perf_counter()
        graph = build_csg_from_schedule(
            instance,
            current.schedule,
            state_id=state_id,
            search_progress=search_progress,
            search_stage=search_stage,
        )
        operations = graph.nodes["OP"]
        makespan = max(float(graph.graph_features["current_makespan"]), 1.0)
        state_feature_summary = {
            "mean_slack_ratio": float(np.mean([
                node.features["operation_slack"] / makespan for node in operations
            ])),
            "mean_w_delay_ratio": float(np.mean([
                node.features["w_delay"] / makespan for node in operations
            ])),
            "mean_f_delay_ratio": float(np.mean([
                node.features["f_delay"] / makespan for node in operations
            ])),
            "mean_island_relative_load": float(np.mean([
                node.features["island_relative_load"] for node in operations
            ])),
            "mean_local_reconfiguration_ratio": float(np.mean([
                node.features["local_reconfiguration"] / makespan for node in operations
            ])),
            "search_progress": float(search_progress),
        }
        csg_ms = (time.perf_counter() - started) * 1000.0
        proposal_started = time.perf_counter()
        generated, records = build_live_proposal_bank(
            instance,
            current,
            state_id=state_id,
            destroy_count=destroy_count,
            seed_namespace=self.proposal_seed_namespace,
        )
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        tensor_started = time.perf_counter()
        sample = NIStateSample(
            self.tensorizer.tensorize(graph),
            tensorize_action_records(graph, records),
            {"search_stage": search_stage},
        )
        batch = batch_state_samples([sample]).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tensor_ms = (time.perf_counter() - tensor_started) * 1000.0
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda" else nullcontext()
        )
        with torch.inference_mode(), autocast:
            inference_started = time.perf_counter()
            node_embeddings, graph_embeddings = self.model.state_encoder(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            scoring_started = time.perf_counter()
            action_embeddings = self.model.action_encoder(
                node_embeddings["OP"], graph_embeddings, batch
            )
            raw_score_tensor = self.model.score_head(action_embeddings).squeeze(-1)
            raw_utility_tensor = self.model.utility_head(action_embeddings).squeeze(-1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        scoring_ms = (time.perf_counter() - scoring_started) * 1000.0
        calibration_started = time.perf_counter()
        raw_scores = raw_score_tensor.detach().float().cpu().numpy()
        raw_utilities = raw_utility_tensor.detach().float().cpu().numpy()
        probabilities = self.probability.predict(raw_scores)
        utilities = self.utility.predict(raw_utilities)
        order = np.argsort(-raw_scores, kind="stable")
        best = int(order[0])
        second_probability = float(probabilities[int(order[1])]) if len(order) > 1 else 0.0
        probability = float(probabilities[best])
        utility = float(utilities[best])
        margin = probability - second_probability
        intervene = bool(
            probability >= float(self.thresholds["confidence"])
            and utility >= float(self.thresholds["predicted_utility"])
            and margin >= float(self.thresholds["decision_margin"])
        )
        calibration_ms = (time.perf_counter() - calibration_started) * 1000.0
        arm = generated.arms[best]
        return InterventionDecision(
            intervene=intervene,
            state_id=state_id,
            selected_target_set_id=arm.target_set_id,
            destroyed_operations=arm.destroyed_operations if intervene else (),
            calibrated_probability=probability,
            calibrated_utility=utility,
            decision_margin=margin,
            fallback_reason=None if intervene else "CALIBRATION_GATE",
            proposal_count=generated.unique_arm_count,
            requested_proposal_count=generated.requested_arm_count,
            duplicate_proposal_count=generated.duplicate_arm_count,
            selected_origin_family=arm.arm_family,
            selected_origin_operator=arm.origin_destroy_operator,
            selected_origin_rules=arm.origin_rules,
            graph_hash=graph.graph_hash,
            timings_ms={
                "csg_build": csg_ms,
                "proposal_bank": proposal_ms,
                "tensorization_and_transfer": tensor_ms,
                "model_inference": inference_ms,
                "action_scoring": scoring_ms,
                "calibration_gate": calibration_ms,
                "total": (time.perf_counter() - started) * 1000.0,
            },
            state_feature_summary=state_feature_summary,
        )
