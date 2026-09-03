"""Frozen three-seed Phase 6I-MR live intervention policy."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.instance import Instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import DecodedCandidate

from .batching import batch_state_samples
from .calibration import FrozenCalibrator
from .dataset import NIStateSample, tensorize_action_records
from .live_inference import FrozenLiveInference
from .live_policy import InterventionDecision
from .phase6i_heads import build_phase6i_head
from .proposal_bank import build_live_proposal_bank


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dag_depth(instance: Instance) -> int:
    depths: dict[str, int] = {}
    pending = set(instance.operations)
    while pending:
        ready = sorted(
            operation
            for operation in pending
            if instance.predecessors[operation].issubset(depths)
        )
        if not ready:
            raise RuntimeError("precedence graph is cyclic")
        for operation in ready:
            depths[operation] = 1 + max(
                (depths[parent] for parent in instance.predecessors[operation]),
                default=0,
            )
            pending.remove(operation)
    return max(depths.values(), default=0)


def _dag_width(instance: Instance) -> int:
    completed: set[str] = set()
    maximum = 0
    while len(completed) < instance.num_operations:
        ready = sorted(
            operation
            for operation in instance.operations
            if operation not in completed
            and instance.predecessors[operation].issubset(completed)
        )
        if not ready:
            raise RuntimeError("precedence graph is cyclic")
        maximum = max(maximum, len(ready))
        completed.add(ready[0])
    return maximum


def _critical_path_proxy(instance: Instance, schedule) -> float:
    features = schedule_features(instance, schedule)
    return float(sum(
        schedule.operation_schedules[operation].processing_time
        for operation in instance.operations
        if features[operation]["is_on_processing_critical_path"]
    ))


@dataclass(frozen=True)
class Phase6IStateEvaluation:
    decision: InterventionDecision
    raw_context: dict[str, float]
    normalized_context: dict[str, float]
    candidate_diagnostics: tuple[dict[str, Any], ...]


class Phase6IMRLiveInference:
    """Run the immutable U1/U2 three-seed head ensemble on the live bank."""

    def __init__(
        self,
        repository_root: Path,
        artifact_path: Path,
        *,
        device: str = "cuda",
        required_status: str | None = None,
    ) -> None:
        self.root = Path(repository_root)
        self.artifact_path = Path(artifact_path)
        self.artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        if required_status is not None and self.artifact.get("status") != required_status:
            raise ValueError("Phase 6I deployment artifact has the wrong status")
        if self.artifact.get("r11_accessed") is not False:
            raise ValueError("Phase 6I deployment artifact has invalid split access")
        if self.artifact.get("model_family") not in {"U1", "U2"}:
            raise ValueError("live head ensemble supports only frozen U1/U2 candidates")
        if self.artifact.get("ensemble_rule") != "ARITHMETIC_MEAN_THREE_TRAINING_SEEDS":
            raise ValueError("invalid Phase 6I ensemble rule")
        if len(self.artifact.get("model_artifacts", ())) != 3:
            raise ValueError("Phase 6I live policy requires exactly three model seeds")

        experiment_freeze = self.root / self.artifact["base_experiment_freeze_path"]
        if sha256_file(experiment_freeze) != self.artifact["base_experiment_freeze_sha256"]:
            raise ValueError("base experiment-freeze hash mismatch")
        self.base = FrozenLiveInference(
            experiment_freeze,
            device=device,
            proposal_seed_namespace=int(self.artifact["proposal_seed_namespace"]),
        )
        if self.base.checkpoint_sha256 != self.artifact["base_checkpoint_sha256"]:
            raise ValueError("base checkpoint hash mismatch")
        if self.base.tensorizer.tensor_schema_hash != self.artifact["tensor_schema_hash"]:
            raise ValueError("tensor-schema hash mismatch")
        self.device = self.base.device
        self.policy_name = str(self.artifact["policy_name"])
        self.proposal_seed_namespace = int(self.artifact["proposal_seed_namespace"])
        self.probability = FrozenCalibrator(**self.artifact["probability_calibrator"])
        self.utility = FrozenCalibrator(**self.artifact["utility_calibrator"])
        self.probability_threshold = float(self.artifact["thresholds"]["probability"])
        self.utility_threshold = float(self.artifact["thresholds"]["utility"])
        self.context_feature_order = tuple(self.artifact["context_feature_order"])
        self.context_normalization = dict(self.artifact["context_normalization"])
        self.support_bounds = dict(self.artifact["support_bounds"])
        if set(self.context_feature_order) != set(self.context_normalization):
            raise ValueError("context feature order/normalization mismatch")

        self.heads = []
        seeds = []
        for record in self.artifact["model_artifacts"]:
            path = self.root / record["checkpoint_path"]
            if sha256_file(path) != record["checkpoint_sha256"]:
                raise ValueError(f"Phase 6I checkpoint hash mismatch: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            seed = int(record["training_seed"])
            if int(checkpoint["training_seed"]) != seed:
                raise ValueError("Phase 6I checkpoint training-seed mismatch")
            if checkpoint["family"] != self.artifact["model_family"]:
                raise ValueError("Phase 6I checkpoint family mismatch")
            head = build_phase6i_head(self.artifact["model_family"], dropout=0.1)
            head.load_state_dict(checkpoint["model_state_dict"])
            head.eval().to(self.device)
            self.heads.append(head)
            seeds.append(seed)
        if tuple(seeds) != tuple(sorted(seeds)) or len(set(seeds)) != 3:
            raise ValueError("Phase 6I model seeds must be unique and sorted")
        self.training_seeds = tuple(seeds)
        self._static_context: dict[str, dict[str, Any]] = {}

    def prepare_instance(self, instance: Instance, h1_schedule) -> None:
        """Cache static U2 context using the solver's already-built H1 schedule."""
        if instance.instance_id in self._static_context:
            return
        h1_makespan = max(
            record.completion_time
            for record in h1_schedule.operation_schedules.values()
        )
        self._static_context[instance.instance_id] = {
            "h1_makespan": float(h1_makespan),
            "h1_critical_path": max(
                _critical_path_proxy(instance, h1_schedule), 1.0
            ),
            "dag_depth": _dag_depth(instance),
            "dag_width": _dag_width(instance),
            "eligibility_density": float(np.mean([
                len(instance.operation_data[operation].eligible_islands)
                / max(1, len(instance.islands))
                for operation in instance.operations
            ])),
        }

    def _instance_context(self, instance: Instance) -> dict[str, Any]:
        cached = self._static_context.get(instance.instance_id)
        if cached is not None:
            return cached
        h1 = solve_dispatching(instance, "H1")
        self.prepare_instance(instance, h1.schedule)
        return self._static_context[instance.instance_id]

    def _context(
        self,
        instance: Instance,
        current: DecodedCandidate,
        graph,
        search_progress: float,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        static = self._instance_context(instance)
        schedule = current.schedule
        makespan = max(float(current.makespan), 1.0)
        operation_records = schedule.operation_schedules.values()
        island_counts = np.asarray(
            list(Counter(record.island_id for record in operation_records).values()),
            dtype=float,
        )
        w_delay = float(sum(
            max(0.0, record.w_ready_time - record.product_ready_time)
            for record in schedule.operation_schedules.values()
        ))
        f_delay = float(sum(
            max(0.0, record.f_ready_time - record.product_ready_time)
            for record in schedule.operation_schedules.values()
        ))
        reconfiguration = float(sum(
            record.reconfiguration_end - record.reconfiguration_start
            for record in schedule.operation_schedules.values()
        ))
        operations = graph.nodes["OP"]
        summary = {
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
                node.features["local_reconfiguration"] / makespan
                for node in operations
            ])),
            "search_progress": float(search_progress),
        }
        node_count = sum(len(values) for values in graph.nodes.values())
        edge_count = sum(len(values) for values in graph.edges.values())
        current_critical = _critical_path_proxy(instance, schedule)
        raw = {
            "log1p_operation_count_r09_robust_z": float(np.log1p(instance.num_operations)),
            "log1p_graph_node_count_r09_robust_z": float(np.log1p(node_count)),
            "log1p_graph_edge_count_r09_robust_z": float(np.log1p(edge_count)),
            "edge_per_node_ratio": float(edge_count / node_count),
            "dag_depth_per_operation": float(static["dag_depth"] / instance.num_operations),
            "dag_width_per_operation": float(static["dag_width"] / instance.num_operations),
            "eligibility_density": float(static["eligibility_density"]),
            "resource_load_cv": float(
                np.std(island_counts) / max(np.mean(island_counts), 1e-12)
            ),
            "current_makespan_over_h1_makespan": float(
                current.makespan / static["h1_makespan"]
            ),
            "critical_path_over_h1_critical_path": float(
                current_critical / static["h1_critical_path"]
            ),
            "w_delay_over_current_makespan": w_delay / makespan,
            "f_delay_over_current_makespan": f_delay / makespan,
            "reconfiguration_over_current_makespan": reconfiguration / makespan,
            **summary,
        }
        if set(raw) != set(self.context_feature_order):
            raise RuntimeError("live Phase 6I context schema mismatch")
        normalized = {}
        for name in self.context_feature_order:
            record = self.context_normalization[name]
            value = np.clip(raw[name], record["winsor_lower"], record["winsor_upper"])
            normalized[name] = float(
                (value - record["median"]) / record["robust_scale"]
            )
        return raw, normalized, summary

    def evaluate(
        self,
        instance: Instance,
        current: DecodedCandidate,
        *,
        state_id: str,
        destroy_count: int,
        search_progress: float,
        search_stage: str,
    ) -> Phase6IStateEvaluation:
        started = time.perf_counter()
        graph = build_csg_from_schedule(
            instance,
            current.schedule,
            state_id=state_id,
            search_progress=search_progress,
            search_stage=search_stage,
        )
        raw_context, normalized_context, summary = self._context(
            instance, current, graph, search_progress
        )
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
            self.base.tensorizer.tensorize(graph),
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
            node_embeddings, graph_embeddings = self.base.model.state_encoder(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            scoring_started = time.perf_counter()
            action_embeddings = self.base.model.action_encoder(
                node_embeddings["OP"], graph_embeddings, batch
            )
            raw_score_tensor = self.base.model.score_head(action_embeddings).squeeze(-1)
        context_tensor = torch.tensor(
            [normalized_context[name] for name in self.context_feature_order],
            dtype=torch.float32,
            device=self.device,
        ).repeat(action_embeddings.shape[0], 1)
        float_embeddings = action_embeddings.float()
        with torch.inference_mode():
            seed_values = [head(float_embeddings, context_tensor) for head in self.heads]
            raw_utility_tensor = torch.stack(seed_values).mean(dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

        calibration_started = time.perf_counter()
        raw_scores = raw_score_tensor.detach().float().cpu().numpy()
        raw_utilities = raw_utility_tensor.detach().float().cpu().numpy()
        probabilities = self.probability.predict(raw_scores)
        utilities = self.utility.predict(raw_utilities)
        outside = sum(
            not (
                float(bounds["minimum"]) <= raw_context[name.removeprefix("raw_context__")]
                <= float(bounds["maximum"])
            )
            for name, bounds in self.support_bounds.items()
        )
        supported = outside == 0
        eligible = [
            index for index, probability in enumerate(probabilities)
            if supported and float(probability) >= self.probability_threshold
        ]
        selected = min(
            eligible,
            key=lambda index: (
                -float(raw_utilities[index]),
                -float(raw_scores[index]),
                generated.arms[index].target_set_id,
            ),
            default=None,
        )
        threshold_pass = bool(
            selected is not None
            and float(utilities[selected]) >= self.utility_threshold
        )
        calibration_ms = (time.perf_counter() - calibration_started) * 1000.0
        arm = None if selected is None else generated.arms[selected]
        fallback_reason = None
        if not threshold_pass:
            if not supported:
                fallback_reason = "SUPPORT_GUARD"
            elif selected is None:
                fallback_reason = "PROBABILITY_GATE"
            else:
                fallback_reason = "UTILITY_GATE"
        diagnostics = tuple({
            "target_set_id": generated.arms[index].target_set_id,
            "raw_score": float(raw_scores[index]),
            "ensemble_raw_utility": float(raw_utilities[index]),
            "calibrated_probability": float(probabilities[index]),
            "calibrated_utility": float(utilities[index]),
            "probability_eligible": bool(
                supported and probabilities[index] >= self.probability_threshold
            ),
        } for index in range(len(generated.arms)))
        decision = InterventionDecision(
            intervene=threshold_pass,
            state_id=state_id,
            selected_target_set_id=None if arm is None else arm.target_set_id,
            destroyed_operations=() if arm is None else arm.destroyed_operations,
            calibrated_probability=(
                None if selected is None else float(probabilities[selected])
            ),
            calibrated_utility=None if selected is None else float(utilities[selected]),
            decision_margin=None,
            fallback_reason=fallback_reason,
            proposal_count=generated.unique_arm_count,
            requested_proposal_count=generated.requested_arm_count,
            duplicate_proposal_count=generated.duplicate_arm_count,
            selected_origin_family=None if arm is None else arm.arm_family,
            selected_origin_operator=(
                None if arm is None else arm.origin_destroy_operator
            ),
            selected_origin_rules=() if arm is None else arm.origin_rules,
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
            state_feature_summary=summary,
            raw_score=None if selected is None else float(raw_scores[selected]),
            raw_probability=(
                None if selected is None else float(
                    1.0 / (1.0 + np.exp(-np.clip(raw_scores[selected], -40.0, 40.0)))
                )
            ),
            raw_utility=None if selected is None else float(raw_utilities[selected]),
            support_in_range=supported,
            support_out_of_range_count=outside,
            policy_name=self.policy_name,
        )
        return Phase6IStateEvaluation(
            decision, raw_context, normalized_context, diagnostics
        )

    def decide(self, instance: Instance, current: DecodedCandidate, **kwargs) -> InterventionDecision:
        return self.evaluate(instance, current, **kwargs).decision
