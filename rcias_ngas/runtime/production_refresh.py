"""Single production authority for complete NGAS C1 refreshes."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import time

# Frozen NGAS training and formal search use this deterministic CUDA contract.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from rcias_ngas.rng import RNGStreams
from rcias_ngas.search.persistent_prior import normalized_prior
from .compact_state import CompactStateBuilder


@dataclass(frozen=True)
class ProductionRefreshResult:
    actions: tuple
    advantage: tuple[float, ...]
    beats_fallback_logit: tuple[float, ...]
    beats_fallback_probability: tuple[float, ...]
    prior: tuple[float, ...]
    ranking: tuple[int, ...]
    sampled_index: int
    critical_signature: str
    dominant_bottleneck: str | None
    bank_summary: dict
    components_ms: dict[str, float]
    state_feature_hash: str
    graph_hash: str
    static_context_hit: bool
    workspace_capacities: dict[str, int]
    workspace_resize_events: int


class ProductionRefreshRuntime:
    """Prepare, build, infer, and rank through one deployable C1 path."""

    def __init__(self, critic, *, prior_advantage_scale: float = .01,
                 prior_uniform_mix: float = .05) -> None:
        if critic.variant != 'C1':
            raise ValueError('A1.5R production runtime requires the frozen C1 critic')
        self.critic = critic
        self.device = critic.device
        if self.device.type == 'cuda':
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.prior_advantage_scale = prior_advantage_scale
        self.prior_uniform_mix = prior_uniform_mix
        self._builders: dict[str, CompactStateBuilder] = {}
        self._prepared_instances: dict[str, object] = {}
        self._cuda_marks = (
            tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
            if self.device.type == 'cuda' else None)

    @property
    def variant(self):
        return self.critic.variant

    @property
    def sha256(self):
        return self.critic.sha256

    def _synchronize(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def prepare_instance(self, instance) -> bool:
        """Prepare immutable context and workspace; return whether it was cached."""
        key = instance.instance_id
        if self._prepared_instances.get(key) is instance:
            return True
        self._builders[key] = CompactStateBuilder(instance)
        self._prepared_instances[key] = instance
        return False

    def structural_identity(self, instance, current) -> tuple[str, str | None]:
        """Evaluate live structure with the same compact analyzer as refresh."""
        self.prepare_instance(instance)
        builder = self._builders[instance.instance_id]
        neural = builder._neural_nodes(current.schedule)
        _, signature, bottleneck = builder._compact_critical(
            current.schedule, neural)
        return signature, bottleneck

    def refresh(self, instance, current, state_id: str, streams: RNGStreams,
                *, sample_seed: int = 0) -> ProductionRefreshResult:
        static_hit = self.prepare_instance(instance)
        builder = self._builders[instance.instance_id]
        self._synchronize()
        total_started = time.perf_counter()
        compact = builder.build(current, state_id, streams)
        timings = dict(compact.component_seconds)
        timings['compact_state_update'] = sum(timings[name] for name in (
            'compact_node_features', 'compact_event_critical_mapping',
            'compact_edge_update'))

        marks = self._cuda_marks
        if marks:
            marks[0].record()
        started = time.perf_counter()
        batch = {name: value.to(self.device) for name, value in compact.cpu_batch.items()}
        timings['h2d_enqueue_wall'] = time.perf_counter() - started
        if marks:
            marks[1].record()

        with torch.inference_mode():
            nodes, pooled = self.critic.model.encode_state(batch)
            if marks:
                marks[2].record()
            output = self.critic.model.score_actions(nodes, pooled, batch)
        if marks:
            marks[3].record()

        started = time.perf_counter()
        advantage = tuple(float(value) for value in output['advantage'].detach().cpu().tolist())
        logits = tuple(
            float(value) for value in output['beats_fallback_logit'].detach().cpu().tolist())
        probability = tuple(
            float(value) for value in torch.sigmoid(
                output['beats_fallback_logit']).detach().cpu().tolist())
        self._synchronize()
        timings['output_and_synchronization_wall'] = time.perf_counter() - started
        if marks:
            timings['h2d_gpu'] = marks[0].elapsed_time(marks[1]) / 1000.
            timings['graph_encoding_gpu'] = marks[1].elapsed_time(marks[2]) / 1000.
            timings['batched_action_scoring_gpu'] = marks[2].elapsed_time(marks[3]) / 1000.
        else:
            timings['h2d_gpu'] = timings['h2d_enqueue_wall']
            timings['graph_encoding_gpu'] = 0.
            timings['batched_action_scoring_gpu'] = 0.
        if not all(math.isfinite(value) for value in (*advantage, *logits, *probability)):
            raise FloatingPointError('Non-finite production critic output')

        started = time.perf_counter()
        prior = normalized_prior(
            advantage, self.prior_advantage_scale, self.prior_uniform_mix)
        timings['prior_construction'] = time.perf_counter() - started
        started = time.perf_counter()
        ranking = tuple(sorted(
            range(len(compact.actions)),
            key=lambda index: (-prior[index], compact.actions[index].action_id)))
        sampled = random.Random(sample_seed).choices(
            range(len(compact.actions)), weights=prior, k=1)[0]
        timings['ranking_sampling'] = time.perf_counter() - started
        timings['complete_refresh'] = time.perf_counter() - total_started
        return ProductionRefreshResult(
            compact.actions, advantage, logits, probability, tuple(prior), ranking,
            sampled, compact.critical_signature, compact.dominant_bottleneck,
            compact.bank_summary,
            {name: value * 1000. for name, value in timings.items()}, '', '',
            static_hit, dict(builder.workspace.capacities),
            builder.workspace.resize_events,
        )
