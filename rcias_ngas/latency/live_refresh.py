"""Semantics-preserving A1.5 implementation of the complete live C1 refresh."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import numpy as np
import torch

from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.search.dabc_chdg import build_generalized_chdg
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS
from rcias_ngas.bank.ngas_bank_v1 import build_bank
from rcias_ngas.csg.critical_mapping import map_critical_events
from rcias_ngas.csg.critical_sync import analyze_graph
from rcias_ngas.csg.revised_features import (
    EDGE_FEATURE_DIM, action_features, state_features_from_components,
)
from rcias_ngas.rng import RNGStreams
from rcias_ngas.search.persistent_prior import normalized_prior


@dataclass(frozen=True)
class LiveRefreshResult:
    actions: tuple[JointAction, ...]
    advantage: tuple[float, ...]
    beats_fallback_probability: tuple[float, ...]
    prior: tuple[float, ...]
    ranking: tuple[int, ...]
    sampled_index: int
    components_ms: dict[str, float]


def _joint_bank(instance, current, state_id: str, streams: RNGStreams, analysis):
    banks = [
        build_bank(instance, current, state_id, size, streams, analysis=analysis)
        for size in SIZE_FRACTIONS
    ]
    return tuple(
        JointAction(bank.size, target, repair)
        for bank in banks
        for target in bank.targets
        for repair in NGAS_REPAIR_IDS
    )


def _tensorize_cpu(state, actions):
    """Create contiguous CPU tensors without walking Python lists twice."""
    def tensor(values, dtype):
        return torch.from_numpy(np.asarray(values, dtype=dtype))

    return {
        'x': tensor(state['node_features'], np.float32),
        'types': tensor(state['node_types'], np.int64),
        'edge_index': tensor(state['edge_index'], np.int64).reshape(-1, 2).T,
        'relations': tensor(state['edge_types'], np.int64),
        'edge_features': tensor(state['edge_features'], np.float32).reshape(-1, EDGE_FEATURE_DIM),
        'operation_nodes': tensor(state['operation_nodes'], np.int64),
        'critical_mask': tensor(state['critical_mask'], np.bool_),
        'membership': tensor(actions['membership'], np.float32),
        'provenance': tensor(actions['provenance'], np.float32),
        'sizes': tensor(actions['sizes'], np.int64),
        'repairs': tensor(actions['repairs'], np.int64),
        'boundary_membership': tensor(actions['boundary_membership'], np.float32),
        'boundary_stats': tensor(actions['boundary_stats'], np.float32),
        'target_critical_overlap': tensor(
            actions['target_critical_overlap'], np.float32).reshape(-1, 1),
    }


class LiveRefreshEngine:
    """Build, score, and sample one complete fixed-horizon refresh boundary.

    One critic is the deployable C1 path. Multiple critics implement the
    development ensemble comparison while sharing all CPU graph work.
    """

    def __init__(self, critics, *, prior_advantage_scale: float = .01,
                 prior_uniform_mix: float = .05) -> None:
        self.critics = tuple(critics)
        if not self.critics:
            raise ValueError('At least one frozen critic is required')
        devices = {critic.device for critic in self.critics}
        variants = {critic.variant for critic in self.critics}
        if len(devices) != 1 or variants != {'C1'}:
            raise ValueError('A1.5 requires C1 critics on one shared device')
        self.device = next(iter(devices))
        self.prior_advantage_scale = prior_advantage_scale
        self.prior_uniform_mix = prior_uniform_mix

    def _synchronize(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def refresh(self, instance, current, state_id: str, streams: RNGStreams,
                *, sample_seed: int = 0) -> LiveRefreshResult:
        self._synchronize()
        total_started = time.perf_counter()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        graph = build_csg_from_schedule(
            instance, current.schedule, state_id=state_id,
            search_progress=0., search_stage='0-20%', attach_hash=False,
        )
        timings['csg_update_build'] = time.perf_counter() - started

        started = time.perf_counter()
        event_graph = build_generalized_chdg(instance, current)
        timings['critical_event_graph'] = time.perf_counter() - started
        started = time.perf_counter()
        analysis = analyze_graph(event_graph)
        timings['critical_extraction'] = time.perf_counter() - started
        started = time.perf_counter()
        mapping = map_critical_events(event_graph, graph, analysis)
        timings['critical_mapping'] = time.perf_counter() - started

        started = time.perf_counter()
        state = state_features_from_components(
            instance, graph, mapping, include_hash=False, include_diagnostics=False,
        )
        timings['state_feature_update'] = time.perf_counter() - started
        started = time.perf_counter()
        actions = _joint_bank(instance, current, state_id, streams, analysis)
        timings['candidate_joint_action_generation'] = time.perf_counter() - started
        if not actions:
            raise ValueError('Live refresh produced an empty action bank')
        started = time.perf_counter()
        features = action_features(state, actions)
        timings['action_feature_update'] = time.perf_counter() - started
        started = time.perf_counter()
        cpu_batch = _tensorize_cpu(state, features)
        timings['cpu_tensorization'] = time.perf_counter() - started

        gpu_marks = None
        if self.device.type == 'cuda':
            gpu_marks = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            gpu_marks[0].record()
        started = time.perf_counter()
        batch = {name: value.to(self.device) for name, value in cpu_batch.items()}
        timings['h2d_enqueue_wall'] = time.perf_counter() - started
        if gpu_marks:
            gpu_marks[1].record()

        encoded, advantages, logits = [], [], []
        with torch.inference_mode():
            for critic in self.critics:
                nodes, pooled = critic.model.encode_state(batch)
                encoded.append((critic, nodes, pooled))
            if gpu_marks:
                gpu_marks[2].record()
            for critic, nodes, pooled in encoded:
                output = critic.model.score_actions(nodes, pooled, batch)
                advantages.append(output['advantage'])
                logits.append(output['beats_fallback_logit'])
        if gpu_marks:
            gpu_marks[3].record()
        mean_advantage = torch.stack(advantages).mean(0)
        mean_probability = torch.sigmoid(torch.stack(logits)).mean(0)
        if gpu_marks:
            gpu_marks[4].record()

        started = time.perf_counter()
        advantage = tuple(float(value) for value in mean_advantage.detach().cpu().tolist())
        probability = tuple(float(value) for value in mean_probability.detach().cpu().tolist())
        self._synchronize()
        timings['output_and_synchronization_wall'] = time.perf_counter() - started
        if gpu_marks:
            timings['h2d_gpu'] = gpu_marks[0].elapsed_time(gpu_marks[1]) / 1000.
            timings['graph_encoding_gpu'] = (
                gpu_marks[1].elapsed_time(gpu_marks[2]) / 1000.)
            timings['batched_action_scoring_gpu'] = (
                gpu_marks[2].elapsed_time(gpu_marks[3]) / 1000.)
            timings['ensemble_reduction_gpu'] = gpu_marks[3].elapsed_time(gpu_marks[4]) / 1000.
        else:
            timings['h2d_gpu'] = timings['h2d_enqueue_wall']
            timings['graph_encoding_gpu'] = 0.
            timings['batched_action_scoring_gpu'] = 0.
            timings['ensemble_reduction_gpu'] = 0.
        if not all(math.isfinite(value) for value in (*advantage, *probability)):
            raise FloatingPointError('Non-finite live critic output')

        started = time.perf_counter()
        prior = normalized_prior(
            advantage, self.prior_advantage_scale, self.prior_uniform_mix)
        timings['prior_construction'] = time.perf_counter() - started
        started = time.perf_counter()
        ranking = tuple(sorted(
            range(len(actions)), key=lambda index: (-prior[index], actions[index].action_id)))
        sampled = random.Random(sample_seed).choices(
            range(len(actions)), weights=prior, k=1)[0]
        timings['ranking_sampling'] = time.perf_counter() - started
        timings['complete_refresh'] = time.perf_counter() - total_started
        return LiveRefreshResult(
            actions, advantage, probability, tuple(prior), ranking, sampled,
            {name: value * 1000. for name, value in timings.items()},
        )
