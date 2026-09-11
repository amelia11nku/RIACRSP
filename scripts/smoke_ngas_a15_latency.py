#!/usr/bin/env python3
"""Pre-freeze CUDA, equivalence, and throughput smoke for NGAS A1.5."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.csg.critical_sync import critical_sync
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.latency.live_refresh import LiveRefreshEngine
from rcias_ngas.rng import RNGStreams
from rcias_ngas.search.ngas_solver import _joint_bank
from rcias_ngas.search.persistent_prior import normalized_prior

RAW = ROOT / ('outputs/ngas_a1/search_integration_c1_r1_v1/raw/C1/'
              'CB1_CAUR_S_CF1_RI2_TI2_R12/seed_746101.json')
INSTANCE_ROOT = ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'
CHECKPOINT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/production/revised_joint_critic.pt'
CHECKPOINT_SHA256 = '448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560'
OUT = ROOT / 'outputs/ngas_a1/latency_qualification_v1/smoke/pre_freeze_gpu_smoke.json'


def _candidate(raw: dict) -> Candidate:
    return Candidate(*(tuple(raw[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('A1.5 CUDA smoke requires the formal GPU environment')
    torch.cuda.reset_peak_memory_stats()
    raw = json.loads(RAW.read_text())
    instance = load_instance(INSTANCE_ROOT / raw['instance']['relative_path'])
    current = decode_candidate(instance, _candidate(raw['best_candidate']))
    if not current.feasible:
        raise RuntimeError('Frozen representative candidate is infeasible')
    critic = FrozenJointCritic(
        CHECKPOINT, 'cuda:0', CHECKPOINT_SHA256, expected_variant='C1')
    state_id = 'ngas-a15-pre-freeze-smoke:S'
    streams = RNGStreams(instance.instance_id, 746101)

    analysis = critical_sync(instance, current)
    actions, _ = _joint_bank(instance, current, state_id, streams, analysis)
    reference, _ = critic.score(instance, current, state_id, actions)
    reference_prior = normalized_prior(reference['advantage'], .01, .05)
    reference_ranking = tuple(sorted(
        range(len(actions)), key=lambda i: (-reference_prior[i], actions[i].action_id)))

    engine = LiveRefreshEngine([critic])
    rows = [engine.refresh(
        instance, current, state_id, streams, sample_seed=746101 + repetition)
            for repetition in range(4)]
    measured = rows[1:]
    last = rows[-1]
    maximum_advantage_error = max(
        abs(a - b) for a, b in zip(reference['advantage'], last.advantage))
    maximum_probability_error = max(abs(a - b) for a, b in zip(
        reference['beats_fallback_probability'], last.beats_fallback_probability))
    maximum_prior_error = max(
        abs(a - b) for a, b in zip(reference_prior, last.prior))
    complete = [row.components_ms['complete_refresh'] for row in measured]
    checks = {
        'cuda_available': True,
        'checkpoint_hash_matches': critic.sha256 == CHECKPOINT_SHA256,
        'representative_replay_feasible': current.feasible,
        'action_identity_and_order_exact': [a.action_id for a in actions]
            == [a.action_id for a in last.actions],
        'advantage_max_abs_error_le_1e-7': maximum_advantage_error <= 1e-7,
        'probability_max_abs_error_le_1e-7': maximum_probability_error <= 1e-7,
        'prior_max_abs_error_le_1e-7': maximum_prior_error <= 1e-7,
        'ranking_exact': reference_ranking == last.ranking,
        'all_outputs_finite': all(math.isfinite(value) for value in (
            *last.advantage, *last.beats_fallback_probability, *last.prior)),
        'three_measured_complete_refreshes': len(complete) == 3,
    }
    payload = {
        'schema': 'ngas-a15-pre-freeze-gpu-smoke-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'device': {
            'name': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        },
        'representative': {
            'instance_id': instance.instance_id,
            'num_operations': instance.num_operations,
            'candidate_source': str(RAW.relative_to(ROOT)),
            'joint_actions': len(actions),
        },
        'checks': checks,
        'numerical_equivalence': {
            'maximum_advantage_absolute_error': maximum_advantage_error,
            'maximum_probability_absolute_error': maximum_probability_error,
            'maximum_prior_absolute_error': maximum_prior_error,
        },
        'complete_refresh_ms': complete,
        'last_component_profile_ms': last.components_ms,
        'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
