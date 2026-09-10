#!/usr/bin/env python3
"""Short CUDA feasibility, determinism, and full-bank smoke test for A1.4."""
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
from rcias_ngas.critic.inference import FrozenJointCritic, file_sha256
from rcias_ngas.search.ngas_solver import (
    MODE_SETTINGS, NGASSearchConfig, solve_ngas,
)

CONFIG = ROOT / 'configs/ngas_a1_search_integration_v1.json'
OUT = ROOT / 'outputs/ngas_a1/search_integration_v1/smoke/smoke.json'


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def signature(result) -> dict:
    return {
        'best_makespan': result.best.makespan,
        'decoder_evaluations': result.decoder_evaluations,
        'iterations': result.iterations,
        'actions': [(row['action_size'], row['target_operations'], row['repair'])
                    for row in result.diagnostics['iterations']],
        'outcomes': [row['outcome_class'] for row in result.diagnostics['iterations']],
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    checkpoint = ROOT / config['representation_boundary']['checkpoint_path']
    if file_sha256(checkpoint) != config['representation_boundary']['checkpoint_sha256']:
        raise RuntimeError('C1 checkpoint hash drift')
    if not torch.cuda.is_available():
        raise RuntimeError('A1.4 formal device CUDA is unavailable')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    critic = FrozenJointCritic(
        checkpoint, 'cuda:0', config['representation_boundary']['checkpoint_sha256'], 'C1')
    smoke_config = NGASSearchConfig(candidate_trials=2, iteration_limit=3)
    tiny = load_instance(ROOT / 'instances/tiny/tiny_03.json')
    modes = {}
    for mode, settings in MODE_SETTINGS.items():
        result = solve_ngas(
            tiny, 30., 746101, mode,
            critic=critic if settings['neural'] else None,
            config=smoke_config)
        modes[mode] = {
            'best_makespan': result.best.makespan,
            'iterations': result.iterations,
            'decoder_evaluations': result.decoder_evaluations,
            'neural_calls': result.diagnostics['telemetry']['termination']['neural_calls'],
            'critic_influence_fraction': result.diagnostics['critic_influence_fraction'],
            'feasible_replay': result.diagnostics['final_replay']['feasible'],
        }
    first = solve_ngas(tiny, 30., 746102, 'PERSISTENT_EVENT_REFRESH', critic, smoke_config)
    second = solve_ngas(tiny, 30., 746102, 'PERSISTENT_EVENT_REFRESH', critic, smoke_config)
    deterministic = signature(first) == signature(second)
    subset = config['development_subset'][0]
    representative = load_instance(
        ROOT / config['instance_source']['root'] / subset['relative_path'])
    representative_result = solve_ngas(
        representative, 60., 746101, 'PERSISTENT_EVENT_REFRESH', critic,
        NGASSearchConfig(candidate_trials=2, iteration_limit=1))
    finite = all(math.isfinite(value) for refresh in representative_result.diagnostics['refreshes']
                 for value in (refresh['refresh_seconds'], refresh['distribution']['entropy']))
    checks = {
        'cuda_available': True,
        'checkpoint_hash_matches': True,
        'all_six_modes_completed': len(modes) == 6,
        'all_final_replays_feasible': all(row['feasible_replay'] for row in modes.values()),
        'event_mode_deterministic': deterministic,
        'representative_R12_full_bank_scored': bool(representative_result.diagnostics['refreshes']),
        'representative_R12_numerically_finite': finite,
        'representative_R12_replay_feasible': representative_result.diagnostics['final_replay']['feasible'],
    }
    payload = {
        'schema': 'ngas-a14-cuda-smoke-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'device': {
            'name': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        },
        'checks': checks, 'mode_results': modes,
        'determinism_signature': signature(first),
        'representative_R12': {
            'instance_id': representative.instance_id,
            'num_operations': representative.num_operations,
            'best_makespan': representative_result.best.makespan,
            'iterations': representative_result.iterations,
            'decoder_evaluations': representative_result.decoder_evaluations,
            'refresh': representative_result.diagnostics['refreshes'][0],
        },
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
