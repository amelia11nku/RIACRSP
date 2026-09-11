#!/usr/bin/env python3
"""Short CUDA and determinism smoke for the matched C1/R1 solver ablation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.search.ngas_solver import NGASSearchConfig, solve_ngas

OUT = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1/smoke/smoke.json'
CHECKPOINTS = {
    'C1': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/C1/seed_746101/fold_0/model.pt',
           '3cd419ada2287d377c194e24adb0b8bf29226ab9e7195e5d1d6704cba3ec591a'),
    'R1': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/R1/seed_746101/fold_0/model.pt',
           '532f9aa448d26c7ffd4e238eaa20daa57ae1eab50b20a405dbc21bbb936dee4f'),
}


def signature(result) -> dict:
    return {
        'best_makespan': result.best.makespan,
        'decoder_evaluations': result.decoder_evaluations,
        'iterations': result.iterations,
        'actions': [(row['action_size'], row['target_operations'], row['repair'])
                    for row in result.diagnostics['iterations']],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('C1/R1 CUDA smoke requires the formal GPU')
    torch.cuda.reset_peak_memory_stats()
    instance = load_instance(ROOT / 'instances/tiny/tiny_03.json')
    config = NGASSearchConfig(candidate_trials=2, iteration_limit=3)
    results = {}
    c1_repeat = None
    for variant, (relative, expected) in CHECKPOINTS.items():
        critic = FrozenJointCritic(ROOT / relative, 'cuda:0', expected, variant)
        result = solve_ngas(
            instance, 30., 746101, 'PERSISTENT_FIXED_REFRESH', critic, config)
        results[variant] = {
            'checkpoint_sha256': critic.sha256,
            'signature': signature(result),
            'feasible_replay': result.diagnostics['final_replay']['feasible'],
            'critic_calls': result.diagnostics['telemetry']['termination']['neural_calls'],
        }
        if variant == 'C1':
            c1_repeat = solve_ngas(
                instance, 30., 746101, 'PERSISTENT_FIXED_REFRESH', critic, config)
    checks = {
        'cuda_available': True,
        'checkpoint_hashes_match': all(
            results[v]['checkpoint_sha256'] == CHECKPOINTS[v][1] for v in CHECKPOINTS),
        'C1_and_R1_replay_feasible': all(row['feasible_replay'] for row in results.values()),
        'both_variants_use_same_solver_mechanism': all(row['critic_calls'] == 1 for row in results.values()),
        'C1_deterministic_repeat': results['C1']['signature'] == signature(c1_repeat),
    }
    payload = {
        'schema': 'ngas-a14-c1-r1-cuda-smoke-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'device': {
            'name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        },
        'checks': checks, 'variants': results,
        'production_variant': 'C1', 'R1_role': 'EXPLANATORY_ONLY',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUT)
    print(json.dumps(payload, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
