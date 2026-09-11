#!/usr/bin/env python3
"""Run one short CUDA feasibility check against the frozen A1.7A-R boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from rcias_ngas.search.ngas_solver import search_config_from_dict  # noqa: E402
from scripts.run_ngas_a17ar_trajectory_collection import (  # noqa: E402
    execute, load_boundary,
)


OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/smoke'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--budget-seconds', type=float, default=8.)
    args = parser.parse_args()
    if args.budget_seconds < 5.:
        raise ValueError('Smoke budget must be at least five seconds')
    config, protocol, protocol_sha = load_boundary(require_clean=True)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('A1.7A-R smoke requires CUDA')
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.empty(1, device=device).fill_(1.)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    critic = FrozenJointCritic(
        ROOT / protocol['checkpoint_path'], args.device,
        protocol['checkpoint_sha256'], 'C1')
    runtime = ProductionRefreshRuntime(
        critic,
        prior_advantage_scale=config['production_solver']['search']['prior_advantage_scale'],
        prior_uniform_mix=config['production_solver']['search']['prior_uniform_mix'])
    task = dict(min(
        (row for row in protocol['instances'] if row['dataset_role'] == 'TRAIN'),
        key=lambda row: (row['num_operations'], row['instance_id'])))
    task['trajectory_seed'] = int(config['trajectory']['seed_namespace']) + 900_000
    task['budget_seconds'] = args.budget_seconds
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    payload = execute(
        task, config, protocol, protocol_sha, critic, runtime,
        search_config_from_dict(config['production_solver']['search']), commit,
        phase='NGAS_A1_7A_R_STAGE6_CUDA_SMOKE',
        access_scope='one short clean non-R12 CUDA feasibility run')
    payload['schema'] = 'ngas-a17ar-cuda-smoke-v1'
    payload['smoke'] = {
        'status': 'PASS',
        'device': args.device,
        'device_name': torch.cuda.get_device_name(device),
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'numerical_stability': 'PASS_FINITE_OUTPUTS_AND_NORMALIZED_PRIORS',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'cuda_feasibility.json'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite smoke evidence: {path}')
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'status': 'PASS', 'path': str(path.relative_to(ROOT)),
        'instance_id': task['instance_id'], 'budget_seconds': args.budget_seconds,
        'elapsed_seconds': payload['end_to_end_seconds'],
        'iterations': payload['iterations'], 'states': len(payload['states']),
        'refreshes': len(payload['search_diagnostics']['refreshes']),
        'peak_memory_allocated_bytes': payload['smoke']['peak_memory_allocated_bytes'],
    }))


if __name__ == '__main__':
    main()
