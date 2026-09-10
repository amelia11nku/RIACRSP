#!/usr/bin/env python3
"""Short outcome-blind CUDA feasibility and throughput test for A1.3."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.joint_critic import JointCritic
from rcias_ngas.critic.losses import joint_loss
from rcias_ngas.critic.train import digest, load_cache, model_for, prepare
from rcias_ngas.evaluation.bks import write_immutable

INPUT = ROOT / 'outputs/ngas_a1/critic_training_v1'
OUT = ROOT / 'outputs/ngas_a1/critic_training_gpu_v1'
CONFIG = ROOT / 'configs/ngas_a1_development_v1.json'


def main() -> None:
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG=:4096:8 is required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one usable CUDA device is required for the GPU smoke test')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    config = json.loads(CONFIG.read_text())
    records = load_cache(INPUT / 'training_cache.json.gz', INPUT / 'training_cache_manifest.json')
    record = max(records, key=lambda row: (
        len(row['state_features']['node_features']), len(row['state_features']['edge_index']),
        len(row['actions']), row['state_id']))
    batch, _ = prepare(record, device)
    action_count = len(record['actions'])
    base = torch.linspace(-.02, .02, action_count, device=device).unsqueeze(1)
    replicate_noise = torch.linspace(-.001, .001, 9, device=device).unsqueeze(0)
    synthetic_labels = base + replicate_noise
    model = model_for(config, 746101, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])

    def training_step() -> dict:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = joint_loss(model(batch), synthetic_labels)
        losses['total'].backward()
        finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                               for parameter in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        optimizer.step()
        finite_parameters = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        return {'loss': float(losses['total'].detach()),
                'finite_gradients': finite_gradients, 'finite_parameters': finite_parameters}

    torch.cuda.reset_peak_memory_stats(device)
    warmup = [training_step() for _ in range(2)]
    torch.cuda.synchronize(device)
    train_times, train_rows = [], []
    for _ in range(5):
        started = time.perf_counter()
        row = training_step()
        torch.cuda.synchronize(device)
        train_times.append(time.perf_counter() - started)
        train_rows.append(row)
    model.eval()
    with torch.inference_mode():
        reference = model(batch)
        repeated = model(batch)
        deterministic_inference = all(torch.equal(reference[key], repeated[key]) for key in reference)
        inference_times = []
        for _ in range(20):
            started = time.perf_counter()
            output = model(batch)
            torch.cuda.synchronize(device)
            inference_times.append(time.perf_counter() - started)
    all_outputs_finite = all(torch.isfinite(value).all().item() for value in output.values())
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_memory = properties.total_memory
    checks = {
        'cuda_tensor_execution': float(torch.ones(1, device=device).item()) == 1.,
        'finite_losses': all(math.isfinite(row['loss']) for row in warmup + train_rows),
        'finite_gradients': all(row['finite_gradients'] for row in warmup + train_rows),
        'finite_parameters': all(row['finite_parameters'] for row in warmup + train_rows),
        'finite_outputs': all_outputs_finite,
        'deterministic_repeated_inference': deterministic_inference,
        'peak_reserved_below_80_percent': peak_reserved / total_memory < .8,
        'memory_headroom_above_1_gib': total_memory - peak_reserved > 2 ** 30,
    }
    report = {
        'schema': 'ngas-a13-gpu-feasibility-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'device': {'name': properties.name, 'index': 0,
                   'total_memory_bytes': total_memory,
                   'compute_capability': f'{properties.major}.{properties.minor}'},
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'cudnn': torch.backends.cudnn.version(),
                        'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                        'tf32_cudnn': torch.backends.cudnn.allow_tf32},
        'representative_state': {'selection': 'maximum nodes, then edges/actions; outcome blind',
                                 'state_id': record['state_id'],
                                 'nodes': len(record['state_features']['node_features']),
                                 'directed_edges': len(record['state_features']['edge_index']),
                                 'joint_actions': action_count},
        'labels': {'source': 'synthetic deterministic linspace plus shared replicate offsets',
                   'development_outcomes_used': False},
        'throughput': {
            'training_steps': len(train_times),
            'mean_training_ms_per_state': 1000 * statistics.fmean(train_times),
            'p90_training_ms_per_state': 1000 * sorted(train_times)[math.ceil(.9 * len(train_times)) - 1],
            'training_joint_actions_per_second': action_count / statistics.fmean(train_times),
            'inference_repeats': len(inference_times),
            'mean_inference_ms_per_state': 1000 * statistics.fmean(inference_times),
            'p90_inference_ms_per_state': 1000 * sorted(inference_times)[math.ceil(.9 * len(inference_times)) - 1],
            'inference_joint_actions_per_second': action_count / statistics.fmean(inference_times),
        },
        'memory': {'peak_allocated_bytes': peak_allocated,
                   'peak_reserved_bytes': peak_reserved,
                   'peak_reserved_fraction': peak_reserved / total_memory,
                   'headroom_bytes': total_memory - peak_reserved},
        'input_hashes': {
            str((INPUT / 'training_cache.json.gz').relative_to(ROOT)): digest(INPUT / 'training_cache.json.gz'),
            str((INPUT / 'training_cache_manifest.json').relative_to(ROOT)): digest(INPUT / 'training_cache_manifest.json'),
            str(CONFIG.relative_to(ROOT)): digest(CONFIG),
        },
        'formal_optimizer_steps': 0,
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_immutable(OUT / 'gpu_feasibility.json', report)
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise RuntimeError('GPU feasibility gate failed')


if __name__ == '__main__':
    main()
