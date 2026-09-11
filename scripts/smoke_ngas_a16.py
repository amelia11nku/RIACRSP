#!/usr/bin/env python3
"""Run one explicitly excluded CUDA smoke check for the A1.6 execution path."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import (  # noqa: E402
    audit_frozen_inputs, digest, load_json,
)
from rcias_ngas.evaluation.a16_io import write_new_json  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from rcias_ngas.search.ngas_solver import search_config_from_dict, solve_ngas  # noqa: E402


CONFIG = ROOT / 'configs/ngas_a16_solver_comparison_v1.json'
OUT = ROOT / 'outputs/ngas_a1/solver_comparison_v1/smoke/smoke.json'


def main() -> None:
    config = load_json(CONFIG)
    if OUT.exists():
        payload = load_json(OUT)
        if (payload.get('schema') != 'ngas-a16-excluded-smoke-v1'
                or payload.get('config_sha256') != digest(CONFIG)
                or payload.get('excluded_from_formal') is not True
                or payload.get('status') != 'PASS'):
            raise RuntimeError('Existing A1.6 smoke artifact is invalid and will not be overwritten')
        print(json.dumps({'status': 'VALID_EXISTING_SMOKE', 'path': str(OUT.relative_to(ROOT))}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('A1.6 smoke requires CUDA')
    frozen = audit_frozen_inputs(ROOT, config)
    if frozen['status'] != 'PASS':
        raise RuntimeError('A1.6 frozen input audit failed before smoke')
    manifest = load_json(ROOT / config['scope']['instance_manifest_path'])
    row = next(item for item in manifest['instances']
               if item['scale'] == 'S' and item['CF_level'] == 'CF1'
               and item['cell_replicate'] == 'C01')
    instance_path = ROOT / config['scope']['instance_root'] / row['relative_path']
    instance = load_instance(instance_path)
    torch.use_deterministic_algorithms(True)
    torch.empty(1, device='cuda:0').fill_(1.)
    torch.cuda.synchronize()
    critic = FrozenJointCritic(
        ROOT / config['production_solver']['checkpoint_path'], 'cuda:0',
        config['production_solver']['checkpoint_sha256'], 'C1')
    runtime = ProductionRefreshRuntime(critic)
    started = time.perf_counter()
    result = solve_ngas(
        instance, config['smoke']['budget_seconds'], config['smoke']['seed'],
        config['production_solver']['mode'], critic,
        search_config_from_dict(config['production_solver']['search']),
        refresh_runtime=runtime,
        budget_accounting=config['budget']['accounting_mode'])
    end_to_end = time.perf_counter() - started
    replay = check_schedule(instance, result.best.schedule)
    preparation = result.diagnostics['runtime_components']['runtime_preparation_seconds']
    checks = {
        'excluded_namespace': config['smoke']['excluded_from_formal'] is True,
        'non_formal_seed': config['smoke']['seed'] not in config['scope']['seeds'],
        'a16_budget_accounting': result.diagnostics['budget_accounting'] == 'A16_INSTANCE_TOTAL',
        'preparation_charged': result.runtime >= preparation,
        'atomic_prestart_rule': result.diagnostics['atomic_budget_audit']['started_after_deadline_count'] == 0,
        'finite_runtime': all(math.isfinite(value) for value in (
            result.runtime, end_to_end, preparation, result.best.makespan)),
        'feasible_replay': result.best.feasible and replay['feasible'],
        'checkpoint_hash': critic.sha256 == config['production_solver']['checkpoint_sha256'],
    }
    payload = {
        'schema': 'ngas-a16-excluded-smoke-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'excluded_from_formal': True,
        'config_sha256': digest(CONFIG),
        'checkpoint_sha256': critic.sha256,
        'instance_id': instance.instance_id,
        'instance_sha256': digest(instance_path),
        'seed': config['smoke']['seed'],
        'budget_seconds': config['smoke']['budget_seconds'],
        'solver_budget_elapsed_seconds': result.runtime,
        'end_to_end_run_elapsed_seconds': end_to_end,
        'runtime_preparation_seconds': preparation,
        'budget_overshoot_seconds': result.diagnostics['budget_overshoot_seconds'],
        'final_makespan': result.best.makespan,
        'decoder_evaluations': result.decoder_evaluations,
        'iterations': result.iterations,
        'incumbent_trace': [asdict(point) for point in result.convergence_trace],
        'checks': checks,
        'device': {
            'name': torch.cuda.get_device_name(0),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
        },
        'locks': config['locks'],
    }
    if payload['status'] != 'PASS':
        raise RuntimeError(payload)
    write_new_json(OUT, payload)
    print(json.dumps({'status': 'PASS', 'path': str(OUT.relative_to(ROOT)),
                      'runtime_seconds': result.runtime, 'iterations': result.iterations}))


if __name__ == '__main__':
    main()
