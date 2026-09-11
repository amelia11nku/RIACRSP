#!/usr/bin/env python3
"""Demonstrate A1.6R observational instrumentation behavior invariance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import digest, load_json  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from rcias_ngas.search.ngas_solver import (  # noqa: E402
    NGASDiagnosticConfig, NGASSearchConfig, solve_ngas,
)


CONFIG = ROOT / 'configs/ngas_a16r_integrity_diagnostic_v1.json'
OUT = ROOT / 'outputs/ngas_a1/solver_comparison_a16r_v1/preregistration/behavior_invariance.json'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def trajectory(result) -> list[dict]:
    fields = (
        'action_id', 'repair', 'candidate_trials', 'candidate_makespan',
        'current_before', 'current_after', 'best_before', 'best_after',
        'accepted', 'outcome_class',
    )
    return [{name: row[name] for name in fields}
            for row in result.diagnostics['iterations']]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the frozen-hardware equivalence audit')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = load_json(CONFIG)
    production = config['production_solver']
    instance = load_instance('instances/tiny/tiny_01.json')
    search = NGASSearchConfig(
        candidate_trials=production['search']['candidate_trials'], iteration_limit=8,
        prior_advantage_scale=production['search']['prior_advantage_scale'],
        prior_uniform_mix=production['search']['prior_uniform_mix'],
        online_exploration=production['search']['online_exploration'],
        portfolio_segment_length=production['search']['portfolio_segment_length'],
        portfolio_reaction=production['search']['portfolio_reaction'],
        portfolio_strength=production['search']['portfolio_strength'])
    critic = FrozenJointCritic(
        ROOT / production['checkpoint_path'], args.device,
        production['checkpoint_sha256'], production['variant'])
    baseline = solve_ngas(
        instance, 120., 746101, production['mode'], critic, search,
        refresh_runtime=ProductionRefreshRuntime(
            critic, prior_advantage_scale=search.prior_advantage_scale,
            prior_uniform_mix=search.prior_uniform_mix),
        budget_accounting=config['budget']['accounting_mode'])
    instrumented = solve_ngas(
        instance, 120., 746101, production['mode'], critic, search,
        refresh_runtime=ProductionRefreshRuntime(
            critic, prior_advantage_scale=search.prior_advantage_scale,
            prior_uniform_mix=search.prior_uniform_mix),
        budget_accounting=config['budget']['accounting_mode'],
        diagnostic_config=NGASDiagnosticConfig(enabled=True))
    baseline_trajectory = trajectory(baseline)
    instrumented_trajectory = trajectory(instrumented)
    checks = {
        'same_h1_solution': (
            baseline.convergence_trace[0].current_best_makespan
            == instrumented.convergence_trace[0].current_best_makespan),
        'same_selected_actions_repairs_candidates_acceptance': (
            baseline_trajectory == instrumented_trajectory),
        'same_final_candidate': baseline.best.candidate == instrumented.best.candidate,
        'same_final_makespan': baseline.best.makespan == instrumented.best.makespan,
        'same_decoder_evaluations': (
            baseline.decoder_evaluations == instrumented.decoder_evaluations),
        'same_iteration_count': baseline.iterations == instrumented.iterations,
        'instrumentation_present': all(
            len(row['a16r_observation']['candidate_trials'])
            == production['search']['candidate_trials']
            for row in instrumented.diagnostics['iterations']),
    }
    payload = {
        'schema': 'ngas-a16r-behavior-invariance-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG),
        'device': args.device,
        'device_name': (torch.cuda.get_device_name(args.device)
                        if args.device.startswith('cuda') else 'CPU'),
        'checkpoint_sha256': critic.sha256,
        'instance_id': instance.instance_id,
        'seed': 746101,
        'budget_accounting': config['budget']['accounting_mode'],
        'candidate_trials': production['search']['candidate_trials'],
        'iteration_limit': 8,
        'checks': checks,
        'baseline_trajectory': baseline_trajectory,
        'instrumented_trajectory': instrumented_trajectory,
        'rng_argument': (
            'identical action/state/trial indices address immutable RNGStreams; '
            'diagnostics only read derived seed values'),
    }
    atomic_json(OUT, payload)
    print(json.dumps({'status': payload['status'], 'checks': checks,
                      'path': str(OUT.relative_to(ROOT))}, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
