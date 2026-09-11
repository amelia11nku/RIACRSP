#!/usr/bin/env python3
"""Short deterministic GPU replay through the authoritative refresh runtime."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from rcias_clgri.data.loader import load_instance
from rcias_ngas.critic.inference import FrozenJointCritic
from rcias_ngas.runtime import ProductionRefreshRuntime
from rcias_ngas.search.ngas_solver import NGASSearchConfig, solve_ngas


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/ngas_a15r_runtime_revision_v1.json'
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/audit/search_integration.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replay_fields(result):
    fields = (
        'action_id', 'candidate_makespan', 'current_after', 'best_after',
        'accepted', 'outcome_class',
    )
    return [tuple(row[field] for field in fields)
            for row in result.diagnostics['iterations']]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Search integration audit requires CUDA')
    config = json.loads(CONFIG.read_text())
    checkpoint = config['production_checkpoint']
    critic = FrozenJointCritic(
        ROOT / checkpoint['path'], 'cuda:0', checkpoint['sha256'], checkpoint['variant'])
    states = json.loads((
        ROOT / config['historical_A1_5']['representative_states']).read_text())['states']
    frozen = next(row for row in states if row['label'] == 'S')
    instance = load_instance(
        ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'
        / frozen['relative_path'])
    search_config = NGASSearchConfig(candidate_trials=1, iteration_limit=8)
    runs = []
    for _ in range(2):
        runtime = ProductionRefreshRuntime(critic)
        runs.append(solve_ngas(
            instance, 60., 746101, 'PERSISTENT_FIXED_REFRESH', critic,
            search_config, refresh_runtime=runtime))
    first, second = runs
    first_runtime = first.diagnostics['runtime_components']
    assertions = {
        'deterministic_iteration_decisions': replay_fields(first) == replay_fields(second),
        'deterministic_best_candidate': first.best.candidate == second.best.candidate,
        'feasibility_100_percent': all(
            run.diagnostics['final_replay']['feasible'] for run in runs),
        'shared_runtime_component_present':
            first_runtime.get('production_refresh_complete_refresh_seconds', 0.) > 0.,
        'legacy_neural_feature_components_absent': not any(
            name in first_runtime for name in (
                'state_feature_seconds', 'action_feature_seconds',
                'tensor_transfer_seconds', 'model_forward_seconds')),
        'one_initial_refresh_per_run': all(
            len(run.diagnostics['refreshes']) == 1 for run in runs),
        'fixed_iteration_count': all(run.iterations == 8 for run in runs),
    }
    payload = {
        'schema': 'ngas-a15r-search-integration-audit-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG), 'checkpoint_sha256': critic.sha256,
        'mode': 'PERSISTENT_FIXED_REFRESH', 'seed': 746101,
        'candidate_trials': 1, 'iteration_limit': 8,
        'assertions': assertions, 'pass': all(assertions.values()),
        'iterations': [dict(zip((
            'action_id', 'candidate_makespan', 'current_after', 'best_after',
            'accepted', 'outcome_class'), row)) for row in replay_fields(first)],
        'runtime_components': first_runtime,
        'historical_A1_5_immutable': True, 'locks': config['locks'],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUT)
    print(json.dumps({
        'output': str(OUT.relative_to(ROOT)), 'assertions': assertions,
        'pass': payload['pass'],
    }, indent=2))


if __name__ == '__main__':
    main()
