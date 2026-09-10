#!/usr/bin/env python3
"""Independently replay and audit the complete A1.3 critic experiment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.joint_critic import JointCritic
from rcias_ngas.critic.train import digest, load_cache, predict
from rcias_ngas.evaluation.bks import write_immutable
from scripts import train_ngas_joint_critic as training

OUT = training.OUT
AUDIT = OUT / 'completion_audit.json'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def model_from_checkpoint(path: Path, device: str) -> tuple[dict, JointCritic]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['model_config']
    model = JointCritic(hidden=int(config['hidden_dim']), layers=int(config['message_passing_layers']))
    model.load_state_dict(checkpoint['model_state'])
    return checkpoint, model.to(device).eval()


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('GPU completion audit requires the frozen CUDA environment')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda'
    protocol = training.validate_protocol()
    protocol_sha = digest(training.PROTOCOL)
    records = load_cache(training.CACHE, training.CACHE_MANIFEST)
    records_by_id = {row['state_id']: row for row in records}
    final = json.loads((OUT / 'final_decision.json').read_text())
    gate = json.loads((OUT / 'oof_gate.json').read_text())
    seed_results = []
    run_rows = []
    replayed_states = 0
    for seed in protocol['config']['training']['seeds']:
        seed_predictions = []
        for fold in range(3):
            require(training.valid_run(seed, fold, protocol_sha), f'Invalid run {seed}/{fold}')
            checkpoint_path, prediction_path, record_path = training.run_paths(seed, fold)
            checkpoint, model = model_from_checkpoint(checkpoint_path, device)
            run_record = json.loads(record_path.read_text())
            held = [row for row in records if row['fold'] == fold]
            stored = training.load_prediction_file(prediction_path)
            replayed = predict(model, held, device)
            require(replayed == stored, f'Checkpoint prediction replay mismatch: {seed}/{fold}')
            require(checkpoint['training_protocol_sha256'] == protocol_sha,
                    'Checkpoint protocol mismatch')
            require(not set(run_record['train_instances']) & set(run_record['held_instances']),
                    'Instance leakage in run record')
            seed_predictions.extend(stored)
            replayed_states += len(held)
            run_rows.append({'seed': seed, 'held_fold': fold,
                             'checkpoint_sha256': digest(checkpoint_path),
                             'predictions_sha256': digest(prediction_path),
                             'record_sha256': digest(record_path),
                             'fixed_final_epoch': run_record['epochs'] == protocol['config']['training']['epochs']})
        seed_results.append(training.evaluate_seed(seed, records_by_id, seed_predictions))
    require(seed_results == gate['seed_results'], 'OOF gate metrics did not reproduce')
    pass_count = sum(row['pass'] for row in seed_results)
    expected_decision = 'READY_FOR_A1_4_SEARCH_INTEGRATION' if pass_count >= 2 else 'NGAS_A1_REVISE_CRITIC'
    require(final['decision'] == expected_decision, 'Final A1.3 decision did not reproduce')
    production = final['production']
    production_replay = False
    if pass_count >= 2:
        require(production is not None, 'Passing OOF gate requires production critic')
        path = ROOT / production['checkpoint_path']
        require(digest(path) == production['checkpoint_sha256'], 'Production checkpoint drift')
        checkpoint, model = model_from_checkpoint(path, device)
        stored = training.load_prediction_file(OUT / 'production/development_predictions.json.gz')
        require(predict(model, records, device) == stored, 'Production prediction replay mismatch')
        require(checkpoint['seed'] == protocol['config']['training']['production_seed'],
                'Production seed drift')
        production_replay = True
    checks = {
        'nine_valid_oof_runs': len(run_rows) == 9,
        'all_checkpoints_fixed_final_epoch': all(row['fixed_final_epoch'] for row in run_rows),
        'all_held_predictions_replayed_exactly': replayed_states == 72 * 3,
        'all_seed_metrics_reproduced': seed_results == gate['seed_results'],
        'gate_and_decision_reproduced': final['decision'] == expected_decision,
        'production_condition_obeyed': (pass_count < 2 and production is None) or production_replay,
        'finite_metrics': all(math.isfinite(value) for result in seed_results
                              for value in result['metrics'].values() if isinstance(value, float)),
        'r13_r14_locked': final['r13'] == final['r14'] == 'LOCKED',
        'zero_gurobi': final['gurobi_run'] is False,
    }
    audit = {
        'schema': 'ngas-a13-gpu-critic-completion-audit-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks, 'decision': expected_decision,
        'training_protocol_sha256': protocol_sha,
        'oof_gate_sha256': digest(OUT / 'oof_gate.json'),
        'final_decision_sha256': digest(OUT / 'final_decision.json'),
        'runs': run_rows, 'seed_results': seed_results,
        'replayed_oof_states': replayed_states,
        'production_replayed': production_replay,
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    if AUDIT.exists():
        require(json.loads(AUDIT.read_text()) == audit, 'Existing completion audit differs')
    else:
        write_immutable(AUDIT, audit)
    print(json.dumps({'status': audit['status'], 'decision': audit['decision'],
                      'seeds_passing': pass_count, 'replayed_oof_states': replayed_states}, indent=2))


if __name__ == '__main__':
    main()
