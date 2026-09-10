#!/usr/bin/env python3
"""Run the frozen NGAS-A1.3 grouped OOF critic experiment and gated final fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.joint_critic import JointCritic
from rcias_ngas.critic.train import digest, fit, load_cache, predict
from rcias_ngas.evaluation.bks import write_immutable
from rcias_ngas.evaluation.ranking import state_metrics, subgroup_summaries, summarize

OUT = ROOT / 'outputs/ngas_a1/critic_training_gpu_v1'
PROTOCOL = OUT / 'training_protocol.json'
CACHE = ROOT / 'outputs/ngas_a1/critic_training_v1/training_cache.json.gz'
CACHE_MANIFEST = ROOT / 'outputs/ngas_a1/critic_training_v1/training_cache_manifest.json'
CONFIG = ROOT / 'configs/ngas_a1_development_v1.json'
GPU_CONFIG = ROOT / 'configs/ngas_a1_critic_training_gpu_v1.json'
REPORT = ROOT / 'docs/reports/ngas_a1/07_joint_critic_training_report.md'


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol.get('schema') != 'ngas-a13-gpu-critic-training-protocol-v1'
            or protocol.get('status') != 'FROZEN_BEFORE_GPU_FORMAL_OPTIMIZER_STEP'
            or protocol.get('formal_optimizer_steps_started') is not False):
        raise RuntimeError('Critic training protocol is not a valid frozen boundary')
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen training source changed: {relative}')
    for relative, expected in protocol['input_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen training input changed: {relative}')
    if any((ROOT / path).exists() for path in (
            'outputs/ngas_a1/r13/access_ledger.json',
            'outputs/ngas_a1/r14/access_ledger.json')):
        raise RuntimeError('NGAS R13/R14 access boundary changed')
    if protocol['gpu_execution']['formal_device'] != 'cuda:0':
        raise RuntimeError('Formal GPU device contract changed')
    return protocol


def run_paths(seed: int, held_fold: int) -> tuple[Path, Path, Path]:
    root = OUT / 'oof' / f'seed_{seed}'
    return (root / f'fold_{held_fold}.pt', root / f'fold_{held_fold}_predictions.json.gz',
            root / f'fold_{held_fold}.json')


def valid_run(seed: int, held_fold: int, protocol_sha: str) -> bool:
    checkpoint, predictions, record_path = run_paths(seed, held_fold)
    if not all(path.exists() for path in (checkpoint, predictions, record_path)):
        return False
    record = json.loads(record_path.read_text())
    return (record.get('schema') == 'ngas-a13-oof-run-v1'
            and record.get('seed') == seed and record.get('held_fold') == held_fold
            and record.get('training_protocol_sha256') == protocol_sha
            and record.get('checkpoint_sha256') == digest(checkpoint)
            and record.get('predictions_sha256') == digest(predictions))


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_prediction_file(path: Path) -> list[dict]:
    return json.loads(gzip.decompress(path.read_bytes()))['predictions']


def save_prediction_file(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
    atomic_bytes(path, gzip.compress(raw, compresslevel=9, mtime=0))


def progress(value: dict) -> None:
    atomic_json(OUT / 'progress.json', value)


def evaluate_seed(seed: int, records_by_id: dict[str, dict], predictions: list[dict]) -> dict:
    prediction_by_id = {row['state_id']: row for row in predictions}
    if set(prediction_by_id) != set(records_by_id):
        raise RuntimeError(f'OOF prediction state coverage mismatch for seed {seed}')
    state_rows = []
    for state_id in sorted(records_by_id):
        prediction = prediction_by_id[state_id]
        record = records_by_id[state_id]
        state_rows.append(state_metrics(record, prediction['predicted_advantage'],
                                        prediction['predicted_beats_fallback_probability']))
    metrics = summarize(state_rows)
    gate = json.loads(CONFIG.read_text())['oof_gate']
    checks = {
        'mean_state_spearman': metrics['mean_state_spearman'] >= gate['mean_state_spearman_min'],
        'material_pair_accuracy': metrics['material_pair_accuracy'] >= gate['material_pair_accuracy_min'],
        'top1_regret_below_uniform': metrics['mean_top1_regret'] < metrics['mean_uniform_expected_regret'],
        'mean_selected_advantage_positive': metrics['mean_selected_advantage'] > 0,
    }
    return {'seed': seed, 'metrics': metrics, 'subgroups': subgroup_summaries(state_rows),
            'state_metrics': state_rows, 'checks': checks, 'pass': all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    args = parser.parse_args()
    protocol = validate_protocol()
    config = protocol['config']
    gpu = protocol['gpu_execution']
    if (not torch.cuda.is_available() or torch.cuda.device_count() != gpu['required_cuda_device_count']
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != gpu['cublas_workspace_config']):
        raise RuntimeError('Frozen CUDA device/count/CUBLAS contract is unavailable')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(0)
    records = load_cache(CACHE, CACHE_MANIFEST)
    records_by_id = {row['state_id']: row for row in records}
    protocol_sha = digest(PROTOCOL)
    seeds = [int(seed) for seed in config['training']['seeds']]
    if (OUT / 'final_decision.json').exists():
        final = json.loads((OUT / 'final_decision.json').read_text())
        if final.get('status') == 'COMPLETE' and final.get('training_protocol_sha256') == protocol_sha:
            print(json.dumps({'event': 'ngas_training_already_complete',
                              'decision': final['decision']}), flush=True)
            return
        raise RuntimeError('Existing final decision does not match the frozen protocol')
    started = time.perf_counter()
    completed = sum(valid_run(seed, fold, protocol_sha) for seed in seeds for fold in range(3))
    print(json.dumps({'event': 'ngas_training_started', 'device': args.device,
                      'completed_runs': completed, 'expected_runs': 9}), flush=True)

    for seed in seeds:
        for held_fold in range(3):
            if valid_run(seed, held_fold, protocol_sha):
                print(json.dumps({'event': 'ngas_oof_run_reused', 'seed': seed,
                                  'held_fold': held_fold}), flush=True)
                continue
            checkpoint, prediction_path, record_path = run_paths(seed, held_fold)
            if record_path.exists():
                raise RuntimeError(f'Existing OOF run record is invalid: {record_path}')
            checkpoint.unlink(missing_ok=True)
            prediction_path.unlink(missing_ok=True)
            train_records = [row for row in records if row['fold'] != held_fold]
            held_records = [row for row in records if row['fold'] == held_fold]
            if ({row['instance_id'] for row in train_records}
                    & {row['instance_id'] for row in held_records}):
                raise RuntimeError('Instance leakage across OOF fold')
            run_started = time.perf_counter()

            def epoch_callback(row: dict) -> None:
                elapsed = time.perf_counter() - run_started
                event = {'event': 'ngas_oof_epoch', 'seed': seed, 'held_fold': held_fold,
                         'elapsed_seconds': elapsed, **row}
                print(json.dumps(event), flush=True)
                progress({'status': 'RUNNING', 'pid': os.getpid(), 'seed': seed,
                          'held_fold': held_fold, 'epoch': row['epoch'],
                          'epochs': config['training']['epochs'], 'completed_runs': completed,
                          'expected_runs': 9, 'protocol_sha256': protocol_sha,
                          'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                          'r13': 'LOCKED', 'r14': 'LOCKED'})

            model, history = fit(train_records, config, seed, args.device, epoch_callback)
            held_predictions = predict(model, held_records, args.device)
            save_checkpoint(checkpoint, {
                'schema': 'ngas-a13-joint-critic-checkpoint-v1',
                'training_protocol_sha256': protocol_sha, 'seed': seed,
                'held_fold': held_fold, 'model_config': config['critic'],
                'formal_device': 'cuda:0', 'precision': 'FP32',
                'model_state': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                'epochs': config['training']['epochs'],
            })
            save_prediction_file(prediction_path, {
                'schema': 'ngas-a13-oof-predictions-v1', 'training_protocol_sha256': protocol_sha,
                'seed': seed, 'held_fold': held_fold, 'predictions': held_predictions})
            write_immutable(record_path, {
                'schema': 'ngas-a13-oof-run-v1', 'training_protocol_sha256': protocol_sha,
                'seed': seed, 'held_fold': held_fold,
                'train_instances': sorted({row['instance_id'] for row in train_records}),
                'held_instances': sorted({row['instance_id'] for row in held_records}),
                'train_states': len(train_records), 'held_states': len(held_records),
                'epochs': config['training']['epochs'], 'history': history,
                'runtime_seconds': time.perf_counter() - run_started,
                'checkpoint_sha256': digest(checkpoint), 'predictions_sha256': digest(prediction_path),
                'checkpoint_selection': 'fixed final epoch',
                'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
            })
            completed += 1
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()

    seed_results = []
    combined_predictions = []
    for seed in seeds:
        seed_predictions = []
        for held_fold in range(3):
            if not valid_run(seed, held_fold, protocol_sha):
                raise RuntimeError('OOF run failed final validation')
            seed_predictions.extend(load_prediction_file(run_paths(seed, held_fold)[1]))
        combined_predictions.extend({'seed': seed, **row} for row in seed_predictions)
        seed_results.append(evaluate_seed(seed, records_by_id, seed_predictions))
    pass_count = sum(row['pass'] for row in seed_results)
    gate_pass = pass_count >= 2
    save_prediction_file(OUT / 'oof_predictions.json.gz', {
        'schema': 'ngas-a13-combined-oof-predictions-v1',
        'training_protocol_sha256': protocol_sha, 'predictions': combined_predictions})
    gate_result = {
        'schema': 'ngas-a13-oof-gate-v1', 'status': 'PASS' if gate_pass else 'FAIL',
        'training_protocol_sha256': protocol_sha, 'seeds_passing': pass_count,
        'seeds_required': 2, 'seed_results': seed_results,
        'decision': 'READY_FOR_PRODUCTION_CRITIC_FIT' if gate_pass else 'NGAS_A1_REVISE_CRITIC',
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'oof_gate.json', gate_result)

    production = None
    if gate_pass:
        production_seed = int(config['training']['production_seed'])
        production_started = time.perf_counter()

        def production_callback(row: dict) -> None:
            print(json.dumps({'event': 'ngas_production_epoch', **row}), flush=True)
            progress({'status': 'PRODUCTION_FIT', 'pid': os.getpid(), 'epoch': row['epoch'],
                      'epochs': config['training']['epochs'], 'completed_runs': 9,
                      'expected_runs': 9, 'protocol_sha256': protocol_sha,
                      'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                      'r13': 'LOCKED', 'r14': 'LOCKED'})

        model, history = fit(records, config, production_seed, args.device, production_callback)
        prediction = predict(model, records, args.device)
        path = OUT / 'production/joint_critic.pt'
        save_checkpoint(path, {
            'schema': 'ngas-a13-production-joint-critic-v1',
            'training_protocol_sha256': protocol_sha, 'seed': production_seed,
            'model_config': config['critic'],
            'formal_device': 'cuda:0', 'precision': 'FP32',
            'model_state': {name: value.detach().cpu() for name, value in model.state_dict().items()},
            'epochs': config['training']['epochs'],
        })
        save_prediction_file(OUT / 'production/development_predictions.json.gz', {
            'schema': 'ngas-a13-production-development-predictions-v1',
            'training_protocol_sha256': protocol_sha, 'predictions': prediction})
        production = {
            'seed': production_seed, 'epochs': config['training']['epochs'],
            'runtime_seconds': time.perf_counter() - production_started,
            'checkpoint_path': str(path.relative_to(ROOT)), 'checkpoint_sha256': digest(path),
            'history': history,
            'scope': 'all R12 DEVELOPMENT; metrics are in-sample diagnostics and not the OOF gate',
        }

    decision = 'READY_FOR_A1_4_SEARCH_INTEGRATION' if gate_pass else 'NGAS_A1_REVISE_CRITIC'
    final = {
        'schema': 'ngas-a13-final-decision-v1', 'status': 'COMPLETE',
        'decision': decision, 'training_protocol_sha256': protocol_sha,
        'oof_gate_sha256': digest(OUT / 'oof_gate.json'), 'seed_results': seed_results,
        'production': production, 'elapsed_seconds': time.perf_counter() - started,
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    progress({'status': 'COMPLETE', 'pid': os.getpid(), 'completed_runs': 9,
              'expected_runs': 9, 'decision': decision, 'protocol_sha256': protocol_sha,
              'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'r13': 'LOCKED', 'r14': 'LOCKED'})
    REPORT.write_text(f'''# A1.3 joint critic training report

Status: **{decision}**. The frozen grouped OOF experiment completed three seeds
and three held-instance folds. {pass_count}/3 seeds passed the preregistered gate;
the gate required at least 2. R12 remains development evidence.

The OOF gate and all per-state, fold, scale and CF diagnostics are in
`outputs/ngas_a1/critic_training_gpu_v1/oof_gate.json`. Fixed final epoch checkpoints
were used; held-fold labels did not select epochs or checkpoints. The single
production critic was {'fit only after the OOF gate passed' if gate_pass else 'not fit because the OOF gate failed'}.

R13/R14 stayed locked. No Gurobi or frozen comparator run was executed.
''')
    atomic_json(OUT / 'final_decision.json', final)
    print(json.dumps({'event': 'ngas_training_complete', 'decision': decision,
                      'seeds_passing': pass_count}), flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
