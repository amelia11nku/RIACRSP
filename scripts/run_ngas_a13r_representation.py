#!/usr/bin/env python3
"""Freeze, smoke-test, train, profile, and decide the A1.3R comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.losses import joint_loss
from rcias_ngas.critic.train import load_cache as load_historical_cache
from rcias_ngas.critic.revised_train import (
    digest, fit, load_cache, model_for, predict, prepare,
)
from rcias_ngas.evaluation.revised_ranking import (
    state_metrics, subgroup_summaries, summarize,
)

OUT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2'
CONFIG = ROOT / 'configs/ngas_a1_critic_training_rthgt_v2.json'
CACHE = OUT / 'data/training_cache_v2.json.gz'
CACHE_MANIFEST = OUT / 'data/training_cache_manifest_v2.json'
IDENTITY_AUDIT = OUT / 'audit/label_identity.json'
CRITICAL_AUDIT = OUT / 'audit/critical_sync.json'
SMOKE = OUT / 'audit/gpu_smoke.json'
PROTOCOL = OUT / 'training_protocol.json'
C0_OUT = ROOT / 'outputs/ngas_a1/critic_training_gpu_v1'
C0_CACHE = ROOT / 'outputs/ngas_a1/critic_training_v1/training_cache.json.gz'
C0_CACHE_MANIFEST = ROOT / 'outputs/ngas_a1/critic_training_v1/training_cache_manifest.json'


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


def save_gzip_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    atomic_bytes(path, gzip.compress(raw, compresslevel=9, mtime=0))


def load_gzip_json(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def save_checkpoint(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def configure_cuda() -> None:
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('A1.3R formal execution requires exactly one CUDA device')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parameter_counts(model: torch.nn.Module) -> dict:
    return {
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
    }


def source_paths() -> list[Path]:
    paths = [
        ROOT / 'rcias_ngas/actions/joint_action.py',
        ROOT / 'rcias_ngas/actions/repair.py',
        ROOT / 'rcias_ngas/critic/losses.py',
        ROOT / 'rcias_ngas/critic/dataset.py',
        ROOT / 'rcias_ngas/critic/development.py',
        ROOT / 'rcias_ngas/critic/pooling.py',
        ROOT / 'rcias_ngas/critic/revised_action_encoder.py',
        ROOT / 'rcias_ngas/critic/revised_critic.py',
        ROOT / 'rcias_ngas/critic/revised_train.py',
        ROOT / 'rcias_ngas/csg/critical_mapping.py',
        ROOT / 'rcias_ngas/csg/critical_sync.py',
        ROOT / 'rcias_ngas/csg/revised_features.py',
        ROOT / 'rcias_ngas/csg/revised_schema.py',
        ROOT / 'rcias_ngas/evaluation/revised_ranking.py',
        ROOT / 'scripts/audit_ngas_action_label_identity.py',
        ROOT / 'scripts/build_ngas_revised_training_cache.py',
        ROOT / 'scripts/run_ngas_a13r_representation.py',
        CONFIG,
    ]
    paths.extend(sorted((ROOT / 'rcias_ngas/critic/encoders').glob('*.py')))
    paths.extend(sorted((ROOT / 'tests/ngas').glob('test_*.py')))
    return sorted(set(paths))


def c0_tree_hash() -> tuple[str, int]:
    files = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(C0_OUT.rglob('*')) if path.is_file()
    }
    value = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value, len(files)


def smoke() -> None:
    configure_cuda()
    config = json.loads(CONFIG.read_text())
    records = load_cache(CACHE, CACHE_MANIFEST)
    largest = max(records, key=lambda row: (
        len(row['state_features']['node_features']), len(row['actions']), row['state_id']))
    variants = {}
    for variant in ('C1', 'R1'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = model_for(config, variant, 746101, 'cuda').train()
        batch, labels = prepare(largest, 'cuda')
        encoder_calls = []
        hook = model.encoder.register_forward_hook(lambda *_: encoder_calls.append(1))
        output = model(batch)
        loss = joint_loss(output, labels)['total']
        loss.backward()
        hook.remove()
        finite_output = all(torch.isfinite(value).all() for value in output.values())
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters())
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.eval()
        with torch.inference_mode():
            first = model(batch)
            second = model(batch)
        deterministic = all(torch.equal(first[key], second[key]) for key in first)
        parameters_unchanged = all(
            torch.equal(before[name], value) for name, value in model.state_dict().items())
        variants[variant] = {
            **parameter_counts(model),
            'encoder_type': config['variants'][variant]['encoder_type'],
            'state_id': largest['state_id'],
            'nodes': len(largest['state_features']['node_features']),
            'edges_with_reverse': len(largest['state_features']['edge_index']),
            'joint_actions': len(largest['actions']),
            'graph_encoder_calls_for_all_actions': len(encoder_calls),
            'finite_outputs': bool(finite_output),
            'finite_gradients': bool(finite_gradients),
            'repeated_inference_bitwise_equal': bool(deterministic),
            'parameters_unchanged_by_inference': bool(parameters_unchanged),
            'peak_reserved_memory_bytes': torch.cuda.max_memory_reserved(),
        }
        del model, batch, labels
    checks = {
        'one_graph_encoding_scores_all_actions':
            all(value['graph_encoder_calls_for_all_actions'] == 1 for value in variants.values()),
        'finite_outputs_and_gradients':
            all(value['finite_outputs'] and value['finite_gradients'] for value in variants.values()),
        'deterministic_repeated_inference':
            all(value['repeated_inference_bitwise_equal'] for value in variants.values()),
        'fp32': True,
        'no_amp': True,
    }
    result = {
        'schema': 'ngas-a13r-gpu-smoke-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks, 'variants': variants,
        'environment': {
            'python': sys.version, 'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0),
            'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(0))),
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        },
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(SMOKE, result)
    report = ROOT / 'docs/reports/ngas_a1/08d_a13r_revised_representation_audit.md'
    report.write_text(f"""# A1.3R revised representation audit

Status: **{result['status']}**.

C1 preserves compact relation-aware mean aggregation while using the revised
typed CPM feature schema. R1 implements type-specific input and Q/K/V
transformations, multi-head relation-specific attention, edge contributions,
residual connections, feed-forward blocks, and per-type normalization. Both
models share type-balanced mean/max/critical pooling and the batched target
mean/max, critical-overlap, graph-boundary, size and repair representation.

The largest cached state ({variants['R1']['nodes']} nodes,
{variants['R1']['joint_actions']} joint actions) required one encoder call for
all actions. CUDA FP32 outputs and gradients were finite and repeated inference
was bitwise equal. C1 has {variants['C1']['parameters']:,} parameters; R1 has
{variants['R1']['parameters']:,}. AMP, TF32 and candidate-wise graph forward
passes are absent.
""")
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(2)


def freeze() -> None:
    if PROTOCOL.exists():
        raise FileExistsError('A1.3R training protocol is already frozen')
    configure_cuda()
    config = json.loads(CONFIG.read_text())
    smoke_result = json.loads(SMOKE.read_text())
    identity = json.loads(IDENTITY_AUDIT.read_text())
    critical = json.loads(CRITICAL_AUDIT.read_text())
    if not (smoke_result['status'] == identity['status'] == critical['status'] == 'PASS'):
        raise RuntimeError('A1.3R pre-training audit has not passed')
    tree_hash, tree_files = c0_tree_hash()
    starting = json.loads((OUT / 'audit/starting_state.json').read_text())
    if (tree_hash, tree_files) != (
            starting['compact_reference']['tree_manifest_sha256'],
            starting['compact_reference']['files']):
        raise ValueError('Historical C0 output tree changed')
    regression = OUT / 'audit/pretraining_regression.txt'
    xml = OUT / 'audit/pretraining_regression.xml'
    with regression.open('w') as stream:
        subprocess.run([
            sys.executable, '-m', 'pytest', '-q', 'tests/ngas',
            '--junitxml', str(xml),
        ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    fold_audit = []
    records = load_cache(CACHE, CACHE_MANIFEST)
    for fold in range(3):
        held = {row['instance_id'] for row in records if row['fold'] == fold}
        fit_instances = {row['instance_id'] for row in records if row['fold'] != fold}
        if held & fit_instances:
            raise ValueError('Instance leakage across OOF fold')
        fold_audit.append({
            'held_fold': fold, 'held_instances': sorted(held),
            'fit_instances': sorted(fit_instances), 'instance_overlap': [],
            'held_states': sum(row['fold'] == fold for row in records),
            'fit_states': sum(row['fold'] != fold for row in records),
        })
    protocol = {
        'schema': 'ngas-a13r-frozen-training-protocol-v2',
        'status': 'FROZEN_BEFORE_FORMAL_OPTIMIZER',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'starting_commit': starting['starting_commit'],
        'repository_head_at_freeze': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config': config,
        'source_hashes': {
            str(path.relative_to(ROOT)): digest(path) for path in source_paths()},
        'data_hashes': {
            str(CACHE.relative_to(ROOT)): digest(CACHE),
            str(CACHE_MANIFEST.relative_to(ROOT)): digest(CACHE_MANIFEST),
            str(IDENTITY_AUDIT.relative_to(ROOT)): digest(IDENTITY_AUDIT),
            str(CRITICAL_AUDIT.relative_to(ROOT)): digest(CRITICAL_AUDIT),
            str(SMOKE.relative_to(ROOT)): digest(SMOKE),
        },
        'historical_C0': {
            'output_tree_sha256': tree_hash, 'files': tree_files,
            'protocol_sha256': digest(C0_OUT / 'training_protocol.json'),
            'oof_predictions_sha256': digest(C0_OUT / 'oof_predictions.json.gz'),
            'oof_gate_sha256': digest(C0_OUT / 'oof_gate.json'),
        },
        'fold_audit': fold_audit,
        'pretraining_regression_sha256': digest(regression),
        'pretraining_regression_xml_sha256': digest(xml),
        'formal_device': 'cuda:0', 'precision': 'FP32',
        'same_labels_folds_seeds_loss_epochs_for_C1_R1': True,
        'checkpoint_selection': 'fixed final epoch',
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        'status': protocol['status'], 'protocol_sha256': digest(PROTOCOL),
        'folds': fold_audit}, indent=2))


def verify_protocol() -> tuple[dict, dict]:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'FROZEN_BEFORE_FORMAL_OPTIMIZER':
        raise ValueError('A1.3R protocol status mismatch')
    for relative, expected in {
            **protocol['source_hashes'], **protocol['data_hashes']}.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen A1.3R boundary changed: {relative}')
    tree_hash, tree_files = c0_tree_hash()
    if (tree_hash, tree_files) != (
            protocol['historical_C0']['output_tree_sha256'],
            protocol['historical_C0']['files']):
        raise ValueError('Historical C0 output tree changed after A1.3R freeze')
    return protocol, protocol['config']


def run_paths(variant: str, seed: int, fold: int) -> tuple[Path, Path, Path]:
    directory = OUT / 'variants' / variant / f'seed_{seed}' / f'fold_{fold}'
    return directory / 'model.pt', directory / 'predictions.json.gz', directory / 'run.json'


def valid_run(variant: str, seed: int, fold: int, protocol_sha: str) -> bool:
    checkpoint, predictions, record = run_paths(variant, seed, fold)
    if not all(path.exists() for path in (checkpoint, predictions, record)):
        return False
    value = json.loads(record.read_text())
    return (
        value.get('training_protocol_sha256') == protocol_sha
        and value.get('checkpoint_sha256') == digest(checkpoint)
        and value.get('predictions_sha256') == digest(predictions)
        and value.get('variant') == variant
    )


def evaluate_seed(seed: int, records_by_id: dict[str, dict],
                  predictions: list[dict], gate: dict) -> dict:
    if len(predictions) != 72 or len({row['state_id'] for row in predictions}) != 72:
        raise ValueError('OOF prediction coverage mismatch')
    rows = [
        state_metrics(
            records_by_id[prediction['state_id']],
            prediction['predicted_advantage'],
            prediction['predicted_beats_fallback_probability'])
        for prediction in sorted(predictions, key=lambda row: row['state_id'])
    ]
    summary = summarize(rows)
    checks = {
        'mean_state_spearman':
            summary['mean_state_spearman'] >= gate['mean_state_spearman_min'],
        'material_pair_accuracy':
            summary['material_pair_accuracy'] >= gate['material_pair_accuracy_min'],
        'regret_below_uniform':
            summary['mean_top1_regret'] < summary['mean_uniform_expected_regret'],
        'selected_advantage_positive': summary['mean_selected_advantage'] > 0,
    }
    return {
        'seed': seed, 'pass': all(checks.values()), 'checks': checks,
        'summary': summary, 'subgroups': subgroup_summaries(rows),
        'state_metrics': rows,
    }


def aggregate_seed_summaries(seed_results: list[dict]) -> dict:
    fields = seed_results[0]['summary']
    return {
        field: statistics.fmean(result['summary'][field] for result in seed_results)
        for field in fields if isinstance(fields[field], (int, float))
    }


def c0_results(records_by_id: dict[str, dict], gate: dict) -> dict:
    values = load_gzip_json(C0_OUT / 'oof_predictions.json.gz')['predictions']
    seed_results = []
    for seed in (746101, 746102, 746103):
        seed_results.append(evaluate_seed(
            seed, records_by_id,
            [{key: value for key, value in row.items() if key != 'seed'}
             for row in values if row['seed'] == seed],
            gate))
    return {
        'name': 'C0_COMPACT_HISTORICAL',
        'seed_results': seed_results,
        'mean_across_seeds': aggregate_seed_summaries(seed_results),
        'seeds_passing': sum(result['pass'] for result in seed_results),
        'source': 'immutable historical A1.3 OOF predictions',
    }


def load_model(checkpoint: Path, config: dict, variant: str,
               device: str = 'cuda'):
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = model_for(config, variant, int(value['seed']), device)
    model.load_state_dict(value['model_state'])
    return model.eval()


def profile_variants(records: list[dict], config: dict) -> dict:
    runtime = config['runtime']
    selected_records = {}
    for scale in ('S', 'M', 'L'):
        candidates = sorted(
            (row for row in records if row['scale'] == scale),
            key=lambda row: (len(row['state_features']['node_features']), row['state_id']))
        selected_records[scale] = candidates[len(candidates) // 2]
    selected_records['MAX'] = max(
        records, key=lambda row: (
            len(row['state_features']['node_features']), len(row['actions']), row['state_id']))
    result = {}
    for variant in ('C1', 'R1'):
        checkpoint = run_paths(
            variant, config['training']['seeds'][0], 0)[0]
        model = load_model(checkpoint, config, variant)
        profiles = {}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for name, record in selected_records.items():
            batch, _ = prepare(record, 'cuda')
            with torch.inference_mode():
                for _ in range(int(runtime['latency_warmup'])):
                    model(batch)
                torch.cuda.synchronize()
                samples = []
                for _ in range(int(runtime['latency_repetitions'])):
                    started = time.perf_counter()
                    model(batch)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - started) * 1000)
            ordered = sorted(samples)

            def percentile(q: float) -> float:
                return ordered[round((len(ordered) - 1) * q)]

            profiles[name] = {
                'state_id': record['state_id'],
                'nodes': len(record['state_features']['node_features']),
                'edges_with_reverse': len(record['state_features']['edge_index']),
                'joint_actions': len(record['actions']),
                'mean_ms': statistics.fmean(samples), 'p50_ms': percentile(.50),
                'p90_ms': percentile(.90), 'p99_ms': percentile(.99),
            }
        result[variant] = {
            **parameter_counts(model), 'states': profiles,
            'worst_representative_p90_ms':
                max(value['p90_ms'] for value in profiles.values()),
            'peak_reserved_memory_bytes': torch.cuda.max_memory_reserved(),
        }
        del model
    return result


def choose(config: dict, variants: dict, profile: dict) -> tuple[str, dict]:
    selection = config['selection']
    c1, r1 = variants['C1']['mean_across_seeds'], variants['R1']['mean_across_seeds']
    margins = selection['compact_matching_absolute_margins']
    improvement = selection['rthgt_material_improvement_any']
    quality_deltas = {
        'mean_state_spearman': r1['mean_state_spearman'] - c1['mean_state_spearman'],
        'material_pair_accuracy':
            r1['material_pair_accuracy'] - c1['material_pair_accuracy'],
        'mean_selected_advantage':
            r1['mean_selected_advantage'] - c1['mean_selected_advantage'],
        'mean_top1_regret_reduction':
            c1['mean_top1_regret'] - r1['mean_top1_regret'],
    }
    material = any(
        quality_deltas[field] >= threshold
        for field, threshold in improvement.items())
    r1_no_material_degradation = (
        quality_deltas['mean_selected_advantage']
        >= -margins['mean_selected_advantage']
        and r1['mean_top1_regret'] <= c1['mean_top1_regret'] + margins['mean_top1_regret'])
    matching = all((
        abs(quality_deltas['mean_state_spearman']) <= margins['mean_state_spearman'],
        abs(quality_deltas['material_pair_accuracy']) <= margins['material_pair_accuracy'],
        abs(quality_deltas['mean_selected_advantage']) <= margins['mean_selected_advantage'],
        abs(r1['mean_top1_regret'] - c1['mean_top1_regret']) <= margins['mean_top1_regret'],
    ))
    c1_latency = profile['C1']['worst_representative_p90_ms']
    r1_latency = profile['R1']['worst_representative_p90_ms']
    latency_budget = selection['latency_budget_p90_ms']
    compact_faster = c1_latency <= (
        1 - selection['compact_substantially_faster_fraction']) * r1_latency
    checks = {
        'R1_material_improvement': material,
        'R1_no_material_value_degradation': r1_no_material_degradation,
        'R1_gate_pass': variants['R1']['gate_pass'],
        'R1_latency_pass': r1_latency <= latency_budget,
        'C1_matches_R1': matching,
        'C1_substantially_faster': compact_faster,
        'C1_gate_pass': variants['C1']['gate_pass'],
        'C1_latency_pass': c1_latency <= latency_budget,
    }
    if all(checks[key] for key in (
            'R1_material_improvement', 'R1_no_material_value_degradation',
            'R1_gate_pass', 'R1_latency_pass')):
        return 'R1', {'checks': checks, 'quality_deltas_R1_minus_C1': quality_deltas}
    if all(checks[key] for key in (
            'C1_matches_R1', 'C1_substantially_faster',
            'C1_gate_pass', 'C1_latency_pass')):
        return 'C1', {'checks': checks, 'quality_deltas_R1_minus_C1': quality_deltas}
    return '', {'checks': checks, 'quality_deltas_R1_minus_C1': quality_deltas}


def train() -> None:
    configure_cuda()
    protocol, config = verify_protocol()
    protocol_sha = digest(PROTOCOL)
    records = load_cache(CACHE, CACHE_MANIFEST)
    records_by_id = {row['state_id']: row for row in records}
    started = time.perf_counter()
    training_results = {}
    total_runs = 2 * 3 * 3
    completed = 0
    for variant in ('C1', 'R1'):
        for seed in config['training']['seeds']:
            for fold in range(3):
                if valid_run(variant, seed, fold, protocol_sha):
                    completed += 1
                    continue
                train_records = [row for row in records if row['fold'] != fold]
                held_records = [row for row in records if row['fold'] == fold]
                checkpoint, prediction_path, run_path = run_paths(variant, seed, fold)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                run_started = time.perf_counter()

                def callback(row: dict) -> None:
                    progress = {
                        'status': 'TRAINING', 'pid': os.getpid(),
                        'variant': variant, 'seed': seed, 'held_fold': fold,
                        'epoch': row['epoch'], 'epochs': config['training']['epochs'],
                        'completed_runs': completed, 'expected_runs': total_runs,
                        'training_protocol_sha256': protocol_sha,
                        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                        'r13': 'LOCKED', 'r14': 'LOCKED',
                    }
                    atomic_json(OUT / 'progress.json', progress)
                    print(json.dumps({'event': 'epoch', **progress, **row}), flush=True)

                model, history = fit(
                    train_records, config, variant, seed, 'cuda', callback)
                predictions = predict(model, held_records, 'cuda')
                runtime_seconds = time.perf_counter() - run_started
                save_checkpoint(checkpoint, {
                    'schema': 'ngas-a13r-oof-checkpoint-v2',
                    'training_protocol_sha256': protocol_sha,
                    'variant': variant, 'seed': seed, 'held_fold': fold,
                    'model_config': config['variants'][variant],
                    'model_state': {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()},
                    'epochs': config['training']['epochs'],
                    'formal_device': 'cuda:0', 'precision': 'FP32',
                    'repair_id_order': ['greedy', 'regret2', 'regret3',
                                        'reconfiguration_aware', 'transport_aware'],
                })
                save_gzip_json(prediction_path, {
                    'schema': 'ngas-a13r-oof-predictions-v2',
                    'training_protocol_sha256': protocol_sha,
                    'variant': variant, 'seed': seed, 'held_fold': fold,
                    'predictions': predictions,
                })
                actions_seen = sum(len(row['actions']) for row in train_records)
                atomic_json(run_path, {
                    'schema': 'ngas-a13r-oof-run-v2',
                    'training_protocol_sha256': protocol_sha,
                    'variant': variant, 'seed': seed, 'held_fold': fold,
                    'train_instances': sorted({
                        row['instance_id'] for row in train_records}),
                    'held_instances': sorted({
                        row['instance_id'] for row in held_records}),
                    'instance_overlap': [],
                    'train_states': len(train_records), 'held_states': len(held_records),
                    'epochs': config['training']['epochs'], 'history': history,
                    'runtime_seconds': runtime_seconds,
                    'training_states_per_second':
                        len(train_records) * config['training']['epochs'] / runtime_seconds,
                    'training_actions_per_second':
                        actions_seen * config['training']['epochs'] / runtime_seconds,
                    'peak_reserved_memory_bytes': torch.cuda.max_memory_reserved(),
                    **parameter_counts(model),
                    'checkpoint_sha256': digest(checkpoint),
                    'predictions_sha256': digest(prediction_path),
                    'checkpoint_selection': 'fixed final epoch',
                    'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
                })
                completed += 1
                del model

    for variant in ('C1', 'R1'):
        seed_results = []
        combined = []
        run_records = []
        for seed in config['training']['seeds']:
            seed_predictions = []
            for fold in range(3):
                if not valid_run(variant, seed, fold, protocol_sha):
                    raise RuntimeError('OOF run artifact failed validation')
                _, prediction_path, run_path = run_paths(variant, seed, fold)
                value = load_gzip_json(prediction_path)
                seed_predictions.extend(value['predictions'])
                run_records.append(json.loads(run_path.read_text()))
            combined.extend({'seed': seed, **row} for row in seed_predictions)
            seed_results.append(evaluate_seed(
                seed, records_by_id, seed_predictions, config['oof_gate']))
        passing = sum(result['pass'] for result in seed_results)
        gate_pass = passing >= 2
        save_gzip_json(OUT / f'variants/{variant}/oof_predictions.json.gz', {
            'schema': 'ngas-a13r-combined-oof-predictions-v2',
            'training_protocol_sha256': protocol_sha,
            'variant': variant, 'predictions': combined,
        })
        result = {
            'schema': 'ngas-a13r-variant-oof-result-v2',
            'variant': variant, 'gate_pass': gate_pass,
            'seeds_passing': passing, 'seeds_required': 2,
            'seed_results': seed_results,
            'mean_across_seeds': aggregate_seed_summaries(seed_results),
            'training_profile': {
                'peak_reserved_memory_bytes':
                    max(row['peak_reserved_memory_bytes'] for row in run_records),
                'mean_training_states_per_second': statistics.fmean(
                    row['training_states_per_second'] for row in run_records),
                'mean_training_actions_per_second': statistics.fmean(
                    row['training_actions_per_second'] for row in run_records),
                'parameters': run_records[0]['parameters'],
                'trainable_parameters': run_records[0]['trainable_parameters'],
            },
            'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
        }
        atomic_json(OUT / f'variants/{variant}/oof_result.json', result)
        training_results[variant] = result

    c0_records = load_historical_cache(C0_CACHE, C0_CACHE_MANIFEST)
    c0_by_id = {row['state_id']: row for row in c0_records}
    historical = c0_results(c0_by_id, config['oof_gate'])
    profile = profile_variants(records, config)
    atomic_json(OUT / 'gpu_profile.json', {
        'schema': 'ngas-a13r-gpu-profile-v2', 'variants': profile,
        'training_protocol_sha256': protocol_sha,
        'precision': 'FP32', 'device': torch.cuda.get_device_name(0),
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    })
    selected, selection_evidence = choose(config, training_results, profile)
    if selected:
        production_seed = config['training']['production_seed']
        production_started = time.perf_counter()

        def production_callback(row: dict) -> None:
            atomic_json(OUT / 'progress.json', {
                'status': 'PRODUCTION_FIT', 'pid': os.getpid(),
                'variant': selected, 'epoch': row['epoch'],
                'epochs': config['training']['epochs'],
                'completed_runs': total_runs, 'expected_runs': total_runs,
                'training_protocol_sha256': protocol_sha,
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                'r13': 'LOCKED', 'r14': 'LOCKED',
            })

        model, history = fit(
            records, config, selected, production_seed, 'cuda', production_callback)
        production_path = OUT / 'production/revised_joint_critic.pt'
        save_checkpoint(production_path, {
            'schema': 'ngas-a13r-production-checkpoint-v2',
            'training_protocol_sha256': protocol_sha,
            'selected_variant': selected, 'seed': production_seed,
            'model_config': config['variants'][selected],
            'model_state': {
                name: value.detach().cpu() for name, value in model.state_dict().items()},
            'epochs': config['training']['epochs'],
            'formal_device': 'cuda:0', 'precision': 'FP32',
            'repair_id_order': ['greedy', 'regret2', 'regret3',
                                'reconfiguration_aware', 'transport_aware'],
        })
        production = {
            'selected_variant': selected, 'seed': production_seed,
            'runtime_seconds': time.perf_counter() - production_started,
            'checkpoint_path': str(production_path.relative_to(ROOT)),
            'checkpoint_sha256': digest(production_path),
            'history': history,
        }
        decision = (
            'NGAS_A1_3R_PASS_RTHGT' if selected == 'R1'
            else 'NGAS_A1_3R_PASS_COMPACT')
    else:
        production = None
        decision = 'NGAS_A1_REVISE_REPRESENTATION'
    comparison = {
        'schema': 'ngas-a13r-oof-representation-comparison-v2',
        'training_protocol_sha256': protocol_sha,
        'C0': historical, 'C1': training_results['C1'],
        'R1': training_results['R1'], 'gpu_profile': profile,
        'selection_evidence': selection_evidence,
        'selected_variant': selected or None, 'decision': decision,
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'representation_comparison.json', comparison)
    final = {
        'schema': 'ngas-a13r-final-decision-v2',
        'status': 'COMPLETE', 'decision': decision,
        'selected_variant': selected or None,
        'training_protocol_sha256': protocol_sha,
        'label_identity_audit_sha256': digest(IDENTITY_AUDIT),
        'critical_sync_audit_sha256': digest(CRITICAL_AUDIT),
        'representation_comparison_sha256':
            digest(OUT / 'representation_comparison.json'),
        'gpu_profile_sha256': digest(OUT / 'gpu_profile.json'),
        'production': production,
        'elapsed_seconds': time.perf_counter() - started,
        'a1_4_preparation_authorized': bool(selected),
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'final_decision.json', final)
    atomic_json(OUT / 'progress.json', {
        'status': 'COMPLETE', 'pid': os.getpid(), 'completed_runs': total_runs,
        'expected_runs': total_runs, 'decision': decision,
        'selected_variant': selected or None,
        'training_protocol_sha256': protocol_sha,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'r13': 'LOCKED', 'r14': 'LOCKED',
    })
    write_reports(comparison, final)
    print(json.dumps({'event': 'a13r_complete', 'decision': decision,
                      'selected_variant': selected}, indent=2), flush=True)


def write_reports(comparison: dict, final: dict) -> None:
    c0 = comparison['C0']['mean_across_seeds']
    c1 = comparison['C1']['mean_across_seeds']
    r1 = comparison['R1']['mean_across_seeds']
    profile = comparison['gpu_profile']
    oof = ROOT / 'docs/reports/ngas_a1/08f_a13r_oof_representation_comparison.md'
    oof.write_text(f"""# A1.3R OOF representation comparison

All variants use the same 72 frozen R12 states, 6,465 joint actions, three
training seeds, three held-instance folds, fixed 60 epochs, and the same joint
loss. C0 is immutable historical output; C1 and R1 use the corrected feature
schema and identical pooling/action heads.

| Variant | Spearman | Pair accuracy | NDCG@1 | NDCG@3 | NDCG@5 | Selected advantage | Top-1 regret | Uniform regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | {c0['mean_state_spearman']:.6f} | {c0['material_pair_accuracy']:.6f} | {c0['mean_ndcg_at_1']:.6f} | {c0['mean_ndcg_at_3']:.6f} | {c0['mean_ndcg_at_5']:.6f} | {c0['mean_selected_advantage']:.6f} | {c0['mean_top1_regret']:.6f} | {c0['mean_uniform_expected_regret']:.6f} |
| C1 | {c1['mean_state_spearman']:.6f} | {c1['material_pair_accuracy']:.6f} | {c1['mean_ndcg_at_1']:.6f} | {c1['mean_ndcg_at_3']:.6f} | {c1['mean_ndcg_at_5']:.6f} | {c1['mean_selected_advantage']:.6f} | {c1['mean_top1_regret']:.6f} | {c1['mean_uniform_expected_regret']:.6f} |
| R1 | {r1['mean_state_spearman']:.6f} | {r1['material_pair_accuracy']:.6f} | {r1['mean_ndcg_at_1']:.6f} | {r1['mean_ndcg_at_3']:.6f} | {r1['mean_ndcg_at_5']:.6f} | {r1['mean_selected_advantage']:.6f} | {r1['mean_top1_regret']:.6f} | {r1['mean_uniform_expected_regret']:.6f} |

C1 passed {comparison['C1']['seeds_passing']}/3 seed gates; R1 passed
{comparison['R1']['seeds_passing']}/3. Full per-state, S/M/L, CF1/CF2/CF3,
fallback diagnostics, parameter, memory and throughput values are in the
machine-readable comparison artifact. Selection follows the margins frozen
before formal optimization.
""")
    gpu = ROOT / 'docs/reports/ngas_a1/08e_a13r_gpu_profile.md'
    gpu.write_text(f"""# A1.3R deterministic GPU profile

Device: NVIDIA GeForce RTX 4060 Ti. Precision: FP32. AMP and TF32 disabled.

| Variant | Parameters | Peak reserved memory (bytes) | Worst representative p90 (ms) | Maximum-state p90 (ms) |
|---|---:|---:|---:|---:|
| C1 | {profile['C1']['parameters']:,} | {profile['C1']['peak_reserved_memory_bytes']:,} | {profile['C1']['worst_representative_p90_ms']:.6f} | {profile['C1']['states']['MAX']['p90_ms']:.6f} |
| R1 | {profile['R1']['parameters']:,} | {profile['R1']['peak_reserved_memory_bytes']:,} | {profile['R1']['worst_representative_p90_ms']:.6f} | {profile['R1']['states']['MAX']['p90_ms']:.6f} |

Each timing includes one complete state encoding and batched scoring of all its
joint actions. Representative S/M/L and maximum-state mean, p50, p90 and p99
values are retained in gpu_profile.json. Smoke tests established deterministic
repeated inference and finite outputs/gradients before protocol freeze.
""")
    decision = ROOT / 'docs/reports/ngas_a1/08g_a13r_final_decision.md'
    decision.write_text(f"""# NGAS-A1.3R final decision

Decision: **{final['decision']}**.

Selected representation: **{final['selected_variant'] or 'none'}**. The
decision uses only frozen R12 development labels and the predeclared OOF,
practical-margin and p90 latency rules. R13/R14 remained locked and Gurobi was
not invoked. A1.4 preparation is
{'authorized' if final['a1_4_preparation_authorized'] else 'not authorized'}.
The production checkpoint was
{'fit after representation selection' if final['production'] else 'not fit'}.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke', action='store_true')
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--train', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke()
    elif args.freeze:
        freeze()
    else:
        train()


if __name__ == '__main__':
    main()
