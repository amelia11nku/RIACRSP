"""R12-only setup and original historical preprocessing for runtime audits."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from rcias_clgri.analysis import phase6i_mr
from rcias_clgri.analysis.phase6i_mr import score_frozen_candidate_bank, select_forced_candidate_roles
from rcias_clgri.analysis.phase6j_caur import build_candidate_source_features, critical_and_bottleneck_operations
from rcias_clgri.data.phase6j_access import load_phase6j_instance
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.search.common import decode_candidate
from scripts import build_phase6j_caur_tensor_cache as cache
from scripts import prepare_phase6j_caur_deployment as deploy
from scripts import run_phase6j_caur_collection as collection
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_preprocessing_decisions import inference_inputs
from scripts.audit_phase6k_start import ROOT, digest, write_once
from scripts.run_phase6j_caur_pilot import load_policy

OUT = ROOT / 'outputs/phase6k_runtime_v1/runtime'
ACTIVE = ROOT / 'configs/phase6k_runtime_v1_amended.json'


@contextmanager
def historical_mode():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def setup():
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.get_num_threads() == 1
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config = r.load_json(r.CONFIG_PATH)
    protocol, parent = deploy.validate_protocol(), r.validate_protocol()
    payload = torch.load(deploy.seed_paths(696101)[0], map_location='cpu', weights_only=False)
    frozen = payload['feature_transform']
    transform = r.FeatureTransform({k: tuple(v) for k, v in frozen['vocabularies'].items()}, frozen['medians'], frozen['iqrs'])
    models = [deploy.load_seed(seed, transform, protocol, parent, torch.device('cuda')) for seed in (696101, 696102, 696103)]
    ensemble = deploy.SharedFrozenCAUREnsemble(models)
    columns = list(dict.fromkeys(['state_id', 'target_set_id', 'scale', 'CF_level', 'frozen_raw_score', *r.CATEGORICAL_COLUMNS, *r.NUMERIC_COLUMNS]))
    frames = r.state_frames(pd.read_parquet(r.SOURCE_PATH, columns=columns))
    assert len(frames) == 288 and sum(map(len, frames.values())) == 6809
    return dict(config=config, protocol=protocol, transform=transform, e0=ensemble,
                policy=load_policy(config, 'cuda'), frames=frames, samples=r.load_samples(),
                replay_paths=cache.replay_index(), instances={}, alns=collection.read_alns_config(config))


def cell_states(env):
    first = {}
    for sid, frame in sorted(env['frames'].items()):
        first.setdefault((str(frame.scale.iloc[0]), str(frame.CF_level.iloc[0])), sid)
    assert len(first) == 9
    return list(first.values())


def context(env, state_id):
    replay = r.load_json(env['replay_paths'][state_id])
    relative = replay['instance_relative_path']
    if relative not in env['instances']:
        path = ROOT / env['config']['instance_suite']['root'] / relative
        assert digest(path) == replay['instance_sha256']
        env['instances'][relative] = load_phase6j_instance(path)
    instance, snapshot = env['instances'][relative], replay['snapshot']
    current = decode_candidate(instance, collection.candidate_from_dict(snapshot['current_candidate']))
    assert abs(current.makespan - snapshot['current_makespan']) < 1e-9
    count = min(max(2, round(instance.num_operations * env['alns'].destroy_fraction)), instance.num_operations)
    return dict(instance=instance, current=current, snapshot=snapshot, count=count, replay=replay, state_id=state_id)


def realize(env, ctx):
    """Fresh complete preprocessing; never substitutes R12 cached graph/features."""
    instance, current, sid = ctx['instance'], ctx['current'], ctx['state_id']
    captured = []
    original_builder = phase6i_mr.build_live_proposal_bank

    def capture_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        captured.append(result)
        return result

    with historical_mode():
        # Capture the membership objects already produced inside the unchanged
        # historical function. The wrapper invokes the original builder once
        # and does not alter its inputs, outputs, operators, dtype, or ordering.
        with patch.object(phase6i_mr, 'build_live_proposal_bank', capture_builder):
            bank = score_frozen_candidate_bank(env['policy'], instance, current, state_id=sid,
                destroy_count=ctx['count'], search_progress=ctx['snapshot']['search_progress'],
                search_stage=collection.search_stage(ctx['snapshot']['search_progress']))
    if len(captured) != 1:
        raise RuntimeError('historical preprocessing must generate exactly one proposal bank')
    generated, records = captured[0]
    started = time.perf_counter()
    fallback = next(x.arm.target_set_id for x in select_forced_candidate_roles(bank.arms) if x.role == 'ALNS_RELATED_FALLBACK')
    critical, bottleneck, _ = critical_and_bottleneck_operations(instance, current)
    features = pd.DataFrame(build_candidate_source_features(generated, state_id=sid,
        operation_count=instance.num_operations, fallback_target_set_id=fallback,
        frozen_scores={a.target_set_id: a.raw_score for a in bank.arms},
        critical_operations=critical, bottleneck_operations=bottleneck)).sort_values('target_set_id').reset_index(drop=True)
    feature_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    sample = NIStateSample(env['policy'].tensorizer.tensorize(bank.graph),
                          tensorize_action_records(bank.graph, sorted(records, key=lambda x: x['target_set_id'])), {})
    packed = inference_inputs(sample, features, env['transform'], 'cpu')
    tensor_ms = (time.perf_counter() - started) * 1000
    return packed, sample, bank, {'historical': bank.timings_ms, 'source_features_ms': feature_ms, 'j1_tensorization_ms': tensor_ms}


def reorder_historical_membership(actions):
    """Reorder already projected action memberships without rebuilding them."""
    source_ids = tuple(actions.target_set_ids)
    order = sorted(range(len(source_ids)), key=source_ids.__getitem__)
    target_ids = tuple(source_ids[index] for index in order)
    memberships = []
    target_action_index = []
    for target_index, source_index in enumerate(order):
        start = int(actions.action_ptr[source_index])
        stop = int(actions.action_ptr[source_index + 1])
        values = actions.target_operation_indices[start:stop]
        memberships.append(values)
        target_action_index.append(torch.full_like(values, target_index))
    return SimpleNamespace(
        target_set_ids=target_ids,
        target_operation_indices=torch.cat(memberships),
        target_action_index=torch.cat(target_action_index),
    )


def realize_reused(env, ctx):
    """Fresh historical preprocessing with same-decision tensor reuse for J1."""
    instance, current, sid = ctx['instance'], ctx['current'], ctx['state_id']
    captured_banks, captured_samples, captured_cuda_batches = [], [], []
    original_builder = phase6i_mr.build_live_proposal_bank
    original_sample = phase6i_mr.NIStateSample

    def capture_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        captured_banks.append(result)
        return result

    def capture_sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        captured_samples.append(result)
        return result

    def capture_cuda_batch(_module, args):
        captured_cuda_batches.append(args[0])

    handle = env['policy'].model.register_forward_pre_hook(capture_cuda_batch)
    try:
        with historical_mode():
            # Both wrappers return the exact original objects. They expose the
            # already materialized inputs after the historical path has used
            # them; no historical operator or numeric result is replaced.
            with patch.object(phase6i_mr, 'build_live_proposal_bank', capture_builder), \
                    patch.object(phase6i_mr, 'NIStateSample', capture_sample):
                bank = score_frozen_candidate_bank(env['policy'], instance, current, state_id=sid,
                    destroy_count=ctx['count'], search_progress=ctx['snapshot']['search_progress'],
                    search_stage=collection.search_stage(ctx['snapshot']['search_progress']))
    finally:
        handle.remove()
    if not (len(captured_banks) == len(captured_samples) == len(captured_cuda_batches) == 1):
        raise RuntimeError('historical preprocessing must materialize one proposal bank and one model input')
    generated, _ = captured_banks[0]
    historical_sample = captured_samples[0]
    started = time.perf_counter()
    fallback = next(x.arm.target_set_id for x in select_forced_candidate_roles(bank.arms)
                    if x.role == 'ALNS_RELATED_FALLBACK')
    critical, bottleneck, _ = critical_and_bottleneck_operations(instance, current)
    features = pd.DataFrame(build_candidate_source_features(generated, state_id=sid,
        operation_count=instance.num_operations, fallback_target_set_id=fallback,
        frozen_scores={a.target_set_id: a.raw_score for a in bank.arms},
        critical_operations=critical, bottleneck_operations=bottleneck)).sort_values('target_set_id').reset_index(drop=True)
    feature_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    sample = SimpleNamespace(
        graph=historical_sample.graph,
        actions=reorder_historical_membership(historical_sample.actions),
    )
    packed = inference_inputs(sample, features, env['transform'], 'cpu')
    reuse_ms = (time.perf_counter() - started) * 1000
    timing = {'historical': bank.timings_ms, 'source_features_ms': feature_ms,
              'j1_same_decision_reuse_ms': reuse_ms}
    return packed, sample, bank, timing, captured_cuda_batches[0]


def transfer(packed):
    batch = packed['batch']
    for name in ('node_features', 'node_batch_index', 'node_ptr', 'edges'):
        setattr(batch, name, {k: v.to('cuda') for k, v in getattr(batch, name).items()})
    for name in ('graph_numeric', 'graph_categorical', 'action_to_state', 'target_operation_indices', 'target_action_index'):
        setattr(batch, name, getattr(batch, name).to('cuda'))
    for name in ('categorical', 'numeric', 'fallback_indices'):
        packed[name] = packed[name].to('cuda')
    return packed


def transfer_reused(packed, historical_cuda_batch):
    """Reuse historical CUDA graph tensors; transfer only J1-specific inputs."""
    batch = packed['batch']
    for name in ('node_features', 'node_batch_index', 'node_ptr', 'edges',
                 'graph_numeric', 'graph_categorical', 'action_to_state'):
        setattr(batch, name, getattr(historical_cuda_batch, name))
    for name in ('target_operation_indices', 'target_action_index'):
        setattr(batch, name, getattr(batch, name).to('cuda'))
    for name in ('categorical', 'numeric', 'fallback_indices'):
        packed[name] = packed[name].to('cuda')
    return packed


def freeze_stage(name, extra, sources):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'protocol.json').exists():
        raise RuntimeError(f'{name} already preregistered; inspect or resume rather than rerun')
    hashes = {}
    for path in [ACTIVE, Path(__file__), *map(Path, sources)]:
        relative = path.resolve().relative_to(ROOT)
        hashes[str(relative)] = digest(path)
        target = directory / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_once(directory / 'protocol.json', {'status': 'FROZEN_BEFORE_EXECUTION', 'source_hashes': hashes,
        'r13_accessed': False, 'r14_accessed': False, **extra})
    return directory


def percentiles(values):
    return {f'p{q}': float(np.percentile(values, q)) for q in (50, 90, 99)}
