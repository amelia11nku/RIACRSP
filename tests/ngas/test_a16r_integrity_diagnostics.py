import json
import socket
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_ngas.critic.revised_critic import RevisedJointCritic
from rcias_ngas.evaluation.a16r_diagnostics import build_all
from rcias_ngas.evaluation.a16_io import validate_raw_contract, write_new_json
from rcias_ngas.execution import (
    ExperimentSessionLock, LockHeldError, recover_stale_lock,
)
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime import CompactStateBuilder, ProductionRefreshRuntime
from rcias_ngas.search.ngas_solver import (
    NGASDiagnosticConfig, NGASSearchConfig, solve_ngas,
)
import scripts.finalize_ngas_a16r_integrity_diagnostic as a16r_finalizer


class TinyCritic:
    variant = 'C1'
    sha256 = 'a16r-test-c1'
    device = torch.device('cpu')

    def __init__(self):
        torch.manual_seed(1601)
        self.model = RevisedJointCritic(
            'rt_hgt', hidden=16, layers=1, heads=4).eval()


def make_lock(tmp_path: Path, heartbeat=.02) -> ExperimentSessionLock:
    return ExperimentSessionLock(
        tmp_path / 'formal.lock', tmp_path / 'sessions',
        experiment='test', stage='A1.6R', implementation_commit='abc',
        command=['test'], formal_owner_id='owner',
        heartbeat_interval_seconds=heartbeat)


def test_exclusive_lock_rejects_second_owner_before_work_and_records_heartbeat(tmp_path):
    first = make_lock(tmp_path).acquire()
    try:
        with pytest.raises(LockHeldError):
            make_lock(tmp_path).acquire()
        first.heartbeat(source='test')
        events = [json.loads(line) for line in first.event_log_path.read_text().splitlines()]
        assert events[0]['event'] == 'lock_acquired'
        assert any(row['event'] == 'heartbeat' for row in events)
        assert json.loads(first.heartbeat_path.read_text())['session_id'] == first.session_id
    finally:
        first.release()
    assert not (tmp_path / 'formal.lock').exists()
    events = [json.loads(line) for line in first.event_log_path.read_text().splitlines()]
    assert events[-1]['event'] == 'lock_released'


def test_stale_lock_recovery_requires_proven_dead_same_host_owner(tmp_path):
    lock_path = tmp_path / 'formal.lock'
    lock_path.write_text(json.dumps({
        'hostname': socket.gethostname(), 'pid': 99999999,
        'process_start_ticks': 1, 'session_id': 'dead',
    }))
    recovered = recover_stale_lock(lock_path, tmp_path / 'recovered')
    assert recovered.exists() and not lock_path.exists()

    live = make_lock(tmp_path).acquire()
    try:
        with pytest.raises(LockHeldError):
            recover_stale_lock(live.lock_path, tmp_path / 'recovered')
    finally:
        live.release()


def test_a16r_raw_contract_is_create_once_and_resumable_only_when_valid(tmp_path):
    task = {'instance_id': 'i1', 'instance_sha256': 'sha', 'seed': 746101,
            'budget_seconds': 10.}
    payload = {
        'schema': 'ngas-a16r-formal-run-v1', 'status': 'COMPLETE',
        'algorithm_id': 'NGAS_A1_6R', 'protocol_sha256': 'protocol',
        'instance_id': 'i1', 'instance_sha256': 'sha', 'seed': 746101,
        'budget_seconds': 10., 'budget_accounting': 'A16_INSTANCE_TOTAL',
        'feasible': True, 'r13_accessed': False, 'r14_accessed': False,
        'gurobi_run': False,
    }
    path = tmp_path / 'raw.json'
    write_new_json(path, payload)
    validate_raw_contract(
        payload, task, 'protocol', schema='ngas-a16r-formal-run-v1',
        algorithm_id='NGAS_A1_6R')
    with pytest.raises(FileExistsError):
        write_new_json(path, payload)
    with pytest.raises(RuntimeError):
        validate_raw_contract(
            {**payload, 'seed': 746102}, task, 'protocol',
            schema='ngas-a16r-formal-run-v1', algorithm_id='NGAS_A1_6R')


def test_diagnostic_instrumentation_preserves_fixed_iteration_search_behavior():
    instance = load_instance('instances/tiny/tiny_01.json')
    config = NGASSearchConfig(candidate_trials=2, iteration_limit=5)
    baseline_critic = TinyCritic()
    baseline = solve_ngas(
        instance, 60., 746101, 'PERSISTENT_FIXED_REFRESH', baseline_critic, config,
        refresh_runtime=ProductionRefreshRuntime(baseline_critic),
        budget_accounting='A16_INSTANCE_TOTAL')
    diagnostic_critic = TinyCritic()
    diagnostic = solve_ngas(
        instance, 60., 746101, 'PERSISTENT_FIXED_REFRESH', diagnostic_critic, config,
        refresh_runtime=ProductionRefreshRuntime(diagnostic_critic),
        budget_accounting='A16_INSTANCE_TOTAL',
        diagnostic_config=NGASDiagnosticConfig(enabled=True))
    fields = (
        'action_id', 'candidate_makespan', 'current_after', 'best_after',
        'accepted', 'outcome_class',
    )
    assert [[row[name] for name in fields] for row in baseline.diagnostics['iterations']] == [
        [row[name] for name in fields] for row in diagnostic.diagnostics['iterations']]
    assert baseline.best.candidate == diagnostic.best.candidate
    assert baseline.best.makespan == diagnostic.best.makespan
    for iteration in diagnostic.diagnostics['iterations']:
        trials = iteration['a16r_observation']['candidate_trials']
        assert len(trials) == 2
        assert [row['trial'] for row in trials] == [1, 2]
        assert all(row['repair_rng_seed'] > 0 for row in trials)
    summaries = build_all([{
        'scale': 'S', 'budget_seconds': 60.,
        'search_diagnostics': diagnostic.diagnostics,
    }])
    assert set(summaries) == {
        'search_stage', 'best_of_k', 'prior_staleness',
        'critic_portfolio', 'passive_critic',
    }
    assert sum(row['iterations'] for row in summaries['search_stage']['rows']) == 5


def test_compact_capacity_bounds_cover_observed_build_without_resize():
    instance = load_instance('instances/tiny/tiny_01.json')
    h1 = solve_dispatching(instance, 'H1')
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    builder = CompactStateBuilder(instance)
    built = builder.build(
        current, 'a16r-capacity', RNGStreams(instance.instance_id, 746101))
    audit = builder.capacity_audit()
    capacity = audit['allocated_capacities']
    assert built.node_count <= capacity['neural_nodes']
    assert built.forward_edge_count <= capacity['forward_edges']
    assert built.target_count <= capacity['targets']
    assert len(built.actions) <= capacity['actions']
    assert audit['required_upper_bounds'] == capacity
    assert audit['resize_events'] == 0
    assert not audit['silent_truncation_allowed']


def test_production_runtime_rejects_cross_thread_mutable_workspace_access():
    instance = load_instance('instances/tiny/tiny_01.json')
    runtime = ProductionRefreshRuntime(TinyCritic())
    observed = []

    def use_from_non_owner():
        try:
            runtime.prepare_instance(instance)
        except Exception as error:
            observed.append(error)

    worker = threading.Thread(target=use_from_non_owner)
    worker.start()
    worker.join()
    assert len(observed) == 1
    assert 'outside its owner' in str(observed[0])


def _session_events(session_id, run_keys):
    events = []
    for event, fields in (
            ('lock_acquired', {}), ('heartbeat', {}), ('heartbeat', {}),
            ('gpu_resource_preflight', {'competing_compute_processes': []})):
        events.append({'event': event, **fields})
    for instance_id, seed in run_keys:
        events.append({'event': 'formal_run_started', 'instance_id': instance_id,
                       'seed': seed})
        events.append({'event': 'formal_run_completed', 'instance_id': instance_id,
                       'seed': seed})
    events.append({'event': 'lock_released'})
    events[0]['implementation_commit'] = 'commit'
    return [{**row, 'sequence': index + 1, 'formal_owner_id': 'owner',
             'session_id': session_id, 'pid': 101, 'hostname': 'host'}
            for index, row in enumerate(events)]


def test_finalizer_integrity_gate_rejects_more_than_one_formal_session(
        tmp_path, monkeypatch):
    out = tmp_path / 'out'
    sessions = out / 'integrity/sessions'
    sessions.mkdir(parents=True)
    run_keys = [(f'i{index:02d}', 746101) for index in range(54)]
    first_events = _session_events('session-one', run_keys)
    (sessions / 'session-one.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in first_events))
    payloads = [{
        'instance_id': instance_id, 'seed': seed, 'source_commit': 'commit',
        'formal_owner_id': 'owner', 'session_id': 'session-one',
    } for instance_id, seed in run_keys]
    monkeypatch.setattr(a16r_finalizer, 'OUT', out)
    monkeypatch.setattr(
        a16r_finalizer.subprocess, 'run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    protocol = {'formal_owner_id': 'owner', 'implementation_commit': 'commit'}
    assert a16r_finalizer.session_integrity_audit(protocol, payloads)['status'] == 'PASS'

    second_events = _session_events('session-two', [])
    (sessions / 'session-two.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in second_events))
    failed = a16r_finalizer.session_integrity_audit(protocol, payloads)
    assert failed['status'] == 'FAIL'
    assert not failed['checks']['exactly_one_process_session']
