import json
from pathlib import Path

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_ngas.rng import RNGStreams
from rcias_ngas.runtime.compact_state import CompactStateBuilder
from scripts.freeze_ngas_a17ar_trajectory_protocol import build_manifest
from scripts.run_ngas_a17ar_trajectory_collection import _compact_features, valid_raw


ROOT = Path(__file__).resolve().parents[2]


def test_clean_trajectory_manifest_has_frozen_counts_and_reserved_disjointness():
    manifest = build_manifest()
    assert manifest['revision'] == 2
    assert manifest['supersedes_protocol_sha256'] == (
        '441b613936fc7f0ef59b3410dfbcd5135af3481e8b5ce43f8ce596766b1b0a73')
    assert manifest['expected_runs'] == 108
    assert manifest['expected_clean_states'] == 540
    assert len(manifest['train_instances']) == 81
    assert len(manifest['validation_instances']) == 27
    assert all(manifest['checks'].values())
    assert manifest['R12'] == 'DEVELOPMENT_EXPOSED_AUDIT_ONLY'
    assert manifest['R13'] == 'LOCKED_FINAL_EVAL_NO_ACCESS'
    assert manifest['R14'] == 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS'
    assert manifest['RCIAS_CB1_CORE45'].startswith('EXTERNAL_BASELINE_ONLY')


def test_clean_split_is_disjoint_from_every_reserved_family_by_id_and_hash():
    manifest = build_manifest()
    registry = json.loads((ROOT / 'configs/dataset_role_registry.json').read_text())
    reserved = [row for row in registry['instances'] if row['family_id'] in {
        'RCIAS_CB1_R12', 'RCIAS_CB1_R13', 'RCIAS_CB1_R14', 'RCIAS_CB1_CORE45'}]
    assert not ({row['instance_id'] for row in manifest['instances']}
                & {row['instance_id'] for row in reserved})
    assert not ({row['content_sha256'] for row in manifest['instances']}
                & {row['content_sha256'] for row in reserved})


def test_candidate_trials_means_eight_trials_for_each_selected_action():
    config = json.loads(
        (ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml').read_text())
    assert config['production_solver']['candidate_trials'] == 8
    assert config['revision'] == 2
    assert config['production_solver']['search']['candidate_trials'] == 8
    assert config['production_solver']['candidate_trials_semantics'] == (
        'stochastic repair/decode realizations per selected joint action')


def test_compact_features_are_rebuilt_from_replayable_candidate():
    instance = load_instance(ROOT / 'instances/tiny/tiny_01.json')
    h1 = solve_dispatching(instance, 'H1')
    candidate = candidate_from_actions(instance, h1.actions)
    current = decode_candidate(instance, candidate)
    state_id = f'{instance.instance_id}:seed817000000:iteration0'
    compact = CompactStateBuilder(instance).build(
        current, state_id, RNGStreams(instance.instance_id, 817000000))
    snapshot = {
        'state_id': state_id, 'current_makespan': current.makespan,
        'current_candidate': {
            'operation_order': list(candidate.operation_order),
            'island_assignment': list(candidate.island_assignment),
            'w_assignment': list(candidate.w_assignment),
            'f_assignment': list(candidate.f_assignment),
        },
        'action_ids': [action.action_id for action in compact.actions],
    }
    features = _compact_features(
        instance, snapshot, 817000000, CompactStateBuilder(instance))
    assert features['node_features']
    assert features['edge_index']
    assert features['state_feature_hash']
    assert features['graph_hash']


def test_raw_resume_contract_requires_role_hashes_and_complete_states(tmp_path):
    task = {
        'instance_id': 'clean-i1', 'content_sha256': 'instance-hash',
        'trajectory_seed': 817000000, 'dataset_role': 'TRAIN',
    }
    state = {
        'dataset_role': 'TRAIN', 'audit_only': False,
        'instance_content_sha256': 'instance-hash',
        'compact_relational_features': {
            'state_feature_hash': 'feature-hash', 'graph_hash': 'graph-hash'},
    }
    payload = {
        'schema': 'ngas-a17ar-clean-trajectory-run-v1', 'status': 'COMPLETE',
        'protocol_sha256': 'protocol-hash', 'instance_id': 'clean-i1',
        'instance_content_sha256': 'instance-hash',
        'trajectory_seed': 817000000, 'states': [dict(state) for _ in range(5)],
        'search_diagnostics': {'final_replay': {'feasible': True}},
    }
    path = tmp_path / 'raw.json'
    path.write_text(json.dumps(payload))
    assert valid_raw(path, task, 'protocol-hash')
    payload['states'][0]['dataset_role'] = 'VALIDATION'
    path.write_text(json.dumps(payload))
    assert not valid_raw(path, task, 'protocol-hash')
