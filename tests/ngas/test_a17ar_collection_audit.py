import copy

from scripts.audit_ngas_a17ar_collection import canonical_hash, validate_state


def _state_and_boundary():
    task = {
        'dataset_role': 'TRAIN', 'instance_id': 'clean-1',
        'content_sha256': 'instance-sha', 'trajectory_seed': 817000000,
    }
    protocol = {
        'candidate_bank_hash': 'bank-sha', 'checkpoint_sha256': 'critic-sha',
        'capture_fractions': [.1, .3, .5, .7, .9],
    }
    features = {
        'schema': 'ngas-compact-relational-state-features-v1',
        'node_features': [[0.]], 'node_types': [0],
        'edge_index': [[], []], 'edge_types': [], 'edge_features': [],
        'operation_nodes': [0], 'critical_mask': [1],
    }
    features['graph_hash'] = canonical_hash({name: features[name] for name in (
        'node_types', 'edge_index', 'edge_types', 'operation_nodes')})
    features['state_feature_hash'] = canonical_hash(features)
    candidate = {
        'operation_order': [0], 'island_assignment': [0],
        'w_assignment': [0], 'f_assignment': [0],
    }
    trial = {
        'candidate_makespan': 10., 'decoder_seconds': .1,
        'repair_seconds': .01, 'trial_seconds': .11,
    }
    state = {
        'schema': 'ngas-a17ar-clean-trajectory-state-v1',
        'dataset_role': 'TRAIN', 'audit_only': False,
        'eligible_for_future_training': True, 'instance_id': 'clean-1',
        'instance_content_sha256': 'instance-sha', 'trajectory_seed': 817000000,
        'candidate_bank_identifier': 'NGAS_BANK_V1_FULL_UNIQUE',
        'candidate_bank_hash': 'bank-sha', 'critic_checkpoint_hash': 'critic-sha',
        'compact_relational_features': features,
        'candidate_action_ids': ['a', 'b'],
        'critic_scores': {
            'advantage': [0., 1.], 'beats_fallback_probability': [.2, .8],
            'neural_prior': [.25, .75], 'neural_prior_entropy': .562,
        },
        'selected_joint_action': {'action_id': 'b'},
        'portfolio_adjusted_scores': {'combined_top1_action_id': 'b'},
        'stochastic_trial_outcomes': [dict(trial, trial=index) for index in range(1, 9)],
        'target_capture_fraction': .5, 'normalized_budget_fraction': .51,
        'search_condition_tags': ['improving', 'high_action_entropy'],
        'replay_metadata': {
            'current_candidate': candidate,
            'candidate_fingerprint': canonical_hash(candidate),
        },
    }
    return state, task, protocol


def test_state_integrity_contract_accepts_valid_full_bank_state():
    state, task, protocol = _state_and_boundary()
    assert validate_state(state, task, protocol) == []


def test_state_integrity_contract_rejects_feature_hash_and_trial_drift():
    state, task, protocol = _state_and_boundary()
    corrupted = copy.deepcopy(state)
    corrupted['compact_relational_features']['node_features'][0][0] = 1.
    corrupted['stochastic_trial_outcomes'].pop()
    errors = validate_state(corrupted, task, protocol)
    assert 'state_feature_hash_mismatch' in errors
    assert 'selected_action_trial_count_not_eight' in errors
