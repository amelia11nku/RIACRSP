from copy import deepcopy

import pytest

from scripts.run_ngas_label_pilot import interaction_diagnostics, persist_or_verify, summarize


def synthetic_rows():
    rows = []
    for state in range(6):
        for size in ('small', 'medium', 'large'):
            for target in range(2):
                for repair_index, repair in enumerate(('greedy', 'regret2', 'regret3', 'reconfiguration_aware', 'transport_aware')):
                    # Repair flips the ranking of two targets.
                    advantage = .01 * (target if repair_index % 2 else 1-target)
                    rows.append({
                        'state_id': str(state), 'action': {
                            'size': size, 'repair': repair,
                            'target': {'target_id': str(target), 'origin_families': list('ABCDE')}},
                        'replicates': [{'crn_seed': i, 'advantage': advantage} for i in range(3)],
                        'advantage_mean': advantage, 'advantage_variance': 0.,
                        'feasible': True, 'runtime_seconds': .1,
                    })
    return rows


def test_pilot_reports_repair_interactions():
    rows = [r for r in synthetic_rows() if r['state_id'] == '0']
    result = interaction_diagnostics(rows)
    assert result['target_repair_residual_rms'] > 0
    assert result['noise_separated_target_rank_reversals_across_repairs'] > 0
    assert result['within_target_size_noise_separated_repair_pairs'] > 0


def test_gate_rejects_flat_noisy_labels():
    config = {'gates': {'min_informative_pair_fraction': .1, 'min_informative_states': 3,
                        'min_positive_mean_advantage_fraction': .05,
                        'max_mean_seconds_per_joint_action_with_three_replicates': 30}}
    rows = synthetic_rows()
    assert summarize(rows, config)['decision'] == 'NGAS_A1_LABEL_PILOT_PASS'
    flat = deepcopy(rows)
    for row in flat:
        row['advantage_mean'] = 0.
        for replicate in row['replicates']:
            replicate['advantage'] = 0.
    gate = summarize(flat, config)
    assert gate['decision'] == 'NGAS_A1_REVISE_LABELS'
    assert not gate['checks']['noise_usable']
    assert not gate['checks']['positive_opportunities']


def test_resume_refuses_changed_artifact(tmp_path):
    path = tmp_path / 'state.json'
    persist_or_verify(path, {'candidate': [1, 2]})
    persist_or_verify(path, {'candidate': [1, 2]})
    with pytest.raises(ValueError, match='mismatch'):
        persist_or_verify(path, {'candidate': [2, 1]})
