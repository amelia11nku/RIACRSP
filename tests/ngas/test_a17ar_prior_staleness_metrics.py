from types import SimpleNamespace

import numpy as np

from scripts.run_ngas_a17ar_prior_staleness import (
    representation_drift, score_drift,
)


def _action(name, operations):
    return SimpleNamespace(
        action_id=name,
        size='small',
        repair='greedy',
        target=SimpleNamespace(operations=tuple(operations)),
    )


def _refresh(actions, prior, advantage):
    ranking = tuple(sorted(
        range(len(actions)), key=lambda index: (-prior[index], actions[index].action_id)))
    return SimpleNamespace(
        actions=tuple(actions), prior=tuple(prior), advantage=tuple(advantage),
        ranking=ranking,
    )


def test_score_drift_matches_actions_by_semantics_instead_of_state_bound_ids():
    stale = _refresh(
        [_action('old-a', ['o1']), _action('old-b', ['o2'])],
        [.8, .2], [.4, -.1])
    fresh = _refresh(
        [_action('new-a', ['o1']), _action('new-b', ['o2'])],
        [.3, .7], [.1, .2])
    result = score_drift(stale, fresh)
    assert result['common_semantic_actions'] == 2
    assert result['semantic_bank_jaccard'] == 1.
    assert result['top1_same_semantics'] is False
    assert result['top5_overlap'] == 2
    assert result['mean_absolute_prior_drift_common'] == .5


def test_representation_drift_reports_feature_and_typed_edge_change():
    base = {
        'operation_features': np.asarray([[1., 0.], [0., 1.]]),
        'graph': {(0, 1, 2)}, 'critical_signature': 'a',
        'dominant_bottleneck': 'W',
    }
    current = {
        'operation_features': np.asarray([[1., 1.], [0., 1.]]),
        'graph': {(0, 1, 2), (1, 0, 3)}, 'critical_signature': 'b',
        'dominant_bottleneck': 'F', 'operation_feature_sha256': 'x',
        'graph_sha256': 'y',
    }
    result = representation_drift(base, current)
    assert result['operation_feature_relative_l2'] > 0.
    assert result['operation_feature_mean_absolute'] == .25
    assert result['graph_edge_jaccard'] == .5
    assert result['critical_signature_changed'] is True
    assert result['dominant_bottleneck_changed'] is True
