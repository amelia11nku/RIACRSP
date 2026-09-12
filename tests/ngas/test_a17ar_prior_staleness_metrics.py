from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_ngas_a17ar_prior_staleness import (
    representation_drift, score_drift,
)


ROOT = Path(__file__).resolve().parents[2]


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


def test_prior_staleness_protocol_is_frozen_on_all_governed_origins():
    path = ROOT / 'artifacts/ngas_a17ar/prior_staleness_protocol_manifest.json'
    protocol = json.loads(path.read_text())
    assert protocol['status'] == 'FROZEN_BEFORE_STALENESS_RESULTS'
    assert protocol['offsets'] == [0, 5, 10, 15, 19]
    assert protocol['refresh_interval'] == 20
    assert len(protocol['states']) == 36
    assert sum(row['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT'
               for row in protocol['states']) == 18
    assert sum(row['dataset_origin'] == 'R12_DEVELOPMENT_EXPOSED'
               for row in protocol['states']) == 18
    assert all(protocol['checks'].values())
    for source, expected_sha256 in protocol['source_hashes'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected_sha256
    assert protocol['boundaries']['R13'] == 'LOCKED_FINAL_EVAL_NO_ACCESS'
    assert protocol['boundaries']['R14'] == 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS'
    assert protocol['boundaries']['adaptive_refresh'] is False
