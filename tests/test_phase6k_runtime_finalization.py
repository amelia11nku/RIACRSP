import copy

import pytest

from scripts.finalize_phase6k_runtime_revision import audit_formal_stream


def fixture():
    rows = []
    for state, scale in [('a', 'S'), ('b', 'M'), ('c', 'L')]:
        for repetition in range(-3, 5):
            rows.append({
                'state_id': state,
                'scale': scale,
                'repetition': repetition,
                'neural_ms': 25.0,
                'neural_cuda_span_ms': 24.9,
                'full_live_ms': 101.0,
                'decision': {'selected_target_set_id': 'target'},
                'preprocessing_timings': {
                    'historical': {'total': 70.0},
                    'source_features_ms': 2.0,
                    'j1_same_decision_reuse_ms': 1.0,
                },
            })
    summary = {scale: {
        'neural_ms': {'p50': 25.0, 'p90': 25.0, 'p99': 25.0},
        'full_live_ms': {'p50': 101.0, 'p90': 101.0, 'p99': 101.0},
    } for scale in ('overall', 'S', 'M', 'L')}
    result = {'summary': summary, 'status': 'MODEL_REVISION_RUNTIME',
              'neural_cap_pass': True, 'full_live_cap_pass': False}
    return rows, result


def test_formal_stream_recomputes_failed_runtime_gate_from_complete_rows():
    rows, result = fixture()
    summary, measured, warmups = audit_formal_stream(rows, result, ['a', 'b', 'c'])
    assert len(measured) == 15 and len(warmups) == 9
    assert summary['overall']['neural_ms']['p90'] == 25.0
    assert summary['overall']['full_live_ms']['p90'] == 101.0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'nonfinite', 'decision',
                                    'scope', 'percentile', 'gate'])
def test_formal_stream_rejects_incomplete_or_inconsistent_terminal_evidence(fault):
    rows, result = fixture()
    rows, result = copy.deepcopy(rows), copy.deepcopy(result)
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1]['repetition'] = 3
    elif fault == 'nonfinite':
        rows[0]['full_live_ms'] = float('nan')
    elif fault == 'decision':
        rows[0]['decision'] = {}
    elif fault == 'scope':
        rows[0]['preprocessing_timings'].pop('historical')
    elif fault == 'percentile':
        result['summary']['overall']['full_live_ms']['p90'] = 99.0
    else:
        result['status'] = 'PASS'
    with pytest.raises(ValueError):
        audit_formal_stream(rows, result, ['a', 'b', 'c'])
