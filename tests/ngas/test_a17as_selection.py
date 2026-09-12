import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'artifacts/ngas_a17as/supplemental_protocol_manifest.json'


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def test_supplemental_selection_is_exact_and_balanced():
    protocol = load_protocol()
    states = protocol['states']
    assert protocol['status'] == 'FROZEN_BEFORE_SUPPLEMENTAL_OUTCOMES'
    assert len(states) == protocol['state_count'] == 27
    assert Counter((row['scale'], row['stage']) for row in states) == Counter({
        (scale, stage): 3 for scale in ('S', 'M', 'L')
        for stage in ('EARLY', 'MIDDLE', 'LATE')})
    assert Counter(
        (row['scale'], row['stage'], row['dataset_role']) for row in states
    ) == Counter({
        **{(scale, stage, 'TRAIN'): 2 for scale in ('S', 'M', 'L')
           for stage in ('EARLY', 'MIDDLE', 'LATE')},
        **{(scale, stage, 'VALIDATION'): 1 for scale in ('S', 'M', 'L')
           for stage in ('EARLY', 'MIDDLE', 'LATE')},
    })


def test_supplemental_selection_has_unique_identity_and_audit_only_labels():
    states = load_protocol()['states']
    assert len({row['state_key'] for row in states}) == 27
    assert len({row['state_payload_sha256'] for row in states}) == 27
    assert all(row['audit_only'] is True for row in states)
    assert all(row['eligible_for_future_training'] is False for row in states)
    assert {row['dataset_role'] for row in states} == {'TRAIN', 'VALIDATION'}
    assert all(
        (row['stage'] == 'EARLY' and 0 <= row['observed_budget_fraction'] < .2)
        or (row['stage'] == 'MIDDLE'
            and .4 <= row['observed_budget_fraction'] < .6)
        or (row['stage'] == 'LATE'
            and .8 <= row['observed_budget_fraction'] <= 1.)
        for row in states)


def test_supplemental_protocol_preserves_forbidden_boundaries():
    protocol = load_protocol()
    assert all(protocol['checks'].values())
    assert protocol['boundaries'] == {
        'C1_retrained': False,
        'R12': 'DEVELOPMENT_EXPOSED_NO_USE',
        'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
        'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
        'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_ACCESS',
        'adaptive_racing_implemented': False,
        'adaptive_refresh_implemented': False,
        'gurobi_run': False,
        'production_solver_changed': False,
    }
