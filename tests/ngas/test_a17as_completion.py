import csv
import hashlib
import json
from pathlib import Path

from rcias_ngas.evaluation.a17as import state_alignment


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17as_v1'
A17AR = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'


def rows(path: Path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_full_bank_completion_and_eight_trial_contract():
    audit = json.loads((OUT / 'audit/full_bank_completion_audit.json').read_text())
    manifest = json.loads((OUT / 'raw/full_bank_manifest.json').read_text())
    assert audit['status'] == 'PASS'
    assert all(audit['checks'].values())
    assert audit['completed_states'] == manifest['completed_states'] == 27
    assert audit['completed_actions'] == manifest['completed_actions'] > 0
    assert audit['completed_direct_trials'] == audit['completed_actions'] * 8
    assert audit['completed_continuation_decoder_evaluations'] == (
        audit['completed_actions'] * 16)
    assert audit['forbidden_family_access_count'] == 0
    assert audit['boundaries']['production_solver_changed'] is False
    assert audit['boundaries']['C1_retrained'] is False
    for relative_path, expected_hash in manifest['files'].items():
        path = ROOT / relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        state = json.loads(path.read_text())
        assert state['audit_only'] is True
        assert state['eligible_for_future_training'] is False
        assert all([trial['trial'] for trial in action['direct_trials']]
                   == list(range(1, 9)) for action in state['actions'])


def test_alignment_rows_are_independently_recomputed_from_actions():
    derived = rows(OUT / 'derived/utility_alignment_state.csv')
    by_key = {(row['state_key'], row['left'], row['right']): row for row in derived}
    for path in sorted((OUT / 'raw/full_bank').glob('*.json')):
        state = json.loads(path.read_text())
        for metric in state_alignment(state['actions']):
            row = by_key[(state['state_key'], metric['left'], metric['right'])]
            assert row['same_top1_all_states'] == str(
                metric['same_top1_all_states'])
            assert int(row['top5_overlap_count']) == metric['top5_overlap_count']
            assert int(row['top10_overlap_count']) == metric['top10_overlap_count']
            expected = metric['spearman']
            assert (row['spearman'] == '') if expected is None else (
                abs(float(row['spearman']) - expected) < 1e-12)


def test_u1_u2_fix_is_consistent_across_summary_report_and_figure_source():
    fix = json.loads((OUT / 'derived/metric_consistency_fix.json').read_text())
    assert fix['status'] == 'PASS'
    assert fix['legacy_clean_positive_U1_same_top1'] == {
        'denominator': 5, 'numerator': 4, 'rate': .8}
    assert fix['corrected_clean_positive_U1_same_top1'] == {
        'denominator': 5, 'numerator': 5, 'rate': 1.}
    assert len(fix['affected_states']) == 1
    corrected = rows(A17AR / 'derived/utility_alignment.csv')
    figure = rows(ROOT / 'reports/figures/ngas_a17ar/source_data/'
                  'cost_normalized_utility.csv')
    expected = [row for row in corrected
                if row['left'] == 'U1_SHORT_HORIZON'
                and row['right'] == 'U2_COST_NORMALIZED']
    assert figure == expected
    report = (ROOT / 'reports/ngas_a17ar_utility_alignment.md').read_text()
    compact_report = ''.join(report.split())
    assert '4/5' in compact_report and '5/5' in compact_report


def test_all_alignment_summary_rates_expose_denominators():
    summary = rows(OUT / 'derived/utility_alignment_summary.csv')
    assert summary
    for row in summary:
        assert int(row['states_all_denominator']) > 0
        assert row['same_top1_all_states_denominator'] == row[
            'states_all_denominator']
        assert int(row['same_top1_pair_informative_denominator']) == int(
            row['valid_spearman_denominator'])


def test_current_production_terminology_audit_passes():
    audit = json.loads(
        (ROOT / 'reports/ngas_a17as_terminology_audit.json').read_text())
    assert audit['status'] == 'PASS'
    assert audit['current_production_encoder'] == 'compact_relational'
    assert audit['inaccurate_matches_remaining'] == []


def test_result_manifest_integrity():
    manifest = json.loads((OUT / 'result_manifest.json').read_text())
    decision = json.loads((OUT / 'final_decision.json').read_text())
    assert manifest['terminal_classification'] == (
        'NGAS_A1_7AS_PASS_SUPPLEMENTAL_CLOSURE')
    assert decision['terminal_classification'] == manifest['terminal_classification']
    assert manifest['artifact_count'] == len(manifest['files'])
    assert manifest['total_bytes'] == sum(
        item['bytes'] for item in manifest['files'].values())
    assert manifest['excluded_self_path'] == (
        'outputs/ngas_a1/trajectory_utility_a17as_v1/result_manifest.json')
    for relative_path, record in manifest['files'].items():
        path = ROOT / relative_path
        assert path.is_file()
        assert path.stat().st_size == record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
