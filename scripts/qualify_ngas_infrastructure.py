#!/usr/bin/env python3
"""Gate A1.0/A1.1 on full regression, frozen hashes and bank validation."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_ngas.evaluation.bks import write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.run_ngas_label_pilot import verify_boundary


def main():
    boundary = verify_boundary()
    out = ROOT / 'outputs/ngas_a1/audit'
    gate_path = out / 'infrastructure_bank_gate.json'
    if gate_path.exists():
        raise FileExistsError('Infrastructure qualification is already frozen')
    xml_path = out / 'regression.xml'
    with (out / 'regression.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(xml_path)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    suite = ET.parse(xml_path).getroot().find('testsuite')
    if int(suite.get('failures')) or int(suite.get('errors')):
        raise ValueError('Full regression failed')
    bank = json.loads((out / 'bank_validation.json').read_text())
    assert bank['status'] == 'PASS' and bank['distinct_cardinalities_all_18']
    phase6f = 'outputs/phase6f/audit/experiment_freeze.json'
    freeze = json.loads((ROOT / phase6f).read_text())
    checkpoint = freeze['selected_checkpoint_path']
    expected = freeze['selected_checkpoint_sha256']
    assert digest(checkpoint) == expected
    policy = json.loads((ROOT / 'outputs/phase6h_calibration/frozen/phase6h_policy.json').read_text())
    assert policy['checkpoint_sha256'] == expected
    supplemental = {phase6f: digest(phase6f), checkpoint: expected}
    # Reproduce the frozen historical per-run RPD table from the materialized v001.
    derived = json.loads((ROOT / 'outputs/ngas_a1/provenance/derived_metric_manifest_v001.json').read_text())
    old_path = ROOT / 'outputs/phase6p_adaptive_portfolio_v1/development/audited_run_summary.csv'
    old = list(csv.DictReader(old_path.open()))
    aliases = {'PHASE6P': 'P1_CSG_ADAPTIVE_PORTFOLIO', 'PHASE6N_TOP1': 'PHASE6N_DETERMINISTIC_TOP1'}
    lookup = {(r['method'], r['instance_id'], int(r['seed'])): float(r['rpd_percent']) for r in old}
    max_difference = max(abs(r['rpd_percent'] - lookup[aliases.get(r['algorithm_id'], r['algorithm_id']), r['instance_id'], r['seed']]) for r in derived['rows'])
    assert max_difference < 1e-12
    verify_boundary()
    gate = {
        'schema': 'ngas-infrastructure-bank-gate-v1', 'status': 'PASS',
        'A1.0': 'PASS', 'A1.1': 'PASS',
        'starting_commit': boundary['starting_commit'],
        'full_regression_tests': int(suite.get('tests')), 'failures': 0, 'errors': 0,
        'regression_seconds': float(suite.get('time')),
        'regression_xml_sha256': digest(xml_path),
        'protected_files_unchanged': len(boundary['protected_hashes']),
        'supplemental_checkpoint_hashes': supplemental,
        'historical_rpd_rows_reproduced': len(derived['rows']),
        'historical_rpd_max_absolute_difference': max_difference,
        'bank_validation_sha256': digest(out / 'bank_validation.json'),
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    write_immutable(gate_path, gate)
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
