#!/usr/bin/env python3
"""Freeze the A1.6 source, artifact, metric, and decision boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a16_integrity import (  # noqa: E402
    audit_frozen_inputs, digest, load_json,
)


OUT = ROOT / 'outputs/ngas_a1/solver_comparison_v1'
CONFIG = ROOT / 'configs/ngas_a16_solver_comparison_v1.json'
PROTOCOL = OUT / 'preregistration/protocol.json'
SOURCE_HASHES = OUT / 'preregistration/source_hashes.json'
PREFORMAL = OUT / 'preregistration/preformal_audit.json'
COMPARATOR_AUDIT = OUT / 'preregistration/comparator_audit.json'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main() -> None:
    if PROTOCOL.exists():
        raise RuntimeError('A1.6 protocol already exists and is immutable')
    if (subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT).returncode != 0
            or subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode != 0):
        raise RuntimeError('Freeze A1.6 only after committing the tracked implementation')
    config = load_json(CONFIG)
    preformal = load_json(PREFORMAL)
    frozen = audit_frozen_inputs(ROOT, config)
    if preformal.get('status') != 'PASS' or frozen.get('status') != 'PASS' \
            or load_json(COMPARATOR_AUDIT) != frozen:
        raise RuntimeError('A1.6 preformal or frozen-input audit is not valid')
    current_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if preformal.get('implementation_commit') != current_commit:
        raise RuntimeError('Tracked implementation changed after A1.6 preformal audit')
    formal_raw = list((OUT / 'raw').glob('*/*.json')) if (OUT / 'raw').exists() else []
    if formal_raw:
        raise RuntimeError('Formal A1.6 raw exists before protocol freeze')
    source_paths = [
        'scripts/run_ngas_a16_solver_comparison.py',
        'scripts/run_ngas_a16_worker.py',
        'scripts/launch_ngas_a16_solver_comparison.py',
        'scripts/smoke_ngas_a16.py',
        'scripts/audit_ngas_a16_preformal.py',
        'scripts/freeze_ngas_a16_solver_comparison.py',
        'scripts/finalize_ngas_a16_solver_comparison.py',
        'rcias_ngas/evaluation/a16.py',
        'rcias_ngas/evaluation/a16_io.py',
        'rcias_ngas/evaluation/a16_integrity.py',
        'rcias_ngas/search/ngas_solver.py',
        'rcias_ngas/search/a14_telemetry.py',
        'rcias_ngas/search/online_portfolio.py',
        'rcias_ngas/search/persistent_prior.py',
        'rcias_ngas/runtime/production_refresh.py',
        'rcias_ngas/runtime/compact_state.py',
        'rcias_ngas/critic/inference.py',
        'rcias_ngas/critic/revised_critic.py',
        'rcias_ngas/critic/encoders/rt_hgt.py',
        'rcias_ngas/csg/revised_features.py',
        'rcias_ngas/bank/ngas_bank_v1.py',
        'rcias_ngas/actions/repair.py',
        'rcias_clgri/search/common.py',
        'rcias_clgri/env/insertion_decoder.py',
        'rcias_clgri/env/feasibility.py',
        'rcias_clgri/heuristic/dispatching.py',
        'tests/ngas/test_a16_solver_comparison.py',
    ]
    hashes = {path: digest(ROOT / path) for path in source_paths}
    atomic_json(SOURCE_HASHES, {
        'schema': 'ngas-a16-source-hashes-v1',
        'implementation_commit': current_commit,
        'files': hashes,
    })
    artifact_paths = [
        'configs/ngas_a16_solver_comparison_v1.json',
        'configs/ngas_a1_search_integration_v1.json',
        'outputs/ngas_a1/runtime_revision_v1/audit/completion_audit.json',
        'outputs/ngas_a1/runtime_revision_v1/result_manifest.json',
        'outputs/ngas_a1/critic_training_rthgt_v2/production/revised_joint_critic.pt',
        'outputs/frozen_2o_baselines/instance_manifest.json',
        'outputs/frozen_2o_baselines/registry.json',
        'outputs/frozen_2o_baselines/alns/manifest.json',
        'outputs/frozen_2o_baselines/alns/summary.json',
        'outputs/frozen_2o_baselines/phase6h/manifest.json',
        'outputs/frozen_2o_baselines/phase6h/summary.json',
        'outputs/frozen_2o_baselines/phase6n/manifest.json',
        'outputs/frozen_2o_baselines/phase6n/summary.json',
        'outputs/frozen_2o_baselines/lg_hga_2o/manifest.json',
        'outputs/frozen_2o_baselines/lg_hga_2o/summary.json',
        'outputs/ngas_a1/provenance/bks_manifest_v001.json',
        'outputs/ngas_a1/solver_comparison_v1/preregistration/source_hashes.json',
        'outputs/ngas_a1/solver_comparison_v1/preregistration/comparator_audit.json',
        'outputs/ngas_a1/solver_comparison_v1/preregistration/preformal_audit.json',
        'outputs/ngas_a1/solver_comparison_v1/preregistration/regression.log',
        'outputs/ngas_a1/solver_comparison_v1/smoke/smoke.json',
    ]
    manifest = load_json(ROOT / config['scope']['instance_manifest_path'])
    nominal = sum(2. * int(row['num_operations']) for row in manifest['instances']) \
        * len(config['scope']['seeds'])
    protocol = {
        'schema': 'ngas-a16-formal-protocol-v1',
        'status': 'FROZEN_BEFORE_FORMAL_RESULTS',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'starting_a15r_commit': config['starting_commit'],
        'implementation_commit': current_commit,
        'a15r_completion_audit_path': preformal['a15r_completion_audit_path'],
        'a15r_completion_audit_sha256': preformal['a15r_completion_audit_sha256'],
        'a15r_result_manifest_path': preformal['a15r_result_manifest_path'],
        'a15r_result_manifest_sha256': preformal['a15r_result_manifest_sha256'],
        'config_path': str(CONFIG.relative_to(ROOT)),
        'config_sha256': digest(CONFIG),
        'source_hashes_path': str(SOURCE_HASHES.relative_to(ROOT)),
        'source_hashes_sha256': digest(SOURCE_HASHES),
        'source_hashes': hashes,
        'artifact_hashes': {path: digest(ROOT / path) for path in artifact_paths},
        'preformal_audit_path': str(PREFORMAL.relative_to(ROOT)),
        'preformal_audit_sha256': digest(PREFORMAL),
        'comparator_audit_path': str(COMPARATOR_AUDIT.relative_to(ROOT)),
        'comparator_audit_sha256': digest(COMPARATOR_AUDIT),
        'registry_sha256': frozen['registry_sha256'],
        'bks_sha256': frozen['bks_sha256'],
        'checkpoint_sha256': frozen['checkpoint_sha256'],
        'instance_manifest_sha256': frozen['instance_manifest_sha256'],
        'instances': manifest['instances'],
        'seeds': config['scope']['seeds'],
        'run_order': config['scope']['execution_order'],
        'formal_run_count': 54,
        'execution_concurrency': 1,
        'nominal_total_budget_seconds': nominal,
        'nominal_total_budget_hours': nominal / 3600.,
        'budget': config['budget'],
        'metrics': config['metrics'],
        'statistics': config['statistics'],
        'decision_rules': config['decision_rules'],
        'comparators': config['comparators'],
        'resume_semantics': 'skip only an existing complete raw after protocol, metadata, finite-value, schedule-replay, candidate-hash, and atomic-budget validation; never overwrite or rerun it',
        'formal_command': [
            '/home/liulei/miniconda3/envs/gnn311/bin/python', '-u',
            'scripts/run_ngas_a16_solver_comparison.py', '--device', 'cuda:0'],
        'finalizer_command': [
            '/home/liulei/miniconda3/envs/gnn311/bin/python',
            'scripts/finalize_ngas_a16_solver_comparison.py'],
        'machine': preformal['machine'],
        'locks': config['locks'],
    }
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        'status': protocol['status'], 'formal_run_count': 54,
        'nominal_total_budget_hours': protocol['nominal_total_budget_hours'],
        'implementation_commit': current_commit,
        'path': str(PROTOCOL.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
