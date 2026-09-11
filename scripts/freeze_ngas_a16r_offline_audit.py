#!/usr/bin/env python3
"""Freeze replayable A1.6R states before counterfactual evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a16_io import write_new_json  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import digest  # noqa: E402
from scripts.run_ngas_a16r_solver_comparison import (  # noqa: E402
    OUT, build_tasks, load_boundary, raw_path, validate_existing_raw,
)


OFFLINE_PROTOCOL = OUT / 'diagnostics/offline_protocol.json'


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    if OFFLINE_PROTOCOL.exists():
        raise RuntimeError('A1.6R offline protocol already exists and is immutable')
    protocol, config, protocol_sha256 = load_boundary(require_clean=True)
    if (OUT / 'integrity/formal.lock').exists():
        raise RuntimeError('formal A1.6R worker is still active')
    tasks = build_tasks(config)
    capture_tasks = {
        (row['instance_id'], int(row['seed']))
        for row in config['diagnostics']['offline_state_capture']['tasks']
    }
    states = []
    for task in tasks:
        path = raw_path(task)
        if not path.exists():
            raise RuntimeError('A1.6R formal raw is incomplete')
        payload = validate_existing_raw(path, task, protocol_sha256, config)
        snapshots = payload['search_diagnostics']['a16r_replayable_states']
        expected = (task['instance_id'], task['seed']) in capture_tasks
        if expected and len(snapshots) != 3:
            raise RuntimeError(f"offline state capture incomplete: {task['instance_id']}")
        if not expected and snapshots:
            raise RuntimeError(f"unexpected offline states: {task['instance_id']}")
        for snapshot in snapshots:
            states.append({
                'state_key': (
                    f"{task['instance_id']}|{task['seed']}|"
                    f"{snapshot['capture_fraction']:.2f}"),
                'instance_id': task['instance_id'],
                'instance_relative_path': task['instance_relative_path'],
                'instance_sha256': task['instance_sha256'],
                'scale': task['scale'], 'CF_level': task['CF_level'],
                'cell_replicate': task['cell_replicate'], 'seed': task['seed'],
                'capture_fraction': snapshot['capture_fraction'],
                'observed_budget_fraction': snapshot['observed_budget_fraction'],
                'formal_raw_path': str(path.relative_to(ROOT)),
                'formal_raw_sha256': digest(path),
                'snapshot_sha256': canonical_hash(snapshot),
            })
    expected_fractions = config['diagnostics']['offline_state_capture']['fractions']
    checks = {
        'exact_state_count': len(states) == 18,
        'all_scales': {row['scale'] for row in states} == {'S', 'M', 'L'},
        'two_cf_levels_per_scale': all(
            len({row['CF_level'] for row in states if row['scale'] == scale}) >= 2
            for scale in ('S', 'M', 'L')),
        'three_search_stages_per_task': all(
            sorted(row['capture_fraction'] for row in states
                   if (row['instance_id'], row['seed']) == key)
            == expected_fractions for key in capture_tasks),
        'no_counterfactual_output_before_freeze': not any(
            (OUT / 'diagnostics/offline_raw').glob('*.json')),
    }
    if not all(checks.values()):
        raise RuntimeError({'invalid_offline_state_freeze': checks})
    source_paths = [
        'scripts/run_ngas_a16r_offline_audit.py',
        'scripts/freeze_ngas_a16r_offline_audit.py',
        'rcias_ngas/search/ngas_solver.py',
        'rcias_ngas/runtime/production_refresh.py',
        'rcias_ngas/runtime/compact_state.py',
        'rcias_ngas/actions/repair.py',
        'rcias_clgri/search/common.py',
        'rcias_clgri/env/insertion_decoder.py',
    ]
    payload = {
        'schema': 'ngas-a16r-offline-counterfactual-protocol-v1',
        'status': 'FROZEN_BEFORE_COUNTERFACTUAL_RESULTS',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'formal_protocol_sha256': protocol_sha256,
        'formal_implementation_commit': protocol['implementation_commit'],
        'freeze_commit_parent': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config_sha256': digest(ROOT / protocol['config_path']),
        'selection_rule': config['diagnostics']['offline_state_capture']['selection_rule'],
        'states': states,
        'state_count': len(states),
        'counterfactual_scope': 'FULL_UNIQUE_PRODUCTION_JOINT_BANK',
        'counterfactual_trials_per_action': 1,
        'matched_random_numbers': (
            'for a state and trial, every action receives a fresh Random object '
            'with the same production neighbor RNG seed'),
        'formal_budget_inclusion': False,
        'ranking_metrics': [
            'critic_top1_realized_rank', 'top1_top5_top10_recall_of_best',
            'ndcg_full', 'spearman_advantage_realized_gain',
            'critic_top1_regret', 'scale_stage_size_repair_breakdowns',
        ],
        'source_hashes': {path: digest(ROOT / path) for path in source_paths},
        'checks': checks,
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    write_new_json(OFFLINE_PROTOCOL, payload)
    print(json.dumps({'status': payload['status'], 'states': len(states),
                      'path': str(OFFLINE_PROTOCOL.relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
