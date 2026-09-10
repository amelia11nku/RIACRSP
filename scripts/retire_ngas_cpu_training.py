#!/usr/bin/env python3
"""Retire the superseded, incomplete CPU A1.3 execution without reading outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.train import digest
from rcias_ngas.evaluation.bks import write_immutable
from scripts.run_ngas_label_pilot import atomic_json

OUT = ROOT / 'outputs/ngas_a1/critic_training_v1'
RETIREMENT = OUT / 'retirement'


def main() -> None:
    progress_path = OUT / 'progress.json'
    launch_path = OUT / 'launch_record.json'
    protocol_path = OUT / 'training_protocol.json'
    progress = json.loads(progress_path.read_text())
    pid = int(progress['pid'])
    if Path(f'/proc/{pid}').exists():
        raise RuntimeError(f'CPU worker is still alive: {pid}')
    removal_roots = [OUT / 'oof', OUT / 'production']
    removal_files = [OUT / name for name in (
        'oof_gate.json', 'oof_predictions.json.gz', 'final_decision.json')]
    files = sorted(
        [path for root in removal_roots if root.exists() for path in root.rglob('*') if path.is_file()]
        + [path for path in removal_files if path.exists()])
    if not files:
        raise RuntimeError('No superseded CPU model artifacts found')
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in files}
    RETIREMENT.mkdir(parents=True, exist_ok=True)
    last_progress = RETIREMENT / 'last_progress_before_retirement.json'
    write_immutable(last_progress, progress)
    record = {
        'schema': 'ngas-a13-cpu-execution-retirement-v1',
        'status': 'SUPERSEDED_INCOMPLETE',
        'reason': 'CUDA availability confirmed; user authorized a fresh GPU protocol before any OOF outcome inspection',
        'retired_at_utc': datetime.now(timezone.utc).isoformat(),
        'cpu_worker_pid': pid,
        'cpu_worker_stopped': True,
        'last_progress_sha256': digest(last_progress),
        'last_progress': {key: progress.get(key) for key in (
            'status', 'completed_runs', 'expected_runs', 'seed', 'held_fold', 'epoch', 'updated_at_utc')},
        'old_protocol_sha256': digest(protocol_path),
        'old_launch_record_sha256': digest(launch_path),
        'outcome_metrics_inspected': False,
        'deleted_artifact_count': len(files),
        'deleted_artifact_hashes': manifest,
        'deleted_categories': ['OOF checkpoints', 'OOF predictions', 'run records',
                               'derived OOF gate/final decision if present'],
        'retained_categories': ['old frozen protocol', 'launch records', 'logs',
                                'training cache', 'retirement evidence'],
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    write_immutable(RETIREMENT / 'retirement_record.json', record)
    for root in removal_roots:
        if root.exists():
            shutil.rmtree(root)
    for path in removal_files:
        path.unlink(missing_ok=True)
    (OUT / '.training.lock').unlink(missing_ok=True)
    atomic_json(progress_path, {
        'status': 'SUPERSEDED_INCOMPLETE', 'pid': pid,
        'old_protocol_sha256': record['old_protocol_sha256'],
        'retirement_record': str((RETIREMENT / 'retirement_record.json').relative_to(ROOT)),
        'deleted_artifact_count': len(files), 'outcome_metrics_inspected': False,
        'updated_at_utc': record['retired_at_utc'],
        'r13': 'LOCKED', 'r14': 'LOCKED',
    })
    if any(path.exists() for path in removal_roots + removal_files):
        raise RuntimeError('Superseded CPU artifacts were not fully removed')
    print(json.dumps({'status': record['status'], 'deleted_artifacts': len(files),
                      'old_protocol_sha256': record['old_protocol_sha256'],
                      'outcome_metrics_inspected': False}, indent=2))


if __name__ == '__main__':
    main()
