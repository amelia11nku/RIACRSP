#!/usr/bin/env python3
"""Freeze the A1.7A-R fixed-refresh prior-staleness diagnostic protocol."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.governance.dataset_roles import sha256_file  # noqa: E402


FULL_BANK_PROTOCOL = ROOT / 'artifacts/ngas_a17ar/full_bank_protocol_manifest.json'
FULL_BANK_AUDIT = ROOT / (
    'outputs/ngas_a1/trajectory_utility_a17ar_v1/audit/'
    'full_bank_completion_audit.json')
FULL_BANK_RAW_MANIFEST = ROOT / (
    'outputs/ngas_a1/trajectory_utility_a17ar_v1/diagnostics/'
    'full_bank_raw_manifest.json')
OUTPUT = ROOT / 'artifacts/ngas_a17ar/prior_staleness_protocol_manifest.json'
REJECTED_V1 = ROOT / (
    'artifacts/ngas_a17ar/prior_staleness_protocol_manifest_rejected_v1.json')
REPORT = ROOT / 'reports/ngas_a17ar_prior_staleness_protocol.md'
RAW = ROOT / (
    'outputs/ngas_a1/trajectory_utility_a17ar_v1/diagnostics/'
    'prior_staleness_raw')


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError(
            'freeze prior-staleness protocol only from a clean committed worktree')
    if (not OUTPUT.exists() or not REJECTED_V1.exists()
            or OUTPUT.read_bytes() != REJECTED_V1.read_bytes()):
        raise RuntimeError(
            'revision 2 requires the exact preserved rejected-v1 protocol')
    if any(RAW.glob('*.json')):
        raise RuntimeError('formal prior-staleness outputs exist before protocol freeze')

    full_protocol = json.loads(FULL_BANK_PROTOCOL.read_text())
    completion = json.loads(FULL_BANK_AUDIT.read_text())
    manifest = json.loads(FULL_BANK_RAW_MANIFEST.read_text())
    if completion.get('status') != 'PASS' or not all(completion['checks'].values()):
        raise RuntimeError('full-bank completion audit has not passed')
    raw_by_state = {}
    for raw_path, expected_sha in manifest['files'].items():
        path = ROOT / raw_path
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f'full-bank raw hash mismatch: {raw_path}')
        payload = json.loads(path.read_text())
        raw_by_state[payload['state_key']] = (raw_path, expected_sha)
    if set(raw_by_state) != {row['state_key'] for row in full_protocol['states']}:
        raise RuntimeError('full-bank raw states do not match the frozen state protocol')

    states = [{
        **row,
        'full_bank_raw_path': raw_by_state[row['state_key']][0],
        'full_bank_raw_sha256': raw_by_state[row['state_key']][1],
    } for row in full_protocol['states']]
    origins = Counter(row['dataset_origin'] for row in states)
    checks = {
        'full_bank_completion_passed': completion['status'] == 'PASS',
        'exact_36_base_states': len(states) == 36,
        'mandatory_origin_split_exact': origins == Counter({
            'CLEAN_NON_R12_DEVELOPMENT': 18,
            'R12_DEVELOPMENT_EXPOSED': 18,
        }),
        'R12_rows_remain_audit_only': all(
            row['audit_only'] is True and row['eligible_for_future_training'] is False
            for row in states
            if row['dataset_origin'] == 'R12_DEVELOPMENT_EXPOSED'),
        'full_bank_raw_identity_complete': len(raw_by_state) == 36,
        'no_formal_output_before_freeze': not any(RAW.glob('*.json')),
    }
    if not all(checks.values()):
        raise RuntimeError({'invalid_prior_staleness_protocol': checks})
    sources = (
        'scripts/run_ngas_a17ar_prior_staleness.py',
        'scripts/freeze_ngas_a17ar_prior_staleness.py',
        'rcias_ngas/evaluation/a17ar.py',
        'rcias_ngas/runtime/production_refresh.py',
        'rcias_ngas/runtime/compact_state.py',
        'rcias_ngas/actions/repair.py',
        'rcias_ngas/bank/ngas_bank_v1.py',
        'rcias_ngas/rng.py',
        'rcias_clgri/search/common.py',
        'rcias_clgri/env/insertion_decoder.py',
    )
    payload = {
        'schema': 'ngas-a17ar-prior-staleness-protocol-v1',
        'revision': 2,
        'status': 'FROZEN_BEFORE_STALENESS_RESULTS',
        'supersedes_rejected_protocol_path': relative(REJECTED_V1),
        'supersedes_rejected_protocol_sha256': sha256_file(REJECTED_V1),
        'revision_reason': (
            'the partial smoke slice now includes every offset-specific archived '
            'selected action and the stale top-1; formal full-bank behavior is unchanged'),
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'freeze_source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'full_bank_protocol_path': relative(FULL_BANK_PROTOCOL),
        'full_bank_protocol_sha256': sha256_file(FULL_BANK_PROTOCOL),
        'full_bank_completion_audit_path': relative(FULL_BANK_AUDIT),
        'full_bank_completion_audit_sha256': sha256_file(FULL_BANK_AUDIT),
        'full_bank_raw_manifest_path': relative(FULL_BANK_RAW_MANIFEST),
        'full_bank_raw_manifest_sha256': sha256_file(FULL_BANK_RAW_MANIFEST),
        'checkpoint_path': full_protocol['checkpoint_path'],
        'checkpoint_sha256': full_protocol['checkpoint_sha256'],
        'candidate_bank_hash': full_protocol['candidate_bank_hash'],
        'refresh_interval': 20,
        'offsets': [0, 5, 10, 15, 19],
        'states': states,
        'state_count': len(states),
        'stale_bank_scope': 'FULL_UNIQUE_BASE_REFRESH_JOINT_BANK',
        'stale_utility_scope': (
            'eight matched repair/decode trials for every persistent base-bank action '
            'at each nonzero offset; offset zero reuses the hash-frozen full-bank U0 trials'),
        'fresh_reference_scope': (
            'hypothetical fixed production refresh at the same current state; semantic '
            'bank/rank/score overlap plus eight matched trials for fresh top-1 only'),
        'trajectory_reconstruction': (
            'replay archived selected actions and all eight stored repair seeds; apply only '
            'archived accepted moves and verify every current-before/current-after objective'),
        'matched_trials_per_action': 8,
        'matched_random_numbers': (
            'within state and offset, every audited action uses the same trial-indexed '
            'neighbor stream; action identity is excluded'),
        'representation_drift': {
            'operation_features': 'relative L2 and mean absolute compact input drift',
            'graph_structure': 'typed directed-edge Jaccard',
            'critical_structure': 'critical-signature and dominant-bottleneck changes',
        },
        'metrics': [
            'stale_U0_spearman', 'stale_top1_realized_rank', 'stale_top1_regret',
            'stale_fresh_top1_top5_top10_overlap', 'semantic_bank_jaccard',
            'prior_advantage_and_percentile_rank_drift', 'representation_drift',
            'stale_fresh_and_archived_selected_action_U0',
        ],
        'statistical_unit': {
            'primary_diagnostic': 'state',
            'uncertainty': 'instance-clustered; multiple R12 states are not independent',
            'solver_generalization_claim': False,
        },
        'source_hashes': {path: sha256_file(ROOT / path) for path in sources},
        'checks': checks,
        'boundaries': full_protocol['boundaries'],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    REPORT.write_text(f'''# NGAS A1.7A-R prior-staleness protocol

Revision 2 freezes the fixed-refresh staleness audit before outcomes. It uses
all 36 already governed full-bank states: 18 clean non-R12 development states and
18 R12 development-exposed audit-only states. At offsets 0, 5, 10, 15, and 19,
the persistent base bank is scored against full U0 counterfactual evidence. A
hypothetical fresh production refresh supplies semantic rank/score drift and a
matched fresh-top-1 reference; it does not change the archived solver trajectory.

- Refresh interval: 20 iterations.
- Trials: eight matched repair/decode realizations per persistent action.
- Checkpoint: `{payload['checkpoint_sha256']}`.
- Freeze source commit: `{payload['freeze_source_commit']}`.
- R13/R14 remain locked; CORE45 is external-only; Gurobi is not run.
- Adaptive refresh and C1-v2 training remain disabled.
- Rejected v1 is preserved at `{relative(REJECTED_V1)}`; the formal full-bank
  execution path is unchanged.
''')
    print(json.dumps({
        'status': payload['status'], 'states': len(states),
        'offsets': payload['offsets'], 'checks': checks,
        'path': relative(OUTPUT),
    }, indent=2))


if __name__ == '__main__':
    main()
