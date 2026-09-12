#!/usr/bin/env python3
"""Freeze exact clean and R12 audit states before A1.7A-R full-bank outcomes."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a16_integrity import digest  # noqa: E402
from rcias_ngas.evaluation.a17ar import canonical_hash  # noqa: E402


TRAJECTORY_PROTOCOL = ROOT / 'artifacts/ngas_a17ar/trajectory_protocol_manifest.json'
COLLECTION_AUDIT = ROOT / (
    'outputs/ngas_a1/trajectory_utility_a17ar_v1/audit/'
    'collection_completion_audit.json')
CLEAN_RAW_MANIFEST = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/raw_manifest.json'
A16R_PROTOCOL = ROOT / 'outputs/ngas_a1/solver_comparison_a16r_v1/preregistration/protocol.json'
R12_STATE_PROTOCOL = ROOT / (
    'outputs/ngas_a1/solver_comparison_a16r_v1/diagnostics/offline_protocol.json')
OUTPUT = ROOT / 'artifacts/ngas_a17ar/full_bank_protocol_manifest.json'
REPORT = ROOT / 'reports/ngas_a17ar_full_bank_protocol.md'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean_states(trajectory: dict, manifest: dict) -> list[dict]:
    tasks = {row['instance_id']: row for row in trajectory['instances']}
    states = []
    for selected in trajectory['full_bank_audit_tasks']:
        task = tasks[selected['instance_id']]
        path = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/raw' \
            / task['dataset_role'].lower() / f"{task['instance_id']}.json"
        if digest(path) != manifest['files'].get(relative(path)):
            raise RuntimeError(f'clean raw hash mismatch: {path}')
        payload = json.loads(path.read_text())
        matches = [row for row in payload['states']
                   if row['target_capture_fraction'] == selected['capture_fraction']]
        if len(matches) != 1:
            raise RuntimeError(f'clean selected state is not unique: {task["instance_id"]}')
        state = matches[0]
        states.append({
            'state_key': f"CLEAN|{state['state_id']}",
            'dataset_origin': 'CLEAN_NON_R12_DEVELOPMENT',
            'dataset_role': task['dataset_role'], 'audit_only': False,
            'eligible_for_future_training': task['dataset_role'] == 'TRAIN',
            'instance_id': task['instance_id'],
            'instance_relative_path': task['relative_path'],
            'instance_sha256': task['content_sha256'],
            'scale': task['scale'], 'CF_level': task['CF_level'],
            'seed': task['trajectory_seed'],
            'capture_fraction': state['target_capture_fraction'],
            'observed_budget_fraction': state['normalized_budget_fraction'],
            'search_iteration': state['search_iteration'],
            'formal_raw_path': relative(path), 'formal_raw_sha256': digest(path),
            'state_sha256': canonical_hash(state),
        })
    return states


def r12_states(a16r: dict, source: dict) -> list[dict]:
    config = json.loads((ROOT / a16r['config_path']).read_text())
    root = ROOT / config['scope']['instance_root']
    states = []
    for state in source['states']:
        formal = ROOT / state['formal_raw_path']
        if digest(formal) != state['formal_raw_sha256']:
            raise RuntimeError(f'R12 formal raw hash mismatch: {formal}')
        instance = root / state['instance_relative_path']
        if digest(instance) != state['instance_sha256']:
            raise RuntimeError(f'R12 instance hash mismatch: {instance}')
        states.append({
            'state_key': f"R12|{state['state_key']}",
            'dataset_origin': 'R12_DEVELOPMENT_EXPOSED',
            'dataset_role': 'DEVELOPMENT_EXPOSED', 'audit_only': True,
            'eligible_for_future_training': False,
            'instance_id': state['instance_id'],
            'instance_relative_path': relative(instance),
            'instance_sha256': state['instance_sha256'],
            'scale': state['scale'], 'CF_level': state['CF_level'],
            'seed': state['seed'],
            'capture_fraction': state['capture_fraction'],
            'observed_budget_fraction': state['observed_budget_fraction'],
            'search_iteration': None,
            'formal_raw_path': state['formal_raw_path'],
            'formal_raw_sha256': state['formal_raw_sha256'],
            'state_sha256': state['snapshot_sha256'],
        })
    return states


def build_manifest() -> dict:
    trajectory = json.loads(TRAJECTORY_PROTOCOL.read_text())
    collection = json.loads(COLLECTION_AUDIT.read_text())
    clean_manifest = json.loads(CLEAN_RAW_MANIFEST.read_text())
    a16r = json.loads(A16R_PROTOCOL.read_text())
    r12_source = json.loads(R12_STATE_PROTOCOL.read_text())
    if collection.get('status') != 'PASS':
        raise RuntimeError('clean trajectory collection has not passed completion audit')
    states = clean_states(trajectory, clean_manifest) + r12_states(a16r, r12_source)
    sources = (
        'scripts/run_ngas_a17ar_full_bank_audit.py',
        'scripts/freeze_ngas_a17ar_full_bank_protocol.py',
        'rcias_ngas/evaluation/a17ar.py',
        'rcias_ngas/search/ngas_solver.py',
        'rcias_ngas/search/online_portfolio.py',
        'rcias_ngas/runtime/production_refresh.py',
        'rcias_ngas/runtime/compact_state.py',
        'rcias_ngas/actions/repair.py',
        'rcias_ngas/bank/ngas_bank_v1.py',
        'rcias_clgri/search/common.py',
        'rcias_clgri/env/insertion_decoder.py',
    )
    clean = [row for row in states
             if row['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT']
    r12 = [row for row in states if row['dataset_origin'] == 'R12_DEVELOPMENT_EXPOSED']
    checks = {
        'clean_collection_passed': collection['status'] == 'PASS',
        'exact_18_clean_states': len(clean) == 18,
        'clean_role_scale_cf_balance': Counter(
            (row['dataset_role'], row['scale'], row['CF_level']) for row in clean)
            == Counter({(role, scale, cf): 1 for role in ('TRAIN', 'VALIDATION')
                        for scale in ('S', 'M', 'L') for cf in ('CF1', 'CF2', 'CF3')}),
        'exact_18_R12_audit_only_states': (
            len(r12) == 18 and all(row['audit_only'] is True
                                   and row['eligible_for_future_training'] is False
                                   for row in r12)),
        'R12_all_scales_and_three_stages': (
            Counter(row['scale'] for row in r12) == Counter({'S': 6, 'M': 6, 'L': 6})
            and Counter(row['capture_fraction'] for row in r12)
            == Counter({.15: 6, .5: 6, .85: 6})),
        'origin_instance_ids_and_hashes_disjoint': (
            not {row['instance_id'] for row in clean} & {row['instance_id'] for row in r12}
            and not {row['instance_sha256'] for row in clean}
            & {row['instance_sha256'] for row in r12}),
        'all_state_keys_unique': len({row['state_key'] for row in states}) == 36,
        'no_formal_output_before_freeze': not any((ROOT /
            'outputs/ngas_a1/trajectory_utility_a17ar_v1/diagnostics/full_bank_raw')
            .glob('*.json')),
    }
    if not all(checks.values()):
        raise RuntimeError({'invalid_full_bank_protocol': checks})
    return {
        'schema': 'ngas-a17ar-full-bank-protocol-v1',
        'status': 'FROZEN_BEFORE_FULL_BANK_RESULTS',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'freeze_source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'trajectory_protocol_sha256': digest(TRAJECTORY_PROTOCOL),
        'collection_audit_path': relative(COLLECTION_AUDIT),
        'collection_audit_sha256': digest(COLLECTION_AUDIT),
        'clean_raw_manifest_sha256': digest(CLEAN_RAW_MANIFEST),
        'R12_state_protocol_path': relative(R12_STATE_PROTOCOL),
        'R12_state_protocol_sha256': digest(R12_STATE_PROTOCOL),
        'checkpoint_path': trajectory['checkpoint_path'],
        'checkpoint_sha256': trajectory['checkpoint_sha256'],
        'candidate_bank_hash': trajectory['candidate_bank_hash'],
        'states': states, 'state_count': len(states),
        'counterfactual_scope': 'FULL_UNIQUE_PRODUCTION_JOINT_BANK',
        'matched_trials_per_action': 8,
        'matched_random_numbers': (
            'within one state and trial index, every action resets the same frozen '
            'production neighbor RNG stream; action identity is excluded from that stream'),
        'utility_definitions': {
            'U0_IMMEDIATE': (
                'positive makespan reduction of the best of eight direct repair/decode trials '
                'relative to the captured current candidate'),
            'U1_SHORT_HORIZON': (
                'positive best makespan reduction relative to the captured current candidate '
                'after forcing the audited best-of-eight action and then executing two more '
                'frozen production iterations'),
            'U2_COST_NORMALIZED': (
                'U1 divided by measured repair-plus-decoder seconds; decoder-normalized '
                'U0/8 and U1/24 are retained as secondary measures'),
            'U3_STOCHASTIC_ROBUSTNESS': (
                'mean signed direct-trial improvement used for ranking, plus improvement '
                'probability, variance, and downside probability across eight trials'),
        },
        'continuation_semantics': {
            'steps_after_forced_action': 2,
            'persistent_prior': True,
            'refresh_inside_horizon': False,
            'reason': 'two steps are strictly inside fixed refresh interval 20',
            'online_portfolio': (
                'reconstructed from every archived pre-state production outcome, then updated '
                'with the forced action and each continuation outcome'),
            'selection_exploration_acceptance': 'unchanged frozen production functions and RNGs',
            'trials_each_continuation_step': 8,
        },
        'ranking_metrics': [
            'top1_realized_rank', 'best_action_hit_and_recall_at_1_5_10',
            'normalized_rank', 'regret', 'ndcg_full', 'spearman',
            'top_k_utility_gap', 'utility_pairwise_agreement',
        ],
        'breakdowns': [
            'dataset_origin', 'scale', 'search_stage', 'search_condition',
            'neighborhood_size', 'candidate_rule_family', 'repair_strategy'],
        'source_hashes': {path: digest(ROOT / path) for path in sources},
        'checks': checks,
        'boundaries': {
            'production_solver_changed': False, 'train_c1_v2': False,
            'adaptive_refresh': False, 'adaptive_trial_racing': False,
            'formal_solver_budget_inclusion': False,
            'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
            'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
            'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_ACCESS',
            'gurobi_run': False,
        },
    }


def main() -> None:
    if subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('freeze full-bank protocol only from a clean committed worktree')
    if OUTPUT.exists():
        raise RuntimeError('full-bank protocol already exists and is immutable')
    payload = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    REPORT.write_text(f'''# NGAS A1.7A-R full-bank utility protocol

This protocol was frozen before full-bank U0-U3 outcomes. It selects exactly
**18 clean non-R12 states** and **18 R12 development-exposed audit-only states**.
Every state evaluates the true frozen production joint-action bank with eight
matched stochastic repair/decode trials per action.

U1 forces the audited best-of-eight action, reconstructs the archived online
portfolio state, and continues exactly two production iterations. Because the
horizon is shorter than the fixed refresh interval of 20, the captured critic
prior and action bank remain persistent and no refresh is introduced.

- Checkpoint: `{payload['checkpoint_sha256']}`
- Candidate bank: `{payload['candidate_bank_hash']}`
- Freeze source commit: `{payload['freeze_source_commit']}`
- R13/R14: locked; CORE45: no access; Gurobi: false.
''')
    print(json.dumps({
        'status': payload['status'], 'states': payload['state_count'],
        'path': relative(OUTPUT), 'checks': payload['checks'],
    }, indent=2))


if __name__ == '__main__':
    main()
