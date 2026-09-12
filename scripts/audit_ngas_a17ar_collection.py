#!/usr/bin/env python3
"""Close the frozen A1.7A-R clean-trajectory collection with integrity evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    DatasetRegistry, sha256_file, verify_exposure_ledger,
)


OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
PROTOCOL = ROOT / 'artifacts/ngas_a17ar/trajectory_protocol_manifest.json'
CONFIG = ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml'
REGISTRY = ROOT / 'configs/dataset_role_registry.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
REPORT = ROOT / 'reports/ngas_a17ar_trajectory_coverage.md'
COVERAGE_CSV = ROOT / 'reports/ngas_a17ar_trajectory_coverage.csv'
FORMAL_OWNER = 'NGAS_A1_7AR_TRAJECTORY_OWNER_V1'
COLLECTION_PHASE = 'NGAS_A1_7A_R_STAGE7'
COLLECTOR_COMMAND = 'scripts/run_ngas_a17ar_trajectory_collection.py'
STAGES = ('0-20%', '20-40%', '40-60%', '60-80%', '80-100%')
CONDITIONS = (
    'improving',
    'plateau_or_stagnating',
    'immediately_post_improvement',
    'bottleneck_or_critical_transition',
    'high_action_entropy',
    'low_action_entropy',
)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def progress_stage(fraction: float) -> str:
    index = min(int(fraction * 5.), 4)
    return STAGES[index]


def validate_state(state: dict, task: dict, protocol: dict) -> list[str]:
    """Return deterministic contract violations for one archived state."""
    errors = []
    expected_scalars = {
        'schema': 'ngas-a17ar-clean-trajectory-state-v1',
        'dataset_role': task['dataset_role'],
        'audit_only': False,
        'eligible_for_future_training': task['dataset_role'] == 'TRAIN',
        'instance_id': task['instance_id'],
        'instance_content_sha256': task['content_sha256'],
        'trajectory_seed': task['trajectory_seed'],
        'candidate_bank_identifier': 'NGAS_BANK_V1_FULL_UNIQUE',
        'candidate_bank_hash': protocol['candidate_bank_hash'],
        'critic_checkpoint_hash': protocol['checkpoint_sha256'],
    }
    for key, expected in expected_scalars.items():
        if state.get(key) != expected:
            errors.append(f'{key}_mismatch')

    features = state.get('compact_relational_features', {})
    if features.get('schema') != 'ngas-compact-relational-state-features-v1':
        errors.append('compact_feature_schema_mismatch')
    feature_body = dict(features)
    claimed_feature_hash = feature_body.pop('state_feature_hash', None)
    if claimed_feature_hash != canonical_hash(feature_body):
        errors.append('state_feature_hash_mismatch')
    graph_identity = {name: features.get(name) for name in (
        'node_types', 'edge_index', 'edge_types', 'operation_nodes')}
    if features.get('graph_hash') != canonical_hash(graph_identity):
        errors.append('graph_hash_mismatch')

    actions = state.get('candidate_action_ids', [])
    if not actions or len(actions) != len(set(actions)):
        errors.append('candidate_bank_empty_or_duplicate')
    critic = state.get('critic_scores', {})
    numerical = []
    for name in ('advantage', 'beats_fallback_probability', 'neural_prior'):
        values = critic.get(name, [])
        if len(values) != len(actions):
            errors.append(f'{name}_shape_mismatch')
        numerical.extend(values)
    entropy = critic.get('neural_prior_entropy')
    if entropy is None:
        errors.append('prior_entropy_missing')
    else:
        numerical.append(entropy)
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in numerical):
        errors.append('critic_non_finite')
    prior = critic.get('neural_prior', [])
    if prior and not math.isclose(math.fsum(prior), 1., rel_tol=1e-6, abs_tol=1e-6):
        errors.append('prior_not_normalized')

    selected = state.get('selected_joint_action', {})
    if selected.get('action_id') not in set(actions):
        errors.append('selected_action_outside_full_bank')
    portfolio = state.get('portfolio_adjusted_scores', {})
    if portfolio.get('combined_top1_action_id') not in set(actions):
        errors.append('portfolio_top1_outside_full_bank')
    trials = state.get('stochastic_trial_outcomes', [])
    if len(trials) != 8 or [row.get('trial') for row in trials] != list(range(1, 9)):
        errors.append('selected_action_trial_count_not_eight')
    for row in trials:
        values = [row.get(name) for name in (
            'candidate_makespan', 'decoder_seconds', 'repair_seconds', 'trial_seconds')]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in values):
            errors.append('trial_non_finite')
            break

    target = state.get('target_capture_fraction')
    observed = state.get('normalized_budget_fraction')
    if target not in protocol['capture_fractions']:
        errors.append('capture_target_outside_protocol')
    if not isinstance(observed, (int, float)) or not target <= observed <= 1.:
        errors.append('observed_capture_fraction_invalid')
    tags = state.get('search_condition_tags', [])
    if not tags or not set(tags) <= set(CONDITIONS):
        errors.append('search_condition_tags_invalid')
    replay = state.get('replay_metadata', {})
    if replay.get('candidate_fingerprint') != canonical_hash(
            replay.get('current_candidate')):
        errors.append('candidate_fingerprint_mismatch')
    return errors


def session_audit(protocol: dict, manifest: dict,
                  collector_commit: str | None) -> tuple[dict, list[Path]]:
    paths = sorted((OUT / 'integrity/sessions').glob('*.jsonl'))
    all_events = []
    session_rows = []
    for path in paths:
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
        all_events.extend(events)
        session_rows.append((path, events))
    starts = [row for row in all_events if row.get('event') == 'trajectory_run_started']
    completions = [row for row in all_events if row.get('event') == 'trajectory_run_completed']
    acquired = [row for row in all_events if row.get('event') == 'lock_acquired']
    released = [row for row in all_events if row.get('event') == 'lock_released']
    finished = [row for row in all_events if row.get('event') == 'formal_collection_complete']
    expected_ids = {row['instance_id'] for row in protocol['instances']}
    completion_hashes = {
        row['raw_path']: row['raw_sha256'] for row in completions
        if row.get('raw_path') and row.get('raw_sha256')}
    checks = {
        'formal_lock_released': not (OUT / 'integrity/formal.lock').exists(),
        'exactly_one_process_session': len(session_rows) == 1,
        'event_sequences_contiguous': all(
            [row['sequence'] for row in events] == list(range(1, len(events) + 1))
            for _, events in session_rows),
        'one_logical_formal_owner': {
            row.get('formal_owner_id') for row in all_events} == {FORMAL_OWNER},
        'one_successful_acquire_release': (
            len(acquired) == len(released) == 1 and released[0].get('status') == 'SUCCESS'),
        'formal_starts_exact_once': (
            len(starts) == len(expected_ids)
            and Counter(row['instance_id'] for row in starts)
            == Counter({key: 1 for key in expected_ids})),
        'formal_completions_exact_once': (
            len(completions) == len(expected_ids)
            and Counter(row['instance_id'] for row in completions)
            == Counter({key: 1 for key in expected_ids})),
        'completion_raw_hashes_match_manifest': completion_hashes == manifest['files'],
        'one_terminal_completion_event': (
            len(finished) == 1
            and finished[0].get('completed_runs') == protocol['expected_runs']
            and finished[0].get('completed_states') == protocol['expected_clean_states']),
        'execution_commit_matches_raw': (
            len(acquired) == 1
            and acquired[0].get('implementation_commit') == collector_commit),
    }
    details = {
        'schema': 'ngas-a17ar-session-integrity-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'session_count': len(session_rows),
        'sessions': [{
            'path': relative(path),
            'session_id': events[0].get('session_id') if events else None,
            'event_count': len(events),
            'heartbeat_count': sum(row.get('event') == 'heartbeat' for row in events),
        } for path, events in session_rows],
    }
    return details, [path for path, _ in session_rows]


def render_report(audit: dict, coverage_rows: list[dict]) -> str:
    coverage = defaultdict(int)
    for row in coverage_rows:
        coverage[(row['dataset_role'], row['scale'], row['search_stage'],
                  row['condition'])] += 1
    lines = [
        '| role | scale | stage | condition | states |',
        '|---|---|---|---|---:|',
    ]
    for key, count in sorted(coverage.items()):
        lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {count} |')
    role_scale = audit['coverage']['states_by_role_and_scale']
    return f'''# NGAS A1.7A-R clean trajectory coverage

## Collection integrity

`{audit['status']}`

The frozen clean non-R12 collection contains **{audit['completed_runs']}/108 runs**
and **{audit['completed_states']}/540 states**. Every raw file was independently
hashed and validated against the frozen revision-3 protocol. The single formal
session released its lock with `SUCCESS`; all 108 starts and completions occurred
exactly once. R13 and R14 remained locked, CORE45 remained excluded, and no Gurobi
run occurred.

The archived states retain the full compact-relational feature payload, full unique
joint-action identities and critic scores. Every selected action has exactly eight
stochastic repair/decode trial records. Candidate-bank size ranges from
**{audit['coverage']['candidate_actions_min']}** to
**{audit['coverage']['candidate_actions_max']}** actions.

## Balance

- TRAIN: S/M/L = `{json.dumps(role_scale['TRAIN'], sort_keys=True)}`.
- VALIDATION: S/M/L = `{json.dumps(role_scale['VALIDATION'], sort_keys=True)}`.
- Capture targets: `{json.dumps(audit['coverage']['states_by_capture_target'], sort_keys=True)}`.
- Medium and Large each contribute **{audit['coverage']['states_by_scale']['M']}**
  and **{audit['coverage']['states_by_scale']['L']}** states, respectively.

Each state can have multiple condition tags, so condition counts below are not
mutually exclusive. Search stage uses the observed normalized budget fraction and
the frozen 20%-wide bins from the A1.7A-R manual.

## Scale × stage × condition

{chr(10).join(lines)}

Machine-readable source data: `{relative(COVERAGE_CSV)}`. The clean origin is
development evidence; this report does not access or claim evidence from R13, R14,
or CORE45.
'''


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    config = json.loads(CONFIG.read_text())
    progress = json.loads((OUT / 'progress.json').read_text())
    manifest = json.loads((OUT / 'raw_manifest.json').read_text())
    protocol_sha = sha256_file(PROTOCOL)
    task_by_id = {row['instance_id']: row for row in protocol['instances']}
    expected_paths = {
        relative(OUT / 'raw' / row['dataset_role'].lower()
                 / f"{row['instance_id']}.json") for row in protocol['instances']}
    raw_paths = sorted((OUT / 'raw').rglob('*.json'))
    raw_summary = []
    state_errors = []
    coverage_rows = []
    seen_state_ids = set()
    capture_counts = Counter()
    role_scale_counts = Counter()
    scale_counts = Counter()
    candidate_counts = []
    trial_counts = []
    overshoots = []
    atomic_bounds = []
    starts_after_deadline = 0
    atomic_budget_contract_complete = True
    all_raw_hashes_match = True
    all_top_contracts_match = True
    all_final_replay_feasible = True
    all_capture_sets_exact = True
    all_state_ids_unique = True

    for path in raw_paths:
        encoded = path.read_bytes()
        actual_sha = hashlib.sha256(encoded).hexdigest()
        expected_sha = manifest.get('files', {}).get(relative(path))
        all_raw_hashes_match &= actual_sha == expected_sha
        payload = json.loads(encoded)
        task = task_by_id.get(payload.get('instance_id'))
        if task is None:
            all_top_contracts_match = False
            continue
        top_contract = {
            'schema': 'ngas-a17ar-clean-trajectory-run-v1',
            'status': 'COMPLETE',
            'protocol_sha256': protocol_sha,
            'instance_id': task['instance_id'],
            'instance_relative_path': task['relative_path'],
            'instance_content_sha256': task['content_sha256'],
            'dataset_role': task['dataset_role'],
            'audit_only': False,
            'trajectory_seed': task['trajectory_seed'],
            'budget_seconds': task['budget_seconds'],
            'R13': 'LOCKED_NO_ACCESS',
            'R14': 'LOCKED_NO_ACCESS',
            'RCIAS_CB1_CORE45': 'EXCLUDED',
            'gurobi_run': False,
        }
        all_top_contracts_match &= all(
            payload.get(key) == value for key, value in top_contract.items())
        states = payload.get('states', [])
        all_capture_sets_exact &= (
            len(states) == 5
            and Counter(row.get('target_capture_fraction') for row in states)
            == Counter(protocol['capture_fractions']))
        for state in states:
            if state.get('state_id') in seen_state_ids:
                all_state_ids_unique = False
            seen_state_ids.add(state.get('state_id'))
            errors = validate_state(state, task, protocol)
            if errors:
                state_errors.append({'state_id': state.get('state_id'), 'errors': errors})
            stage = progress_stage(float(state['normalized_budget_fraction']))
            for condition in state['search_condition_tags']:
                coverage_rows.append({
                    'dataset_origin': 'CLEAN_NON_R12_DEVELOPMENT',
                    'dataset_role': task['dataset_role'], 'scale': task['scale'],
                    'CF_level': task['CF_level'], 'search_stage': stage,
                    'condition': condition, 'state_id': state['state_id'],
                })
            capture_counts[str(state['target_capture_fraction'])] += 1
            role_scale_counts[(task['dataset_role'], task['scale'])] += 1
            scale_counts[task['scale']] += 1
            candidate_counts.append(len(state['candidate_action_ids']))
            trial_counts.append(len(state['stochastic_trial_outcomes']))
        diagnostics = payload.get('search_diagnostics', {})
        replay = diagnostics.get('final_replay', {})
        all_final_replay_feasible &= (
            replay.get('feasible') is True
            and replay.get('violation_count') == 0
            and replay.get('makespan') == payload.get('final_makespan'))
        atomic = diagnostics.get('atomic_budget_audit', {})
        atomic_budget_contract_complete &= (
            atomic.get('rule') == 'A16_STRICT_PRESTART_CHECKS'
            and isinstance(atomic.get('maximum_operation_seconds'), dict)
            and bool(atomic.get('maximum_operation_seconds'))
            and isinstance(atomic.get('started_after_deadline_count'), int))
        maximum_atomic = max(
            (float(value) for value in atomic.get('maximum_operation_seconds', {}).values()),
            default=0.)
        overshoot = float(diagnostics.get('budget_overshoot_seconds', math.inf))
        overshoots.append(overshoot)
        atomic_bounds.append(maximum_atomic)
        starts_after_deadline += int(atomic.get('started_after_deadline_count', 0))
        raw_summary.append({
            'instance_id': task['instance_id'],
            'collector_git_sha': payload.get('collector_git_sha'),
        })

    collector_commits = {row['collector_git_sha'] for row in raw_summary}
    collector_commit = next(iter(collector_commits), None)
    session, session_paths = session_audit(protocol, manifest, collector_commit)
    ledger_entries = verify_exposure_ledger(LEDGER)
    collection_ledger = [row for row in ledger_entries
                         if row.get('phase') == COLLECTION_PHASE
                         and row.get('command') == COLLECTOR_COMMAND]
    registry = DatasetRegistry(REGISTRY, ROOT)
    reserved = [row for row in registry.payload['instances']
                if row['family_id'] in set(config['reserved_families'])]
    selected_ids = {row['instance_id'] for row in protocol['instances']}
    selected_hashes = {row['content_sha256'] for row in protocol['instances']}
    commit_exists = bool(collector_commit) and subprocess.run(
        ['git', 'cat-file', '-e', f'{collector_commit}^{{commit}}'], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    frozen_ancestor = commit_exists and subprocess.run(
        ['git', 'merge-base', '--is-ancestor', protocol['freeze_source_commit'],
         collector_commit], cwd=ROOT).returncode == 0
    maximum_budget_excess = max(
        (over - bound for over, bound in zip(overshoots, atomic_bounds)), default=math.inf)

    checks = {
        'protocol_status_and_revision_frozen': (
            protocol.get('status') == 'FROZEN_BEFORE_FORMAL_COLLECTION'
            and protocol.get('revision') == 3),
        'protocol_config_registry_checkpoint_hashes_match': (
            protocol.get('config_sha256') == sha256_file(CONFIG)
            and protocol.get('registry_sha256') == sha256_file(REGISTRY)
            and protocol.get('checkpoint_sha256')
            == sha256_file(ROOT / protocol['checkpoint_path'])),
        'frozen_collection_source_hashes_match': all(
            sha256_file(ROOT / path) == expected
            for path, expected in protocol['source_hashes'].items()),
        'progress_marks_exact_completion': (
            progress.get('status') == 'FORMAL_COLLECTION_COMPLETE'
            and progress.get('completed_runs') == progress.get('expected_runs') == 108
            and progress.get('completed_states') == progress.get('expected_states') == 540
            and progress.get('protocol_sha256') == protocol_sha
            and progress.get('current_task') is None),
        'raw_manifest_scope_exact': (
            manifest.get('schema') == 'ngas-a17ar-trajectory-raw-manifest-v1'
            and manifest.get('protocol_sha256') == protocol_sha
            and manifest.get('completed_runs') == 108
            and manifest.get('completed_states') == 540
            and set(manifest.get('files', {})) == expected_paths
            and {relative(path) for path in raw_paths} == expected_paths),
        'all_raw_sha256_match': all_raw_hashes_match,
        'all_raw_top_level_contracts_match': all_top_contracts_match,
        'exact_540_unique_states': len(seen_state_ids) == 540 and all_state_ids_unique,
        'every_run_has_exact_capture_set': all_capture_sets_exact,
        'all_state_feature_action_trial_contracts_match': not state_errors,
        'all_final_schedule_replays_feasible': all_final_replay_feasible,
        'budget_atomic_prestart_and_overshoot_contract_pass': (
            atomic_budget_contract_complete
            and starts_after_deadline == 0
            and maximum_budget_excess <= .01
            and statistics.median(overshoots) <= statistics.median(atomic_bounds)),
        'single_committed_execution_revision': (
            len(collector_commits) == 1 and commit_exists and frozen_ancestor),
        'exposure_ledger_hash_chain_and_scope_pass': (
            len(collection_ledger) == 108
            and Counter(row['instance_id'] for row in collection_ledger)
            == Counter({key: 1 for key in selected_ids})
            and all(row.get('permitted') is True
                    and row.get('requested_purpose') == 'diagnostic'
                    and row.get('checkpoint_identifier') == protocol['checkpoint_sha256']
                    and row.get('git_sha') == collector_commit
                    and row.get('dataset_role') == task_by_id[row['instance_id']]['dataset_role']
                    for row in collection_ledger)),
        'train_validation_reserved_disjoint_by_id_and_hash': (
            not selected_ids & {row['instance_id'] for row in reserved}
            and not selected_hashes & {row['content_sha256'] for row in reserved}),
        'R13_R14_locked_CORE45_excluded_no_gurobi': (
            progress.get('R13') == 'LOCKED_NO_ACCESS'
            and progress.get('R14') == 'LOCKED_NO_ACCESS'
            and progress.get('RCIAS_CB1_CORE45') == 'EXCLUDED'),
        'formal_session_integrity_pass': session['status'] == 'PASS',
        'medium_and_large_not_underrepresented': (
            scale_counts == Counter({'S': 180, 'M': 180, 'L': 180})),
        'candidate_trials_semantics_preserved': (
            set(trial_counts) == {8}
            and config['production_solver']['candidate_trials'] == 8
            and config['production_solver']['search']['candidate_trials'] == 8),
    }
    status = 'PASS' if all(checks.values()) else 'FAIL'
    audit = {
        'schema': 'ngas-a17ar-collection-completion-audit-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'checks': checks,
        'protocol_sha256': protocol_sha,
        'collector_git_sha': collector_commit,
        'checkpoint_sha256': protocol['checkpoint_sha256'],
        'candidate_bank_hash': protocol['candidate_bank_hash'],
        'completed_runs': len(raw_summary), 'completed_states': len(seen_state_ids),
        'raw_bytes': sum(path.stat().st_size for path in raw_paths),
        'raw_manifest_sha256': sha256_file(OUT / 'raw_manifest.json'),
        'formal_log_sha256': sha256_file(OUT / 'logs/formal.log'),
        'session_integrity': session,
        'session_log_sha256': {
            relative(path): sha256_file(path) for path in session_paths},
        'exposure_ledger_sha256': sha256_file(LEDGER),
        'budget': {
            'median_overshoot_seconds': statistics.median(overshoots),
            'maximum_overshoot_seconds': max(overshoots),
            'median_maximum_atomic_seconds': statistics.median(atomic_bounds),
            'maximum_atomic_seconds': max(atomic_bounds),
            'maximum_overshoot_minus_atomic_bound_seconds': maximum_budget_excess,
            'atomic_starts_after_deadline': starts_after_deadline,
        },
        'coverage': {
            'states_by_role_and_scale': {
                role: {scale: role_scale_counts[(role, scale)] for scale in ('S', 'M', 'L')}
                for role in ('TRAIN', 'VALIDATION')},
            'states_by_scale': {scale: scale_counts[scale] for scale in ('S', 'M', 'L')},
            'states_by_capture_target': dict(sorted(capture_counts.items())),
            'candidate_actions_min': min(candidate_counts),
            'candidate_actions_max': max(candidate_counts),
            'candidate_action_observations': sum(candidate_counts),
            'selected_action_trial_records': sum(trial_counts),
        },
        'state_validation_errors': state_errors[:100],
        'boundaries': {
            'R12': 'DEVELOPMENT_EXPOSED_AUDIT_ONLY',
            'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
            'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
            'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_ACCESS',
            'production_solver_changed': False,
            'train_c1_v2': False,
            'gurobi_run': False,
        },
        'next_stage': ('FULL_BANK_DIAGNOSTIC_AUDIT'
                       if status == 'PASS' else 'STOP_COLLECTION_INTEGRITY_FAILURE'),
    }
    atomic_csv(COVERAGE_CSV, coverage_rows)
    atomic_json(OUT / 'audit/collection_completion_audit.json', audit)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(audit, coverage_rows))
    print(json.dumps({
        'status': status, 'completed_runs': audit['completed_runs'],
        'completed_states': audit['completed_states'], 'checks': checks,
        'audit': relative(OUT / 'audit/collection_completion_audit.json'),
        'report': relative(REPORT),
    }, indent=2))
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
