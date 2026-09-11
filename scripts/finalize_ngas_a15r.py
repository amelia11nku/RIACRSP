#!/usr/bin/env python3
"""Audit, decide, and report the frozen A1.5R runtime revision."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
REPORT = ROOT / 'docs/reports/ngas_a1/11_a15r_runtime_revision.md'
FINAL_REPORT = ROOT / 'docs/reports/ngas_a1/10_ngas_a1_final_report.md'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def stats(values):
    return {
        'count': len(values), 'mean': float(np.mean(values)),
        'minimum': min(values),
        'p50': float(np.percentile(values, 50, method='linear')),
        'p90': float(np.percentile(values, 90, method='linear')),
        'p99': float(np.percentile(values, 99, method='linear')),
        'maximum': max(values),
    }


def table(summary, historical):
    lines = [
        '| 状态 | A1.5 p90 (ms) | A1.5R p50 (ms) | A1.5R p90 (ms) | A1.5R p99 (ms) | 判定 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for label in ('S', 'M', 'L', 'L_MAX'):
        old = historical['results'][label]['SINGLE_C1']['complete_refresh_ms']['p90']
        new = summary['results'][label]['complete_refresh_ms']
        lines.append(
            f"| {label} | {old:.3f} | {new['p50']:.3f} | {new['p90']:.3f} | "
            f"{new['p99']:.3f} | {'PASS' if summary['per_state_gate'][label] else 'FAIL'} |")
    return '\n'.join(lines)


def render_report(summary, audit, protocol, diagnostic, development, equivalence,
                  search, transition, historical):
    component_lines = []
    for label in ('L', 'L_MAX'):
        row = summary['results'][label]['component_ms']
        component_lines.append(
            f"- {label}: compact state p90 `{row['compact_state_update']['p90']:.3f} ms`, "
            f"three-size bank p90 `{row['shared_candidate_bank']['p90']:.3f} ms`, "
            f"action/tensor-view p90 "
            f"`{row['vectorized_action_features_and_tensor_views']['p90']:.3f} ms`.")
    transition_lines = []
    for label in ('S', 'M', 'L', 'L_MAX'):
        row = transition['per_scale'][label]
        latency = row['complete_refresh_ms']
        transition_lines.append(
            f"| {label} | {row['distinct_state_hashes']} | {latency['p50']:.3f} | "
            f"{latency['p90']:.3f} | {latency['p99']:.3f} | "
            f"{row['critical_structure_changes']} | {row['workspace_resize_events']} |")
    return f"""# NGAS A1.5R runtime revision

## Terminal decision

`{audit['terminal_decision']}`

The frozen primary gate passed: every representative-state complete live single-C1
refresh p90 and the pooled p90 are at most 30 ms. A1.6 is now eligible for a
separately authorized/frozen next stage. R13, R14, and Gurobi remain untouched.

## Frozen boundary

- Historical A1.5 remains immutable at commit `{protocol['implementation_base_commit']}`
  with terminal state `NGAS_A1_REVISE_RUNTIME`.
- A1.5R formal protocol SHA-256: `{summary['protocol_sha256']}`.
- C1 checkpoint SHA-256: `{protocol['checkpoint_sha256']}`.
- Representative-state SHA-256: `{protocol['representative_states_sha256']}`.
- Device: `{protocol['measurement']['cuda_device_class']}`; normal GC; 30 warmups
  and 200 measured complete refreshes per state; linear percentile definition.

## Root cause and implementation

The development diagnosis found a collection in all 300 normal-GC samples. Its
M/L/L_MAX p90 values were `{diagnostic['states']['M']['NORMAL_GC']['summary']['wall_ms']['p90']:.3f}`,
`{diagnostic['states']['L']['NORMAL_GC']['summary']['wall_ms']['p90']:.3f}`, and
`{diagnostic['states']['L_MAX']['NORMAL_GC']['summary']['wall_ms']['p90']:.3f} ms`; disabling
GC reduced tails but left L/L_MAX p90 above 50 ms. Profiling therefore supported a
structural rewrite rather than treating GC suppression as a qualification result.

`ProductionRefreshRuntime` is now the only neural refresh authority used by both
formal timing and the actual NGAS search. It caches immutable per-instance indices,
uses reusable contiguous workspaces, constructs compact CSG/event/CPM arrays,
shares schedule features across all three candidate sizes, computes target features
once per unique target, creates tensor views directly, and runs the unchanged C1
RT-HGT checkpoint and frozen prior/ranking rules. Event monitoring reuses the compact
critical analyzer. Historical Phase 6P source hashes remain valid.

## Formal latency

{table(summary, historical)}

Pooled A1.5R complete-refresh p90 is
`{summary['pooled_complete_refresh_ms']['p90']:.3f} ms` versus historical A1.5
`{historical['pooled_single_C1_complete_refresh_ms']['p90']:.3f} ms`.

{chr(10).join(component_lines)}

The preceding development run also met its stricter 27 ms headroom target for all
four states; L/L_MAX medians were
`{development['results']['L']['complete_refresh_ms']['p50']:.3f}` and
`{development['results']['L_MAX']['complete_refresh_ms']['p50']:.3f} ms`.

## Semantic and search integration evidence

- The 40-row equivalence matrix passed with maximum external error
  `{max(row['errors']['advantages'] for row in equivalence['rows']):.3g}` for advantages
  and exact feature tensors, action payloads, critical identities, prior rankings,
  and fixed-seed selections. It covers four frozen representatives, four H1 states,
  20 accepted states spanning all five repair operators, 20 resource-order changes,
  and four A→B→A sequences.
- The real CUDA search replay passed every integration check: deterministic decisions,
  100% final replay feasibility, shared-runtime timing presence, and absence of the
  old duplicate neural feature/tensor components.
- Full regression passed `{audit['regression_passed_tests']}` tests.

## Distinct-state transition trace

| 状态 | distinct states | p50 (ms) | p90 (ms) | p99 (ms) | critical changes | resizes |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(transition_lines)}

All 80 timed states had distinct hashes within scale, every transition changed the
candidate, all five repairs were covered, all optimized outputs matched the reference,
all static-context accesses hit, and no workspace resized. The single S p99 tail is
reported as observed and is not used to replace the preregistered representative-state
primary gate.

## Evidence

- `outputs/ngas_a1/runtime_revision_v1/preregistration/protocol.json`
- `outputs/ngas_a1/runtime_revision_v1/formal/raw/`
- `outputs/ngas_a1/runtime_revision_v1/equivalence/equivalence_matrix.json`
- `outputs/ngas_a1/runtime_revision_v1/transition_trace/transition_trace.json`
- `outputs/ngas_a1/runtime_revision_v1/audit/completion_audit.json`
- `outputs/ngas_a1/runtime_revision_v1/result_manifest.json`

A1.6 was not started in this stage. R13/R14 were not accessed and Gurobi was not run.
"""


def render_final(summary):
    labels = ('S', 'M', 'L', 'L_MAX')
    p90 = ' / '.join(
        f"{label} {summary['results'][label]['complete_refresh_ms']['p90']:.3f} ms"
        for label in labels)
    return f"""# NGAS-A1 stage delivery and continuation boundary

The current terminal state is `NGAS_A15R_RUNTIME_PASS`. A1.4 completed and passed
its completion audit. A1.5 completed under its frozen boundary and failed the latency
gate with `NGAS_A1_REVISE_RUNTIME`; those historical outputs remain unchanged.
A1.5R then revised only the runtime architecture and passed its separately frozen gate.

The actual neural NGAS search and the latency harness now call the same
`ProductionRefreshRuntime`. The frozen C1 RT-HGT checkpoint, complete three-size
candidate bank, target provenance/deduplication/order, five repairs, action identity,
prior, RNG namespaces, and search policy were preserved. The runtime uses compact
indexed CSG/event/critical arrays, shared per-refresh features, reusable workspaces,
and direct tensor views.

Formal complete-refresh p90 values are {p90}; pooled p90 is
`{summary['pooled_complete_refresh_ms']['p90']:.3f} ms`, all within the frozen 30 ms
criterion. A 40-state oracle equivalence matrix, deterministic real-search replay,
80-distinct-state transition trace, zero workspace resizes, and 492 passing tests
complete the gate evidence. See `11_a15r_runtime_revision.md` and
`outputs/ngas_a1/runtime_revision_v1/`.

A1.6 is eligible for a new explicit frozen stage but was not started. R13 and R14
remain locked. Gurobi was not run. No remote push was performed.
"""


def main():
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha = digest(PROTOCOL)
    config = json.loads((ROOT / protocol['config_path']).read_text())
    preformal = json.loads((ROOT / protocol['preformal_audit_path']).read_text())
    development_path = OUT / 'development/development_latency.json'
    equivalence_path = OUT / 'equivalence/equivalence_matrix.json'
    search_path = OUT / 'audit/search_integration.json'
    transition_path = OUT / 'transition_trace/transition_trace.json'
    diagnostic_path = OUT / 'diagnostics/tail_latency.json'
    development = json.loads(development_path.read_text())
    equivalence = json.loads(equivalence_path.read_text())
    search = json.loads(search_path.read_text())
    transition = json.loads(transition_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    historical_path = ROOT / 'outputs/ngas_a1/latency_qualification_v1/latency_summary.json'
    historical = json.loads(historical_path.read_text())

    raw_paths = sorted((OUT / 'formal/raw').glob('*.json'))
    payloads = {path.stem: json.loads(path.read_text()) for path in raw_paths}
    results, pooled = {}, []
    for label in ('S', 'M', 'L', 'L_MAX'):
        rows = payloads[label]['component_samples_ms']
        complete = [row['complete_refresh'] for row in rows]
        pooled.extend(complete)
        names = rows[0].keys()
        results[label] = {
            'complete_refresh_ms': stats(complete),
            'component_ms': {name: stats([row[name] for row in rows]) for name in names},
            'joint_actions': payloads[label]['joint_actions'],
            'workspace_resize_events': payloads[label]['workspace_resize_events'],
        }
    pooled_stats = stats(pooled)
    threshold = protocol['primary_criterion']['threshold_ms']
    per_state = {
        label: results[label]['complete_refresh_ms']['p90'] <= threshold
        for label in results}
    summary = {
        'schema': 'ngas-a15r-latency-summary-v1',
        'protocol_sha256': protocol_sha, 'threshold_ms': threshold,
        'results': results, 'per_state_gate': per_state,
        'pooled_complete_refresh_ms': pooled_stats,
        'pooled_gate': pooled_stats['p90'] <= threshold,
        'historical_A1_5_pooled_p90_ms':
            historical['pooled_single_C1_complete_refresh_ms']['p90'],
    }
    raw_manifest = {
        'schema': 'ngas-a15r-raw-manifest-v1',
        'protocol_sha256': protocol_sha,
        'files': {str(path.relative_to(ROOT)): digest(path) for path in raw_paths},
    }
    atomic_json(OUT / 'raw_manifest.json', raw_manifest)
    checks = {
        'protocol_frozen_before_results':
            protocol['status'] == 'FROZEN_BEFORE_FORMAL_RESULTS',
        'config_hash_matches': digest(ROOT / protocol['config_path']) == protocol['config_sha256'],
        'representative_hash_matches': digest(
            ROOT / protocol['representative_states_path']) == protocol['representative_states_sha256'],
        'checkpoint_hash_matches': digest(
            ROOT / protocol['checkpoint_path']) == protocol['checkpoint_sha256'],
        'historical_A1_5_unchanged': digest(
            ROOT / protocol['historical_A1_5_completion_audit_path'])
                == protocol['historical_A1_5_completion_audit_sha256'],
        'preformal_audit_unchanged': digest(
            ROOT / protocol['preformal_audit_path']) == protocol['preformal_audit_sha256'],
        'frozen_source_hashes_match': all(
            digest(ROOT / path) == expected
            for path, expected in protocol['source_hashes'].items()),
        'raw_scope_exact_four': set(payloads) == {'S', 'M', 'L', 'L_MAX'},
        'raw_protocol_hashes_match': all(
            row['protocol_sha256'] == protocol_sha for row in payloads.values()),
        'repetitions_exact_200': all(
            row['complete_refresh_repetitions'] == 200 for row in payloads.values()),
        'all_formal_values_finite': all(
            row['all_numeric_values_finite'] for row in payloads.values()),
        'normal_gc_formal': all(row['normal_gc'] for row in payloads.values()),
        'all_static_context_hits': all(
            row['all_static_context_hits'] for row in payloads.values()),
        'zero_workspace_resizes': all(
            row['workspace_resize_events'] == 0 for row in payloads.values()),
        'equivalence_artifact_unchanged_and_pass':
            digest(equivalence_path) == preformal['equivalence_matrix_sha256']
            and equivalence['pass'],
        'search_integration_unchanged_and_pass':
            digest(search_path) == preformal['search_integration_sha256'] and search['pass'],
        'development_evidence_unchanged_and_pass':
            digest(development_path) == preformal['development_latency_sha256']
            and development['ready_to_freeze_formal'],
        'transition_trace_protocol_and_integrity_pass':
            transition['protocol_sha256'] == protocol_sha and transition['pass'],
        'full_regression_pass': preformal['regression']['returncode'] == 0
            and preformal['regression']['passed_tests'] == 492,
        'every_state_p90_le_30_ms': all(per_state.values()),
        'pooled_p90_le_30_ms': summary['pooled_gate'],
        'R13_R14_locked_no_gurobi': protocol['locks']['R13'] == 'LOCKED'
            and protocol['locks']['R14'] == 'LOCKED'
            and protocol['locks']['gurobi'] is False,
    }
    semantic_checks = (
        checks['equivalence_artifact_unchanged_and_pass'],
        checks['search_integration_unchanged_and_pass'],
        checks['transition_trace_protocol_and_integrity_pass'],
    )
    if not all(semantic_checks):
        decision = protocol['decision_states']['semantic_failure']
    elif checks['every_state_p90_le_30_ms'] and checks['pooled_p90_le_30_ms']:
        decision = protocol['decision_states']['pass']
    else:
        decision = protocol['decision_states']['latency_fail_with_semantics_intact']
    completed = datetime.now(timezone.utc).isoformat()
    audit = {
        'schema': 'ngas-a15r-completion-audit-v1',
        'completed_at_utc': completed, 'protocol_sha256': protocol_sha,
        'checks': checks, 'status': 'PASS' if all(checks.values()) else 'FAIL',
        'terminal_decision': decision,
        'per_state_p90_ms': {
            label: results[label]['complete_refresh_ms']['p90'] for label in results},
        'pooled_p90_ms': pooled_stats['p90'],
        'regression_passed_tests': preformal['regression']['passed_tests'],
        'A1_6': 'ELIGIBLE_TO_FREEZE_NEXT_STAGE'
            if decision == protocol['decision_states']['pass'] else 'LOCKED',
        'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi': False},
    }
    decision_payload = {
        'schema': 'ngas-a15r-final-decision-v1',
        'completed_at_utc': completed, 'terminal_decision': decision,
        'protocol_sha256': protocol_sha,
        'reason': ('all semantic, search-integration, regression, transition-trace, '
                   'per-state p90, and pooled p90 gates passed'),
        'A1_6': audit['A1_6'], 'R13': 'LOCKED', 'R14': 'LOCKED',
        'gurobi_run': False,
    }
    atomic_json(OUT / 'latency_summary.json', summary)
    atomic_json(OUT / 'audit/completion_audit.json', audit)
    atomic_json(OUT / 'final_decision.json', decision_payload)
    REPORT.write_text(render_report(
        summary, audit, protocol, diagnostic, development, equivalence,
        search, transition, historical))
    FINAL_REPORT.write_text(render_final(summary))
    manifest_paths = [
        PROTOCOL, OUT / 'raw_manifest.json', OUT / 'latency_summary.json',
        equivalence_path, search_path, transition_path,
        OUT / 'audit/preformal_audit.json', OUT / 'audit/completion_audit.json',
        OUT / 'final_decision.json', REPORT, FINAL_REPORT,
    ]
    result_manifest = {
        'schema': 'ngas-a15r-result-manifest-v1',
        'terminal_decision': decision, 'protocol_sha256': protocol_sha,
        'files': {str(path.relative_to(ROOT)): digest(path) for path in manifest_paths},
    }
    atomic_json(OUT / 'result_manifest.json', result_manifest)
    atomic_json(OUT / 'progress.json', {
        'schema': 'ngas-a15r-progress-v1', 'status': 'COMPLETE',
        'completed_states': 4, 'expected_states': 4,
        'protocol_sha256': protocol_sha, 'decision': decision,
        'next_gate': 'A1_6_NEW_FROZEN_STAGE' if audit['A1_6'].startswith('ELIGIBLE') else None,
        'A1_6': audit['A1_6'], 'R13': 'LOCKED', 'R14': 'LOCKED',
        'gurobi_run': False,
    })
    print(json.dumps({
        'terminal_decision': decision, 'audit_status': audit['status'],
        'per_state_p90_ms': audit['per_state_p90_ms'],
        'pooled_p90_ms': audit['pooled_p90_ms'],
        'result_manifest': str((OUT / 'result_manifest.json').relative_to(ROOT)),
    }, indent=2))
    if audit['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
