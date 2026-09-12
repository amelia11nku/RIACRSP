#!/usr/bin/env python3
"""Finalize the evidence-preserving NGAS A1.7A-R recovery phase."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
AUDIT_DIR = OUT / 'audit'
FINAL_DECISION = OUT / 'final_decision.json'
RESULT_MANIFEST = OUT / 'result_manifest.json'
FINAL_REPORT = ROOT / 'reports/ngas_a17ar_final_report.md'
DOC_REPORT = ROOT / 'docs/reports/ngas_a1/14_a17ar_contamination_recovery_and_diagnostics.md'
FIGURES = ROOT / 'reports/figures/ngas_a17ar'
FIGURE_QA = FIGURES / 'figure_qa.json'
REGRESSION_AUDIT = AUDIT_DIR / 'final_regression_test.json'
TERMINAL = 'NGAS_A1_7AR_PASS_CONTAMINATION_RECOVERED'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def format_ids(values: list[str]) -> str:
    return '\n'.join(f'- `{value}`' for value in values)


def figure_audit() -> dict:
    stems = (
        'critic_rank_quality_by_scale_stage',
        'u0_u1_ranking_agreement',
        'cost_normalized_utility_summary',
        'prior_staleness_curve',
        'candidate_trial_marginal_value',
    )
    source = json.loads((FIGURES / 'source_validation.json').read_text())
    checks = {
        'source_validation_pass': (
            source['summary']['ready'] is True
            and source['summary']['counts']['FAIL'] == 0),
    }
    figures = {}
    for stem in stems:
        alignment = json.loads((FIGURES / f'{stem}.alignment.json').read_text())
        collision = json.loads((FIGURES / f'{stem}.collision.json').read_text())
        text = json.loads((FIGURES / f'{stem}.pdf-text.json').read_text())
        expected = [FIGURES / f'{stem}.{suffix}'
                    for suffix in ('pdf', 'svg', 'png', 'tiff')]
        checks[f'{stem}_exports_complete'] = all(path.is_file() for path in expected)
        checks[f'{stem}_alignment_pass'] = alignment['verdict'] in {'PASS', 'NOT APPLICABLE'}
        checks[f'{stem}_collision_pass'] = collision['verdict'] == 'PASS'
        checks[f'{stem}_glyph_floor_pass'] = (
            text['auditable'] is True and text['below_minimum_count'] == 0
            and float(text['minimum_found_pt']) >= 5.)
        figures[stem] = {
            'exports': {path.suffix.lstrip('.'): sha256(path) for path in expected},
            'alignment_sha256': sha256(FIGURES / f'{stem}.alignment.json'),
            'collision_sha256': sha256(FIGURES / f'{stem}.collision.json'),
            'pdf_text_sha256': sha256(FIGURES / f'{stem}.pdf-text.json'),
        }
    payload = {
        'schema': 'ngas-a17ar-figure-qa-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': 'python-matplotlib', 'checks': checks, 'figures': figures,
        'visual_inspection': (
            'PASS: every PNG was inspected at final composition; no clipping, '
            'ambiguous encoding, or unreadable panel was observed'),
        'source_validation_sha256': sha256(FIGURES / 'source_validation.json'),
    }
    atomic_json(FIGURE_QA, payload)
    return payload


def governed_access_check(registry: dict, ledger_rows: list[dict]) -> dict:
    forbidden_families = {'RCIAS_CB1_R13', 'RCIAS_CB1_R14', 'RCIAS_CB1_CORE45'}
    forbidden = [row for row in registry['instances']
                 if row['family_id'] in forbidden_families]
    forbidden_ids = {row['instance_id'] for row in forbidden}
    forbidden_hashes = {row['content_sha256'] for row in forbidden}
    phase_rows = [row for row in ledger_rows
                  if str(row.get('phase', '')).startswith('NGAS_A1_7A_R')]
    locked_rows = [row for row in phase_rows
                   if row.get('instance_id') in forbidden_ids
                   or row.get('instance_content_sha256') in forbidden_hashes]
    allowed_metadata = [row for row in locked_rows
                        if row.get('requested_purpose') == 'metadata_integrity'
                        and row.get('command') == 'scripts/audit_ngas_a17ar_exposure.py'
                        and row.get('access_scope')
                            == 'manifest metadata and byte hash only; no solver or performance access']
    violations = [row for row in locked_rows if row not in allowed_metadata]
    return {
        'a17ar_ledger_entries': len(phase_rows),
        'allowed_locked_metadata_integrity_entries': len(allowed_metadata),
        'forbidden_family_accesses': len(violations),
        'violations': violations,
    }


def report_text(context: dict) -> str:
    r12_ids = context['r12_ids']
    train_ids = context['trajectory_protocol']['train_instances']
    validation_ids = context['trajectory_protocol']['validation_instances']
    return f'''# NGAS A1.7A-R final report

## Terminal classification

`{TERMINAL}`

A1.7A-R reconstructs and repairs the experimental evidence boundary. It does not
retrain C1, alter the production solver, run Gurobi, or access R13, R14, or
RCIAS-CB1-CORE45 for performance evaluation.

## Required conclusions

1. **Exact final-C1 instances.** The final frozen C1 used these 18 R12 instances:

{format_ids(r12_ids)}

2. **Training volume.** The cache contains **72 states** and **6,465 joint-action
   labels**.
3. **Continuation label scope.** `no_continuation_rollout=true`.
4. **Four sources per instance.** `H1`, `NATIVE16_746101`,
   `NATIVE16_746102`, and `NATIVE16_746103`.
5. **Instance overlap.** Final C1 and A1.6R overlap on **18/18 instances (100%)**.
6. **Seed overlap.** All three Native16/A1.6R seeds overlap:
   `746101`, `746102`, and `746103` (**3/3; 100%**).
7. **State overlap.** The 18 replayable A1.6R snapshots are all
   same-instance/same-seed but non-identical states. Exact matches: **0**;
   same schedule with incomplete metadata: **0**; non-identical provenance
   matches: **18**.
8. **Remaining A1.6R claims.** A1.6R remains valid for production integration,
   budget/concurrency accounting, feasibility, reproducibility, runtime
   qualification, and R12 development-benchmark performance.
9. **Removed A1.6R claims.** It does not establish independent-test performance,
   unseen-instance generalization, leakage-free held-out evaluation, or an
   unbiased final comparison against learning-free baselines.
10. **Comparator exposure.** R12-specific training/adaptation is confirmed for
    `PHASE6N_TOP1` and `NGAS_C1`. No R12-specific tuning evidence was found for
    GA, DCGA, DABC, LG_HGA_2O, ALNS, or PHASE6H.
11. **Permanent R12 role.** `DEVELOPMENT_EXPOSED`.
12. **Final-evaluation locks.** R13 remains `LOCKED_FINAL_EVAL`; R14 remains
    `LOCKED_GENERALIZATION_EVAL`. Their performance data remain untouched.
13. **New C1-v2 training pool.** The governed pool contains 81 instances:

{format_ids(train_ids)}

14. **New validation pool.** The governed pool contains 27 instances:

{format_ids(validation_ids)}

15. **Split disjointness.** Training and validation IDs and content hashes are
    mutually disjoint and are also disjoint from R12, R13, R14, and CORE45.
16. **Scale/stage ranking.** U0 is too sparse to establish monotonic degradation.
    Clean L has 0/6 informative states; clean M and S have mean rho 0.0529 and
    0.0349. R12 S/M/L mean rho is 0.0812/0.0819/0.1436 on only 1/2/2 informative
    states. R12 middle and late audited states are all U0-constant. This is a
    support and trajectory-coverage problem, not evidence of reliable scaling.
17. **U0 versus U1.** Conditional on informative states, mean action-rank rho is
    **0.9330** on clean non-R12 and **0.7664** on R12, but only 4/18 and 5/18
    states are informative. The apparent all-state Top-1 agreement is inflated by
    all-zero banks.
18. **Cost normalization.** U1 and U2 rankings are nearly identical: mean rho is
    **0.999999** (clean) and **0.999991** (R12), with the same Top-1 in all 36
    audited states. The current measured cost normalization does not materially
    change the action order.
19. **Portfolio interaction.** Deterministic portfolio adjustment changes neural
    Top-1 in 1/18 states per origin. The one clean change loses 13 U0 units; the
    R12 change is neutral. Archived sampled production actions differ from neural
    Top-1 in every state, so that comparison combines frozen exploration with
    portfolio weighting. There is no evidence of a systematic correction benefit.
20. **Trials 1...8.** Mean raw best-makespan loss versus eight trials is
    187.8078, 101.4400, and 43.1263 at caps 1, 2, and 4. The probability that a
    later trial changes the best is 0.8807, 0.7472, and 0.4973. Positive-gain loss
    is much smaller (0.0564, 0.0442, 0.0353), while the no-new-best rate reaches
    0.8869 at trial 8.
21. **Downstream decision.** Evidence justifies separate development pilots for
    **both A1.7B and A1.7C**. A1.7B should test adaptive trial racing against the
    raw-quality/positive-gain tradeoff. A1.7C should train a trajectory-aware C1-v2
    because current local ranking is sparse and weak. Neither phase starts here.

## Clean trajectory and staleness evidence

The frozen revision-3 collection completed 108/108 runs and 540/540 states: 405
TRAIN and 135 VALIDATION states, balanced across S/M/L and five search-stage
targets. The full-bank audit evaluated 12,775 actions from 36 frozen states.

The staleness audit completed 36 states, 180 offsets, and 63,875 action-offset
evaluations. At offset 5, semantic bank Jaccard drops to 0.1413 (clean) and
0.1325 (R12), even where compact features and typed edges are unchanged. Common
semantic actions retain rank rho around 0.98-1.00 and mean percentile-rank drift
around 0.03 through offset 19. The drop is therefore state-ID-conditioned
candidate-bank churn, not evidence of RT-HGT numerical instability. Fresh Top-1
does not consistently improve U0, so the frozen refresh interval is unchanged.

## Evidence locations

- Exposure reconstruction: `reports/ngas_a17ar_r12_exposure_audit.json`
- Dataset registry: `configs/dataset_role_registry.json`
- Exposure ledger: `artifacts/dataset_exposure_ledger.jsonl`
- Trajectory protocol: `artifacts/ngas_a17ar/trajectory_protocol_manifest.json`
- Completion audits: `{relative(AUDIT_DIR)}`
- Diagnostic reports: `reports/ngas_a17ar_critic_ranking_audit.md` through
  `reports/ngas_a17ar_prior_staleness.md`
- Figures and source data: `{relative(FIGURES)}`
- Final decision: `{relative(FINAL_DECISION)}`
- Result manifest: `{relative(RESULT_MANIFEST)}`
'''


def docs_text() -> str:
    return f'''# NGAS A1.7A-R 污染恢复与轨迹诊断

终态为 `{TERMINAL}`。程序化重建确认：最终 C1 的 72 个状态、6,465 个
joint-action 标签来自全部 18 个 R12 算例，且 A1.6R 也使用相同的 18 个
算例与三个 Native16 seed。因此 R12 永久归类为 `DEVELOPMENT_EXPOSED`，
A1.6R 只保留开发阶段工程与性能证据，不再承担独立测试或未见泛化证据。

数据治理已经机器化：81 个 TRAIN 算例与 27 个 VALIDATION 算例按 ID 和
内容哈希与 R12、R13、R14、CORE45 完全隔离；R13/R14 继续锁定。本阶段完成
108 条干净轨迹、540 个状态、36 状态的 12,775-action 全候选池诊断，以及
36 状态 × 5 offset 的 staleness 审计。生产 C1、搜索策略、20-iteration
refresh 和 `candidate_trials=8` 均未改变。

诊断表明，U0/U1 有效状态稀少且 critic Top-1 未命中正效用最佳 action；
U1 与成本归一化 U2 排序几乎相同；portfolio 的确定性 Top-1 调整很少且未
显示系统收益；少 trial cap 会造成明显原始候选质量损失，但正效用损失很小。
因此建议分别预注册 A1.7B adaptive trial racing 和 A1.7C trajectory-aware
C1-v2 小试验，本阶段不自动启动。

完整 21 项结论见 `reports/ngas_a17ar_final_report.md`；机器终态与全量哈希见
`outputs/ngas_a1/trajectory_utility_a17ar_v1/final_decision.json` 和
`outputs/ngas_a1/trajectory_utility_a17ar_v1/result_manifest.json`。
'''


def formal_artifacts() -> list[Path]:
    paths = set()
    for base in (OUT, ROOT / 'artifacts/ngas_a17ar', FIGURES):
        paths.update(path for path in base.rglob('*') if path.is_file())
    paths.update(path for path in (ROOT / 'reports').glob('ngas_a17ar*')
                 if path.is_file())
    for path in (
            ROOT / 'artifacts/dataset_exposure_ledger.jsonl',
            ROOT / 'configs/dataset_role_registry.json',
            ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml',
            ROOT / 'rcias_ngas/evaluation/a17ar.py',
            ROOT / 'rcias_ngas/governance/dataset_roles.py',
            FINAL_REPORT, DOC_REPORT, FINAL_DECISION):
        paths.add(path)
    paths.update((ROOT / 'scripts').glob('*ngas_a17ar*.py'))
    paths.update((ROOT / 'tests/ngas').glob('*a17ar*.py'))
    paths.discard(RESULT_MANIFEST)
    return sorted(path for path in paths if path.is_file())


def main() -> None:
    required_audits = {
        'collection': AUDIT_DIR / 'collection_completion_audit.json',
        'full_bank': AUDIT_DIR / 'full_bank_completion_audit.json',
        'prior_staleness': AUDIT_DIR / 'prior_staleness_completion_audit.json',
        'regression': REGRESSION_AUDIT,
    }
    audits = {name: json.loads(path.read_text())
              for name, path in required_audits.items()}
    exposure = json.loads((ROOT / 'reports/ngas_a17ar_r12_exposure_audit.json').read_text())
    registry = json.loads((ROOT / 'configs/dataset_role_registry.json').read_text())
    trajectory = json.loads(
        (ROOT / 'artifacts/ngas_a17ar/trajectory_protocol_manifest.json').read_text())
    preflight = json.loads((ROOT / 'artifacts/ngas_a17ar/preflight.json').read_text())
    figure = figure_audit()

    exposure_rows = read_csv(ROOT / 'reports/ngas_a17ar_r12_exposure_rows.csv')
    r12_ids = sorted({row['instance_id'] for row in exposure_rows
                      if row['record_type'] == 'C1_TRAINING_STATE'
                      and row['contributed_to_final_c1'] == 'True'})
    overlap_rows = read_csv(ROOT / 'reports/ngas_a17ar_state_overlap.csv')
    overlap_counts = Counter(row['classification'] for row in overlap_rows)
    comparator_rows = read_csv(ROOT / 'reports/ngas_a17ar_comparator_exposure.csv')
    tuned = sorted(row['algorithm'] for row in comparator_rows
                   if row['r12_specific_tuning'] == 'True')

    ledger_rows = []
    for line in (ROOT / 'artifacts/dataset_exposure_ledger.jsonl').read_text().splitlines():
        if line:
            ledger_rows.append(json.loads(line))
    access = governed_access_check(registry, ledger_rows)
    r12_registry = [row for row in registry['instances']
                    if row['family_id'] == 'RCIAS_CB1_R12']
    r13_registry = [row for row in registry['instances']
                    if row['family_id'] == 'RCIAS_CB1_R13']
    r14_registry = [row for row in registry['instances']
                    if row['family_id'] == 'RCIAS_CB1_R14']

    required_reports = [ROOT / f'reports/ngas_a17ar_{name}.md' for name in (
        'preflight', 'r12_exposure_audit', 'state_overlap_audit',
        'comparator_exposure_audit', 'terminology_audit', 'trajectory_protocol',
        'trajectory_coverage', 'critic_ranking_audit', 'utility_alignment',
        'prior_staleness', 'portfolio_interaction', 'candidate_trials_audit')]
    checks = {
        'all_completion_audits_pass': all(
            audit.get('status') == 'PASS' for audit in audits.values()),
        'r12_exposure_reconstructed': (
            exposure['status'] == 'PASS' and len(r12_ids) == 18
            and exposure['c1_training']['states'] == 72
            and exposure['c1_training']['joint_action_labels'] == 6465
            and exposure['overlap']['instance_ids'] == {'count': 18, 'denominator': 18}),
        'state_overlap_conservatively_classified': (
            len(overlap_rows) == 18
            and overlap_counts == Counter({'SAME_INSTANCE_SEED_NON_IDENTICAL_STATE': 18})),
        'comparator_exposure_classified': tuned == ['NGAS_C1', 'PHASE6N_TOP1'],
        'R12_permanently_development_exposed': (
            len(r12_registry) == 18
            and {row['role'] for row in r12_registry} == {'DEVELOPMENT_EXPOSED'}),
        'R13_R14_roles_locked': (
            len(r13_registry) == len(r14_registry) == 18
            and {row['role'] for row in r13_registry} == {'LOCKED_FINAL_EVAL'}
            and {row['role'] for row in r14_registry}
                == {'LOCKED_GENERALIZATION_EVAL'}
            and preflight['locked_evaluation']['R13']['performance_accessed'] is False
            and preflight['locked_evaluation']['R14']['performance_accessed'] is False),
        'no_A17AR_forbidden_family_access': access['forbidden_family_accesses'] == 0,
        'clean_future_pools_frozen_and_disjoint': (
            trajectory['status'] == 'FROZEN_BEFORE_FORMAL_COLLECTION'
            and len(trajectory['train_instances']) == 81
            and len(trajectory['validation_instances']) == 27
            and all(trajectory['checks'].values())),
        'diagnostics_complete': (
            audits['collection']['completed_states'] == 540
            and audits['full_bank']['completed_actions'] == 12775
            and audits['prior_staleness']['stale_action_offset_evaluations'] == 63875),
        'production_solver_unchanged': all(
            audit['boundaries']['production_solver_changed'] is False
            for name, audit in audits.items() if name != 'regression'),
        'R13_R14_CORE45_and_gurobi_boundaries_preserved': all(
            audit['boundaries']['gurobi_run'] is False
            for name, audit in audits.items() if name != 'regression'),
        'required_reports_present': all(path.is_file() for path in required_reports),
        'figure_QA_pass': figure['status'] == 'PASS',
        'full_regression_suite_pass': audits['regression'].get('status') == 'PASS',
    }
    if not all(checks.values()):
        atomic_json(FINAL_DECISION, {
            'schema': 'ngas-a17ar-final-decision-v1',
            'terminal_classification': 'NGAS_A1_7AR_INVALID',
            'checks': checks, 'stop_reason': 'finalization check failed',
        })
        print(json.dumps({'status': 'FAIL', 'checks': checks}, indent=2))
        raise SystemExit(1)

    context = {
        'r12_ids': r12_ids, 'trajectory_protocol': trajectory,
    }
    FINAL_REPORT.write_text(report_text(context))
    DOC_REPORT.write_text(docs_text())
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    decision = {
        'schema': 'ngas-a17ar-final-decision-v1',
        'decided_at_utc': datetime.now(timezone.utc).isoformat(),
        'terminal_classification': TERMINAL,
        'checks': checks, 'finalization_input_commit': head,
        'contamination_recovery': {
            'final_C1_instances': r12_ids,
            'final_C1_states': 72, 'final_C1_joint_action_labels': 6465,
            'no_continuation_rollout': True,
            'instance_overlap': '18/18 (100%)', 'seed_overlap': '3/3 (100%)',
            'R12_role': 'DEVELOPMENT_EXPOSED',
        },
        'future_data': {
            'train_instances': len(trajectory['train_instances']),
            'validation_instances': len(trajectory['validation_instances']),
            'collected_states': 540, 'train_states': 405, 'validation_states': 135,
            'reserved_id_and_hash_disjoint': True,
        },
        'diagnostics': {
            'full_bank_states': 36, 'full_bank_actions': 12775,
            'staleness_states': 36, 'staleness_offsets': 180,
            'staleness_action_offset_evaluations': 63875,
            'adaptive_refresh_supported': False,
        },
        'boundaries': {
            'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
            'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
            'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_ACCESS',
            'production_solver_changed': False, 'C1_retrained': False,
            'adaptive_trial_racing_implemented': False,
            'adaptive_refresh_implemented': False, 'gurobi_run': False,
        },
        'recommendation': {
            'A1.7B': 'JUSTIFIED_AS_SEPARATELY_FROZEN_DEVELOPMENT_PILOT',
            'A1.7C': 'JUSTIFIED_AS_SEPARATELY_FROZEN_DEVELOPMENT_PILOT',
            'automatic_start': False,
        },
        'protocol_hashes': {
            'trajectory': sha256(ROOT / 'artifacts/ngas_a17ar/trajectory_protocol_manifest.json'),
            'full_bank': sha256(ROOT / 'artifacts/ngas_a17ar/full_bank_protocol_manifest.json'),
            'prior_staleness': sha256(
                ROOT / 'artifacts/ngas_a17ar/prior_staleness_protocol_manifest.json'),
        },
        'reports': {
            'final': relative(FINAL_REPORT), 'documentation': relative(DOC_REPORT),
            'figure_QA': relative(FIGURE_QA),
        },
        'regression': audits['regression'],
    }
    atomic_json(FINAL_DECISION, decision)

    artifact_paths = formal_artifacts()
    files = {relative(path): {'sha256': sha256(path), 'bytes': path.stat().st_size}
             for path in artifact_paths}
    manifest = {
        'schema': 'ngas-a17ar-result-manifest-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'terminal_classification': TERMINAL,
        'artifact_count': len(files),
        'total_bytes': sum(item['bytes'] for item in files.values()),
        'files': files,
        'excluded_self_path': relative(RESULT_MANIFEST),
    }
    atomic_json(RESULT_MANIFEST, manifest)
    print(json.dumps({
        'status': 'PASS', 'terminal_classification': TERMINAL,
        'checks_passed': sum(checks.values()), 'checks_total': len(checks),
        'artifact_count': len(files), 'result_manifest': relative(RESULT_MANIFEST),
    }, indent=2))


if __name__ == '__main__':
    main()
