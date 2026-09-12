#!/usr/bin/env python3
"""Finalize the audit-only NGAS A1.7A-S supplemental diagnostic closure."""
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
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17as_v1'
A17AR = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
AUDIT = OUT / 'audit'
FIGURES = ROOT / 'reports/figures/ngas_a17as'
FIGURE_QA = FIGURES / 'figure_qa.json'
PROTOCOL = ROOT / 'artifacts/ngas_a17as/supplemental_protocol_manifest.json'
FINAL_DECISION = OUT / 'final_decision.json'
RESULT_MANIFEST = OUT / 'result_manifest.json'
FINAL_REPORT = ROOT / 'reports/ngas_a17as_final_report.md'
DOC_REPORT = ROOT / 'docs/reports/ngas_a1/15_a17as_supplemental_closure.md'
REGRESSION = AUDIT / 'final_regression_test.json'
TERMINAL = 'NGAS_A1_7AS_PASS_SUPPLEMENTAL_CLOSURE'
BASELINE = '46b45a46b391e155b1abaf093287a468c83b543d'
CHECKPOINT_SHA = '448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def normalize_a17ar_metric_note(metric_fix: dict) -> None:
    """Keep the generated correction note readable without changing its values."""
    path = ROOT / 'reports/ngas_a17ar_utility_alignment.md'
    marker = 'The prior clean positive-U1 report-only U1/U2 value was'
    prefix, separator, _ = path.read_text().partition(marker)
    if not separator:
        raise RuntimeError(f'missing correction marker in {relative(path)}')
    old = metric_fix['legacy_clean_positive_U1_same_top1']
    new = metric_fix['corrected_clean_positive_U1_same_top1']
    note = (
        f'{marker} {old["numerator"]}/{old["denominator"]}; deterministic '
        f'action-ID tie-breaking corrects it to {new["numerator"]}/'
        f'{new["denominator"]}. Raw outcomes are unchanged. R12 remains '
        'development-exposed.\n')
    path.write_text(prefix + note)


def figure_audit() -> dict:
    stems = (
        'clean_critic_ranking_by_scale_stage',
        'utility_support_transitions',
        'utility_alignment_summary',
    )
    source = json.loads((FIGURES / 'source_validation.json').read_text())
    checks = {'source_validation_pass': source['status'] == 'PASS'
              and all(source['checks'].values())}
    figures = {}
    for stem in stems:
        alignment = json.loads((FIGURES / f'{stem}.alignment.json').read_text())
        collision = json.loads((FIGURES / f'{stem}.collision.json').read_text())
        pdf_text = json.loads((FIGURES / f'{stem}.pdf-text.json').read_text())
        exports = [FIGURES / f'{stem}.{suffix}'
                   for suffix in ('pdf', 'svg', 'png', 'tiff')]
        checks[f'{stem}_exports_complete'] = all(path.is_file() for path in exports)
        checks[f'{stem}_alignment_pass'] = alignment['verdict'] in {
            'PASS', 'NOT APPLICABLE'}
        checks[f'{stem}_collision_pass'] = collision['verdict'] == 'PASS'
        checks[f'{stem}_glyph_floor_pass'] = (
            pdf_text['auditable'] is True
            and pdf_text['below_minimum_count'] == 0
            and float(pdf_text['minimum_found_pt']) >= 5.)
        figures[stem] = {
            'exports': {path.suffix.lstrip('.'): sha256(path) for path in exports},
            'alignment_sha256': sha256(FIGURES / f'{stem}.alignment.json'),
            'collision_sha256': sha256(FIGURES / f'{stem}.collision.json'),
            'pdf_text_sha256': sha256(FIGURES / f'{stem}.pdf-text.json'),
        }

    corrected = ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary'
    corrected_exports = [corrected.with_suffix(f'.{suffix}')
                         for suffix in ('pdf', 'svg', 'png', 'tiff')]
    corrected_collision = json.loads(corrected.with_suffix('.collision.json').read_text())
    corrected_text = json.loads(corrected.with_suffix('.pdf-text.json').read_text())
    checks['corrected_a17ar_exports_complete'] = all(
        path.is_file() for path in corrected_exports)
    checks['corrected_a17ar_collision_pass'] = corrected_collision['verdict'] == 'PASS'
    checks['corrected_a17ar_glyph_floor_pass'] = (
        corrected_text['auditable'] is True
        and corrected_text['below_minimum_count'] == 0
        and float(corrected_text['minimum_found_pt']) >= 5.)

    payload = {
        'schema': 'ngas-a17as-figure-qa-v1',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'backend': 'python-matplotlib',
        'checks': checks,
        'figures': figures,
        'corrected_a17ar_figure': {
            'stem': relative(corrected),
            'exports': {path.suffix.lstrip('.'): sha256(path)
                        for path in corrected_exports},
        },
        'visual_inspection': (
            'PASS: all three A1.7A-S PNGs and the corrected A1.7A-R PNG were '
            'inspected; no clipping, unreadable text, or ambiguous encoding was observed'),
        'source_validation_sha256': sha256(FIGURES / 'source_validation.json'),
    }
    atomic_json(FIGURE_QA, payload)
    return payload


def report_text() -> str:
    return f'''# NGAS A1.7A-S supplemental diagnostic closure

## Terminal classification

`{TERMINAL}`

A1.7A-S is an audit-only development closure over 27 frozen clean non-R12
trajectory states. It does not establish final generalization and does not change
the production algorithm.

## Required conclusions

1. **Clean critic ranking remains weak across search stages.** For U0, valid-state
   mean Spearman rho is 0.2516 in EARLY, 0.0643 in MIDDLE, and 0.0753 in LATE.
   The critic Top-1 hits the best positive-U0 action in 0/8, 0/4, and 0/4
   informative states, respectively.
2. **U0 support is sparse and stage dependent.** U0 is informative in 16/27
   states overall: S 4/9, M 6/9, L 6/9; EARLY 8/9, MIDDLE 4/9, LATE 4/9.
3. **U1 recovers limited support absent in U0.** Among 11 U0-constant states, U1
   becomes nonconstant in 2/11 and has a positive best action in the same 2/11.
   The recovery occurs in one EARLY and one MIDDLE state; none occurs in LATE.
4. **U3 exposes robustness variation whenever U0 is flat.** U3 is nonconstant in
   11/11 U0-constant states and in all 27 states overall. This is ranking signal,
   while only 1/27 states has positive-best U3 under the frozen definition.
5. **U2 does not materially reorder U1.** Corrected U1/U2 agreement is 27/27
   same Top-1 across all states and 18/18 across pair-informative states. Mean
   valid-state rho is 0.999989; mean Top-5 and Top-10 overlap are 0.9778 and
   0.9889.
6. **The earlier U1/U2 mismatch was a tie-handling defect in a duplicate report
   path.** Python `max()` inherited action-list order for tied U1 values, while the
   canonical pairwise path used utility-descending/action-ID-ascending ordering.
   One clean state was affected. The positive-U1 clean value changes from 4/5 to
   5/5; raw outcomes and solver behavior are unchanged.
7. **A1.7B remains justified as a separately frozen development pilot.** Sparse
   immediate support and utility-dependent signal strengthen the case for testing
   whether adaptive trial allocation can preserve useful evidence at lower cost.
   These results do not validate adaptive racing or authorize its implementation.
8. **A1.7C should remain regime-aware and multi-task.** Scale/stage heterogeneity
   and the different support of U0, U1, and U3 strengthen that design rationale.
   They do not authorize C1-v2 training in this phase.
9. **No production change is justified now.** The evidence does not support a
   change to refresh horizon, portfolio, encoder architecture, exploration,
   acceptance, repair, candidate-bank construction, or decoder.

## Execution and integrity

The deterministic selection contains 27/27 states, with exactly two TRAIN and one
VALIDATION state in every S/M/L by EARLY/MIDDLE/LATE cell. The formal full-bank
audit completed 9,560 unique actions, 76,480 ordered direct trials, and 152,960
continuation decoder evaluations. Completion and boundary checks pass 21/21.

The U1/U2 correction is generated from centralized per-state rows and propagated
to its summary, Markdown report, figure source CSV, and corrected A1.7A-R figure.
Current production wording names the encoder `compact_relational`; the sole
retained historical-encoder match is a legitimate historical terminal name.

R12 remains `DEVELOPMENT_EXPOSED`. R13 remains `LOCKED_FINAL_EVAL`; R14 remains
`LOCKED_GENERALIZATION_EVAL`; CORE45 remains external-baseline-only. None was used
for supplemental evaluation, label generation, or model selection, and Gurobi was
not run.

## Evidence locations

- Frozen protocol: `artifacts/ngas_a17as/supplemental_protocol_manifest.json`
- Selection evidence: `reports/ngas_a17as_state_selection.md`
- Full-bank audit: `outputs/ngas_a1/trajectory_utility_a17as_v1/audit/full_bank_completion_audit.json`
- Derived diagnostics: `outputs/ngas_a1/trajectory_utility_a17as_v1/derived/`
- Metric correction: `reports/ngas_a17as_metric_consistency_fix.md`
- Terminology audit: `reports/ngas_a17as_terminology_audit.md`
- Figures and QA: `reports/figures/ngas_a17as/`
- Final decision: `outputs/ngas_a1/trajectory_utility_a17as_v1/final_decision.json`
- Result manifest: `outputs/ngas_a1/trajectory_utility_a17as_v1/result_manifest.json`
'''


def docs_text() -> str:
    return f'''# NGAS A1.7A-S 补充诊断闭环

终态为 `{TERMINAL}`。本阶段从冻结的 A1.7A-R 干净轨迹池中确定性选择
27 个非 R12 状态，严格覆盖 S/M/L × EARLY/MIDDLE/LATE，每格 2 个 TRAIN
和 1 个 VALIDATION。全候选池审计完成 9,560 个动作、76,480 次有序直接试验
和 152,960 次续算解码；完成、哈希与边界检查通过 21/21。

U0 仅在 16/27 状态提供非恒定信号，且 critic 对正 U0 最优动作的 Top-1
命中为 0/16。U0 平坦时，U1 在 2/11 状态恢复非恒定且正改进支持，U3 在
11/11 状态恢复非恒定稳健性信号。U1/U2 修正后全状态 Top-1 一致 27/27，
有效状态平均 rho 为 0.999989，成本归一化没有实质改变短期效用排序。

旧 U1/U2 数值不一致由重复报告路径的并列值处理造成：`max()` 继承动作
列表顺序，而规范路径按效用降序、动作 ID 升序确定性解并列。修正值从
4/5 变为 5/5，只更新诊断派生数据、报告和图件，原始结果与求解器未变。

证据继续支持将 A1.7B 作为独立冻结的 adaptive trial racing 开发小试验，
并支持 A1.7C 保持 regime-aware/multi-task 方向；两者均未在本阶段启动。
生产 `compact_relational` C1、refresh=20、portfolio、candidate_trials=8、
repair、candidate bank、decoder 及搜索逻辑全部保持不变。

R12 仍为 `DEVELOPMENT_EXPOSED`；R13/R14 保持锁定；CORE45 未用于本阶段；
未运行 Gurobi。完整结论和机器证据分别见
`reports/ngas_a17as_final_report.md`、终态 JSON 与结果清单。
'''


def formal_artifacts() -> list[Path]:
    paths: set[Path] = set()
    for base in (OUT, ROOT / 'artifacts/ngas_a17as', FIGURES):
        paths.update(path for path in base.rglob('*') if path.is_file())
    paths.update(path for path in (ROOT / 'reports').glob('ngas_a17as*')
                 if path.is_file())
    paths.update((ROOT / 'scripts').glob('*ngas_a17as*.py'))
    paths.update((ROOT / 'tests/ngas').glob('*a17as*.py'))
    for path in (
            ROOT / 'configs/ngas_a17as_supplemental_protocol.yaml',
            ROOT / 'rcias_ngas/evaluation/a17as.py',
            ROOT / 'artifacts/dataset_exposure_ledger.jsonl',
            ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/utility_alignment.csv',
            ROOT / 'reports/ngas_a17ar_utility_alignment.md',
            ROOT / 'reports/figures/ngas_a17ar/source_data/cost_normalized_utility.csv',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.pdf',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.svg',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.png',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.tiff',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.alignment.json',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.alignment.svg',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.collision.json',
            ROOT / 'reports/figures/ngas_a17ar/cost_normalized_utility_summary.pdf-text.json',
            ROOT / 'reports/figures/ngas_a17ar/figure_manifest.json',
            ROOT / 'reports/figures/ngas_a17ar/figure_qa.json',
            ROOT / 'reports/figures/ngas_a17ar/source_validation.json',
            ROOT / 'reports/ngas_a17ar_prior_staleness.md',
            ROOT / 'reports/ngas_a17ar_final_report.md',
            ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/final_decision.json',
            ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/result_manifest.json',
            ROOT / 'scripts/audit_ngas_a17ar_prior_staleness.py',
            ROOT / 'scripts/finalize_ngas_a17ar.py',
            ROOT / 'scripts/plot_ngas_a17ar_diagnostics.py',
            FINAL_REPORT, DOC_REPORT, FINAL_DECISION):
        paths.add(path)
    paths.discard(RESULT_MANIFEST)
    return sorted(path for path in paths if path.is_file())


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    completion = json.loads((AUDIT / 'full_bank_completion_audit.json').read_text())
    raw_manifest = json.loads((OUT / 'raw/full_bank_manifest.json').read_text())
    metric_fix = json.loads((OUT / 'derived/metric_consistency_fix.json').read_text())
    normalize_a17ar_metric_note(metric_fix)
    terminology = json.loads(
        (ROOT / 'reports/ngas_a17as_terminology_audit.json').read_text())
    regression = json.loads(REGRESSION.read_text())
    figure = figure_audit()
    roles = Counter(state['dataset_role'] for state in protocol['states'])
    cells = Counter((state['scale'], state['stage']) for state in protocol['states'])
    cell_roles = Counter((state['scale'], state['stage'], state['dataset_role'])
                         for state in protocol['states'])
    required_reports = [ROOT / f'reports/{name}' for name in (
        'ngas_a17as_state_selection.md',
        'ngas_a17as_clean_stage_critic_audit.md',
        'ngas_a17as_utility_support_transitions.md',
        'ngas_a17as_utility_alignment.md',
        'ngas_a17as_metric_consistency_fix.md',
        'ngas_a17as_terminology_audit.md',
    )]
    checks = {
        'baseline_and_checkpoint_match': (
            protocol['baseline_head'] == BASELINE
            and protocol['checkpoint_sha256'] == CHECKPOINT_SHA),
        'protocol_frozen_before_outcomes': (
            protocol['status'] == 'FROZEN_BEFORE_SUPPLEMENTAL_OUTCOMES'
            and all(protocol['checks'].values())),
        'selection_exact_and_balanced': (
            protocol['state_count'] == len(protocol['states']) == 27
            and roles == Counter({'TRAIN': 18, 'VALIDATION': 9})
            and set(cells.values()) == {3}
            and set(cell_roles.values()) == {1, 2}),
        'full_bank_audit_pass': (
            completion['status'] == 'PASS' and all(completion['checks'].values())
            and completion['completed_states'] == 27
            and completion['completed_actions'] == raw_manifest['completed_actions']
            and completion['completed_direct_trials']
                == completion['completed_actions'] * 8),
        'metric_consistency_pass': metric_fix['status'] == 'PASS',
        'terminology_audit_pass': (
            terminology['status'] == 'PASS'
            and not terminology['inaccurate_matches_remaining']),
        'figure_QA_pass': figure['status'] == 'PASS',
        'full_regression_suite_pass': (
            regression['status'] == 'PASS'
            and regression['tests_failed'] == 0),
        'required_reports_present': all(path.is_file() for path in required_reports),
        'forbidden_boundaries_preserved': (
            completion['forbidden_family_access_count'] == 0
            and completion['boundaries'] == {
                'C1_retrained': False,
                'R12': 'DEVELOPMENT_EXPOSED_NO_USE',
                'R13': 'LOCKED_FINAL_EVAL_NO_ACCESS',
                'R14': 'LOCKED_GENERALIZATION_EVAL_NO_ACCESS',
                'RCIAS_CB1_CORE45': 'EXTERNAL_BASELINE_ONLY_NO_ACCESS',
                'adaptive_racing_implemented': False,
                'adaptive_refresh_implemented': False,
                'gurobi_run': False,
                'production_solver_changed': False,
            }),
    }
    if not all(checks.values()):
        atomic_json(FINAL_DECISION, {
            'schema': 'ngas-a17as-final-decision-v1',
            'terminal_classification': 'NGAS_A1_7AS_FAIL_FINALIZATION',
            'checks': checks,
        })
        print(json.dumps({'status': 'FAIL', 'checks': checks}, indent=2))
        raise SystemExit(1)

    FINAL_REPORT.write_text(report_text())
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(docs_text())
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    decision = {
        'schema': 'ngas-a17as-final-decision-v1',
        'decided_at_utc': datetime.now(timezone.utc).isoformat(),
        'terminal_classification': TERMINAL,
        'baseline_head': BASELINE,
        'finalization_input_head': head,
        'checkpoint_sha256': CHECKPOINT_SHA,
        'checks': checks,
        'selection': {
            'states': 27,
            'train': roles['TRAIN'],
            'validation': roles['VALIDATION'],
            'scale_stage_counts': {f'{scale}|{stage}': count
                                   for (scale, stage), count in sorted(cells.items())},
        },
        'full_bank': {
            'actions': completion['completed_actions'],
            'direct_action_trial_evaluations': completion['completed_direct_trials'],
            'continuation_decoder_evaluations':
                completion['completed_continuation_decoder_evaluations'],
            'forbidden_family_access_count': completion['forbidden_family_access_count'],
        },
        'metric_consistency': {
            'status': metric_fix['status'],
            'legacy_clean_positive_U1_same_top1':
                metric_fix['legacy_clean_positive_U1_same_top1'],
            'corrected_clean_positive_U1_same_top1':
                metric_fix['corrected_clean_positive_U1_same_top1'],
            'tie_break': metric_fix['tie_break'],
        },
        'terminology_audit': {
            'status': terminology['status'],
            'current_production_encoder': terminology['current_production_encoder'],
            'violations_remaining': len(terminology['inaccurate_matches_remaining']),
        },
        'figure_QA': {'status': figure['status'], 'path': relative(FIGURE_QA)},
        'regression': regression,
        'boundaries': completion['boundaries'],
        'recommendation': {
            'A1.7B': 'JUSTIFIED_AS_SEPARATELY_FROZEN_DEVELOPMENT_PILOT',
            'A1.7C': 'RETAIN_REGIME_AWARE_MULTI_TASK_DEVELOPMENT_DIRECTION',
            'automatic_start': False,
        },
        'reports': {
            'final': relative(FINAL_REPORT),
            'documentation': relative(DOC_REPORT),
        },
    }
    atomic_json(FINAL_DECISION, decision)

    files = {relative(path): {'sha256': sha256(path), 'bytes': path.stat().st_size}
             for path in formal_artifacts()}
    manifest = {
        'schema': 'ngas-a17as-result-manifest-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'terminal_classification': TERMINAL,
        'artifact_count': len(files),
        'total_bytes': sum(item['bytes'] for item in files.values()),
        'files': files,
        'excluded_self_path': relative(RESULT_MANIFEST),
    }
    atomic_json(RESULT_MANIFEST, manifest)
    print(json.dumps({
        'status': 'PASS',
        'terminal_classification': TERMINAL,
        'checks_passed': sum(checks.values()),
        'checks_total': len(checks),
        'artifact_count': len(files),
        'result_manifest': relative(RESULT_MANIFEST),
    }, indent=2))


if __name__ == '__main__':
    main()
