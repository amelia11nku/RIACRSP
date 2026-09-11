#!/usr/bin/env python3
"""Audit frozen A1.5 timings, decide the gate, and write the required report."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/latency_qualification_v1'
PROTOCOL = OUT / 'preregistration/protocol_v2.json'
PROGRESS = OUT / 'progress.json'
REPORT = ROOT / 'docs/reports/ngas_a1/09_latency_qualification.md'
A14_RAW = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1/raw/C1'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values):
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError('Latency samples must be finite and non-empty')
    return {
        'count': len(values), 'mean': statistics.fmean(values),
        'p50': percentile(values, .50), 'p90': percentile(values, .90),
        'p99': percentile(values, .99), 'minimum': min(values), 'maximum': max(values),
    }


def summarize_raw(payload):
    complete = [row['complete_refresh'] for row in payload['component_samples_ms']]
    names = sorted(payload['component_samples_ms'][0])
    return {
        'state': payload['state'], 'mode': payload['mode'],
        'critic_count': payload['critic_count'], 'joint_actions': payload['joint_actions'],
        'complete_refresh_ms': stats(complete),
        'raw_inference_ms': stats(payload['raw_inference_samples_ms']),
        'components_ms': {
            name: stats([row[name] for row in payload['component_samples_ms']])
            for name in names
        },
    }


def historical_amortization():
    grouped = {'S': [], 'M': [], 'L': []}
    for path in sorted(A14_RAW.glob('*/seed_*.json')):
        raw = json.loads(path.read_text())
        diagnostics = raw['diagnostics']
        grouped[raw['instance']['scale']].append({
            'critic_calls': diagnostics['telemetry']['termination']['neural_calls'],
            'guided_iterations': diagnostics['guided_iterations'],
            'guided_iterations_per_call': diagnostics['guided_iterations_per_critic_call'],
            'historical_effective_neural_overhead_seconds':
                diagnostics['effective_neural_overhead_seconds'],
            'historical_complete_refresh_seconds':
                diagnostics['runtime_components']['refresh_total_seconds'],
        })
    if any(len(rows) != 3 for rows in grouped.values()):
        raise RuntimeError('Expected three frozen A1.4 C1 runs per S/M/L scale')
    return {
        scale: {
            'runs': len(rows),
            'mean_critic_calls_per_run': statistics.fmean(row['critic_calls'] for row in rows),
            'mean_guided_iterations_per_call': statistics.fmean(
                row['guided_iterations_per_call'] for row in rows),
            'mean_guided_iterations_per_run': statistics.fmean(
                row['guided_iterations'] for row in rows),
            'mean_historical_effective_neural_overhead_seconds': statistics.fmean(
                row['historical_effective_neural_overhead_seconds'] for row in rows),
            'mean_historical_complete_refresh_seconds_per_call': statistics.fmean(
                row['historical_complete_refresh_seconds'] / row['critic_calls'] for row in rows),
        }
        for scale, rows in grouped.items()
    }


def render_report(summary, audit):
    lines = [
        '# NGAS A1.5 完整在线刷新延迟资格报告', '',
        '## 冻结边界', '',
        f"- 协议 SHA256：`{summary['protocol_sha256']}`",
        f"- 实现提交：`{summary['implementation_commit']}`",
        '- 生产路径：单个 C1 critic；开发对照：三个 C1 checkpoint 的共享特征 ensemble。',
        '- 完整边界包含 CSG 更新、关键同步提取与映射、动作生成、特征更新、CPU 张量化、H2D、图编码、批量动作打分、prior、排序/采样及同步。',
        '- raw inference 仅计准备好的 GPU batch 上的模型推理，并在每次调用后同步。',
        '', '## 主要结果', '',
        '| 状态 | 操作数 | 路径 | actions | 完整 mean / p50 / p90 / p99 (ms) | raw p90 (ms) |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for label in ('S', 'M', 'L', 'L_MAX'):
        for mode in ('SINGLE_C1', 'TEACHER_ENSEMBLE'):
            row = summary['results'][label][mode]
            complete, raw = row['complete_refresh_ms'], row['raw_inference_ms']
            lines.append(
                f"| {label} | {row['state']['num_operations']} | {mode} | {row['joint_actions']} | "
                f"{complete['mean']:.3f} / {complete['p50']:.3f} / {complete['p90']:.3f} / {complete['p99']:.3f} | "
                f"{raw['p90']:.3f} |")
    lines += ['', '## 单 C1 完整刷新分项 p90', '',
              '| 状态 | CSG | event DAG | critical | mapping | bank | action feature | tensor | H2D GPU | encoder GPU | scorer GPU | prior | rank/sample |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    component_names = (
        'csg_update_build', 'critical_event_graph', 'critical_extraction',
        'critical_mapping', 'candidate_joint_action_generation', 'action_feature_update',
        'cpu_tensorization', 'h2d_gpu', 'graph_encoding_gpu',
        'batched_action_scoring_gpu', 'prior_construction', 'ranking_sampling')
    for label in ('S', 'M', 'L', 'L_MAX'):
        row = summary['results'][label]['SINGLE_C1']['components_ms']
        lines.append('| ' + label + ' | ' + ' | '.join(
            f"{row[name]['p90']:.3f}" for name in component_names) + ' |')
    lines += [
        '',
        'GPU event 分项与 `output_and_synchronization_wall` 会在完整 wall time 内重叠，因此分项不能直接相加。完整延迟以 `complete_refresh` wall time 为准。',
        '', '## 搜索摊销', '',
        '| 尺度 | A1.4 critic calls/run | guided iterations/call | 旧完整 refresh/call (ms) | 新完整 refresh mean (ms) | 新完整成本/guided iteration (ms) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for label, scale in (('S', 'S'), ('M', 'M'), ('L', 'L'), ('L_MAX', 'L')):
        old = summary['historical_A1_4_amortization'][scale]
        new = summary['results'][label]['SINGLE_C1']['complete_refresh_ms']['mean']
        lines.append(
            f"| {label} | {old['mean_critic_calls_per_run']:.2f} | "
            f"{old['mean_guided_iterations_per_call']:.2f} | "
            f"{1000 * old['mean_historical_complete_refresh_seconds_per_call']:.3f} | "
            f"{new:.3f} | {new / old['mean_guided_iterations_per_call']:.3f} |")
    lines += [
        '', '## 判定', '',
        f"- 冻结门槛：每个 S/M/L/L_MAX 状态及 pooled 单 C1 完整刷新 `p90 <= {summary['latency_threshold_ms']:.1f} ms`。",
        f"- pooled 单 C1 p90：`{summary['pooled_single_C1_complete_refresh_ms']['p90']:.3f} ms`。",
        f"- 逐状态门槛：`{json.dumps(summary['per_state_latency_gate'], sort_keys=True)}`。",
        f"- A1.5 是否通过：`{str(summary['A1_5_pass']).lower()}`。",
        f"- 终止判定：`{summary['terminal_decision']}`。",
        '- A1.6 保持锁定；R13、R14 未访问，未运行 Gurobi。',
        '',
        '单 C1 raw forward 已处于毫秒级；剩余瓶颈主要位于随规模增长的 CSG、动作 bank、动作特征和张量构造。由于正式冻结结果未通过 30 ms 完整边界，不在本协议结果后追加 distillation、AMP 或有利重跑。',
        '', '## 审计', '',
        f"- 完成审计：`{audit['status']}`",
        f"- 原始结果：`{audit['raw_file_count']}/8`",
        f"- 原始 manifest SHA256：`{audit['raw_manifest_sha256']}`",
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha256 = digest(PROTOCOL)
    state_manifest = json.loads((ROOT / protocol['representative_states_path']).read_text())
    expected = [(state['label'], mode) for state in state_manifest['states']
                for mode in ('SINGLE_C1', 'TEACHER_ENSEMBLE')]
    payloads, manifest = {}, []
    errors = []
    for label, mode in expected:
        path = OUT / 'raw' / label / f'{mode}.json'
        if not path.is_file():
            errors.append(f'missing:{path.relative_to(ROOT)}')
            continue
        raw = json.loads(path.read_text())
        if raw.get('protocol_sha256') != protocol_sha256:
            errors.append(f'protocol:{path.relative_to(ROOT)}')
        if not raw.get('all_numeric_values_finite'):
            errors.append(f'nonfinite:{path.relative_to(ROOT)}')
        payloads[label, mode] = raw
        manifest.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    if errors or len(payloads) != 8:
        raise RuntimeError(f'A1.5 raw result audit failed: {errors}')
    source_checks = {
        path: digest(ROOT / path) == expected_hash
        for path, expected_hash in protocol['source_hashes'].items()
    }
    manifest_payload = {
        'schema': 'ngas-a15-raw-result-manifest-v1',
        'protocol_sha256': protocol_sha256,
        'files': sorted(manifest, key=lambda row: row['path']),
    }
    manifest_path = OUT / 'result_manifest.json'
    atomic_json(manifest_path, manifest_payload)
    results = {
        label: {mode: summarize_raw(payloads[label, mode])
                for mode in ('SINGLE_C1', 'TEACHER_ENSEMBLE')}
        for label in ('S', 'M', 'L', 'L_MAX')
    }
    production_samples = [
        row['complete_refresh']
        for label in ('S', 'M', 'L', 'L_MAX')
        for row in payloads[label, 'SINGLE_C1']['component_samples_ms']
    ]
    pooled = stats(production_samples)
    threshold = float(protocol['latency_gate']['threshold_ms'])
    per_state_gate = {
        label: results[label]['SINGLE_C1']['complete_refresh_ms']['p90'] <= threshold
        for label in ('S', 'M', 'L', 'L_MAX')
    }
    latency_pass = all(per_state_gate.values()) and pooled['p90'] <= threshold
    historical = historical_amortization()
    decision = 'NGAS_A1_5_PASS' if latency_pass else 'NGAS_A1_REVISE_RUNTIME'
    summary = {
        'schema': 'ngas-a15-latency-summary-v1',
        'protocol_sha256': protocol_sha256,
        'implementation_commit': protocol['implementation_commit'],
        'results': results,
        'pooled_single_C1_complete_refresh_ms': pooled,
        'per_state_latency_gate': per_state_gate,
        'latency_threshold_ms': threshold,
        'historical_A1_4_amortization': historical,
        'A1_5_pass': latency_pass,
        'A1_6': 'UNLOCKED' if latency_pass else 'LOCKED',
        'terminal_decision': decision,
    }
    atomic_json(OUT / 'latency_summary.json', summary)
    checks = {
        'A1_4_completion_audit_pass': json.loads(
            (ROOT / protocol['A1_4_completion_audit_path']).read_text())['status'] == 'PASS',
        'protocol_frozen_before_formal_results': all(
            raw['formal_started_at_utc'] > protocol['frozen_at_utc'] for raw in payloads.values()),
        'protocol_and_state_hashes_match':
            digest(ROOT / protocol['config_path']) == protocol['config_sha256']
            and digest(ROOT / protocol['representative_states_path'])
                == protocol['representative_states_sha256'],
        'source_hashes_match': all(source_checks.values()),
        'raw_scope_exact_8': len(list((OUT / 'raw').glob('*/*.json'))) == 8,
        'raw_manifest_hashes_match': all(
            digest(ROOT / row['path']) == row['sha256'] for row in manifest_payload['files']),
        'all_samples_finite': all(raw['all_numeric_values_finite'] for raw in payloads.values()),
        'production_repetitions_exact': all(
            payloads[label, 'SINGLE_C1']['complete_refresh_repetitions']
                == protocol['measurement']['production_complete_repetitions']
            for label in ('S', 'M', 'L', 'L_MAX')),
        'ensemble_repetitions_exact': all(
            payloads[label, 'TEACHER_ENSEMBLE']['complete_refresh_repetitions']
                == protocol['measurement']['ensemble_complete_repetitions']
            for label in ('S', 'M', 'L', 'L_MAX')),
        'R13_R14_locked_and_no_gurobi': all(
            raw['locks'] == {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False}
            for raw in payloads.values()),
    }
    audit = {
        'schema': 'ngas-a15-completion-audit-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks, 'source_checks': source_checks,
        'protocol_sha256': protocol_sha256,
        'raw_file_count': len(manifest),
        'raw_manifest_sha256': digest(manifest_path),
        'latency_gate_pass': latency_pass,
        'per_state_latency_gate': per_state_gate,
        'pooled_p90_ms': pooled['p90'], 'threshold_ms': threshold,
        'A1_6': 'UNLOCKED' if latency_pass else 'LOCKED',
        'terminal_decision': decision,
        'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False},
    }
    atomic_json(OUT / 'audit/completion_audit.json', audit)
    atomic_json(OUT / 'final_decision.json', {
        'schema': 'ngas-a15-final-decision-v1',
        'A1_5_pass': latency_pass, 'A1_6': audit['A1_6'],
        'terminal_decision': decision,
        'reason': ('complete live single-C1 refresh p90 satisfied the frozen 30 ms gate'
                   if latency_pass else
                   'one or more representative complete live single-C1 refresh p90 values exceeded the frozen 30 ms gate'),
        'protocol_sha256': protocol_sha256,
        'completion_audit_path': 'outputs/ngas_a1/latency_qualification_v1/audit/completion_audit.json',
    })
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(summary, audit))
    atomic_json(PROGRESS, {
        'schema': 'ngas-a15-latency-progress-v1',
        'status': 'COMPLETE', 'completed_units': 8, 'expected_units': 8,
        'protocol_sha256': protocol_sha256,
        'decision': decision,
        'next_gate': 'STOP_RUNTIME_REVISION_REQUIRED' if not latency_pass else 'A1_6',
        'A1_6': audit['A1_6'], 'R13': 'LOCKED', 'R14': 'LOCKED',
        'gurobi_run': False,
    })
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
