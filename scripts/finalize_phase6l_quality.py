#!/usr/bin/env python3
"""Run final regression and close Phase 6L at the failed L4 quality gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6l_quality as audit  # noqa: E402


NAMESPACE = ROOT / "outputs/phase6l_legacy_score_decoupling_v1"
QUALITY_PATH = NAMESPACE / "quality/development_quality.json"
FINAL = NAMESPACE / "final"
REPORTS = ROOT / "docs/reports"
STARTING_COMMIT = "c52a37f64573d3fa822282960a6a9cbf6222f375"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace terminal evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    text = value.rstrip() + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace terminal report: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def process_exited(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return False
    except OSError:
        return True


def access_audit() -> dict:
    forbidden = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        NAMESPACE / "r13_selection/access_ledger.json",
        NAMESPACE / "r14_holdout/access_ledger.json",
    ]
    found = [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
    require(not found, f"R13/R14 access evidence exists: {found}")
    return {
        "status": "PASS_LOCKED_NOT_ACCESSED",
        "checked_paths": [str(path.relative_to(ROOT)) for path in forbidden],
        "existing_access_ledgers": found,
        "r13_accessed": False,
        "r14_accessed": False,
    }


def run_regression() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    match = re.search(r"(\d+) passed in ([0-9.]+)s", result.stdout)
    require(result.returncode == 0 and match is not None, "full repository regression failed")
    return {
        "schema": "phase6l-terminal-regression-v1",
        "status": "PASS",
        "command": f"{sys.executable} -m pytest -q",
        "passed": int(match.group(1)),
        "failed": 0,
        "seconds": float(match.group(2)),
        "stdout_tail": result.stdout.strip().splitlines()[-1],
    }


def render_quality(quality: dict) -> str:
    metrics = quality["metrics"]
    ablation = quality["ablation"]
    comparison = quality["phase6j_j1_comparison"]
    old = comparison["authoritative_existing_oof"]
    common = comparison["common_reanchored_target_diagnostic"]
    return f"""# Phase 6L L4 开发质量与消融报告

## 判定

L4 的正式判定为 **`MODEL_REVISION_QUALITY`**。primary 的 raw ranking/utility essential checks 全部通过，但 18 个冻结 gate 组合没有一个满足 Phase 6J intervention-readiness 门槛。Phase 6L 因此在 L4 终止，不进入 L5 开发运行时、L6 单次 R12 资格、bundle、solver、R13 或 R14。

这里把两个层级明确分开：raw 模型存在有效排序信号；可部署 gate 资格失败。不能用前者替代后者。

## Primary 三 seed outer OOF

| 指标 | Phase 6L |
| --- | ---: |
| States / candidates | {metrics['state_count']} / {metrics['action_count']} |
| Overall Spearman | {metrics['overall_spearman']:.6f} |
| Pairwise accuracy | {metrics['pairwise_accuracy']:.6f} |
| NDCG@1 | {metrics['ndcg_at_1']:.6f} |
| Raw selected lift | {metrics['selected_lift']:.6f} |
| Raw selected lift 95% LCB | {metrics['selected_lift_lcb']:.6f} |
| Selection regret | {metrics['selection_regret']:.6f} |
| Selected-winner ECE | {quality['calibration']['metrics'][0]['expected_calibration_error']:.6f} |
| Candidate / winner support | {quality['support']['candidate_support_rate']:.2%} / {quality['support']['selected_winner_support_rate']:.2%} |

S/M/L mean Spearman 为 {metrics['mean_spearman_by_scale']['S']:.6f} / {metrics['mean_spearman_by_scale']['M']:.6f} / {metrics['mean_spearman_by_scale']['L']:.6f}，均未发生符号反转。三项 preferred 诊断阈值（0.25 Spearman、0.60 pairwise、0.75 NDCG@1）均未达到；它们不单独决定 essential 资格。

## Gate 与 action frequency

冻结网格为 3 个 `p_min` × 2 个 `lambda` × 3 个 `delta_min`，共 18 个组合，retained 数为 **0**。因此 `selected_gate=null`，正式可部署 intervention 为 0/288；若保持 fail-closed 行为，288/288 均回退。神经 argmax 本身选择 fallback 2 次、非 fallback 286 次，这不代表通过 gate。

`p_min=0.55` 的组合产生 42 次 intervention，S/M/L 为 19/13/10，低于每 scale 20 的冻结下限，且 M lift 为 -0.000233。`p_min=0.65` 产生 20 次，S/M/L 为 11/5/4；lift 非负但 coverage 仍失败。`p_min=0.75` 仅 2 次且总体 lift 为负。所有 forced-abstention exception 均失败。

失败项为：`retained_gate_with_scale_coverage`、`positive_gated_lift_lcb`、`nonnegative_gated_scale_lift`。完整网格保存在 `outputs/phase6l_legacy_score_decoupling_v1/training/gate_grid.csv`。

## 与冻结 Phase 6J J1 对照

| 指标 | Phase 6J J1 | Phase 6L score-free |
| --- | ---: | ---: |
| Overall Spearman | {old['metrics']['overall_spearman']:.6f} | {metrics['overall_spearman']:.6f} |
| Pairwise accuracy | {old['metrics']['pairwise_accuracy']:.6f} | {metrics['pairwise_accuracy']:.6f} |
| NDCG@1 | {old['metrics']['ndcg_at_1']:.6f} | {metrics['ndcg_at_1']:.6f} |
| Raw selected lift | {old['metrics']['selected_lift']:.6f} | {metrics['selected_lift']:.6f} |
| Raw selected lift LCB | {old['metrics']['selected_lift_lcb']:.6f} | {metrics['selected_lift_lcb']:.6f} |
| Retained gates | >=1 | 0 |

Phase 6L 的 raw 指标与 J1 相近或略高，但冻结 J1 有一个 94-intervention retained gate，S/M/L 为 25/35/34，gated lift 0.004976、LCB 0.001926。将既有 J1 OOF 预测仅按 Phase 6L canonical fallback 标签重评分的诊断仍得到 {common['eligible_gate_count']} 个 retained gate，最佳组合 84 次 intervention。该诊断没有重跑 J1；10 个 fallback 改变状态仍保留历史 J1 输入上下文，因此只用于说明 gate 差异并非单纯由标签重锚定解释。

## 预注册 feature-removal 消融

单 seed 706101、不可选择的 `L1_NO_FALLBACK_CONTEXT_ABLATION` 完成 3 folds。它将 `fallback_overlap_fraction`、`fallback_jaccard`、`is_fallback` 三个模型输入置零，但保留 fallback identity 的标签和决策语义。

| 指标 | Primary seed 706101 | 移除 fallback context | 差值 |
| --- | ---: | ---: | ---: |
| Spearman | {ablation['primary_seed_706101']['overall_spearman']:.6f} | {ablation['no_fallback_context_seed_706101']['overall_spearman']:.6f} | {ablation['ablation_minus_primary']['overall_spearman']:.6f} |
| Pairwise | {ablation['primary_seed_706101']['pairwise_accuracy']:.6f} | {ablation['no_fallback_context_seed_706101']['pairwise_accuracy']:.6f} | {ablation['ablation_minus_primary']['pairwise_accuracy']:.6f} |
| NDCG@1 | {ablation['primary_seed_706101']['ndcg_at_1']:.6f} | {ablation['no_fallback_context_seed_706101']['ndcg_at_1']:.6f} | {ablation['ablation_minus_primary']['ndcg_at_1']:.6f} |
| Selected lift | {ablation['primary_seed_706101']['selected_lift']:.6f} | {ablation['no_fallback_context_seed_706101']['selected_lift']:.6f} | {ablation['ablation_minus_primary']['selected_lift']:.6f} |

fallback context 对 Spearman 与 pairwise 有小幅正贡献，但并不能解释全部 gate coverage 缺失。该消融耗时 {ablation['elapsed_seconds']:.2f} 秒，严格保持不可晋级。

## 完整性与分层证据

完整 bank 审计覆盖 288 states、6,809 个唯一 state/candidate identity；每状态请求 24 条规则，去重后候选数 21–24，全部 archived repair 均可行，每状态恰有一个 canonical fallback。S/M/L、CF1/2/3 与 search-stage 分层指标在 `quality/stratified_metrics.csv`；support 与 origin 分布分别在 `support_by_regime.csv`、`origin_selection_by_scale.csv`。

保护审计重新验证 Phase 6I/6J 6,636 个文件、Phase 6K 21,365 个文件和 139 个 tracked predecessor 文件。R13/R14 未访问。
"""


def render_runtime(quality: dict, phase6k: dict) -> str:
    return f"""# Phase 6L 运行时报告

## 状态

**L5 development runtime 与 L6 formal runtime 均未运行。** L4 已按预注册 quality gate 得出 `MODEL_REVISION_QUALITY`；协议要求在该点停止，因此没有为失败模型构建 live integration、没有计时样本、没有 deployable bundle，也没有 runtime pass 声明。

Phase 6L 已验证的数据与 feature builder 边界记录历史 scorer forward calls 为 0，但这不等同于完整 live runtime 资格。

## 继承的 Phase 6K 事实

Phase 6K E4R 在 288 states、1,440 个正式测量中 neural p90 为 {phase6k['formal_runtime']['neural_p90_ms']:.6f} ms，通过 30 ms；完整 live p90 为 {phase6k['formal_runtime']['full_live_p90_ms']:.6f} ms，未通过 100 ms。该结果说明历史前处理主导剩余运行时问题，是 Phase 6L 的起点；它不是 Phase 6L runtime 测量，也不能替 Phase 6L 通过 L5/L6。

R13/R14 保持锁定。
"""


def render_r12(quality: dict) -> str:
    return f"""# Phase 6L R12 Go/No-Go

判定：**NO-GO，`MODEL_REVISION_QUALITY`**。

Phase 6L 复用了协议已授权的冻结 R12 CAUR-FIT full-bank 数据执行 grouped outer OOF 开发审计；没有启动 L6 的单一 bundle 正式资格流程，也没有运行 R12 matched-budget solver。原因是 L4 的 18 个 gate 组合 retained 数为 {quality['eligible_gate_count']}，Phase 6J intervention-readiness 三项检查均失败。

raw OOF 的 Spearman {quality['metrics']['overall_spearman']:.6f}、selected lift {quality['metrics']['selected_lift']:.6f} 与 LCB {quality['metrics']['selected_lift_lcb']:.6f} 为正，但不能替代 retained gate、scale coverage 与 gated-lift 要求。

没有 deployable bundle；R13/R14 不得开放。
"""


def render_final(quality: dict, phase6k: dict, regression: dict, access: dict,
                 protected: dict, head: str, launch: dict) -> str:
    metrics = quality["metrics"]
    return f"""# Phase 6L Legacy-Score Decoupling 最终报告

## 最终决策

**`MODEL_REVISION_QUALITY`**，停止边界为 **L4、早于 development runtime 与新的 R12 qualification**。

唯一 primary `L1_SCORE_FREE_CONT_FROZEN` 完成 3 seeds × 3 outer folds，完整性审计通过。它在不使用历史 score 输入的条件下保留了正的 raw 排序与 utility 信号：Spearman {metrics['overall_spearman']:.6f}，selected lift {metrics['selected_lift']:.6f}，grouped-bootstrap LCB {metrics['selected_lift_lcb']:.6f}，ECE {quality['calibration']['metrics'][0]['expected_calibration_error']:.6f}。但是冻结 gate 网格 retained 数为 0，无法形成满足每 scale coverage 与 gated-lift 条件的干预策略，因此不具备晋级资格。

## 证据边界

- 继承的 Phase 6K：E4R neural p90 {phase6k['formal_runtime']['neural_p90_ms']:.6f} ms 通过，完整 live p90 {phase6k['formal_runtime']['full_live_p90_ms']:.6f} ms 失败；Phase 6K 终态为 `MODEL_REVISION_RUNTIME`。
- Phase 6L 新证据：288 states、6,809 candidates、三 seed OOF、单 seed不可选择消融、full-bank/support/origin/分层与 Phase 6J J1 对照均已完成。
- L5/L6 runtime：未授权执行，因为 L4 quality 失败。
- solver：未运行；deployable bundle 未创建。
- R13/R14：{access['status']}。

冻结 Phase 6J J1 与 score-free primary 的 raw 指标接近；J1 保有覆盖 S/M/L 的 retained gate，而 score-free 模型没有。诊断性 canonical-fallback 重评分仍保留 J1 gate，说明差异不能只归因于 10 个状态的标签重锚定。下一轮应作为全新、预注册的模型修订研究 score-free confidence/support 表征；不得在本轮调 gate 网格、seed 或 coverage 下限。

## 完整性与复现

完整回归：**{regression['passed']} passed in {regression['seconds']:.2f} s**。保护审计通过：Phase 6I/6J {protected['phase6i_phase6j']['files']:,} files，Phase 6K {protected['phase6k']['files']:,} files，tracked predecessor {protected['tracked_predecessor']['files']} files。L3 worker PID {launch['pid']} 已退出；无仍在运行的 Phase 6L job。

起始 commit：`{STARTING_COMMIT}`。生成终局证据前的最新冻结 stage commit：`{head}`。最终 evidence commit 为包含本报告与 `outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json` 的后续本地 commit。
"""


def render_handoff(quality: dict, regression: dict, access: dict,
                   protected: dict, head: str, launch: dict) -> str:
    return f"""# Phase 6L 项目交接

当前终态为 **`MODEL_REVISION_QUALITY`**。Phase 6L 已在 L4 结束；不要继续 L5/L6、solver、R13 或 R14。

## 当前边界

- Starting commit：`{STARTING_COMMIT}`
- L0：`15d466fc539f2e9db0d72ffaef95f153d235bed4`
- L1：`875e76b2a931646b8569fa3a7752f5aca70a183b`
- L2：`a8adc02`
- L3 launcher：`48ec809b3c1dd430cd741c19faa0bc9af275d06a`
- L3 evidence：`45de6a8`
- Terminal bundle 前 HEAD：`{head}`
- Ending evidence commit：包含本交接与 terminal JSON 的本地 commit；用 `git log -1 --oneline` 获取其不可自引用哈希。

## 决策依据

primary raw essential checks 全通过，但 retained gate 为 0。失败项是 `retained_gate_with_scale_coverage`、`positive_gated_lift_lcb`、`nonnegative_gated_scale_lift`。不得用 raw lift 为正覆盖 gate 失败，也不得调整冻结阈值后续跑。

关键证据：

- `outputs/phase6l_legacy_score_decoupling_v1/training/completion_integrity_audit.json`
- `outputs/phase6l_legacy_score_decoupling_v1/quality/development_quality.json`
- `outputs/phase6l_legacy_score_decoupling_v1/quality/ablation_comparison.json`
- `outputs/phase6l_legacy_score_decoupling_v1/quality/phase6j_j1_comparison.json`
- `outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json`

## 锁定与环境

回归为 {regression['passed']} passed in {regression['seconds']:.2f} s。保护审计验证 {protected['phase6i_phase6j']['files']:,} + {protected['phase6k']['files']:,} predecessor output files 和 {protected['tracked_predecessor']['files']} 个 tracked files。{access['status']}。PID {launch['pid']} 已退出，没有后台任务、checkpoint/resume 或 ETA。

后续只能从新的科学假设与新的预注册边界开始。建议重点研究 score-free winner support/calibration，而不是降低 coverage 门槛；本轮所有 R12 OOF 和失败 gate 结果必须保持冻结。

## 新会话开场提示

```text
阅读 docs/reports/phase6l_project_handoff.md 与 outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json。Phase 6L 已以 MODEL_REVISION_QUALITY 在 L4 终止，R13/R14 锁定。先审计当前 git status、保护 manifests 与终态哈希，再分析一个全新且需单独预注册的 score-free confidence/support 模型修订；不要复用 Phase 6L 结果调 gate 或打开 holdout。
```
"""


def main() -> None:
    quality = json.loads(QUALITY_PATH.read_text())
    require(quality["decision"] == "MODEL_REVISION_QUALITY", "L4 is not terminal quality failure")
    require(quality["eligible_gate_count"] == 0 and quality["selected_gate"] is None,
            "terminal quality evidence requires no retained gate")
    protected = audit.verify_protected_evidence()
    require(protected == quality["protected_evidence"], "protected evidence audit changed after L4")
    access = access_audit()
    launch = json.loads((NAMESPACE / "training/launch_record.json").read_text())
    require(process_exited(int(launch["pid"])), "L3 worker is still alive")
    regression = run_regression()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    phase6k_path = ROOT / "outputs/phase6k_runtime_v1/final/final_decision_v2.json"
    phase6k = json.loads(phase6k_path.read_text())
    decision = {
        "schema": "phase6l-terminal-decision-v1",
        "status": "COMPLETE",
        "decision": "MODEL_REVISION_QUALITY",
        "reason": "raw OOF quality has signal, but no frozen gate satisfies retained scale coverage and gated-lift readiness",
        "stop_boundary": "L4_BEFORE_DEVELOPMENT_RUNTIME_AND_R12_QUALIFICATION",
        "primary_model": "L1_SCORE_FREE_CONT_FROZEN",
        "primary_model_qualified": False,
        "raw_essential_checks_pass": all(quality["raw_essential_checks"].values()),
        "intervention_readiness_checks_pass": all(quality["phase6j_intervention_readiness_checks"].values()),
        "failed_checks": quality["failed_checks"],
        "eligible_gate_count": 0,
        "selected_gate": None,
        "deployable_bundle": "NOT_CREATED_QUALITY_GATE_FAILED",
        "development_runtime": "NOT_RUN_STOPPED_ON_L4_QUALITY",
        "formal_r12_qualification": "NOT_RUN_STOPPED_ON_L4_QUALITY",
        "r12_solver_gate": "NOT_RUN_QUALITY_GATE_FAILED",
        "r13_accessed": False,
        "r14_accessed": False,
        "r13_locked": True,
        "r14_locked": True,
        "protected_evidence_unchanged": True,
        "background_processes": {"training_pid": launch["pid"], "training_pid_exited": True, "active_jobs": []},
        "regression": regression,
        "starting_commit": STARTING_COMMIT,
        "head_before_terminal_evidence_commit": head,
        "quality_audit_path": str(QUALITY_PATH.relative_to(ROOT)),
        "quality_audit_sha256": digest(QUALITY_PATH),
        "inherited_phase6k_decision_path": str(phase6k_path.relative_to(ROOT)),
        "inherited_phase6k_decision_sha256": digest(phase6k_path),
        "next_step": "a separately preregistered model revision focused on score-free confidence/support; do not retune this frozen result or access R13/R14",
    }
    atomic_json(FINAL / "regression.json", regression)
    atomic_json(FINAL / "access_audit.json", access)
    atomic_json(FINAL / "final_decision.json", decision)
    atomic_text(REPORTS / "phase6l_development_quality_report.md", render_quality(quality))
    atomic_text(REPORTS / "phase6l_runtime_report.md", render_runtime(quality, phase6k))
    atomic_text(REPORTS / "phase6l_r12_go_no_go.md", render_r12(quality))
    atomic_text(REPORTS / "phase6l_final_report.md", render_final(
        quality, phase6k, regression, access, protected, head, launch
    ))
    atomic_text(REPORTS / "phase6l_project_handoff.md", render_handoff(
        quality, regression, access, protected, head, launch
    ))
    print(json.dumps({
        "status": decision["status"], "decision": decision["decision"],
        "stop_boundary": decision["stop_boundary"],
        "regression": regression, "access": access["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
