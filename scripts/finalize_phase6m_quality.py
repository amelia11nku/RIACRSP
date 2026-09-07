#!/usr/bin/env python3
"""Run final regression and close Phase 6M at the failed M4 quality gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6m_failure_attribution as m0_audit  # noqa: E402
from scripts import audit_phase6m_training as m3_audit  # noqa: E402
from scripts import train_phase6m_selective_confidence as training  # noqa: E402


NAMESPACE = ROOT / "outputs/phase6m_selective_confidence_v1"
QUALITY_PATH = NAMESPACE / "quality/development_quality.json"
FINAL = NAMESPACE / "final"
REPORTS = ROOT / "docs/reports"
STARTING_COMMIT = "8f0d37397af82e6a8bfdb437c88459712cdf5ec7"
STAGE_COMMITS = {
    "M0_failure_attribution": "6f99e2d",
    "M1_preregistration": "1c42559",
    "M2_implementation": "f99cdc4",
    "M3_training": "be74fe2",
    "M4_quality": "4af628e",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def access_audit() -> dict:
    paths = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        NAMESPACE / "r13_selection/access_ledger.json",
        NAMESPACE / "r14_holdout/access_ledger.json",
    ]
    existing = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    require(not existing, f"R13/R14 access evidence exists: {existing}")
    return {
        "status": "PASS_LOCKED_NOT_ACCESSED",
        "checked_paths": [str(path.relative_to(ROOT)) for path in paths],
        "existing_access_ledgers": existing,
        "r13_accessed": False, "r14_accessed": False,
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
        "schema": "phase6m-terminal-regression-v1", "status": "PASS",
        "command": f"{sys.executable} -m pytest -q",
        "passed": int(match.group(1)), "failed": 0,
        "seconds": float(match.group(2)),
        "stdout_tail": result.stdout.strip().splitlines()[-1],
    }


def phase6m_manifest() -> dict:
    paths = sorted(path for path in NAMESPACE.rglob("*") if path.is_file())
    return {
        str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in paths if FINAL not in path.parents
    }


def render_runtime(quality: dict) -> str:
    return f"""# Phase 6M 运行时报告

## 状态

**M5 development runtime 与 M6 formal runtime 均未运行。** M4 按预注册质量门槛得出 `{quality['decision']}`，协议要求在此停止。没有构建 live integration、没有生成 latency 样本、没有冻结 deployable bundle，也没有 neural 或 complete-live runtime pass 声明。

M2/M3 已确认新 feature/training 路径的 historical frozen-score online forward calls 为 0。这是依赖边界，不是完整 live latency 资格。Phase 6K 的历史运行时结果也不能替代 Phase 6M 测量。

R13/R14 保持锁定。
"""


def render_r12(quality: dict) -> str:
    raw = quality["raw_ranking"]
    return f"""# Phase 6M R12 Go/No-Go

判定：**NO-GO，`{quality['decision']}`**。

Phase 6M 使用授权的 R12 CAUR-FIT development split 完成 288-state、6,809-candidate、三 selector seed 的 nested OOF 质量审计。Raw score-free ranker 继续满足正排序与 utility 条件：Spearman {raw['overall_spearman']:.6f}，selected lift {raw['selected_lift']:.6f}，Phase 6M seed 下 grouped-bootstrap LCB {raw['selected_lift_lcb']:.6f}。

18 个预注册 gate 的 retained 数为 {quality['eligible_gate_count']}，所有组合的 intervention 数均为 0。最宽松 `lambda=0.5` 的 selector LCB 最大值仍为 {quality['selector_lcb_diagnostics']['lambda_0_5_max']:.6f}，因此没有满足 cross-scale coverage 与正 gated-lift 的策略。

没有进入 M5/M6、没有创建 deployable bundle、没有运行 matched-budget R12 solver；R13/R14 不得开放。
"""


def render_final(quality: dict, regression: dict, access: dict, protected: dict,
                 predecessor: dict, head: str, launch: dict) -> str:
    raw = quality["raw_ranking"]
    confidence = quality["confidence"]
    support = quality["support"]
    return f"""# Phase 6M Score-Free Selective Confidence 最终报告

## 最终决策

**`MODEL_REVISION_QUALITY`**，停止边界为 **M4、早于 development runtime**。

唯一 promotable family `M1_SCORE_FREE_SELECTIVE_RISK` 已完成 18 个 inner ranker runs、9 个 selector runs 和完整 nested outer OOF。它精确保留 Phase 6L 的候选、winner 与 state-level raw ranking 结果：Spearman {raw['overall_spearman']:.6f}、pairwise accuracy {raw['pairwise_accuracy']:.6f}、NDCG@1 {raw['ndcg_at_1']:.6f}、raw lift {raw['selected_lift']:.6f}。Phase 6M 预注册 bootstrap seed 得到 raw-lift LCB {raw['selected_lift_lcb']:.6f}；Phase 6L 原 seed 的存档 LCB 为 {quality['comparison']['phase6l_score_free']['raw_metrics']['selected_lift_lcb']:.6f}，两者均为正。

新的 support 表示达到了候选 {support['candidate_support_rate']:.3%}、winner {support['selected_winner_support_rate']:.3%}，消除了旧 support 对 winner 的拒绝。新的 confidence/scale 没有恢复选择能力：ECE {confidence['expected_calibration_error']:.6f}、AUROC {confidence['auroc']:.6f}、AUPRC {confidence['auprc']:.6f}、resolution {confidence['resolution']:.6f}；只有 {confidence['fraction_above']['0.55']:.3%} winner 达到 `p>=0.55`，且最宽松 selector LCB 的 288 个值全部为负。18 个 gate 因而全部 0 intervention、0 retained。

## 科学解释

Phase 6L 已证明 score-free continuation ranking 信号存在，Phase 6M 进一步证明单独改造当前 selector/support 方案仍不足以形成可部署的 cross-scale intervention。Support 修订有效，但不是当前阻塞点。主要阻塞是预测 scale 相对 mean 过大，使预注册 lower bound 全部为负；probability sharpness 也从 Phase 6L 的 {quality['comparison']['phase6l_score_free']['confidence']['probability_std']:.6f} 降到 {confidence['probability_std']:.6f}，AUROC 未提高。

下一轮必须作为新的预注册模型修订。应先诊断 outer-fold scale shift、两条 CRN outcome 对 aleatoric scale 的可识别性、selector scale 与实际误差的覆盖关系，以及仍只有 84/288 通过的 frozen immediate-utility head。若研究 archived historical score 信息，只能采用说明书允许的独立、不可晋级 offline-teacher/distillation 诊断；不得回头调本轮 gate、seed、LCB 系数或 coverage 下限。

## 完整性与停止边界

- M3 连续 worker 耗时 {quality['artifact_sha256'] and json.loads((NAMESPACE / 'training/progress.json').read_text())['elapsed_seconds']:.2f} 秒，PID {launch['pid']} 已退出，无后台 Phase 6M 任务。
- 完整回归：**{regression['passed']} passed in {regression['seconds']:.2f} s**。
- Phase 6L 保护 manifest：{protected['files']} files / {protected['bytes']:,} bytes，全部通过。
- 前代保护：Phase 6I/6J {predecessor['phase6i_phase6j']['files']:,} files，Phase 6K {predecessor['phase6k']['files']:,} files，tracked predecessor {predecessor['tracked_predecessor']['files']} files。
- historical score online forward calls：0；Gurobi：未运行；{access['status']}。
- M5/M6 runtime、bundle、R12 solver、R13、R14：均未运行。

起始 commit：`{STARTING_COMMIT}`。终局证据生成前 HEAD：`{head}`。最终 evidence commit 是包含本报告与 `outputs/phase6m_selective_confidence_v1/final/final_decision.json` 的后续本地 commit。
"""


def render_handoff(quality: dict, regression: dict, access: dict, protected: dict,
                   predecessor: dict, head: str, launch: dict) -> str:
    commits = "\n".join(f"- {stage}: `{commit}`" for stage, commit in STAGE_COMMITS.items())
    return f"""# Phase 6M 项目交接

当前终态为 **`MODEL_REVISION_QUALITY`**。Phase 6M 已在 M4 结束；不得继续 M5/M6、bundle、solver、R13 或 R14。

## Commit 与运行状态

- Starting commit：`{STARTING_COMMIT}`
{commits}
- Terminal bundle 前 HEAD：`{head}`
- Ending evidence commit：包含本交接与 terminal JSON 的本地 commit；用 `git log -1 --oneline` 获取。

M3 worker PID {launch['pid']} 已退出，没有活动 job。完整回归为 {regression['passed']} passed in {regression['seconds']:.2f} s。Phase 6L 的 {protected['files']} 个冻结文件和前代 Phase 6I/6J、Phase 6K manifests 均复核通过。{access['status']}。

## 决策依据

Raw score-free ranker 的 Spearman、所有 scale Spearman、selected lift 与 bootstrap LCB 均为正。修订 support 达到 candidate {quality['support']['candidate_support_rate']:.3%}、winner 100%。但预注册 gate 0/18 retained，最宽松 selector LCB 最大值 {quality['selector_lcb_diagnostics']['lambda_0_5_max']:.6f}，正式 intervention 为 0/288。因此 quality gate 失败，不能以 raw ranking 或 support 改善替代 intervention readiness。

关键证据：

- `outputs/phase6m_selective_confidence_v1/training/completion_integrity_audit.json`
- `outputs/phase6m_selective_confidence_v1/quality/development_quality.json`
- `outputs/phase6m_selective_confidence_v1/quality/gate_grid.csv`
- `outputs/phase6m_selective_confidence_v1/final/final_decision.json`
- `docs/reports/phase6m_final_report.md`

下一步只能从新的科学假设与新的预注册边界开始。优先诊断 selector scale、fold shift 和 immediate-utility head；不得对 Phase 6M 的 18 个 gate 做事后扩展或降低门槛。

## 新会话开场提示

```text
阅读 docs/reports/phase6m_project_handoff.md 与 outputs/phase6m_selective_confidence_v1/final/final_decision.json。Phase 6M 已以 MODEL_REVISION_QUALITY 在 M4 终止，R13/R14 锁定。先复核 git status、终态哈希与保护 manifests，再为下一轮 selector scale/fold shift、immediate-utility 或不可晋级 offline-teacher 诊断定义一个全新预注册边界；不要调整 Phase 6M 的冻结 gate。
```
"""


def main() -> None:
    quality = json.loads(QUALITY_PATH.read_text())
    require(quality["status"] == "M4_COMPLETE", "M4 quality audit is incomplete")
    require(quality["decision"] == "MODEL_REVISION_QUALITY", "M4 is not terminal quality failure")
    require(quality["eligible_gate_count"] == 0 and quality["selected_gate"] is None,
            "terminal evidence requires no retained gate")
    require(all(quality["raw_quality_checks"].values()), "raw quality integrity changed")
    require(not all(quality["formal_intervention_readiness_checks"].values()),
            "terminal evidence requires failed intervention readiness")
    config, implementation, implementation_sha256 = training.validate_boundary()
    del config
    completion = json.loads((NAMESPACE / "training/completion_integrity_audit.json").read_text())
    require(completion["status"] == "PASS", "M3 completion audit failed")
    require(completion["implementation_protocol_sha256"] == implementation_sha256,
            "M3/M4 implementation boundary changed")
    protected = m3_audit.verify_phase6l_protection(implementation)
    predecessor = m0_audit.verify_predecessor_evidence()
    access = access_audit()
    launch = json.loads((NAMESPACE / "training/launch_record.json").read_text())
    require(not m3_audit.process_alive(int(launch["pid"])), "M3 worker is still alive")
    regression = run_regression()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head.startswith(STAGE_COMMITS["M4_quality"]), "M4 commit is not current HEAD")
    manifest = phase6m_manifest()
    decision = {
        "schema": "phase6m-terminal-decision-v1", "status": "COMPLETE",
        "decision": "MODEL_REVISION_QUALITY",
        "reason": "raw score-free ranking and revised support pass, but all preregistered selector lower bounds are negative and no gate retains any intervention",
        "stop_boundary": "M4_BEFORE_DEVELOPMENT_RUNTIME",
        "primary_model": "M1_SCORE_FREE_SELECTIVE_RISK",
        "primary_model_qualified": False,
        "raw_quality_checks_pass": all(quality["raw_quality_checks"].values()),
        "intervention_readiness_checks_pass": all(
            quality["formal_intervention_readiness_checks"].values()
        ),
        "failed_checks": quality["failed_checks"],
        "eligible_gate_count": 0, "selected_gate": None,
        "deployable_bundle": "NOT_CREATED_QUALITY_GATE_FAILED",
        "development_runtime": "NOT_RUN_STOPPED_ON_M4_QUALITY",
        "formal_runtime": "NOT_RUN_STOPPED_ON_M4_QUALITY",
        "r12_solver_gate": "NOT_RUN_QUALITY_GATE_FAILED",
        "historical_score_online_forward_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False, "r14_accessed": False,
        "r13_locked": True, "r14_locked": True,
        "protected_evidence_unchanged": True,
        "background_processes": {
            "training_pid": int(launch["pid"]), "training_pid_exited": True,
            "active_jobs": [],
        },
        "regression": regression,
        "starting_commit": STARTING_COMMIT,
        "stage_commits": STAGE_COMMITS,
        "head_before_terminal_evidence_commit": head,
        "quality_audit_path": str(QUALITY_PATH.relative_to(ROOT)),
        "quality_audit_sha256": digest(QUALITY_PATH),
        "phase6m_preterminal_manifest_path": str((FINAL / "preterminal_manifest.json").relative_to(ROOT)),
        "next_step": "a separately preregistered model revision targeting selector scale, fold shift, immediate utility, or a non-promotable offline-teacher diagnostic",
    }
    atomic_json(FINAL / "preterminal_manifest.json", manifest)
    atomic_json(FINAL / "regression.json", regression)
    atomic_json(FINAL / "access_audit.json", access)
    atomic_json(FINAL / "final_decision.json", decision)
    atomic_text(REPORTS / "phase6m_runtime_report.md", render_runtime(quality))
    atomic_text(REPORTS / "phase6m_r12_go_no_go.md", render_r12(quality))
    atomic_text(REPORTS / "phase6m_final_report.md", render_final(
        quality, regression, access, protected, predecessor, head, launch
    ))
    atomic_text(REPORTS / "phase6m_project_handoff.md", render_handoff(
        quality, regression, access, protected, predecessor, head, launch
    ))
    print(json.dumps({
        "status": decision["status"], "decision": decision["decision"],
        "stop_boundary": decision["stop_boundary"],
        "regression": regression, "access": access["status"],
        "phase6m_manifest_files": len(manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
