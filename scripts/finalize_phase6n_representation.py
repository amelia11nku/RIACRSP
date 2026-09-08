#!/usr/bin/env python3
"""Close Phase 6N at the preregistered N5 representation gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6n_architecture_data as n0_audit  # noqa: E402


NAMESPACE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1"
N4_PATH = NAMESPACE / "training/completion_integrity_audit.json"
N5_PATH = NAMESPACE / "quality/raw_representation_gate.json"
FINAL = NAMESPACE / "final"
REPORTS = ROOT / "docs/reports"
STARTING_COMMIT = "8efe396e6cca44493a18c07720c4effaf2add7c1"
STAGE_COMMITS = {
    "N0_architecture_data_audit": "f38725a61e6340bd3047401d2e570fe88fcb7493",
    "N1_preregistration": "bd0b245",
    "N2_data_implementation": "c662750f5c5dc6d2986ca1a1faae8dfe07edbad7",
    "N2_data_completion": "21d5b57",
    "N3_model_implementation": "101309cfb0683626ef6e7f12be34b8087a2eee34",
    "N3_smoke_qualification": "8e96d5634fefeed3c0175b06e7d7e9b5e17acb46",
    "N4_training_and_N5_representation": "c2e7718b019aa221dabd4e5ba97e5035a18654b9",
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


def process_matches(pid: int, marker: str) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return False
    return marker in cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace")


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
        "schema": "phase6n-access-audit-v1",
        "status": "PASS_LOCKED_NOT_ACCESSED",
        "checked_paths": [str(path.relative_to(ROOT)) for path in paths],
        "existing_access_ledgers": existing,
        "r13_accessed": False,
        "r14_accessed": False,
    }


def verify_protected_evidence() -> dict:
    frozen_path = NAMESPACE / "audit/protected_phase6l_phase6m_evidence.json"
    frozen = json.loads(frozen_path.read_text())
    current = n0_audit.protected_phase6l_phase6m()
    require(frozen == current, "protected Phase 6L/6M evidence changed")
    chained = n0_audit.verify_predecessor_evidence()
    return {
        "status": "PASS",
        "phase6l_phase6m_manifest_path": str(frozen_path.relative_to(ROOT)),
        "phase6l_phase6m_manifest_sha256": digest(frozen_path),
        "phase6l_phase6m_files": len(frozen),
        "phase6l_phase6m_bytes": sum(item["bytes"] for item in frozen.values()),
        "pre_phase6l_chained_manifest_verification": chained,
    }


def run_regression() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    match = re.search(r"(\d+) passed in ([0-9.]+)s", result.stdout)
    require(result.returncode == 0 and match is not None, "full regression failed")
    return {
        "schema": "phase6n-terminal-regression-v1",
        "status": "PASS",
        "command": f"{sys.executable} -m pytest -q",
        "passed": int(match.group(1)),
        "failed": 0,
        "seconds": float(match.group(2)),
        "stdout_tail": result.stdout.strip().splitlines()[-1],
    }


def preterminal_manifest() -> dict:
    files = sorted(path for path in NAMESPACE.rglob("*") if path.is_file())
    return {
        str(path.relative_to(ROOT)): {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in files
        if FINAL not in path.parents
    }


def render_decision_report(n5: dict) -> str:
    return f"""# Phase 6N 决策质量报告

状态：**未运行，N5 表示质量门槛终止。**

N5 得出 `{n5['decision']}`。按预注册协议，只有 raw critic quality 通过后才能执行 N6 empirical residual calibration 与 direct-decision gate。因此没有拟合 residual quantile、没有选择 decision threshold、没有生成 intervention coverage 或 gated-lift 结果，也没有以 Phase 6M selector 替代该步骤。

N7 的四组 non-promotable representation ablations 同样未运行。它们不能用于事后改选已冻结 primary，也不能覆盖 N5 的 paired utility 硬失败。

R13/R14 保持锁定。
"""


def render_runtime_report(n5: dict) -> str:
    return f"""# Phase 6N 运行时报告

状态：**未运行，N5 表示质量门槛终止。**

N5 得出 `{n5['decision']}`，所以没有进入 N8 runtime qualification。没有构建 deployable bundle，也没有执行 CSG construction、one-pass RT-HGT、candidate-conditioned pooling、critic、calibration decision 或 action extraction 的正式 continuous-process timing。

因此本阶段没有 neural p90 或 complete-live p90 资格结论，Phase 6K 的历史测量也不替代 Phase 6N 测量。R13/R14 保持锁定。
"""


def render_r12_report(n5: dict) -> str:
    common = n5["metrics"]["common_original_288"]
    paired = n5["metrics"]["paired_improvement"]
    return f"""# Phase 6N R12 Go/No-Go

判定：**NO-GO，`{n5['decision']}`**。

N4 完成 3 seeds × 3 folds 的 whole-instance nested OOF。N5 common 288-state 结果为 Spearman {common['spearman']:.6f}、raw selected lift {common['selected_lift']:.6f}；相对 Phase 6L 的 paired lift 改善为 {paired['mean_improvement']:.6f}，18-instance bootstrap 95% 区间为 [{paired['instance_grouped_lcb']:.6f}, {paired['instance_grouped_ucb']:.6f}]。预注册硬门槛要求该 LCB 严格大于 0，实际未通过。

没有进入 N6 direct-decision、N8 runtime 或 N9 matched-budget R12 solver gate。没有运行 Gurobi，没有打开 R13/R14，也不能声明 `CSG-NI v1 FROZEN`。
"""


def render_final_report(
    n4: dict,
    n5: dict,
    regression: dict,
    protected: dict,
    access: dict,
    data_progress: dict,
    head: str,
) -> str:
    common = n5["metrics"]["common_original_288"]
    expanded = n5["metrics"]["expanded_864"]
    phase6l = n5["metrics"]["phase6l_common_reference"]
    paired = n5["metrics"]["paired_improvement"]
    return f"""# Phase 6N Candidate-Conditioned CSG 最终报告

## 最终决策

**`{n5['decision']}`**，停止边界为 **N5、早于 empirical calibration 与 direct-decision gate**。

Phase 6N 完成了架构/数据审计、预注册、576 个扩展状态采集、864-state candidate-conditioned CSG 训练及完整 outer OOF。N4 审计为 `{n4['status']}`：9/9 runs、864 states、20,441 candidates、61,323 outer seed-candidate rows 和相同数量的 inner-validation rows 均完整；候选身份、truth、feasibility、whole-instance isolation 与工件哈希全部通过。

## N5 科学结果

| 范围 | Spearman | Pairwise | NDCG@1 | Raw lift | Lift LCB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 6N common 288 | {common['spearman']:.6f} | {common['pairwise_accuracy']:.6f} | {common['ndcg_at_1']:.6f} | {common['selected_lift']:.6f} | {common['selected_lift_lcb']:.6f} |
| Phase 6N expanded 864 | {expanded['spearman']:.6f} | {expanded['pairwise_accuracy']:.6f} | {expanded['ndcg_at_1']:.6f} | {expanded['selected_lift']:.6f} | {expanded['selected_lift_lcb']:.6f} |
| Phase 6L common 288 | {phase6l['spearman']:.6f} | {phase6l['pairwise_accuracy']:.6f} | {phase6l['ndcg_at_1']:.6f} | {phase6l['selected_lift']:.6f} | {phase6l['selected_lift_lcb']:.6f} |

Candidate-conditioned representation 将 common Spearman 提高 {n5['metrics']['common_spearman_improvement']:.6f}，且 Spearman 与 pairwise 两项优于 Phase 6L；expanded Spearman 达到 0.25 preferred target。它没有证明主要选择效用优于 Phase 6L：paired raw-lift 改善均值 {paired['mean_improvement']:.6f}，95% 区间 [{paired['instance_grouped_lcb']:.6f}, {paired['instance_grouped_ucb']:.6f}]。唯一失败硬项为 `{n5['failed_hard_checks'][0]}`。

这说明当前表示改善了候选排序相关性，但其 top-ranked candidate 没有在冻结 common evidence 上形成可验证的增量 utility。后续研究需建立新的预注册边界，针对 ranking surrogate 与 top-selection utility 的错配、L-scale utility 回落以及 candidate-conditioned pooling/训练目标进行诊断；不得调整本轮 bootstrap、硬门槛或事后选择 repetition。

## 完整性与停止边界

- N2：576/576 新状态、13,632 candidates，wall time {data_progress['process_elapsed_seconds']:.2f} 秒。
- N4：9/9 runs，wall time {n4['training_seconds']:.2f} 秒；training PID 已退出。
- 完整回归：**{regression['passed']} passed in {regression['seconds']:.2f} s**。
- Phase 6L/6M 保护：{protected['phase6l_phase6m_files']} files / {protected['phase6l_phase6m_bytes']:,} bytes，逐文件复核通过；前代保护链通过。
- historical score online forward calls：0；Gurobi：未运行；{access['status']}。
- N6 calibration/decision、N7 ablations、N8 runtime、N9 solver、R13、R14：均未运行。

起始 commit：`{STARTING_COMMIT}`。终局证据生成前 HEAD：`{head}`。最终 evidence commit 是包含本报告及 terminal JSON 的后续本地 commit。
"""


def render_handoff(
    n5: dict,
    regression: dict,
    protected: dict,
    access: dict,
    data_launch: dict,
    training_launch: dict,
    head: str,
) -> str:
    paired = n5["metrics"]["paired_improvement"]
    commits = "\n".join(f"- {stage}: `{commit}`" for stage, commit in STAGE_COMMITS.items())
    return f"""# Phase 6N 项目交接

当前终态为 **`{n5['decision']}`**。Phase 6N 已在 N5 结束；不得继续 N6–N9、R13 或 R14。

## Commit 与运行状态

- Starting commit：`{STARTING_COMMIT}`
{commits}
- Terminal bundle 前 HEAD：`{head}`
- Ending evidence commit：包含本交接与 terminal JSON 的本地 commit；用 `git log -1 --oneline` 获取。

N2 worker PID {data_launch['pid']} 与 N4 worker PID {training_launch['pid']} 均已退出，没有活动 Phase 6N job。完整回归为 {regression['passed']} passed in {regression['seconds']:.2f} s。Phase 6L/6M 的 {protected['phase6l_phase6m_files']} 个冻结文件及更早保护链均复核通过。{access['status']}。

## 决策依据

N4 的 9-run outer OOF 工件完整。N5 common Spearman 相对 Phase 6L 提高 0.033946，但 paired raw-lift 改善为 {paired['mean_improvement']:.6f}，18-instance bootstrap LCB 为 {paired['instance_grouped_lcb']:.6f}。这是唯一失败硬项，协议要求直接判为 `MODEL_REVISION_REPRESENTATION`。

关键证据：

- `outputs/phase6n_candidate_conditioned_csg_v1/training/completion_integrity_audit.json`
- `outputs/phase6n_candidate_conditioned_csg_v1/quality/raw_representation_gate.json`
- `outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json`
- `docs/reports/phase6n_representation_quality_report.md`
- `docs/reports/phase6n_final_report.md`

下一轮只能从新的科学假设和新的预注册边界开始。优先分析相关性与 top-selection utility 的错配、L-scale 负向 paired delta，以及 pooling/训练目标是否把排序信号转化为错误的 top candidate；不得对 Phase 6N 的 gate、seed 或 bootstrap 做事后修改。

## 新会话开场提示

```text
阅读 docs/reports/phase6n_project_handoff.md 与 outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json。Phase 6N 已以 MODEL_REVISION_REPRESENTATION 在 N5 终止，N6-N9 与 R13/R14 均未运行。先复核 git status、终态哈希及保护 manifests，再为 ranking-to-selection utility 错配诊断定义一个全新预注册边界；不要调整 Phase 6N 的冻结 gate。
```
"""


def main() -> None:
    n4 = json.loads(N4_PATH.read_text())
    n5 = json.loads(N5_PATH.read_text())
    require(n4["status"] == "PASS" and all(n4["checks"].values()), "N4 audit failed")
    require(n5["status"] == "FAIL", "N5 is not a terminal failure")
    require(n5["decision"] == "MODEL_REVISION_REPRESENTATION", "unexpected N5 decision")
    require(
        n5["failed_hard_checks"] == ["paired_phase6l_lift_improvement_lcb_positive"],
        "N5 failed-check set changed",
    )
    require(n5["metrics"]["paired_improvement"]["instance_grouped_lcb"] <= 0, "N5 failure disappeared")
    require(n5["historical_score_calls"] == 0, "historical score calls detected")

    data_progress = json.loads((NAMESPACE / "data/progress.json").read_text())
    data_launch = json.loads((NAMESPACE / "data/launch_record.json").read_text())
    training_progress = json.loads((NAMESPACE / "training/progress.json").read_text())
    training_launch = json.loads((NAMESPACE / "training/launch_record.json").read_text())
    require(data_progress["status"] == "COMPLETE", "N2 is incomplete")
    require(training_progress["status"] == "COMPLETE", "N4 is incomplete")
    require(not process_matches(int(data_launch["pid"]), "run_phase6n_data_generation.py"), "N2 worker is active")
    require(not process_matches(int(training_launch["pid"]), "train_phase6n_candidate_conditioned.py"), "N4 worker is active")

    protected = verify_protected_evidence()
    access = access_audit()
    regression = run_regression()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == STAGE_COMMITS["N4_training_and_N5_representation"], "N5 commit is not current HEAD")
    manifest = preterminal_manifest()

    decision = {
        "schema": "phase6n-terminal-decision-v1",
        "status": "COMPLETE",
        "decision": "MODEL_REVISION_REPRESENTATION",
        "reason": "candidate-conditioned ranking correlation improves, but paired raw selected-lift improvement over Phase 6L has non-positive mean and LCB",
        "stop_boundary": "N5_BEFORE_EMPIRICAL_CALIBRATION_AND_DIRECT_DECISION",
        "primary_family": "N1_CANDIDATE_CONDITIONED_CSG",
        "primary_family_qualified": False,
        "n4_training_integrity": "PASS",
        "n5_representation_gate": "FAIL",
        "failed_hard_checks": n5["failed_hard_checks"],
        "empirical_calibration": "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "direct_decision_gate": "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "paper_ablations": "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "runtime_qualification": "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "deployable_bundle": "NOT_CREATED_REPRESENTATION_GATE_FAILED",
        "r12_solver_gate": "NOT_RUN_REPRESENTATION_GATE_FAILED",
        "historical_score_online_forward_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "r13_locked": True,
        "r14_locked": True,
        "protected_evidence_unchanged": True,
        "background_processes": {
            "data_pid": int(data_launch["pid"]),
            "data_pid_exited": True,
            "training_pid": int(training_launch["pid"]),
            "training_pid_exited": True,
            "active_jobs": [],
        },
        "regression": regression,
        "starting_commit": STARTING_COMMIT,
        "stage_commits": STAGE_COMMITS,
        "head_before_terminal_evidence_commit": head,
        "n4_audit_path": str(N4_PATH.relative_to(ROOT)),
        "n4_audit_sha256": digest(N4_PATH),
        "n5_gate_path": str(N5_PATH.relative_to(ROOT)),
        "n5_gate_sha256": digest(N5_PATH),
        "preterminal_manifest_path": str((FINAL / "preterminal_manifest.json").relative_to(ROOT)),
        "preterminal_manifest_files": len(manifest),
        "next_step": "a separately preregistered revision addressing ranking-to-selection utility mismatch; Phase 6N remains frozen failed evidence",
    }

    atomic_json(FINAL / "preterminal_manifest.json", manifest)
    atomic_json(FINAL / "regression.json", regression)
    atomic_json(FINAL / "protected_evidence_audit.json", protected)
    atomic_json(FINAL / "access_audit.json", access)
    atomic_json(FINAL / "final_decision.json", decision)
    atomic_text(REPORTS / "phase6n_decision_quality_report.md", render_decision_report(n5))
    atomic_text(REPORTS / "phase6n_runtime_report.md", render_runtime_report(n5))
    atomic_text(REPORTS / "phase6n_r12_go_no_go.md", render_r12_report(n5))
    atomic_text(
        REPORTS / "phase6n_final_report.md",
        render_final_report(n4, n5, regression, protected, access, data_progress, head),
    )
    atomic_text(
        REPORTS / "phase6n_project_handoff.md",
        render_handoff(n5, regression, protected, access, data_launch, training_launch, head),
    )
    print(json.dumps({
        "status": decision["status"],
        "decision": decision["decision"],
        "stop_boundary": decision["stop_boundary"],
        "regression": regression,
        "access": access["status"],
        "protected": protected["status"],
        "preterminal_manifest_files": len(manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
