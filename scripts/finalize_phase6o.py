#!/usr/bin/env python3
"""Freeze the Phase 6O terminal decision after the Route B OOF gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phase6o_top_selection as o0_audit  # noqa: E402


NAMESPACE = ROOT / "outputs/phase6o_neural_shortlist_v1"
REPORTS = ROOT / "docs/reports"
FINAL = NAMESPACE / "final"
STARTING_COMMIT = "3ec6996123737e034595236b9bea81dc1ca6817a"
OOF_GATE_COMMIT = "4b5560b1df1d7241867b3ba518c92c57f73b0035"
STAGE_COMMITS = {
    "O0_implementation": "f09710c",
    "O0_route_decision": "93ee70e",
    "O1_preregistration": "b5da2f3",
    "targeted_relabeling_implementation": "30e1875",
    "preoutcome_boundary_fix": "20fe195",
    "relabeling_recovery": "490bd8d",
    "relabeling_recovery_amendment": "ea6067a",
    "targeted_relabeling_completion": "1d06a20",
    "top_utility_training_implementation": "fcbf8e6",
    "training_protocol_freeze": "e628272",
    "training_and_oof_gate": OOF_GATE_COMMIT,
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


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


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
        "schema": "phase6o-terminal-regression-v1",
        "status": "PASS",
        "command": f"{sys.executable} -m pytest -q",
        "passed": int(match.group(1)),
        "failed": 0,
        "seconds": float(match.group(2)),
        "stdout_tail": result.stdout.strip().splitlines()[-1],
    }


def access_audit() -> dict:
    paths = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json",
        NAMESPACE / "r13_selection/access_ledger.json",
        NAMESPACE / "r14_holdout/access_ledger.json",
    ]
    existing = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    require(not existing, f"R13/R14 access evidence exists: {existing}")
    return {
        "schema": "phase6o-access-audit-v1",
        "status": "PASS_LOCKED_NOT_ACCESSED",
        "checked_paths": [str(path.relative_to(ROOT)) for path in paths],
        "existing_access_ledgers": existing,
        "r13_accessed": False,
        "r14_accessed": False,
    }


def downstream_audit() -> dict:
    paths = {
        "direct_decision": NAMESPACE / "direct_decision",
        "development_solver_pilot": NAMESPACE / "solver_pilot",
        "formal_runtime": NAMESPACE / "runtime",
        "formal_r12_solver": NAMESPACE / "r12_solver",
        "r13": NAMESPACE / "r13_selection",
        "r14": NAMESPACE / "r14_holdout",
    }
    existing = {
        name: [str(path.relative_to(ROOT)) for path in directory.rglob("*") if path.is_file()]
        for name, directory in paths.items()
        if directory.exists()
    }
    require(not existing, f"downstream Phase 6O artifacts exist after OOF fail: {existing}")
    launch = json.loads((NAMESPACE / "training/launch_record.json").read_text())
    progress = json.loads((NAMESPACE / "training/progress.json").read_text())
    pid = int(launch["pid"])
    command_line = Path(f"/proc/{pid}/cmdline")
    marker_running = command_line.exists() and "train_phase6o_top_utility.py" in (
        command_line.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    )
    require(not marker_running, "Phase 6O training process is still running")
    require(
        launch["status"] == "COMPLETE_VERIFIED"
        and progress["status"] == "COMPLETE"
        and progress["runs_complete"] == progress["runs_expected"] == 9,
        "Phase 6O completion record changed",
    )
    return {
        "schema": "phase6o-downstream-stop-audit-v1",
        "status": "PASS_STOPPED_AT_OOF_GATE",
        "absent_downstream_paths": {
            name: str(path.relative_to(ROOT)) for name, path in paths.items()
        },
        "training_pid": pid,
        "training_process_running": False,
        "training_runs_complete": 9,
        "background_job_status": "INACTIVE_COMPLETE",
    }


def preterminal_manifest() -> dict:
    paths = set(
        path
        for path in NAMESPACE.rglob("*")
        if path.is_file() and FINAL not in path.parents
    )
    paths.update(REPORTS.glob("phase6o*.md"))
    paths.add(ROOT / "configs/phase6o_neural_shortlist_v1.json")
    paths.update((ROOT / "scripts").glob("*phase6o*.py"))
    paths.update((ROOT / "tests").glob("*phase6o*.py"))
    return {
        str(path.relative_to(ROOT)): {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    }


def main() -> None:
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "Phase 6O starting commit is not an ancestor of HEAD",
    )
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", OOF_GATE_COMMIT, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "Phase 6O OOF gate commit is not an ancestor of HEAD",
    )
    completion = json.loads(
        (NAMESPACE / "training/completion_integrity_audit.json").read_text()
    )
    quality = json.loads((NAMESPACE / "quality/oof_quality_gate.json").read_text())
    require(
        completion["status"] == "PASS" and all(completion["checks"].values()),
        "Phase 6O training completion audit did not pass",
    )
    require(
        quality["status"] == "FAIL"
        and quality["decision"] == "MODEL_REVISION_TOP_UTILITY"
        and quality["failed_hard_checks"]
        == [
            "common_raw_selected_lift_at_least_phase6l",
            "common_top1_regret_at_most_phase6l",
        ],
        "Phase 6O terminal OOF decision changed",
    )
    protected = o0_audit.verify_starting_boundary()
    access = access_audit()
    downstream = downstream_audit()
    regression = run_regression()
    common = quality["metrics"]["common_original_288"]
    expanded = quality["metrics"]["expanded_864"]
    paired = quality["metrics"]["paired_improvement_same_relabel_truth"]
    preterminal_head = git("rev-parse", "HEAD")

    atomic_text(
        REPORTS / "phase6o_solver_pilot_report.md",
        f"""# Phase 6O Development Solver Pilot

状态：**未运行，OOF 质量门槛终止。**

Route B common raw selected lift 为 {common['selected_lift']:.8f}，低于冻结 Phase 6L 门槛 {quality['thresholds']['common_raw_selected_lift_min']:.8f}；common regret 为 {common['selection_regret']:.8f}，高于上限 {quality['thresholds']['common_top1_regret_max']:.8f}。同一 targeted-relabel truth 上的 paired delta 为 {paired['mean_improvement']:.8f}，并且 L、M 两个规模均为负。

因此不能使用“显著性不足本身不阻断 pilot”的例外：该例外要求 point estimate 更好且无规模实质退化，本次不满足。没有运行 3-seed、2N development solver，也没有根据 solver outcome 调整模型、epoch、阈值或候选集合。
""",
    )
    atomic_text(
        REPORTS / "phase6o_runtime_report.md",
        """# Phase 6O 运行时报告

状态：**未运行，OOF 质量门槛终止。**

冻结协议只允许在 promising development solver pilot 之后进行正式 runtime qualification。本阶段在 pilot 之前已由 `MODEL_REVISION_TOP_UTILITY` 终止，因此没有构建资格 bundle，也没有产生 neural p50/p90/p99 或 complete-live p50/p90/p99。30 ms neural gate 与 100 ms complete-live gate 均保持未测试，不能使用训练 smoke 或历史 Phase 6K 延迟替代。

R13/R14 保持锁定。
""",
    )
    atomic_text(
        REPORTS / "phase6o_r12_go_no_go.md",
        f"""# Phase 6O R12 Go/No-Go

判定：**NO-GO，`MODEL_REVISION_TOP_UTILITY`**。

Route B 完成 3 seeds × 3 whole-instance folds 的完整 OOF，但 common 288 的 lift/regret 两个冻结硬门槛失败。Expanded lift {expanded['selected_lift']:.6f}、grouped LCB {expanded['selected_lift_lcb']:.6f} 及 S/M/L lift 均为正，说明模型仍有信号；这些诊断不能覆盖 common top-selection 硬失败。

没有运行 direct-decision calibration、development solver pilot、runtime qualification 或 formal matched-budget R12。没有运行 Gurobi，没有打开 R13/R14，不能声明 `CSG-NI v1 FROZEN`。
""",
    )

    manifest = preterminal_manifest()
    final_decision = {
        "schema": "phase6o-final-decision-v1",
        "status": "TERMINAL",
        "decision": "MODEL_REVISION_TOP_UTILITY",
        "stop_stage": "ROUTE_B_OOF_QUALITY_GATE",
        "starting_commit": STARTING_COMMIT,
        "preterminal_head": preterminal_head,
        "stage_commits": STAGE_COMMITS,
        "training_completion_audit_path": "outputs/phase6o_neural_shortlist_v1/training/completion_integrity_audit.json",
        "training_completion_audit_sha256": digest(
            NAMESPACE / "training/completion_integrity_audit.json"
        ),
        "oof_quality_gate_path": "outputs/phase6o_neural_shortlist_v1/quality/oof_quality_gate.json",
        "oof_quality_gate_sha256": digest(
            NAMESPACE / "quality/oof_quality_gate.json"
        ),
        "failed_hard_checks": quality["failed_hard_checks"],
        "development_solver_pilot": "NOT_RUN_OOF_GATE_FAILED",
        "formal_runtime": "NOT_RUN_OOF_GATE_FAILED",
        "formal_r12_solver": "NOT_RUN_OOF_GATE_FAILED",
        "r13_accessed": False,
        "r14_accessed": False,
        "gurobi_run": False,
        "historical_score_calls": 0,
        "background_job_status": "INACTIVE_COMPLETE",
        "regression_passed": regression["passed"],
        "protected_predecessor_status": "PASS",
    }
    atomic_json(FINAL / "access_audit.json", access)
    atomic_json(FINAL / "downstream_stop_audit.json", downstream)
    atomic_json(FINAL / "regression.json", regression)
    atomic_json(FINAL / "preterminal_manifest.json", manifest)
    atomic_json(FINAL / "final_decision.json", final_decision)

    atomic_text(
        REPORTS / "phase6o_final_report.md",
        f"""# Phase 6O Neural Shortlist Screening and Top-Utility Alignment 最终报告

## 最终决策

**`MODEL_REVISION_TOP_UTILITY`**，停止边界为 **Route B OOF quality gate**。

O0 使用冻结 Phase 6L/6N 证据否决了 k≤6 的 Route A，并在新预注册下执行 Route B。定向重标注完成 864 states、4,786 candidates 和 14,358 additional seed rows；随后完成冻结全 Phase 6F encoder 的 3 seeds × 3 whole-instance OOF。训练完整性审计为 PASS：9/9 runs、61,323 seed-candidate predictions、20,441 ensemble candidates、所有检查点哈希、candidate identity/order、full-bank、feasibility、fold isolation 和 FP32 score-free 边界均通过。

| 范围 | Spearman | Pairwise | NDCG@1 | Raw lift | Lift LCB | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 6O common 288 | {common['spearman']:.6f} | {common['pairwise_accuracy']:.6f} | {common['ndcg_at_1']:.6f} | {common['selected_lift']:.6f} | {common['selected_lift_lcb']:.6f} | {common['selection_regret']:.6f} |
| Phase 6O expanded 864 | {expanded['spearman']:.6f} | {expanded['pairwise_accuracy']:.6f} | {expanded['ndcg_at_1']:.6f} | {expanded['selected_lift']:.6f} | {expanded['selected_lift_lcb']:.6f} | {expanded['selection_regret']:.6f} |

Common raw lift 必须不低于冻结 Phase 6L 的 {quality['thresholds']['common_raw_selected_lift_min']:.6f}，实际为 {common['selected_lift']:.6f}；common regret 必须不高于 {quality['thresholds']['common_top1_regret_max']:.6f}，实际为 {common['selection_regret']:.6f}。两项均失败。Expanded 正 lift/LCB、所有规模非负 lift 和五类 selected-origin diversity 均通过，说明本轮目标仍保留可测信号，但没有把它转化为优于冻结 Phase 6L 的 common top selection。

在同一 Phase 6O targeted-relabel truth 上，Phase 6L 的 lift 为 {paired['phase6l_selected_lift_on_phase6o_truth']:.6f}，Phase 6O paired delta 为 {paired['mean_improvement']:.6f}，18-instance bootstrap 95% CI 为 [{paired['instance_grouped_lcb']:.6f}, {paired['instance_grouped_ucb']:.6f}]。这一诊断没有改变预注册常数门槛。

## 完整性与停止边界

- 正式训练 wall time：{completion['training_seconds']:.2f} 秒；后台任务已完成并退出。
- 完整回归：**{regression['passed']} passed in {regression['seconds']:.2f} s**。
- 受保护 Phase 6N 证据及其 Phase 6L/6M/predecessor chain 复核通过，共 {protected['protected_phase6n_files']} 个 Phase 6N 文件。
- historical scorer calls = 0；Gurobi 未运行；R13/R14 未访问。
- direct decision、development pilot、runtime、formal R12 均因 OOF gate fail 而未运行。

本阶段不能继续增加 selector、放宽 lift/regret 门槛或从当前 OOF 事后挑选 seed/epoch。任何后续模型修订都需要新的科学假设与新预注册边界。
""",
    )
    atomic_text(
        REPORTS / "phase6o_project_handoff.md",
        f"""# Phase 6O 项目交接

当前终态：**`MODEL_REVISION_TOP_UTILITY`**。Starting commit 为 `{STARTING_COMMIT}`；终止证据父提交为 `{preterminal_head}`，包含本交接的最终本地提交以 `git rev-parse HEAD` 为准。

Phase 6O 已完成 O0、Route B 预注册、定向重标注、正式 OOF 训练和独立质量审计。OOF 完整性通过，但 common selected lift {common['selected_lift']:.8f} 未达到 {quality['thresholds']['common_raw_selected_lift_min']:.8f}，common regret {common['selection_regret']:.8f} 超过 {quality['thresholds']['common_top1_regret_max']:.8f}。不得运行 development solver pilot、runtime、formal R12、R13 或 R14。

关键证据：

- `outputs/phase6o_neural_shortlist_v1/training/completion_integrity_audit.json`
- `outputs/phase6o_neural_shortlist_v1/quality/oof_quality_gate.json`
- `outputs/phase6o_neural_shortlist_v1/final/final_decision.json`
- `docs/reports/phase6o_final_report.md`

阶段提交：`{json.dumps(STAGE_COMMITS, sort_keys=True)}`。

后续新会话开场提示：

> 阅读 `docs/reports/phase6o_project_handoff.md` 和 `outputs/phase6o_neural_shortlist_v1/final/final_decision.json`。Phase 6O 已冻结为 `MODEL_REVISION_TOP_UTILITY`；保留所有 predecessor/Phase 6O 证据，R13/R14 继续锁定。先基于 common top-selection lift/regret 失败提出一个新的可证伪假设和独立预注册，不得在 Phase 6O OOF 上继续调参、改门槛或运行被禁止的 solver pilot。
""",
    )
    print(json.dumps(final_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
