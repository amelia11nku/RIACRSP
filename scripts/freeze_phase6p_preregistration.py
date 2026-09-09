#!/usr/bin/env python3
"""Freeze Phase 6P policy, critic, instance, and command boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import grouped_oof_fold  # noqa: E402


CONFIG = ROOT / "configs/phase6p_adaptive_portfolio_v1.json"
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/preregistration"
REPORT = ROOT / "docs/reports/phase6p_preregistered_protocol.md"
EXPECTED_HEAD = "f18896fdf05884a5507174392aa166ef65350058"
PYTHON = "/home/liulei/miniconda3/envs/gnn311/bin/python"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def checkpoint_manifest() -> dict:
    training = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training"
    protocol = load_json(training / "training_protocol.json")
    runs = []
    for seed in protocol["training"]["seeds"]:
        for held_fold in range(3):
            path = training / f"oof/seed_{seed}/fold_{held_fold}.pt"
            record_path = path.with_suffix(".json")
            record = load_json(record_path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            actual = digest(path)
            require(actual == record["checkpoint_sha256"], f"checkpoint drift: {path}")
            require(
                checkpoint["schema"] == "phase6n-oof-checkpoint-v1"
                and checkpoint["held_fold"] == held_fold
                and checkpoint["training_seed"] == seed,
                f"checkpoint contract drift: {path}",
            )
            runs.append({
                "training_seed": seed,
                "held_fold": held_fold,
                "path": str(path.relative_to(ROOT)),
                "sha256": actual,
                "feature_transform_sha256": hashlib.sha256(
                    json.dumps(
                        checkpoint["feature_transform"], sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            })
    require(len(runs) == 9, "Phase 6P requires nine Phase 6N OOF checkpoints")
    return {
        "schema": "phase6p-phase6n-checkpoint-manifest-v1",
        "status": "FROZEN_BEFORE_PHASE6P_SOLVER_OUTCOMES",
        "family": "N1_CANDIDATE_CONDITIONED_CSG",
        "ensemble": "arithmetic mean of three seed predictions within routed held fold",
        "fold_route": {
            f"{scale}_{cf}": grouped_oof_fold(scale, cf)
            for scale in ("S", "M", "L")
            for cf in ("CF1", "CF2", "CF3")
        },
        "checkpoints": runs,
        "phase6n_training_protocol_sha256": digest(training / "training_protocol.json"),
        "phase6n_completion_audit_sha256": digest(training / "completion_integrity_audit.json"),
        "phase6n_final_decision_sha256": digest(
            ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json"
        ),
        "qualification_boundary": "frozen OOF critic evidence; no Phase 6N deployable bundle was created",
        "whole_instance_routing": True,
        "historical_score_calls": 0,
        "precision": "FP32_ONLY",
    }


def instance_manifest(config: dict) -> dict:
    source_path = (
        ROOT
        / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_instance_manifest.csv"
    )
    source = pd.read_csv(source_path)
    fit = source[source.caur_split.eq("CAUR_FIT")].copy().sort_values("instance_id")
    require(len(fit) == 18 and set(fit.replicate) == {"R12"}, "R12 FIT scope changed")
    root = ROOT / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14"
    records = []
    for row in fit.itertuples(index=False):
        path = root / str(row.relative_path)
        require(digest(path) == row.sha256, f"instance hash drift: {row.instance_id}")
        records.append({
            "instance_id": str(row.instance_id),
            "scale": str(row.scale),
            "CF_level": str(row.CF_level),
            "cell_replicate": str(row.cell_replicate),
            "relative_path": str(row.relative_path),
            "sha256": str(row.sha256),
            "num_operations": len(load_json(path)["operations"]),
        })
    return {
        "schema": "phase6p-development-instance-manifest-v1",
        "status": "FROZEN_BEFORE_PHASE6P_SOLVER_OUTCOMES",
        "split": "R12_CAUR_FIT",
        "source_manifest_path": str(source_path.relative_to(ROOT)),
        "source_manifest_sha256": digest(source_path),
        "instances": records,
        "instance_count": len(records),
        "development_seeds": config["development"]["seeds"],
        "formal_seeds": config["formal_r12"]["canonical_seeds"],
        "budget": "2 * num_operations seconds",
    }


def command_manifest(config: dict) -> dict:
    commands = [
        {
            "stage": "P2_smoke",
            "command": f"{PYTHON} scripts/run_phase6p_smoke.py",
        },
        {
            "stage": "P3_comparator_reuse_audit",
            "command": f"{PYTHON} scripts/audit_phase6p_comparator_reuse.py",
        },
        *[
            {
                "stage": f"P3_development_{method}",
                "command": f"{PYTHON} scripts/run_phase6p_development.py --method {method}",
            }
            for method in config["development"]["methods"]
        ],
        {
            "stage": "P3_development_summary",
            "command": f"{PYTHON} scripts/summarize_phase6p_development.py",
        },
    ]
    return {
        "schema": "phase6p-command-manifest-v1",
        "working_directory": str(ROOT),
        "python": PYTHON,
        "commands": commands,
        "comparator_guard": "do not execute any comparator before comparator_reuse_audit",
    }


def source_hashes() -> dict:
    paths = (
        Path("configs/phase6p_adaptive_portfolio_v1.json"),
        Path("configs/phase5c_alns.json"),
        Path("configs/lghga_2o_baseline.json"),
        Path("rcias_clgri/search/alns.py"),
        Path("rcias_clgri/search/csgni.py"),
        Path("rcias_clgri/search/phase6c.py"),
        Path("rcias_clgri/search/lghga_2o.py"),
        Path("rcias_clgri/ni/phase6n_candidate_conditioned.py"),
        Path("scripts/train_phase6n_candidate_conditioned.py"),
        Path("outputs/phase6p_adaptive_portfolio_v1/audit/live_search_integration.json"),
        Path("outputs/phase6p_adaptive_portfolio_v1/audit/candidate_bank_semantics.json"),
        Path("outputs/phase6p_adaptive_portfolio_v1/audit/multi_origin_target_audit.csv"),
        Path("outputs/phase6n_candidate_conditioned_csg_v1/training/training_protocol.json"),
        Path("outputs/phase6n_candidate_conditioned_csg_v1/training/completion_integrity_audit.json"),
        Path("outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json"),
        Path("outputs/phase6h_calibration/frozen/phase6h_policy.json"),
        Path("outputs/phase6k_runtime_v1/audit/lghga_2o_real_clock_smoke.json"),
    )
    return {
        "schema": "phase6p-preregistered-source-hashes-v1",
        "files": {str(path): digest(ROOT / path) for path in paths},
    }


def write_report(protocol: dict, instances: dict, checkpoints: dict) -> None:
    config = protocol["frozen_policy_summary"]
    text = f"""# Phase 6P P1 预注册协议

状态：**`{protocol['status']}`**。本协议冻结于任何 Phase 6P solver-quality outcome 之前，P0 commit 为 `{protocol['head_at_freeze']}`。R13/R14 保持锁定，禁止 Gurobi 和新神经训练。

## 主策略

`P1_CSG_ADAPTIVE_PORTFOLIO` 沿用 H1 初始化与 Phase 6H 的确定性 20% neural eligibility；非 eligible iteration 保持原 ALNS。每次神经决策生成完整 24-rule bank 并按 destroyed-operation set 去重，对所有 unique targets 使用冻结 Phase 6N critic 评分，然后按预测 continuation advantage 排序，target ID 作为固定 tie-break，保留 top-6。

候选权重为 `1/r × mapped destroy weight`。P0 在 864 states 中发现 2 个跨 destroy-operator 去重 target，因此 mapped weight 冻结为全部唯一来源 destroy 权重的算术均值；target 产生真实结果后，每个唯一来源 operator 各执行一次原 ALNS 更新。repair 独立按五个当前权重 roulette，保留 8 trials、模拟退火、5/1/0.1-floor 奖励和 `reaction_factor=0.2`。

canonical related fallback 始终保留在 critic 的 fallback-relative 上下文中；它在 top-6 时正常参与采样，评分不完整、非有限或在采样前异常时作为安全回退。主策略没有 confidence、immediate-utility 或 support hard gate。

## 冻结 critic

Phase 6N critic 由 **{len(checkpoints['checkpoints'])} 个 whole-instance OOF checkpoints** 组成。实例按 `Scale × CF` 路由到 held fold，再对三个训练 seed 的 `predicted_continuation_advantage` 取算术平均。模型严格 FP32，历史 scorer 与 Phase 6O critic 调用均为 0。该对象只作为冻结 OOF critic 证据使用；Phase 6N 的 N5 失败及未创建 deployable bundle 的结论不变。

## 开发与正式边界

开发范围冻结为全部 **{instances['instance_count']} 个 R12 CAUR-FIT instances**，三 seeds 为 `{config['development_seeds']}`；所有方法使用包含初始化和全部 live overhead 的 `2*|O|` 总 wall-clock。正式 R12 使用同一实例集与五 seeds `{config['formal_seeds']}`，仅在开发和 runtime gates 通过后运行。

任何 comparator 执行前必须先建立 standing `outputs/frozen_2o_baselines/` registry 并产出 reuse audit。已存在结果只有在实例、代码、checkpoint、配置、seed、预算、计时边界、decoder、指标和环境全部相同时才可复用；禁止有利重跑。

P2 只能做 tiny smoke 与不变量验证。P2 通过后直接进入 P3，不增加 surrogate/OOF quality gate。runtime 只在 P3 promising 后执行，门槛仍为 neural p90 ≤30 ms、complete-live p90 ≤100 ms。

机器冻结证据位于 `outputs/phase6p_adaptive_portfolio_v1/preregistration/`。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    require(git("rev-parse", "HEAD") == EXPECTED_HEAD, "P1 must freeze on the P0 commit")
    status_paths = {
        line[3:] for line in git("status", "--porcelain").splitlines() if line
    }
    require(
        status_paths.issubset({
            "configs/phase6p_adaptive_portfolio_v1.json",
            "scripts/freeze_phase6p_preregistration.py",
        }),
        f"unexpected worktree changes before P1 freeze: {sorted(status_paths)}",
    )
    config = load_json(CONFIG)
    require(
        config["status"] == "PREREGISTERED_BEFORE_ANY_PHASE6P_SOLVER_QUALITY_OUTCOME"
        and config["starting_commit"] == EXPECTED_HEAD,
        "Phase 6P config boundary changed",
    )
    require(config["development"]["seeds"] == [746101, 746102, 746103], "development seeds changed")
    require(config["formal_r12"]["canonical_seeds"] == [746101, 746102, 746103, 746104, 746105], "formal seeds changed")
    require(config["primary"]["candidate_bank"]["shortlist_k"] == 6, "shortlist k changed")
    require(config["primary"]["repair"]["candidate_trials"] == 8, "candidate trials changed")

    checkpoints = checkpoint_manifest()
    instances = instance_manifest(config)
    commands = command_manifest(config)
    sources = source_hashes()
    atomic_json(OUT / "phase6n_checkpoint_manifest.json", checkpoints)
    atomic_json(OUT / "development_instance_manifest.json", instances)
    atomic_json(OUT / "command_manifest.json", commands)
    atomic_json(OUT / "source_hashes.json", sources)
    protocol = {
        "schema": "phase6p-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_PHASE6P_SOLVER_QUALITY_OUTCOME",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "head_at_freeze": EXPECTED_HEAD,
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": digest(CONFIG),
        "manual_sha256": "f1fb0a7eaafdbcd82afc6d68d713d973dcc2f831f07a0e5a13304796d7084040",
        "checkpoint_manifest_sha256": digest(OUT / "phase6n_checkpoint_manifest.json"),
        "instance_manifest_sha256": digest(OUT / "development_instance_manifest.json"),
        "command_manifest_sha256": digest(OUT / "command_manifest.json"),
        "source_hashes_sha256": digest(OUT / "source_hashes.json"),
        "freeze_script_sha256": digest(Path(__file__)),
        "frozen_policy_summary": {
            "primary": config["primary_family"],
            "top_k": 6,
            "rank_prior": "1/r",
            "cross_origin_destroy_weight": "mean",
            "candidate_trials": 8,
            "neural_eligibility_percent": 20,
            "development_seeds": config["development"]["seeds"],
            "formal_seeds": config["formal_r12"]["canonical_seeds"],
            "budget": config["development"]["budget_seconds"],
        },
        "solver_quality_outcomes_observed": False,
        "optimizer_steps_started": False,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT / "preregistration.json", protocol)
    write_report(protocol, instances, checkpoints)
    print(json.dumps({
        "status": protocol["status"],
        "checkpoints": len(checkpoints["checkpoints"]),
        "instances": instances["instance_count"],
        "development_seeds": config["development"]["seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
