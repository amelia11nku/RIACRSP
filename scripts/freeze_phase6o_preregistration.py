#!/usr/bin/env python3
"""Freeze the Phase 6O Route B protocol before relabeling or optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6o_neural_shortlist_v1.json"
AUDIT = ROOT / "outputs/phase6o_neural_shortlist_v1/audit/loss_alignment_audit.json"
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/preregistration"
REPORT = ROOT / "docs/reports/phase6o_preregistered_protocol.md"
PHASE6N = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training/ensemble_oof.parquet"
PHASE6L = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet"
EXPECTED_O0_COMMIT = "93ee70e"


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
        raise RuntimeError(f"refusing to replace frozen Phase 6O protocol: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace frozen Phase 6O candidate union: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def ranked_top(group: pd.DataFrame, k: int) -> list[str]:
    return group.sort_values(
        ["ensemble_advantage_mean", "target_set_id"],
        ascending=[False, True],
        kind="stable",
    ).target_set_id.astype(str).head(k).tolist()


def build_candidate_union(config: dict) -> pd.DataFrame:
    phase6n = pd.read_parquet(PHASE6N)
    phase6l = {
        str(state): group for state, group in pd.read_parquet(PHASE6L).groupby("state_id")
    }
    rows = []
    for state_id, group in phase6n.groupby("state_id", sort=True):
        roles: dict[str, set[str]] = {}
        for target in ranked_top(group, 4):
            roles.setdefault(target, set()).add("PHASE6N_TOP4")
        if str(state_id) in phase6l:
            for target in ranked_top(phase6l[str(state_id)], 4):
                roles.setdefault(target, set()).add("PHASE6L_TOP4")
        fallback = group[group.is_fallback.astype(bool)]
        require(len(fallback) == 1, f"fallback mismatch: {state_id}")
        fallback_id = str(fallback.target_set_id.iloc[0])
        roles.setdefault(fallback_id, set()).add("CANONICAL_FALLBACK")
        known = set(group.target_set_id.astype(str))
        require(set(roles).issubset(known), f"candidate union escaped full bank: {state_id}")
        for target in sorted(roles):
            source = str(group.phase6n_data_origin.iloc[0])
            rows.append({
                "state_id": str(state_id),
                "target_set_id": target,
                "instance_id": str(group.instance_id.iloc[0]),
                "phase6n_data_origin": source,
                "scale": str(group.scale.iloc[0]),
                "CF_level": str(group.CF_level.iloc[0]),
                "oof_fold": int(group.oof_fold.iloc[0]),
                "selection_roles": "|".join(sorted(roles[target])),
                "is_fallback": target == fallback_id,
            })
    union = pd.DataFrame(rows).sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    expected = config["targeted_relabeling"]["candidate_union"]
    require(union.state_id.nunique() == expected["states"], "candidate-union state count changed")
    require(len(union) == expected["state_candidate_rows"], "candidate-union row count changed")
    source_counts = union.groupby("phase6n_data_origin").size().to_dict()
    require(source_counts["ORIGINAL_PHASE6J_CAUR"] == expected["original_state_candidate_rows"], "original union count changed")
    require(source_counts["NEW_ALNS_EXPANSION"] == expected["new_state_candidate_rows"], "new union count changed")
    require(union.groupby("state_id").is_fallback.sum().eq(1).all(), "union lacks one fallback per state")
    return union


def main() -> None:
    config = json.loads(CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    require(audit["status"] == "PASS", "O0 audit did not pass")
    require(audit["route"]["decision"] == "PROCEED_TOP_UTILITY_RETRAIN", "O0 did not select Route B")
    require(config["route"]["decision"] == audit["route"]["decision"], "config route differs from O0")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head.startswith(EXPECTED_O0_COMMIT), "O0 evidence commit is not current HEAD")
    require(not (ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling").exists(), "relabel outcomes already exist")
    require(not (ROOT / "outputs/phase6o_neural_shortlist_v1/training").exists(), "optimizer outputs already exist")
    ledgers = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        ROOT / "outputs/phase6o_neural_shortlist_v1/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6o_neural_shortlist_v1/r14_holdout/access_ledger.json",
    ]
    require(not any(path.exists() for path in ledgers), "R13/R14 access detected")
    union = build_candidate_union(config)
    union_path = OUT / "targeted_relabel_candidate_union.csv"
    atomic_csv(union_path, union)
    sources = {
        path: digest(ROOT / path)
        for path in (
            config["locked_inputs"]["phase6n_predictions"],
            config["locked_inputs"]["phase6n_grouped_labels"],
            config["locked_inputs"]["phase6n_seed_labels"],
            config["locked_inputs"]["phase6n_training_protocol"],
            config["locked_inputs"]["phase6l_predictions"],
            config["locked_inputs"]["phase6f_base_checkpoint"],
            "outputs/phase6o_neural_shortlist_v1/audit/loss_alignment_audit.json",
            "outputs/phase6o_neural_shortlist_v1/audit/protected_phase6n_evidence.json",
        )
    }
    source_record = {
        "schema": "phase6o-source-hashes-v1",
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": digest(CONFIG),
        "sources": sources,
        "manual_path": "/home/liulei/下载/phase6o_neural_shortlist_top_utility_codex_instructions.md",
        "manual_sha256": digest(Path("/home/liulei/下载/phase6o_neural_shortlist_top_utility_codex_instructions.md")),
    }
    atomic_json(OUT / "source_hashes.json", source_record)
    relabel_plan = {
        "schema": "phase6o-targeted-relabeling-plan-v1",
        "status": "FROZEN_BEFORE_ADDITIONAL_CONTINUATION",
        **config["targeted_relabeling"],
        "candidate_union_path": str(union_path.relative_to(ROOT)),
        "candidate_union_sha256": digest(union_path),
        "actual_state_candidate_rows": len(union),
        "actual_additional_continuation_rows": len(union) * len(config["targeted_relabeling"]["additional_crn_seeds"]),
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT / "targeted_relabeling_plan.json", relabel_plan)
    preregistration = {
        "schema": "phase6o-preregistration-v1",
        "status": "FROZEN_BEFORE_TARGETED_RELABEL_OR_OPTIMIZER_STEP",
        "starting_commit": config["starting_commit"],
        "o0_evidence_commit": head,
        "route": config["route"]["decision"],
        "primary_family": config["route"]["primary_family"],
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": digest(CONFIG),
        "source_hashes_path": str((OUT / "source_hashes.json").relative_to(ROOT)),
        "source_hashes_sha256": digest(OUT / "source_hashes.json"),
        "targeted_relabeling_plan_path": str((OUT / "targeted_relabeling_plan.json").relative_to(ROOT)),
        "targeted_relabeling_plan_sha256": digest(OUT / "targeted_relabeling_plan.json"),
        "candidate_union_sha256": digest(union_path),
        "additional_outcomes_started": False,
        "optimizer_steps_started": False,
        "live_solver_runs_started": False,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT / "preregistration.json", preregistration)
    report = f"""# Phase 6O 预注册协议

状态：**`FROZEN_BEFORE_TARGETED_RELABEL_OR_OPTIMIZER_STEP`**。冻结路线：**`PROCEED_TOP_UTILITY_RETRAIN`**。

O0 证明 k≤6 的 neural shortlist exact/near-best recall 不足，因此 Route A 未启动。Primary 保留 Phase 6N candidate-conditioned pooler/fusion/heads，冻结完整 Phase 6F RT-HGT encoder；训练只使用 FP32、score-free inputs 和 whole-instance folds。

## Targeted relabeling

两条现有 CRN seeds 在 501/864 states 上给出不同 winner。冻结 union 为：original states 使用 Phase 6L top-4 ∪ Phase 6N top-4 ∪ fallback，new states 使用 Phase 6N top-4 ∪ fallback。实际 {len(union):,} state-candidate rows，追加 seeds `{config['targeted_relabeling']['additional_crn_seeds']}`，共 {relabel_plan['actual_additional_continuation_rows']:,} additional continuation rows。所有 864 states 必须完成，不得按 outcome 停止；新标签单独保存，绝不覆盖 Phase 6L/6N truth。

Targeted candidates 训练时使用 5-seed mean，其他 candidates 保持 2-seed mean。Noise margin 仅从 training-fold CRN outcomes 拟合：candidate noise 为 1.4826×MAD，fold margin 为其 q75 并截断到 [0.0025, 0.03]。Held fold 不参与 margin、temperature、utility/regret scale、normalization 或 epoch selection。

## Primary objective

每个 source 获得 50% aggregate state weight，source 内再按 instance/state 平衡；opportunity weight 截断 [0.5, 2.0] 并在 source-instance 内归一化。Loss 为 top-set CE + near-best-vs-rest regret logistic + normalized soft expected utility，各权重 1.0；Huber 与 beats-fallback BCE 各 0.1。Phase 6N broad all-pairs loss和 standard-z ListNet 不用于 primary。

每个 outer fold/seed 使用对称 two-way inner validation，按 mean selected lift、near-best hit、top-1 regret、joint loss、earliest epoch 的冻结词典序选择一个 epoch，再在两个 outer-training folds 上 refit。

## Gate 与锁

Common raw lift 必须不低于 Phase 6L 的 0.0075163542，common regret 不高于 0.0338655744；expanded lift 与 grouped LCB 必须为正，S/M/L 不得反号，并保持 origin diversity、full bank、feasibility 和 fold isolation。失败为 `MODEL_REVISION_TOP_UTILITY`。

只有 OOF gate 通过后才能运行 3-seed、2N 的 non-promotable R12 development solver pilot。R13/R14 继续锁定；不运行 Gurobi。
"""
    REPORT.write_text(report)
    print(json.dumps({
        "status": preregistration["status"],
        "route": preregistration["route"],
        "candidate_union_rows": len(union),
        "additional_continuation_rows": relabel_plan["actual_additional_continuation_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
