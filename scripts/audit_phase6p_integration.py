#!/usr/bin/env python3
"""Audit the frozen search and candidate semantics before Phase 6P execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.search.alns import (  # noqa: E402
    ALNSConfig,
    DESTROY,
    REPAIR,
    solve_alns,
)
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402


EXPECTED_START = "936836445b8e7646574076645750f836a53b6d8d"
MANUAL = Path("/home/liulei/下载/phase6p_csg_adaptive_portfolio_codex_instructions_v3.md")
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit"
REPORT = ROOT / "docs/reports/phase6p_live_search_integration_audit.md"
PHASE6N_CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
PHASE6N_LABELS = (
    ROOT
    / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
)
PHASE6N_TRAINING = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training"
REPLAY_ROOTS = (
    ROOT / "outputs/phase6j_caur/r12_collection/state_replays",
    ROOT
    / "outputs/phase6n_candidate_conditioned_csg_v1/data/new_r12_collection/state_replays",
)

EXPECTED_RULES = (
    "operator_random",
    "operator_critical",
    "operator_overloaded_island",
    "operator_high_reconfiguration",
    "operator_w_bottleneck",
    "operator_f_bottleneck",
    "operator_related",
    "related_variant_1",
    "related_variant_2",
    "related_variant_3",
    "related_variant_4",
    "matched_random_1",
    "matched_random_2",
    "matched_random_3",
    "one_operation_swap",
    "two_operation_swap",
    "related_replace_25",
    "related_replace_50",
    "near_same_product",
    "near_precedence_neighbor",
    "near_same_island_chain",
    "near_high_W_delay",
    "near_high_F_delay",
    "near_low_slack",
)


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


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def candidate_from_dict(payload: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(payload["operation_order"]),
        tuple(payload["island_assignment"]),
        tuple(payload["w_assignment"]),
        tuple(payload["f_assignment"]),
    )


def read_replays() -> dict[str, tuple[dict, str]]:
    result: dict[str, tuple[dict, str]] = {}
    for root in REPLAY_ROOTS:
        origin = "ORIGINAL_PHASE6J_CAUR" if "phase6j_caur" in root.parts else "NEW_ALNS_EXPANSION"
        for path in sorted(root.glob("*.json")):
            replay = load_json(path)
            state_id = str(replay["state_id"])
            require(state_id not in result, f"duplicate replay state: {state_id}")
            result[state_id] = (replay, origin)
    return result


def audit_candidate_bank(config: dict) -> tuple[dict, list[dict]]:
    labels = pd.read_parquet(PHASE6N_LABELS)
    replays = read_replays()
    label_states = set(labels.state_id.astype(str))
    require(len(replays) == 864 and label_states == set(replays), "R12 replay scope mismatch")
    require(set(labels.phase6n_data_origin.astype(str)) == {
        "ORIGINAL_PHASE6J_CAUR", "NEW_ALNS_EXPANSION"
    }, "Phase 6N data origins changed")

    labels_by_state = {
        str(state_id): group.copy()
        for state_id, group in labels.groupby("state_id", sort=False)
    }
    instance_root = ROOT / config["locked_inputs"]["instance_root"]
    proposal_namespace = int(config["data_generation"]["proposal_seed_namespace"])
    destroy_fraction = float(load_json(ROOT / "configs/phase5c_alns.json")["destroy_fraction"])
    instances = {}
    rows: list[dict] = []
    rule_counts: Counter[str] = Counter()
    unique_counts: Counter[int] = Counter()
    duplicate_counts: Counter[int] = Counter()
    cross_destroy_states: set[str] = set()
    replay_mismatches: list[str] = []
    label_mismatches: list[str] = []

    for state_id in sorted(replays):
        replay, origin = replays[state_id]
        relative = str(replay["instance_relative_path"])
        if relative not in instances:
            path = instance_root / relative
            require(digest(path) == replay["instance_sha256"], f"instance hash drift: {relative}")
            instances[relative] = load_phase6j_instance(path)
        instance = instances[relative]
        current = decode_candidate(
            instance, candidate_from_dict(replay["snapshot"]["current_candidate"])
        )
        require(
            abs(current.makespan - float(replay["current_replay_makespan"])) <= 1e-9,
            f"incumbent replay mismatch: {state_id}",
        )
        destroy_count = min(
            max(2, round(instance.num_operations * destroy_fraction)),
            instance.num_operations,
        )
        generated = generate_revised_target_arms(
            instance, current, state_id, destroy_count, proposal_namespace
        )
        expected_ids = replay.get(
            "full_bank_target_ids_in_generator_order", replay.get("full_bank_target_ids")
        )
        actual_ids = [arm.target_set_id for arm in generated.arms]
        if actual_ids != expected_ids or any(
            list(arm.destroyed_operations)
            != replay["full_bank_target_operations"].get(arm.target_set_id)
            for arm in generated.arms
        ):
            replay_mismatches.append(state_id)
        group = labels_by_state[state_id]
        if set(group.target_set_id.astype(str)) != set(actual_ids):
            label_mismatches.append(state_id)
        label_by_id = {
            str(row.target_set_id): row for row in group.itertuples(index=False)
        }

        by_operations = {}
        for proposal in generated.proposals:
            rule_counts[proposal.origin_rule] += 1
            by_operations.setdefault(proposal.destroyed_operations, []).append(proposal)
        unique_counts[generated.unique_arm_count] += 1
        duplicate_counts[generated.duplicate_arm_count] += 1
        require(generated.requested_arm_count == 24, f"non-24 bank: {state_id}")
        require(len(by_operations) == len(generated.arms), f"dedup mismatch: {state_id}")

        for generator_index, arm in enumerate(generated.arms):
            origins = by_operations[arm.destroyed_operations]
            rules = tuple(item.origin_rule for item in origins)
            destroy_operators = tuple(dict.fromkeys(
                item.origin_destroy_operator for item in origins
            ))
            families = tuple(dict.fromkeys(item.arm_family for item in origins))
            cross_destroy = len(destroy_operators) > 1
            if cross_destroy:
                cross_destroy_states.add(state_id)
            stored = label_by_id[arm.target_set_id]
            stored_rules = tuple(json.loads(str(stored.origin_rules)))
            stored_families = tuple(json.loads(str(stored.origin_families)))
            metadata_match = (
                str(stored.origin_destroy_operator) == arm.origin_destroy_operator
                and stored_rules == arm.origin_rules == rules
                and stored_families == arm.origin_families == families
            )
            if not metadata_match and state_id not in label_mismatches:
                label_mismatches.append(state_id)
            rows.append({
                "state_id": state_id,
                "instance_id": str(replay["instance_id"]),
                "data_origin": origin,
                "target_set_id": arm.target_set_id,
                "generator_order_index": generator_index,
                "all_origin_rules": json.dumps(rules, separators=(",", ":")),
                "all_origin_destroy_operators": json.dumps(
                    destroy_operators, separators=(",", ":")
                ),
                "all_origin_families": json.dumps(families, separators=(",", ":")),
                "current_stored_first_origin_destroy_operator": arm.origin_destroy_operator,
                "origin_rule_count": len(rules),
                "origin_destroy_operator_count": len(destroy_operators),
                "cross_destroy_duplicate": cross_destroy,
                "stored_metadata_match": metadata_match,
            })

    cross_rows = sum(bool(row["cross_destroy_duplicate"]) for row in rows)
    checks = {
        "authorized_r12_states_exact": len(replays) == 864,
        "authorized_r12_instances_exact": len(instances) == 18,
        "all_replays_reconstruct": not replay_mismatches,
        "all_label_candidate_sets_match": not label_mismatches,
        "requested_rules_exact": set(rule_counts) == set(EXPECTED_RULES)
        and all(rule_counts[rule] == 864 for rule in EXPECTED_RULES),
        "requested_count_24_every_state": sum(rule_counts.values()) == 864 * 24,
        "unique_count_range_21_to_24": set(unique_counts).issubset({21, 22, 23, 24}),
        "stored_first_origin_and_aggregates_match": all(
            bool(row["stored_metadata_match"]) for row in rows
        ),
    }
    result = {
        "schema": "phase6p-candidate-bank-semantics-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "authorized_scope": "R12_CAUR_FIT_PHASE6N_864_STATES",
        "states": len(replays),
        "instances": len(instances),
        "requested_proposals": sum(rule_counts.values()),
        "unique_targets": len(rows),
        "rule_order": list(EXPECTED_RULES),
        "rule_counts": dict(rule_counts),
        "unique_targets_per_state_distribution": {
            str(key): value for key, value in sorted(unique_counts.items())
        },
        "duplicates_per_state_distribution": {
            str(key): value for key, value in sorted(duplicate_counts.items())
        },
        "cross_destroy_duplicate_targets": cross_rows,
        "cross_destroy_duplicate_states": len(cross_destroy_states),
        "cross_destroy_resolution": (
            "MEAN_CURRENT_ALNS_DESTROY_WEIGHT_ACROSS_UNIQUE_ORIGINS_REQUIRED"
            if cross_rows else "SINGLE_STORED_ORIGIN_RETAINED_WITH_ASSERTION"
        ),
        "replay_mismatch_states": replay_mismatches,
        "label_mismatch_states": label_mismatches,
        "proposal_seed_namespace": proposal_namespace,
        "destroy_count": "min(max(2, round(instance.num_operations * 0.15)), instance.num_operations)",
        "candidate_identity": "target_set_id hashes state_id plus sorted destroyed-operation set",
        "deduplication": "exact destroyed-operation tuple in proposal insertion order",
        "source_hashes": {
            "phase6n_labels": digest(PHASE6N_LABELS),
            "phase6c_generator": digest(ROOT / "rcias_clgri/search/phase6c.py"),
        },
        "r13_accessed": False,
        "r14_accessed": False,
    }
    return result, rows


def audit_integration(config: dict, bank: dict) -> dict:
    head = git("rev-parse", "HEAD")
    phase6o = ROOT / "outputs/phase6o_neural_shortlist_v1/final/final_decision.json"
    phase6o_decision = load_json(phase6o)
    alns_config_path = ROOT / "configs/phase5c_alns.json"
    alns_raw = load_json(alns_config_path)
    default_config = asdict(ALNSConfig())
    frozen_config = {
        key: alns_raw[key] for key in ALNSConfig.__dataclass_fields__ if key in alns_raw
    }
    solve_source = inspect.getsource(solve_alns)
    checkpoints = []
    for seed in (726101, 726102, 726103):
        for fold in range(3):
            path = PHASE6N_TRAINING / f"oof/seed_{seed}/fold_{fold}.pt"
            record = load_json(path.with_suffix(".json"))
            checkpoints.append({
                "training_seed": seed,
                "held_fold": fold,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "record_checkpoint_sha256": record["checkpoint_sha256"],
            })
    phase6n_final = load_json(
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json"
    )
    phase6h_config = load_json(ROOT / "configs/phase6h_live_calibration.json")
    phase6h_policy = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
    lghga_config = ROOT / "configs/lghga_2o_baseline.json"
    lghga_smoke = ROOT / "outputs/phase6k_runtime_v1/audit/lghga_2o_real_clock_smoke.json"

    alns_checks = {
        "destroy_operators_exact": DESTROY == (
            "random", "critical", "overloaded_island", "high_reconfiguration",
            "w_bottleneck", "f_bottleneck", "related",
        ),
        "repair_operators_exact": REPAIR == (
            "greedy", "regret2", "regret3", "reconfiguration_aware", "transport_aware",
        ),
        "frozen_config_matches_defaults": frozen_config == {
            key: value for key, value in default_config.items() if key != "iteration_limit"
        },
        "destroy_and_repair_selected_independently": (
            "destroy = _roulette(DESTROY, weights, rng)" in solve_source
            and "repair = _roulette(REPAIR, weights, rng)" in solve_source
        ),
        "candidate_trials_inherited": "range(config.candidate_trials)" in solve_source,
        "simulated_annealing_acceptance_inherited": (
            "math.exp(-delta / max(temperature, 1e-12))" in solve_source
        ),
        "reward_and_update_inherited": (
            "score = 1.0" in solve_source
            and "score = 5.0" in solve_source
            and "max(score, 0.1)" in solve_source
        ),
        "current_and_best_separate": (
            "current = candidate" in solve_source
            and "best = candidate" in solve_source
            and "candidate.makespan < best.makespan" in solve_source
        ),
    }
    checks = {
        "expected_start_head": head == EXPECTED_START,
        "phase6o_terminal": phase6o_decision.get("decision") == "MODEL_REVISION_TOP_UTILITY",
        "phase6o_final_hash": digest(phase6o)
        == "9bb6e06c78b5fdccf0830e67b638fbb9bfcaea4ddcfddae477baa73d9fc371b9",
        "manual_hash": digest(MANUAL)
        == "f1fb0a7eaafdbcd82afc6d68d713d973dcc2f831f07a0e5a13304796d7084040",
        "alns_contract": all(alns_checks.values()),
        "candidate_bank_contract": bank["status"] == "PASS",
        "phase6h_policy_frozen": phase6h_policy.is_file()
        and digest(phase6h_policy) == "4d2da03e13a036569bebf3897135e5da139e292e3260a2f452754d5c9ae3d239",
        "phase6n_oof_checkpoint_grid_complete": len(checkpoints) == 9
        and all(item["sha256"] == item["record_checkpoint_sha256"] for item in checkpoints),
        "phase6n_whole_instance_isolation": all(
            not row["instance_overlap"]
            for row in load_json(PHASE6N_TRAINING / "training_protocol.json")["fold_audit"]
        ),
        "lghga_2o_runnable_smoke": lghga_smoke.is_file()
        and load_json(lghga_smoke).get("status") == "PASS",
        "r13_r14_locked": all(not path.exists() for path in (
            ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
            ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
            ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json",
            ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json",
        )),
    }
    return {
        "schema": "phase6p-live-search-integration-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "repository": {
            "expected_start_commit": EXPECTED_START,
            "head_at_audit": head,
            "phase6o_final_decision": phase6o_decision["decision"],
            "phase6o_final_decision_sha256": digest(phase6o),
            "manual_sha256": digest(MANUAL),
        },
        "alns": {
            "entry_point": "rcias_clgri.search.alns.solve_alns",
            "source_sha256": digest(ROOT / "rcias_clgri/search/alns.py"),
            "config_path": str(alns_config_path.relative_to(ROOT)),
            "config_sha256": digest(alns_config_path),
            "config": frozen_config,
            "destroy_operators": list(DESTROY),
            "repair_operators": list(REPAIR),
            "destroy_count_source": "solve_alns; max(2, round(N * config.destroy_fraction)), capped at N when passed to _destroy",
            "candidate_trials_source": "ALNSConfig.candidate_trials=8",
            "roulette": "destroy and repair independently use current operator weights",
            "acceptance": "delta <= 0 or U < exp(-delta / max(temperature, 1e-12))",
            "rewards": {"new_global_best": 5.0, "accepted_non_best": 1.0, "rejected_raw": 0.0, "update_floor": 0.1},
            "weight_update": "w <- (1 - 0.2) * w + 0.2 * max(score, 0.1)",
            "weight_update_from_initial_one": {"best": 1.8, "accepted": 1.0, "rejected": 0.82},
            "best_current_semantics": "accepted candidates update current; strict global improvement updates best; return best",
            "checks": alns_checks,
        },
        "phase6h": {
            "live_entry_points": [
                "rcias_clgri.ni.live_inference.FrozenLiveInference.decide",
                "rcias_clgri.search.csgni.solve_csgni",
            ],
            "policy_path": str(phase6h_policy.relative_to(ROOT)),
            "policy_sha256": digest(phase6h_policy),
            "intervention_rate": int(phase6h_config["search"]["intervention_rate"]),
            "destroy_count_source": "solve_csgni; min(max(2, round(N * alns_config.destroy_fraction)), N)",
            "candidate_trials_source": "frozen phase5c ALNS config through solve_csgni; 8",
            "budget": "2 * instance.num_operations seconds in Phase 6H validation entry",
        },
        "phase6n": {
            "development_entry_points": [
                "rcias_clgri.search.phase6c.generate_revised_target_arms",
                "scripts.run_phase6n_data_generation.collect_state",
                "scripts.train_phase6n_candidate_conditioned.predict",
            ],
            "production_live_entry_point": None,
            "production_live_entry_status": "MUST_BE_IMPLEMENTED_IN_PHASE6P",
            "destroy_count_source": "Phase 6N data generation; min(max(2, round(N * frozen ALNS destroy_fraction)), N)",
            "candidate_trials_source": "Phase 6N frozen data plan candidate_trials_per_target=8",
            "critic_bundle_interpretation": "nine frozen whole-instance OOF checkpoints routed by held structural fold and ensembled across three training seeds",
            "checkpoint_grid": checkpoints,
            "precision": "FP32_ONLY",
            "terminal_decision": phase6n_final["decision"],
            "deployable_bundle": phase6n_final["deployable_bundle"],
            "qualification_boundary": "frozen OOF critic evidence only; Phase 6N did not pass its representation/deployment gate",
        },
        "candidate_bank": {
            "requested_rules": 24,
            "deduplication": bank["deduplication"],
            "authorized_states": bank["states"],
            "unique_targets": bank["unique_targets"],
            "cross_destroy_duplicate_targets": bank["cross_destroy_duplicate_targets"],
            "cross_destroy_resolution": bank["cross_destroy_resolution"],
        },
        "lghga_2o": {
            "entry_point": "rcias_clgri.search.lghga_2o.solve_lghga_2o",
            "budget_entry_point": "rcias_clgri.search.lghga_2o.operation_budget",
            "budget": "2 * instance.num_operations seconds, including population initialization",
            "adapter_sha256": digest(ROOT / "rcias_clgri/search/lghga_2o.py"),
            "config_path": str(lghga_config.relative_to(ROOT)),
            "config_sha256": digest(lghga_config),
            "config_status": load_json(lghga_config)["status"],
            "real_clock_smoke_path": str(lghga_smoke.relative_to(ROOT)),
            "real_clock_smoke_sha256": digest(lghga_smoke),
            "canonical_registry_status": "ABSENT_REQUIRES_P3_COMPARATOR_REUSE_AUDIT",
        },
        "constraints": {
            "new_neural_training_before_pilot": False,
            "phase6m_selector": False,
            "phase6o_critic": False,
            "gurobi": False,
            "r13_accessed": False,
            "r14_accessed": False,
        },
        "next_stage": "P1_PREREGISTRATION" if all(checks.values()) else "HOLD_FIX_P0",
    }


def write_report(integration: dict, bank: dict) -> None:
    status = integration["status"]
    cross = bank["cross_destroy_duplicate_targets"]
    resolution = (
        "检测到跨 destroy-operator 去重，P2 实现必须暴露全部来源，并以这些来源当前权重的均值作为候选 destroy 权重。"
        if cross else
        "未检测到跨 destroy-operator 去重；保留首来源表示，并在运行时加入断言。"
    )
    text = f"""# Phase 6P P0 在线搜索集成审计

状态：**`{status}`**。起始 HEAD 与说明书要求一致，Phase 6O 终态及哈希保持不变；R13/R14 仍锁定，未运行 Gurobi 或任何新神经训练。

## 候选生成与多来源结论

对 Phase 6N 已授权的全部 **{bank['states']} 个 R12 CAUR-FIT states** 进行了当前候选生成器的真实回放，共重建 {bank['requested_proposals']:,} 个原始 proposals、{bank['unique_targets']:,} 个去重 target。每个 state 均生成 24 条规则，去重后数量分布为 `{bank['unique_targets_per_state_distribution']}`；候选 ID、生成顺序、destroyed-operation set 以及存档聚合来源均与回放证据一致。

跨 destroy-operator 的去重 target 数为 **{cross}**，涉及 {bank['cross_destroy_duplicate_states']} 个 states。{resolution}

完整逐 target 来源表见 `outputs/phase6p_adaptive_portfolio_v1/audit/multi_origin_target_audit.csv`。

## 搜索语义

冻结 ALNS 仍使用 7 个 destroy 与 5 个 repair 算子，二者独立按当前权重 roulette；`destroy_fraction=0.15`、`candidate_trials=8`、`reaction_factor=0.2`。奖励为新全局最优 5、接受非最优 1、拒绝 0 并在更新时设 0.1 floor；权重更新为 `(1-0.2)w + 0.2*max(score,0.1)`。候选先按模拟退火更新 current，严格改善才更新 best，最终返回 best。

Phase 6H 的真实入口是 `FrozenLiveInference.decide` 加 `solve_csgni`，destroy count 由 `solve_csgni` 使用继承 ALNS 比例计算，8 trials 同样来自冻结 ALNS 配置。

Phase 6N 当前只有数据生成和 OOF 推理入口，没有正式 live solver 或 deployable bundle。可复用对象是 9 个冻结 whole-instance OOF checkpoints：按 structural held fold 路由，再对三个训练 seed 集成。Phase 6N 自身的 N5 失败结论不变，Phase 6P 的 P1 必须将这项复用边界及全部 checkpoint 哈希预注册。

LG_HGA_2O 的 `solve_lghga_2o` 与 `operation_budget` 可运行，既有真实时钟 smoke 通过，预算从 population initialization 前开始并使用 `2*|O|`。当前还没有 standing canonical baseline registry，需在 P3 comparator audit 中判定并冻结，不能把 tiny smoke 当成正式 comparator 结果。

## P0 判定

机器可读检查见 `live_search_integration.json` 与 `candidate_bank_semantics.json`。P0 {('通过，可以进入 P1 预注册。' if status == 'PASS' else '未通过，必须修复失败检查后才能进入 P1。')}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    config = load_json(PHASE6N_CONFIG)
    bank, rows = audit_candidate_bank(config)
    integration = audit_integration(config, bank)
    atomic_csv(OUT / "multi_origin_target_audit.csv", rows)
    atomic_json(OUT / "candidate_bank_semantics.json", bank)
    atomic_json(OUT / "live_search_integration.json", integration)
    write_report(integration, bank)
    print(json.dumps({
        "status": integration["status"],
        "states": bank["states"],
        "unique_targets": bank["unique_targets"],
        "cross_destroy_duplicate_targets": bank["cross_destroy_duplicate_targets"],
        "next_stage": integration["next_stage"],
    }, indent=2))
    if integration["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
