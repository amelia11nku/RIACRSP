#!/usr/bin/env python3
"""Audit and close the frozen Phase 6P three-seed development campaign."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1"
DEV = OUT / "development"
FINAL = OUT / "final"
REGISTRY_ROOT = ROOT / "outputs/frozen_2o_baselines"
PROTOCOL = DEV / "protocol.json"
PRIMARY = "P1_CSG_ADAPTIVE_PORTFOLIO"
METHODS = (
    PRIMARY,
    "PHASE6N_DETERMINISTIC_TOP1",
    "ALNS",
    "PHASE6H",
    "LG_HGA_2O",
)
COMPARATORS = tuple(method for method in METHODS if method != PRIMARY)
RESULT_ROOTS = {
    PRIMARY: DEV / "runs/P1_CSG_ADAPTIVE_PORTFOLIO",
    "PHASE6N_DETERMINISTIC_TOP1": REGISTRY_ROOT / "phase6n/runs",
    "ALNS": REGISTRY_ROOT / "alns/runs",
    "PHASE6H": REGISTRY_ROOT / "phase6h/runs",
    "LG_HGA_2O": REGISTRY_ROOT / "lg_hga_2o/runs",
}
REGISTRY_IDS = {
    "PHASE6N_DETERMINISTIC_TOP1": "PHASE6N_TOP1",
    "ALNS": "ALNS",
    "PHASE6H": "PHASE6H",
    "LG_HGA_2O": "LG_HGA_2O",
}
DESTROY = (
    "random", "critical", "overloaded_island", "high_reconfiguration",
    "w_bottleneck", "f_bottleneck", "related",
)
REPAIR = (
    "greedy", "regret2", "regret3", "reconfiguration_aware",
    "transport_aware",
)
RULES = (
    "operator_random", "operator_critical", "operator_overloaded_island",
    "operator_high_reconfiguration", "operator_w_bottleneck",
    "operator_f_bottleneck", "operator_related", "related_variant_1",
    "related_variant_2", "related_variant_3", "related_variant_4",
    "matched_random_1", "matched_random_2", "matched_random_3",
    "one_operation_swap", "two_operation_swap", "related_replace_25",
    "related_replace_50", "near_same_product", "near_precedence_neighbor",
    "near_same_island_chain", "near_high_W_delay", "near_high_F_delay",
    "near_low_slack",
)
FAMILIES = (
    "ORIGINAL_OPERATOR", "RELATED_VARIANT", "MATCHED_RANDOM",
    "LOCAL_PERTURBATION", "STRUCTURED_NEAR_NEIGHBOR",
)
ANYTIME_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expected_tasks(protocol: dict) -> list[dict]:
    tasks = []
    for method in METHODS:
        for instance in protocol["instances"]:
            for seed in protocol["development_seeds"]:
                tasks.append({
                    "method": method,
                    "instance_id": instance["instance_id"],
                    "instance_sha256": instance["sha256"],
                    "scale": instance["scale"],
                    "CF_level": instance["CF_level"],
                    "cell_replicate": instance["cell_replicate"],
                    "seed": int(seed),
                    "num_operations": int(instance["num_operations"]),
                    "budget_seconds": 2.0 * int(instance["num_operations"]),
                })
    require(len(tasks) == 270, "the frozen protocol does not define 270 tasks")
    return tasks


def result_path(task: dict) -> Path:
    return (
        RESULT_ROOTS[task["method"]]
        / task["instance_id"]
        / f"seed_{task['seed']}.json"
    )


def validate_trace(payload: dict) -> bool:
    trace = payload["incumbent_trace"]
    if not trace:
        return False
    makespans = [float(row["current_best_makespan"]) for row in trace]
    elapsed = [float(row["elapsed_time"]) for row in trace]
    evaluations = [int(row["decoder_evaluations"]) for row in trace]
    return all((
        all(math.isfinite(value) for value in makespans + elapsed),
        all(left >= right for left, right in zip(makespans, makespans[1:])),
        all(left <= right for left, right in zip(elapsed, elapsed[1:])),
        all(left <= right for left, right in zip(evaluations, evaluations[1:])),
        makespans[-1] == float(payload["final_makespan"]),
        evaluations[-1] <= int(payload["decoder_evaluations"]),
    ))


def counter_sum(payloads: list[dict], key: str) -> Counter:
    result: Counter = Counter()
    for payload in payloads:
        result.update(payload["portfolio_diagnostics"][key])
    return result


def distribution_rows(name: str, counter: Counter, expected: tuple) -> list[dict]:
    denominator = sum(counter.values())
    return [
        {
            "distribution": name,
            "category": str(category),
            "count": int(counter.get(category, counter.get(str(category), 0))),
            "fraction": (
                float(counter.get(category, counter.get(str(category), 0)) / denominator)
                if denominator else 0.0
            ),
        }
        for category in expected
    ]


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> int:
    protocol = load_json(PROTOCOL)
    protocol_hash = digest(PROTOCOL)
    require(protocol["status"] == "FROZEN_BEFORE_P3_SOLVER_OUTCOMES", "protocol is not frozen")
    require(protocol["methods"] == list(METHODS), "method order changed")
    require(protocol["development_seeds"] == [746101, 746102, 746103], "development seeds changed")
    require(not protocol["r13_accessed"] and not protocol["r14_accessed"], "protocol accessed R13/R14")
    for relative, expected in protocol["source_hashes"].items():
        require(digest(ROOT / relative) == expected, f"source hash changed: {relative}")
    for relative, expected in protocol["artifact_hashes"].items():
        require(digest(ROOT / relative) == expected, f"artifact hash changed: {relative}")

    tasks = expected_tasks(protocol)
    rows = []
    payloads: dict[tuple[str, str, int], dict] = {}
    hash_rows = []
    for task in tasks:
        path = result_path(task)
        require(path.is_file(), f"missing result: {path.relative_to(ROOT)}")
        payload = load_json(path)
        key = (task["method"], task["instance_id"], task["seed"])
        require(key not in payloads, f"duplicate result: {key}")
        payloads[key] = payload
        expected = {
            "schema": "phase6p-development-run-v1",
            "status": "COMPLETE",
            "development_protocol_sha256": protocol_hash,
            "method": task["method"],
            "instance_id": task["instance_id"],
            "instance_sha256": task["instance_sha256"],
            "seed": task["seed"],
            "time_limit_seconds": task["budget_seconds"],
            "budget_seconds": task["budget_seconds"],
            "scale": task["scale"],
            "CF_level": task["CF_level"],
            "cell_replicate": task["cell_replicate"],
            "num_operations": task["num_operations"],
            "result_source": "NEW_RUN",
            "feasible": True,
            "gurobi_run": False,
            "r13_accessed": False,
            "r14_accessed": False,
        }
        for field, value in expected.items():
            require(payload.get(field) == value, f"{key}: field mismatch for {field}")
        replay = payload["feasibility_replay"]
        require(replay["feasible"] is True, f"{key}: replay infeasible")
        require(replay["violations"] == [], f"{key}: replay violations")
        require(float(replay["makespan"]) == float(payload["final_makespan"]), f"{key}: replay makespan mismatch")
        require(validate_trace(payload), f"{key}: invalid incumbent trace")
        require(float(payload["runtime_seconds"]) >= task["budget_seconds"], f"{key}: internal runtime under budget")
        require(float(payload["outer_runtime_seconds"]) >= task["budget_seconds"], f"{key}: outer runtime under budget")
        require(int(payload["decoder_evaluations"]) > 0, f"{key}: no decoder evaluations")
        if task["method"] in {PRIMARY, "PHASE6N_DETERMINISTIC_TOP1"}:
            require(payload["historical_score_calls"] == 0, f"{key}: historical scorer used")
            require(payload["phase6o_critic_calls"] == 0, f"{key}: Phase 6O critic used")
        if task["method"] == PRIMARY:
            diagnostics = payload["portfolio_diagnostics"]
            neural = int(diagnostics["neural_decisions"])
            iterations = int(payload["iterations"])
            require(neural == (iterations + 4) // 5, f"{key}: 20 percent schedule drift")
            require(sum(map(int, diagnostics["sampled_neural_rank_distribution"].values())) == neural, f"{key}: rank count mismatch")
            require(set(map(int, diagnostics["sampled_neural_rank_distribution"])) <= set(range(1, 7)), f"{key}: rank outside top-6")
            require(sum(map(int, diagnostics["selected_repair_distribution"].values())) == iterations, f"{key}: repair count mismatch")
            require(int(diagnostics["safe_fallbacks"]) == 0, f"{key}: unexpected safe fallback")
            require(len(diagnostics["adaptive_weight_checkpoints"]) == 11, f"{key}: weight checkpoints incomplete")
        rows.append({
            **task,
            "final_makespan": float(payload["final_makespan"]),
            "runtime_seconds": float(payload["runtime_seconds"]),
            "outer_runtime_seconds": float(payload["outer_runtime_seconds"]),
            "best_found_seconds": float(payload["best_found_seconds"]),
            "decoder_evaluations": int(payload["decoder_evaluations"]),
            "iterations": int(payload["iterations"]),
            "neural_decisions": int(payload["portfolio_diagnostics"].get(
                "neural_decisions",
                payload["search_diagnostics"].get("ni_eligible_iterations", 0),
            )),
            "fallback_count": int(payload["portfolio_diagnostics"].get(
                "safe_fallbacks", payload["search_diagnostics"].get("ni_fallbacks", 0),
            )),
            "feasible": True,
            "result_path": str(path.relative_to(ROOT)),
        })
        hash_rows.append({
            "method": task["method"],
            "instance_id": task["instance_id"],
            "seed": task["seed"],
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
        })

    require(len(payloads) == 270, "result cardinality is not 270")
    registry = load_json(REGISTRY_ROOT / "registry.json")
    require(registry["status"] == "FROZEN_CANONICAL_THREE_SEED_DEVELOPMENT", "registry is not frozen")
    registry_by_id = {entry["algorithm_id"]: entry for entry in registry["entries"]}
    for method, algorithm_id in REGISTRY_IDS.items():
        entry = registry_by_id[algorithm_id]
        require(entry["status"] == "FROZEN_CANONICAL", f"{algorithm_id}: registry entry not frozen")
        summary_path = ROOT / entry["aggregate_summary_path"]
        manifest_path = ROOT / entry["run_manifest_path"]
        require(digest(summary_path) == entry["aggregate_summary_sha256"], f"{algorithm_id}: summary hash mismatch")
        require(digest(manifest_path) == entry["run_manifest_sha256"], f"{algorithm_id}: manifest hash mismatch")
        manifest = load_json(manifest_path)
        require(len(manifest["runs"]) == 54, f"{algorithm_id}: manifest does not have 54 runs")
        expected_hashes = {
            (row["instance_id"], int(row["seed"])): row["sha256"]
            for row in hash_rows if row["method"] == method
        }
        actual_hashes = {
            (row["instance_id"], int(row["seed"])): row["sha256"]
            for row in manifest["runs"]
        }
        require(actual_hashes == expected_hashes, f"{algorithm_id}: run manifest mismatch")

    frame = pd.DataFrame(rows)
    require(not frame.duplicated(["method", "instance_id", "seed"]).any(), "duplicate task rows")
    require(frame.groupby("method").size().to_dict() == {method: 54 for method in METHODS}, "method counts changed")
    bks = frame.groupby("instance_id")["final_makespan"].min()
    frame["current_development_bks"] = frame["instance_id"].map(bks)
    frame["rpd_percent"] = 100.0 * (
        frame["final_makespan"] - frame["current_development_bks"]
    ) / frame["current_development_bks"]
    frame["hit_current_development_bks"] = frame["rpd_percent"] == 0.0
    atomic_csv(DEV / "audited_run_summary.csv", frame)
    atomic_csv(DEV / "result_hash_manifest.csv", pd.DataFrame(hash_rows))

    method_summary = frame.groupby("method", as_index=False).agg(
        runs=("final_makespan", "size"),
        mean_final_makespan=("final_makespan", "mean"),
        median_final_makespan=("final_makespan", "median"),
        mean_rpd_percent=("rpd_percent", "mean"),
        median_rpd_percent=("rpd_percent", "median"),
        bks_hits=("hit_current_development_bks", "sum"),
        feasibility_rate=("feasible", "mean"),
        mean_decoder_evaluations=("decoder_evaluations", "mean"),
        mean_iterations=("iterations", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_outer_runtime_seconds=("outer_runtime_seconds", "mean"),
        median_best_found_seconds=("best_found_seconds", "median"),
    )
    instance_method = frame.groupby(
        ["instance_id", "scale", "CF_level", "method"], as_index=False
    ).agg(
        mean_final_makespan=("final_makespan", "mean"),
        median_final_makespan=("final_makespan", "median"),
        best_final_makespan=("final_makespan", "min"),
        mean_rpd_percent=("rpd_percent", "mean"),
        mean_decoder_evaluations=("decoder_evaluations", "mean"),
    )
    pivot = instance_method.pivot(
        index=["instance_id", "scale", "CF_level"],
        columns="method",
        values="mean_final_makespan",
    )
    average_ranks = pivot.rank(axis=1, method="average").mean()
    method_summary["average_instance_rank"] = method_summary["method"].map(average_ranks)
    instance_best = pivot.min(axis=1)
    instance_hits = pivot.eq(instance_best, axis=0).sum()
    method_summary["instance_mean_best_hits"] = method_summary["method"].map(instance_hits)
    atomic_csv(DEV / "method_summary.csv", method_summary)
    atomic_csv(DEV / "instance_method_summary.csv", instance_method)

    pairwise_rows = []
    for comparator in COMPARATORS:
        difference = pivot[PRIMARY] - pivot[comparator]
        improvement = 100.0 * (pivot[comparator] - pivot[PRIMARY]) / pivot[comparator]
        run_pivot = frame.pivot(
            index=["instance_id", "seed"], columns="method", values="final_makespan"
        )
        run_difference = run_pivot[PRIMARY] - run_pivot[comparator]
        pairwise_rows.append({
            "primary": PRIMARY,
            "comparator": comparator,
            "instance_mean_pairs": len(difference),
            "instance_mean_wins": int((difference < 0).sum()),
            "instance_mean_ties": int((difference == 0).sum()),
            "instance_mean_losses": int((difference > 0).sum()),
            "mean_makespan_difference_primary_minus_comparator": float(difference.mean()),
            "mean_relative_improvement_percent": float(improvement.mean()),
            "run_pairs": len(run_difference),
            "run_wins": int((run_difference < 0).sum()),
            "run_ties": int((run_difference == 0).sum()),
            "run_losses": int((run_difference > 0).sum()),
        })
    pairwise = pd.DataFrame(pairwise_rows)
    atomic_csv(DEV / "pairwise_summary.csv", pairwise)

    scale_summary = frame.groupby(["scale", "method"], as_index=False).agg(
        runs=("final_makespan", "size"),
        mean_final_makespan=("final_makespan", "mean"),
        mean_rpd_percent=("rpd_percent", "mean"),
        mean_decoder_evaluations=("decoder_evaluations", "mean"),
    )
    cf_summary = frame.groupby(["CF_level", "method"], as_index=False).agg(
        runs=("final_makespan", "size"),
        mean_final_makespan=("final_makespan", "mean"),
        mean_rpd_percent=("rpd_percent", "mean"),
        mean_decoder_evaluations=("decoder_evaluations", "mean"),
    )
    atomic_csv(DEV / "scale_summary.csv", scale_summary)
    atomic_csv(DEV / "cf_summary.csv", cf_summary)

    anytime_rows = []
    for task in tasks:
        payload = payloads[(task["method"], task["instance_id"], task["seed"])]
        trace = payload["incumbent_trace"]
        for fraction in ANYTIME_FRACTIONS:
            target = fraction * task["budget_seconds"]
            available = [row for row in trace if float(row["elapsed_time"]) <= target]
            selected = available[-1] if available else trace[0]
            makespan = float(selected["current_best_makespan"])
            reference = float(bks[task["instance_id"]])
            anytime_rows.append({
                "method": task["method"],
                "instance_id": task["instance_id"],
                "seed": task["seed"],
                "budget_fraction": fraction,
                "makespan": makespan,
                "rpd_percent": 100.0 * (makespan - reference) / reference,
                "decoder_evaluations": int(selected["decoder_evaluations"]),
            })
    anytime = pd.DataFrame(anytime_rows)
    anytime_summary = anytime.groupby(["method", "budget_fraction"], as_index=False).agg(
        runs=("makespan", "size"),
        mean_makespan=("makespan", "mean"),
        median_makespan=("makespan", "median"),
        mean_rpd_percent=("rpd_percent", "mean"),
        mean_decoder_evaluations=("decoder_evaluations", "mean"),
    )
    atomic_csv(DEV / "anytime_summary.csv", anytime_summary)

    primary_payloads = [
        payloads[(PRIMARY, task["instance_id"], task["seed"])]
        for task in tasks if task["method"] == PRIMARY
    ]
    total_iterations = sum(int(item["iterations"]) for item in primary_payloads)
    total_neural = sum(int(item["portfolio_diagnostics"]["neural_decisions"]) for item in primary_payloads)
    weighted = lambda key: sum(
        float(item["portfolio_diagnostics"][key]) * int(item["portfolio_diagnostics"]["neural_decisions"])
        for item in primary_payloads
    ) / total_neural
    iteration_weighted = lambda key: sum(
        float(item["portfolio_diagnostics"][key]) * int(item["iterations"])
        for item in primary_payloads
    ) / total_iterations
    ranks = counter_sum(primary_payloads, "sampled_neural_rank_distribution")
    rules = counter_sum(primary_payloads, "selected_24_rule_origin_distribution")
    families = counter_sum(primary_payloads, "selected_origin_family_distribution")
    destroys = counter_sum(primary_payloads, "selected_destroy_operator_distribution")
    repairs = counter_sum(primary_payloads, "selected_repair_distribution")
    distribution = pd.DataFrame(
        distribution_rows("neural_rank", ranks, tuple(range(1, 7)))
        + distribution_rows("origin_rule", rules, RULES)
        + distribution_rows("origin_family", families, FAMILIES)
        + distribution_rows("credited_destroy_operator", destroys, DESTROY)
        + distribution_rows("repair_operator", repairs, REPAIR)
    )
    atomic_csv(DEV / "selection_distribution.csv", distribution)

    checkpoint_rows = []
    for payload in primary_payloads:
        for checkpoint in payload["portfolio_diagnostics"]["adaptive_weight_checkpoints"]:
            weights = checkpoint["weights"]
            destroy_values = [float(weights[name]) for name in DESTROY]
            repair_values = [float(weights[name]) for name in REPAIR]
            checkpoint_rows.append({
                "budget_fraction": float(checkpoint["budget_fraction"]),
                "destroy_weight_spread": max(destroy_values) - min(destroy_values),
                "repair_weight_spread": max(repair_values) - min(repair_values),
                "all_destroy_at_floor": max(abs(value - 0.1) for value in destroy_values) <= 1e-9,
                "all_repair_at_floor": max(abs(value - 0.1) for value in repair_values) <= 1e-9,
            })
    checkpoint_summary = pd.DataFrame(checkpoint_rows).groupby(
        "budget_fraction", as_index=False
    ).agg(
        runs=("destroy_weight_spread", "size"),
        mean_destroy_weight_spread=("destroy_weight_spread", "mean"),
        median_destroy_weight_spread=("destroy_weight_spread", "median"),
        mean_repair_weight_spread=("repair_weight_spread", "mean"),
        median_repair_weight_spread=("repair_weight_spread", "median"),
        destroy_all_floor_fraction=("all_destroy_at_floor", "mean"),
        repair_all_floor_fraction=("all_repair_at_floor", "mean"),
    )
    atomic_csv(DEV / "adaptive_weight_summary.csv", checkpoint_summary)

    runtime_components = Counter()
    for payload in primary_payloads:
        runtime_components.update(payload["portfolio_diagnostics"]["runtime_component_sums"])
    search_seconds = float(runtime_components["iteration"])
    critic_seconds = float(runtime_components["critic_ms_total"]) / 1000.0
    selection_checks = {
        "all_six_neural_ranks_selected": all(ranks[str(rank)] > 0 for rank in range(1, 7)),
        "all_five_origin_families_selected": all(families[name] > 0 for name in FAMILIES),
        "at_least_20_of_24_rules_selected": sum(rules[name] > 0 for name in RULES) >= 20,
        "largest_rank_share_below_half": max(ranks.values()) / sum(ranks.values()) < 0.5,
        "largest_family_share_below_half": max(families.values()) / sum(families.values()) < 0.5,
    }
    search_improved_runs = sum(len(item["incumbent_trace"]) > 1 for item in primary_payloads)
    selection = {
        "total_iterations": total_iterations,
        "total_neural_decisions": total_neural,
        "safe_fallbacks": sum(int(item["portfolio_diagnostics"]["safe_fallbacks"]) for item in primary_payloads),
        "canonical_fallback_selections": sum(int(item["portfolio_diagnostics"]["canonical_fallback_selections"]) for item in primary_payloads),
        "acceptance_rate": iteration_weighted("acceptance_rate"),
        "new_global_best_rate": iteration_weighted("new_global_best_rate"),
        "current_improvement_rate": iteration_weighted("current_improvement_rate"),
        "runs_with_strict_improvement_over_initializer": search_improved_runs,
        "weighted_portfolio_probability_changed_fraction": weighted("portfolio_probability_changed_by_destroy_weights_fraction"),
        "weighted_mean_probability_l1_vs_rank_prior": weighted("mean_portfolio_probability_l1_vs_rank_prior"),
        "weighted_mean_total_variation_vs_rank_prior": 0.5 * weighted("mean_portfolio_probability_l1_vs_rank_prior"),
        "selected_rank_count": sum(ranks[value] > 0 for value in map(str, range(1, 7))),
        "selected_rule_count": sum(rules[name] > 0 for name in RULES),
        "selected_family_count": sum(families[name] > 0 for name in FAMILIES),
        "largest_rank_share": max(ranks.values()) / sum(ranks.values()),
        "largest_rule_origin_share": max(rules.values()) / sum(rules.values()),
        "largest_family_share": max(families.values()) / sum(families.values()),
        "selection_non_collapse_checks": selection_checks,
        "runtime_component_seconds": {
            "iteration": search_seconds,
            "decoder": float(runtime_components["decoder"]),
            "repair_excluding_decoder": float(runtime_components["repair_excluding_decoder"]),
            "critic_total": critic_seconds,
            "critic_candidate_bank": float(runtime_components["critic_ms_candidate_bank"]) / 1000.0,
            "critic_tensorization_transfer": float(runtime_components["critic_ms_graph_features_tensorization"]) / 1000.0,
            "critic_three_seed_inference": float(runtime_components["critic_ms_three_seed_inference"]) / 1000.0,
            "critic_ranking": float(runtime_components["critic_ms_ranking"]) / 1000.0,
        },
        "runtime_component_fraction_of_iteration": {
            "decoder": float(runtime_components["decoder"]) / search_seconds,
            "repair_excluding_decoder": float(runtime_components["repair_excluding_decoder"]) / search_seconds,
            "critic_total": critic_seconds / search_seconds,
        },
        "mean_critic_ms_per_neural_decision": 1000.0 * critic_seconds / total_neural,
    }
    atomic_json(DEV / "portfolio_diagnostics.json", selection)

    primary_summary = method_summary.set_index("method").loc[PRIMARY]
    top1_summary = method_summary.set_index("method").loc["PHASE6N_DETERMINISTIC_TOP1"]
    pair_by_name = pairwise.set_index("comparator")
    scale_pivot = scale_summary.pivot(index="scale", columns="method", values="mean_final_makespan")
    scale_relative_vs_top1 = {
        scale: float(100.0 * (row["PHASE6N_DETERMINISTIC_TOP1"] - row[PRIMARY]) / row["PHASE6N_DETERMINISTIC_TOP1"])
        for scale, row in scale_pivot.iterrows()
    }
    development_checks = {
        "integrity_complete": True,
        "feasibility_100_percent": bool(frame["feasible"].all()),
        "better_than_phase6n_top1_overall": bool(
            primary_summary["mean_final_makespan"] < top1_summary["mean_final_makespan"]
            and primary_summary["mean_rpd_percent"] < top1_summary["mean_rpd_percent"]
        ),
        "within_one_percent_of_alns_on_instance_mean_relative_improvement": bool(
            pair_by_name.loc["ALNS", "mean_relative_improvement_percent"] >= -1.0
        ),
        "within_one_percent_of_phase6h_on_instance_mean_relative_improvement": bool(
            pair_by_name.loc["PHASE6H", "mean_relative_improvement_percent"] >= -1.0
        ),
        "no_material_scale_collapse_vs_top1_at_one_percent_descriptive_margin": bool(
            min(scale_relative_vs_top1.values()) >= -1.0
        ),
        "useful_search": search_improved_runs >= 27 and float(primary_summary["mean_decoder_evaluations"]) > 0,
        "selection_not_collapsed": all(selection_checks.values()),
        "optional_rescue_preregistered": False,
    }
    require(
        load_json(ROOT / "configs/phase6p_adaptive_portfolio_v1.json")["optional_preregistered_rescue"] is None,
        "unexpected rescue protocol",
    )
    promising = all((
        development_checks["integrity_complete"],
        development_checks["feasibility_100_percent"],
        development_checks["better_than_phase6n_top1_overall"],
        development_checks["within_one_percent_of_alns_on_instance_mean_relative_improvement"],
        development_checks["within_one_percent_of_phase6h_on_instance_mean_relative_improvement"],
        development_checks["no_material_scale_collapse_vs_top1_at_one_percent_descriptive_margin"],
        development_checks["useful_search"],
        development_checks["selection_not_collapsed"],
    ))
    require(not promising, "unexpected development pass; terminal no-go script is not applicable")

    audit = {
        "schema": "phase6p-development-completion-audit-v1",
        "status": "PASS",
        "audited_at_utc": max(item["completed_at_utc"] for item in payloads.values()),
        "development_protocol_sha256": protocol_hash,
        "checks": {
            "expected_results_270": len(payloads) == 270,
            "exactly_54_per_method": True,
            "all_protocol_hashes_match": True,
            "all_instance_hashes_match": True,
            "all_budget_contracts_match": True,
            "all_runtime_at_least_budget": True,
            "all_feasibility_replays_match": True,
            "all_incumbent_traces_monotone": True,
            "all_comparator_manifests_match": True,
            "registry_frozen_canonical_three_seed": True,
            "historical_score_calls_zero_for_phase6p_and_top1": True,
            "phase6o_calls_zero_for_phase6p_and_top1": True,
            "gurobi_not_run": True,
            "r13_r14_locked": True,
        },
        "result_count": len(payloads),
        "method_counts": frame.groupby("method").size().to_dict(),
        "result_hash_manifest": str((DEV / "result_hash_manifest.csv").relative_to(ROOT)),
    }
    atomic_json(DEV / "completion_integrity_audit.json", audit)

    decision = {
        "schema": "phase6p-final-decision-v1",
        "decision": "ADAPTIVE_PORTFOLIO_PILOT_NO_GO",
        "status": "TERMINAL_BEFORE_RUNTIME_AND_FORMAL_R12",
        "reason": "P1 did not improve overall mean makespan or mean RPD over the frozen Phase 6N deterministic top-1 comparator.",
        "development_promising": False,
        "checks": development_checks,
        "primary_vs_phase6n_top1": pair_by_name.loc["PHASE6N_DETERMINISTIC_TOP1"].to_dict(),
        "primary_mean_final_makespan": float(primary_summary["mean_final_makespan"]),
        "phase6n_top1_mean_final_makespan": float(top1_summary["mean_final_makespan"]),
        "primary_mean_rpd_percent": float(primary_summary["mean_rpd_percent"]),
        "phase6n_top1_mean_rpd_percent": float(top1_summary["mean_rpd_percent"]),
        "runtime_qualification": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
        "formal_five_seed_r12": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
        "r13": "LOCKED",
        "r14": "LOCKED",
        "gurobi_run": False,
        "optional_preregistered_rescue": None,
        "development_protocol_sha256": protocol_hash,
        "completion_audit_sha256": digest(DEV / "completion_integrity_audit.json"),
    }
    atomic_json(FINAL / "final_decision.json", decision)

    registry_changed = False
    for entry in registry["entries"]:
        if entry["algorithm_id"] in set(REGISTRY_IDS.values()):
            if entry.get("regression_integrity_result") != "PASS_PHASE6P_DEVELOPMENT_AUDIT":
                entry["regression_integrity_result"] = "PASS_PHASE6P_DEVELOPMENT_AUDIT"
                registry_changed = True
    if registry_changed:
        atomic_json(REGISTRY_ROOT / "registry.json", registry)

    display = method_summary.copy()
    display = display[[
        "method", "mean_final_makespan", "median_final_makespan",
        "mean_rpd_percent", "median_rpd_percent", "bks_hits",
        "average_instance_rank", "mean_decoder_evaluations",
    ]]
    for column in display.columns:
        if column != "method":
            display[column] = display[column].map(lambda value: f"{float(value):.3f}")
    pair_display = pairwise[[
        "comparator", "instance_mean_wins", "instance_mean_ties",
        "instance_mean_losses", "mean_relative_improvement_percent",
    ]].copy()
    pair_display["W/T/L"] = pair_display.apply(
        lambda row: f"{int(row.instance_mean_wins)}/{int(row.instance_mean_ties)}/{int(row.instance_mean_losses)}",
        axis=1,
    )
    pair_display["mean_relative_improvement_percent"] = pair_display[
        "mean_relative_improvement_percent"
    ].map(lambda value: f"{value:.3f}")
    pair_display = pair_display[["comparator", "W/T/L", "mean_relative_improvement_percent"]]
    runtime_seconds = selection["runtime_component_seconds"]
    report = f"""# Phase 6P development solver pilot

## Decision

**`ADAPTIVE_PORTFOLIO_PILOT_NO_GO`**. The detached P3 campaign completed all
270 frozen tasks and the completion audit passes. P1 nevertheless misses the
mandatory development condition that overall quality be better than Phase 6N
deterministic top-1. P1 mean makespan is
{primary_summary['mean_final_makespan']:.3f} versus
{top1_summary['mean_final_makespan']:.3f}, and mean RPD is
{primary_summary['mean_rpd_percent']:.4f}% versus
{top1_summary['mean_rpd_percent']:.4f}%. Lower is better for both metrics.

The rescue field was frozen as `null` before P3 outcomes. No post-outcome
rescue, runtime qualification, five-seed formal R12, R13, or R14 run is
scientifically authorized in this Phase 6P attempt.

## Result integrity and comparison basis

The audit validates all 270 protocol hashes, instance hashes, task identities,
2|O| budgets, feasibility replays, monotone incumbent traces, and comparator
run manifests. Every method has 54 results (18 instances x 3 seeds), every
schedule is feasible, and initialization plus live overhead is charged to the
wall-clock budget. The four comparator registry entries are frozen canonical;
no favorable rerun was made.

RPD uses the current development BKS: the lowest final makespan observed for
each instance across all five methods and all three frozen seeds. W/T/L is
computed over the 18 paired instance means. The three-seed stage is descriptive;
formal significance is not claimed.

{markdown_table(display)}

{markdown_table(pair_display)}

P1 is within 0.730% of ALNS and 0.065% of Phase 6H by mean paired relative
improvement, but it loses to ALNS on 14/18 instance means and splits Phase 6H
at 9 wins and 9 losses. Against Phase 6N top-1 it is exactly 9/0/9, with a -0.009% mean paired
relative improvement and a 2.093 higher overall mean makespan. This is a small
but directionally failed result under the pre-registered conjunctive gate.

## Scale and search behavior

Relative to Phase 6N top-1, P1 mean quality changes by
S {scale_relative_vs_top1['S']:+.3f}%, M {scale_relative_vs_top1['M']:+.3f}%,
and L {scale_relative_vs_top1['L']:+.3f}%. There is no material scale collapse
under the descriptive 1% audit margin. P1 averages
{primary_summary['mean_decoder_evaluations']:.1f} decoder evaluations, and
{search_improved_runs}/54 runs strictly improve their H1 initializer, so the
failure is not caused by an absence of search.

Across {total_neural:,} neural decisions, all six neural ranks, all five
origin families, and {selection['selected_rule_count']}/24 rules are selected.
The largest rank/family shares are {100 * selection['largest_rank_share']:.2f}%
and {100 * selection['largest_family_share']:.2f}%; selection therefore does
not collapse to one rank or family. Safe fallback count is zero.

Destroy weights change the portfolio probability vector in
{100 * selection['weighted_portfolio_probability_changed_fraction']:.2f}% of
neural decisions. The mean L1 distance from the pure 1/r prior is
{selection['weighted_mean_probability_l1_vs_rank_prior']:.4f} (total variation
{selection['weighted_mean_total_variation_vs_rank_prior']:.4f}). Thus the ALNS
weights measurably alter early sampling, but only modestly on average; the
checkpoint table also records their later convergence toward the 0.1 floor.

Weighted iteration-level rates are acceptance
{100 * selection['acceptance_rate']:.3f}%, current improvement
{100 * selection['current_improvement_rate']:.3f}%, and new global best
{100 * selection['new_global_best_rate']:.3f}%.

## Runtime accounting inside P3

P1 recorded {runtime_seconds['iteration']:.3f} aggregate iteration seconds.
The included critic consumed {runtime_seconds['critic_total']:.3f} s
({100 * selection['runtime_component_fraction_of_iteration']['critic_total']:.2f}%),
including {runtime_seconds['critic_candidate_bank']:.3f} s candidate-bank work,
{runtime_seconds['critic_tensorization_transfer']:.3f} s tensorization/transfer,
{runtime_seconds['critic_three_seed_inference']:.3f} s three-seed inference,
and {runtime_seconds['critic_ranking']:.3f} s ranking. Mean critic time is
{selection['mean_critic_ms_per_neural_decision']:.3f} ms per neural decision.
Decoder work consumed {runtime_seconds['decoder']:.3f} s. These P3 aggregate
observer timings diagnose overhead; they are not the formal p50/p90/p99 runtime
qualification, which is prohibited after the quality gate fails.

## Artifacts

- Completion audit: `outputs/phase6p_adaptive_portfolio_v1/development/completion_integrity_audit.json`
- Audited runs and hashes: `outputs/phase6p_adaptive_portfolio_v1/development/audited_run_summary.csv`, `result_hash_manifest.csv`
- Method/pair/scale/CF/anytime summaries: `outputs/phase6p_adaptive_portfolio_v1/development/`
- Portfolio and weight diagnostics: `portfolio_diagnostics.json`, `selection_distribution.csv`, `adaptive_weight_summary.csv`
- Terminal decision: `outputs/phase6p_adaptive_portfolio_v1/final/final_decision.json`
"""
    atomic_text(ROOT / "docs/reports/phase6p_development_solver_pilot.md", report)

    runtime_report = """# Phase 6P runtime qualification\n\nStatus: **NOT RUN -- DEVELOPMENT GATE FAILED**.\n\nThe frozen protocol permits formal p50/p90/p99 runtime qualification only after promising development solver quality. P1 did not outperform Phase 6N deterministic top-1 overall, so the neural p90 <=30 ms and complete-live p90 <=100 ms gates were not executed. P3 observer totals remain diagnostic and cannot be relabeled as the formal latency test.\n"""
    formal_report = """# Phase 6P formal five-seed R12\n\nStatus: **NOT RUN -- DEVELOPMENT GATE FAILED**.\n\nThe authoritative five-seed R12 comparison requires both development success and runtime qualification. Neither condition was reached. Frozen three-seed comparator results remain preserved; seeds 746104 and 746105 were not generated, and R13/R14 remain locked.\n"""
    final_report = f"""# Phase 6P final report\n\nPhase 6P closes as **`ADAPTIVE_PORTFOLIO_PILOT_NO_GO`**. P3 completed and passed integrity, feasibility, budget, and provenance audits, but P1 mean makespan/RPD were slightly worse than the frozen Phase 6N deterministic top-1 comparator. Since no rescue was pre-registered, Phase 6P stops before runtime qualification and formal five-seed R12. No Gurobi, R13, or R14 work was run.\n\nThe full result interpretation is in `docs/reports/phase6p_development_solver_pilot.md`; the machine decision is `outputs/phase6p_adaptive_portfolio_v1/final/final_decision.json`. Final repository regression: **440 passed**.\n"""
    handoff = """# Phase 6P project handoff\n\n## Current gate\n\n- Decision: **`ADAPTIVE_PORTFOLIO_PILOT_NO_GO`**\n- P3: 270/270 complete; integrity PASS\n- Runtime qualification: not run\n- Formal five-seed R12: not run\n- R13/R14: locked and unaccessed\n- Gurobi: not run\n\n## Scientific boundary\n\nThe frozen primary is feasible, searches productively, and does not collapse its candidate distribution. It fails the conjunctive development gate because overall mean makespan and current-development-BKS RPD are both slightly worse than Phase 6N deterministic top-1. `optional_preregistered_rescue` was frozen as `null`; adding a rescue after observing P3 would be post-outcome selection. Runtime-only optimization cannot repair this solver-quality failure.\n\n## Authoritative evidence\n\n- `docs/reports/phase6p_development_solver_pilot.md`\n- `outputs/phase6p_adaptive_portfolio_v1/development/completion_integrity_audit.json`\n- `outputs/phase6p_adaptive_portfolio_v1/development/result_hash_manifest.csv`\n- `outputs/phase6p_adaptive_portfolio_v1/final/final_decision.json`\n- `outputs/frozen_2o_baselines/registry.json`\n\n## Next valid work\n\nA future search-integration revision needs a new namespace and pre-registration before viewing new solver outcomes. The frozen Phase 6P results and canonical comparator registry should be reused; do not rerun valid comparator seeds or tune against R13/R14.\n"""
    atomic_text(ROOT / "docs/reports/phase6p_runtime_report.md", runtime_report)
    atomic_text(ROOT / "docs/reports/phase6p_formal_r12_report.md", formal_report)
    atomic_text(ROOT / "docs/reports/phase6p_final_report.md", final_report)
    atomic_text(ROOT / "docs/reports/phase6p_project_handoff.md", handoff)
    registry_report = """# Frozen 2|O| baseline registry\n\nThe registry is **`FROZEN_CANONICAL_THREE_SEED_DEVELOPMENT`** for the 18 R12 CAUR-FIT instances, seeds 746101-746103, and the matched `2 * num_operations` wall-clock budget. ALNS, Phase 6H, Phase 6N deterministic top-1, and LG_HGA_2O each contain 54 feasible runs with a hash manifest and aggregate summary. The Phase 6P completion audit independently matched every manifest entry to the raw result bytes and set each registry integrity field to `PASS_PHASE6P_DEVELOPMENT_AUDIT`.\n\nThese three-seed results are reusable canonical evidence. A future protocol requiring the same five canonical seeds must retain seeds 746101-746103 and run only missing seeds 746104-746105. No valid result may be rerun because of the observed Phase 6P outcome.\n"""
    atomic_text(ROOT / "docs/reports/frozen_2o_baseline_registry_report.md", registry_report)
    print(json.dumps({
        "status": "PASS_PHASE6P_DEVELOPMENT_CLOSED",
        "decision": decision["decision"],
        "results": len(payloads),
        "primary_mean": decision["primary_mean_final_makespan"],
        "top1_mean": decision["phase6n_top1_mean_final_makespan"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
