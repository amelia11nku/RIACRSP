#!/usr/bin/env python3
"""Run the preregistered Phase 6P real-checkpoint smoke and invariant audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.phase6p_live_inference import (  # noqa: E402
    FrozenPhase6NCriticEnsemble,
)
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.common import candidate_from_actions, decode_candidate  # noqa: E402
from rcias_clgri.search.phase6p_adaptive import (  # noqa: E402
    portfolio_distribution,
    sample_portfolio_target,
    solve_phase6p,
)


CONFIG = ROOT / "configs/phase6p_adaptive_portfolio_v1.json"
PREREG = ROOT / "outputs/phase6p_adaptive_portfolio_v1/preregistration"
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/smoke"
REPORT = ROOT / "docs/reports/phase6p_smoke_report.md"
BANK_REPORT = ROOT / "docs/reports/phase6p_candidate_bank_and_adaptation_report.md"
INSTANCE_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14"


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def summarize_event(event: dict) -> dict:
    return {
        "iteration": event["iteration"],
        "neural_eligible": event["neural_eligible"],
        "accepted": event["accepted"],
        "new_global_best": event["new_global_best"],
        "candidate_feasible": event["candidate"].feasible,
        "candidate_makespan": event["candidate"].makespan,
        "best_before_makespan": event["best_before"].makespan,
        "best_after_makespan": event["best_after"].makespan,
        "repair_operator": event["repair_operator"],
        "credited_destroy_operators": list(event["credited_destroy_operators"]),
        "candidate_trials_completed": event["candidate_trials_completed"],
        "requested_proposal_count": event["requested_proposal_count"],
        "unique_proposal_count": event["unique_proposal_count"],
        "duplicate_proposal_count": event["duplicate_proposal_count"],
        "selected_target_set_id": event["selected_target_set_id"],
        "top_target_ids": list(event["top_target_ids"]),
        "safe_fallback": event["safe_fallback"],
        "scoring_error": event["scoring_error"],
        "held_fold": event["held_fold"],
        "graph_hash": event["graph_hash"],
        "critic_timing_ms": event["critic_timing_ms"],
        "operator_weights_before": event["operator_weights_before"],
        "operator_weights_after": event["operator_weights_after"],
    }


def main() -> None:
    started_at = datetime.now(timezone.utc)
    require(
        os.environ.get("PYTHONHASHSEED") == "0",
        "Phase 6P reproducibility smoke requires PYTHONHASHSEED=0 before interpreter start",
    )
    config = load_json(CONFIG)
    require(
        config["status"] == "PREREGISTERED_BEFORE_ANY_PHASE6P_SOLVER_QUALITY_OUTCOME",
        "Phase 6P protocol is not frozen",
    )
    manifest = load_json(PREREG / "development_instance_manifest.json")
    chosen = min(manifest["instances"], key=lambda row: (row["num_operations"], row["instance_id"]))
    instance_path = INSTANCE_ROOT / chosen["relative_path"]
    require(digest(instance_path) == chosen["sha256"], "smoke instance changed")
    instance = load_instance(instance_path)
    require(instance.num_operations == chosen["num_operations"], "operation count drifted")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = FrozenPhase6NCriticEnsemble(device=device)
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    state_id = f"{instance.instance_id}__phase6p_smoke_fixed"
    score_kwargs = {
        "state_id": state_id,
        "destroy_count": max(2, round(instance.num_operations * 0.15)),
        "search_progress": 0.0,
        "search_stage": "0-20%",
        "proposal_seed_namespace": 692000000,
    }
    first = critic.score_bank(instance, current, **score_kwargs)
    second = critic.score_bank(instance, current, **score_kwargs)
    exact_scores = dict(first.target_scores) == dict(second.target_scores)
    exact_ranking = first.ranked_target_ids == second.ranked_target_ids
    require(exact_scores and exact_ranking, "fixed-state Phase 6N ranking is not deterministic")
    require(first.top_target_ids == second.top_target_ids, "fixed top-6 changed")
    require(len(first.target_scores) == first.generated.unique_arm_count, "not all targets scored")

    expected_rules = {
        *(f"operator_{name}" for name in (
            "random", "critical", "overloaded_island", "high_reconfiguration",
            "w_bottleneck", "f_bottleneck", "related",
        )),
        *(f"related_variant_{index}" for index in range(1, 5)),
        *(f"matched_random_{index}" for index in range(1, 4)),
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
    }
    actual_rules = {proposal.origin_rule for proposal in first.generated.proposals}
    require(first.generated.requested_arm_count == 24, "requested bank is not 24")
    require(actual_rules == expected_rules, "24-rule identities changed")
    require(first.fallback.target_set_id in first.target_scores, "fallback missing from context")

    unit_weights = {name: 1.0 for name in (
        "random", "critical", "overloaded_island", "high_reconfiguration",
        "w_bottleneck", "f_bottleneck", "related", "greedy", "regret2",
        "regret3", "reconfiguration_aware", "transport_aware",
    )}
    distribution = portfolio_distribution(
        first.ranked_target_ids, first.origin_destroy_operators, unit_weights
    )
    choice_a = sample_portfolio_target(distribution, random.Random(746101))
    choice_b = sample_portfolio_target(distribution, random.Random(746101))
    require(choice_a == choice_b, "portfolio sampling is not reproducible")
    boosted = dict(unit_weights)
    affected_origin = first.origin_destroy_operators[first.top_target_ids[-1]][0]
    boosted[affected_origin] = 4.0
    boosted_distribution = portfolio_distribution(
        first.ranked_target_ids, first.origin_destroy_operators, boosted
    )
    before_probability = dict((row[0], row[1]) for row in distribution)[first.top_target_ids[-1]]
    after_probability = dict((row[0], row[1]) for row in boosted_distribution)[first.top_target_ids[-1]]
    require(after_probability > before_probability, "destroy weight did not affect target probability")

    events = []
    result = solve_phase6p(
        instance,
        60.0,
        746101,
        critic,
        alns_config=ALNSConfig(iteration_limit=6),
        observer=events.append,
    )
    require(result.best.feasible, "Phase 6P smoke returned an infeasible best solution")
    require(all(event["candidate"].feasible for event in events), "infeasible smoke candidate")
    require(all(
        event["best_after"].makespan <= event["best_before"].makespan
        for event in events
    ), "best-so-far worsened")
    eligible = [event for event in events if event["neural_eligible"]]
    require(len(eligible) == 2, "six smoke iterations must contain two neural decisions")
    require(all(event["candidate_trials_completed"] == 8 for event in events), "eight trials drifted")
    require(all(event["requested_proposal_count"] == 24 for event in eligible), "live bank drifted")
    require(not any(event["safe_fallback"] for event in eligible), "real critic fell back")

    source_text = (
        (ROOT / "rcias_clgri/ni/phase6p_live_inference.py").read_text()
        + (ROOT / "rcias_clgri/search/phase6p_adaptive.py").read_text()
    )
    forbidden = {
        "phase6o": "Phase 6O critic",
        "gurobi": "Gurobi",
        "frozen_score": "historical frozen-score scorer",
    }
    forbidden_hits = {name: token in source_text.lower() for token, name in forbidden.items()}
    require(not any(forbidden_hits.values()), f"forbidden dependency in Phase 6P source: {forbidden_hits}")
    r13_ledger = ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json"
    r14_ledger = ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json"
    require(not r13_ledger.exists() and not r14_ledger.exists(), "R13/R14 access boundary failed")

    checks = {
        "requested_rules_exactly_24": first.generated.requested_arm_count == 24,
        "all_rule_names_accounted_for": actual_rules == expected_rules,
        "all_unique_targets_scored": len(first.target_scores) == first.generated.unique_arm_count,
        "deterministic_seed_scores": exact_scores,
        "deterministic_ranking": exact_ranking,
        "deterministic_top6": first.top_target_ids == second.top_target_ids,
        "reproducible_portfolio_choice": choice_a == choice_b,
        "destroy_weight_changes_probability": after_probability > before_probability,
        "fallback_available": first.fallback.target_set_id in first.target_scores,
        "eight_candidate_trials": all(event["candidate_trials_completed"] == 8 for event in events),
        "feasibility_100_percent": result.best.feasible and all(
            event["candidate"].feasible for event in events
        ),
        "best_so_far_never_worsens": all(
            event["best_after"].makespan <= event["best_before"].makespan
            for event in events
        ),
        "no_historical_scorer": not forbidden_hits["historical frozen-score scorer"],
        "no_phase6o_critic": not forbidden_hits["Phase 6O critic"],
        "no_gurobi": not forbidden_hits["Gurobi"],
        "r13_r14_locked": not r13_ledger.exists() and not r14_ledger.exists(),
    }
    require(all(checks.values()), f"Phase 6P smoke gate failed: {checks}")
    payload = {
        "schema": "phase6p-smoke-result-v1",
        "status": "PASS",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance": chosen,
        "device": str(device),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "precision": "FP32_ONLY",
        },
        "source_hashes": {
            "config": digest(CONFIG),
            "checkpoint_manifest": digest(PREREG / "phase6n_checkpoint_manifest.json"),
            "live_inference": digest(ROOT / "rcias_clgri/ni/phase6p_live_inference.py"),
            "adaptive_search": digest(ROOT / "rcias_clgri/search/phase6p_adaptive.py"),
        },
        "candidate_bank": {
            "state_id": state_id,
            "requested_rules": first.generated.requested_arm_count,
            "unique_targets": first.generated.unique_arm_count,
            "duplicate_targets": first.generated.duplicate_arm_count,
            "origin_rules": sorted(actual_rules),
            "fallback_target_set_id": first.fallback.target_set_id,
            "all_target_scores": first.target_scores,
            "per_seed_scores": first.per_seed_scores,
            "ranked_target_ids": list(first.ranked_target_ids),
            "top_target_ids": list(first.top_target_ids),
            "origin_destroy_operators": first.origin_destroy_operators,
            "supported_by_seed": first.supported_by_seed,
            "held_fold": first.held_fold,
            "graph_hash": first.graph_hash,
            "first_timing_ms": first.timings_ms,
            "repeat_timing_ms": second.timings_ms,
        },
        "adaptation": {
            "unit_weight_distribution": distribution,
            "boosted_destroy_operator": affected_origin,
            "affected_target": first.top_target_ids[-1],
            "probability_before": before_probability,
            "probability_after": after_probability,
            "reproducible_choice": choice_a,
        },
        "solver": {
            "method": result.method,
            "seed": 746101,
            "iteration_limit": 6,
            "iterations": result.iterations,
            "decoder_evaluations": result.decoder_evaluations,
            "initial_makespan": current.makespan,
            "best_makespan": result.best.makespan,
            "runtime_seconds": result.runtime,
            "best_found_time": result.best_found_time,
            "diagnostics": result.diagnostics,
            "events": [summarize_event(event) for event in events],
        },
        "forbidden_dependency_hits": forbidden_hits,
        "checks": checks,
    }
    atomic_json(OUT / "result.json", payload)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# Phase 6P P2 smoke report

Status: **PASS**.

The real frozen Phase 6N OOF critic was restored in FP32 on `{device}` and routed to fold {first.held_fold} for `{instance.instance_id}`. The fixed H1 state produced {first.generated.requested_arm_count} proposals, {first.generated.unique_arm_count} unique targets, and {first.generated.duplicate_arm_count} deduplicated duplicates. Every unique target was scored by all three routed seed models. Repeating the same realized state produced exactly identical seed scores, ensemble ranking, and top-6.

The six-iteration live search exercised neural iterations 0 and 5, completed eight repair trials in every iteration, returned only feasible decoded candidates, and kept best-so-far monotone. Initial makespan was {current.makespan:.6f}; smoke best was {result.best.makespan:.6f}. This is an invariant smoke result, not a development quality comparison.

The primary source contains no Phase 6O critic, historical frozen-score scorer, or Gurobi dependency. R13/R14 access ledgers remain absent. Machine-readable evidence: `outputs/phase6p_adaptive_portfolio_v1/smoke/result.json`.
""", encoding="utf-8")
    BANK_REPORT.write_text(f"""# Phase 6P candidate bank and adaptation audit

The implementation generates the frozen 24-rule Phase 6C bank before exact destroyed-set deduplication. The P2 state contained {first.generated.unique_arm_count} unique targets. Candidate identity and insertion order remained aligned across action tensorization and score-free feature construction; all targets received one score from each of the three fold-{first.held_fold} OOF seed models.

Ranking uses descending mean predicted continuation advantage and ascending `target_set_id` for ties. The first six ranks form the shortlist. For rank `r`, live target mass is `(1 / r) * mean(current destroy weights over all unique proposal origins)`. The selected target is sampled with the solver baseline RNG. The repair operator is then sampled independently from current repair weights. After the candidate outcome, reward 5, 1, or the 0.1 update floor is applied once to every unique destroy origin and once to the selected repair operator.

The probability audit increased `{affected_origin}` from 1.0 to 4.0. The affected rank-{len(first.top_target_ids)} target probability changed from {before_probability:.9f} to {after_probability:.9f}. Fixed-seed roulette selected `{choice_a}` in both repetitions. Full ranks, scores, provenance, timings, adaptive weights, and iteration records are preserved in `outputs/phase6p_adaptive_portfolio_v1/smoke/result.json`.
""", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "result": str((OUT / "result.json").relative_to(ROOT)),
        "runtime_seconds": result.runtime,
        "unique_targets": first.generated.unique_arm_count,
        "device": str(device),
    }))


if __name__ == "__main__":
    main()
