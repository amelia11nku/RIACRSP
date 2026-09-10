#!/usr/bin/env python3
"""Independently audit the completed NGAS-A1.3R formal artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.revised_train import load_cache, model_for, prepare

OUT = ROOT / "outputs/ngas_a1/critic_training_rthgt_v2"
C0_OUT = ROOT / "outputs/ngas_a1/critic_training_gpu_v1"
PROTOCOL = OUT / "training_protocol.json"
CACHE = OUT / "data/training_cache_v2.json.gz"
CACHE_MANIFEST = OUT / "data/training_cache_manifest_v2.json"
AUDIT = OUT / "audit/completion_audit.json"
REPLAY = OUT / "audit/production_replay.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_gzip_json(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(actual: list[float], predicted: list[float]) -> float:
    left, right = average_ranks(actual), average_ranks(predicted)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right))
    return numerator / denominator if denominator else 0.0


def material_pairs(replicates: list[list[float]], minimum_gap: float,
                   se_multiplier: float) -> list[tuple[int, int, int]]:
    result = []
    for left in range(len(replicates)):
        for right in range(left + 1, len(replicates)):
            differences = [
                a - b for a, b in zip(replicates[left], replicates[right])]
            mean = statistics.fmean(differences)
            standard_error = (
                statistics.stdev(differences) / math.sqrt(len(differences))
                if len(differences) > 1 else math.inf)
            if abs(mean) > max(minimum_gap, se_multiplier * standard_error):
                result.append((left, right, 1 if mean > 0 else -1))
    return result


def state_metrics(record: dict, prediction: dict, objective: dict) -> dict:
    predicted = prediction["predicted_advantage"]
    probabilities = prediction["predicted_beats_fallback_probability"]
    replicates = record["replicate_advantages"]
    require(len(predicted) == len(record["actions"]), "advantage/action size mismatch")
    require(len(probabilities) == len(record["actions"]), "probability/action size mismatch")
    require(all(math.isfinite(value) for value in predicted), "non-finite advantage")
    require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities),
            "invalid beats-fallback probability")
    require(all(math.isfinite(value) for value in prediction["predicted_beats_fallback_logit"]),
            "non-finite beats-fallback logit")
    actual = [statistics.fmean(values) for values in replicates]
    pairs = material_pairs(
        replicates, float(objective["minimum_material_gap"]),
        float(objective["paired_SE_multiplier"]))
    correct = 0.0
    for left, right, direction in pairs:
        predicted_direction = (
            (predicted[left] > predicted[right])
            - (predicted[left] < predicted[right]))
        correct += 1.0 if predicted_direction == direction else (
            0.5 if predicted_direction == 0 else 0.0)
    order = sorted(range(len(actual)), key=lambda index: (-predicted[index], index))
    ideal = sorted(range(len(actual)), key=lambda index: (-actual[index], index))
    relevance = [value - min(actual) for value in actual]

    def ndcg(cutoff: int) -> float:
        def dcg(indices: list[int]) -> float:
            return sum(
                relevance[index] / math.log2(rank + 2)
                for rank, index in enumerate(indices[:cutoff]))
        denominator = dcg(ideal)
        return dcg(order) / denominator if denominator else 1.0

    selected = order[0]
    best = max(actual)
    observed = record["beats_fallback_frequency"]
    return {
        "spearman": spearman(actual, predicted),
        "material_pairs": len(pairs),
        "material_pair_correct": correct,
        "ndcg_at_1": ndcg(1),
        "ndcg_at_3": ndcg(min(3, len(actual))),
        "ndcg_at_5": ndcg(min(5, len(actual))),
        "selected_advantage": actual[selected],
        "top1_regret": best - actual[selected],
        "uniform_expected_regret": best - statistics.fmean(actual),
        "continuation_lift": actual[selected] - statistics.fmean(actual),
        "beats_fallback_brier": statistics.fmean(
            (probability - frequency) ** 2
            for probability, frequency in zip(probabilities, observed)),
        "beats_fallback_accuracy": statistics.fmean(
            float((probability >= 0.5) == (frequency > 0))
            for probability, frequency in zip(probabilities, observed)),
        "selected_beats_fallback_probability": probabilities[selected],
        "selected_beats_fallback_frequency": observed[selected],
    }


def summarize(rows: list[dict]) -> dict:
    pairs = sum(row["material_pairs"] for row in rows)
    correct = sum(row["material_pair_correct"] for row in rows)

    def mean(field: str) -> float:
        return statistics.fmean(row[field] for row in rows)

    return {
        "states": len(rows),
        "mean_state_spearman": mean("spearman"),
        "material_pairs": pairs,
        "material_pair_accuracy": correct / pairs,
        "mean_ndcg_at_1": mean("ndcg_at_1"),
        "mean_ndcg_at_3": mean("ndcg_at_3"),
        "mean_ndcg_at_5": mean("ndcg_at_5"),
        "mean_selected_advantage": mean("selected_advantage"),
        "mean_top1_regret": mean("top1_regret"),
        "mean_uniform_expected_regret": mean("uniform_expected_regret"),
        "mean_continuation_lift": mean("continuation_lift"),
        "beats_fallback_brier": mean("beats_fallback_brier"),
        "beats_fallback_accuracy": mean("beats_fallback_accuracy"),
        "mean_selected_beats_fallback_probability":
            mean("selected_beats_fallback_probability"),
        "mean_selected_beats_fallback_frequency":
            mean("selected_beats_fallback_frequency"),
    }


def compare_summary(observed: dict, expected: dict, label: str) -> None:
    for field, value in observed.items():
        require(field in expected, f"{label}: missing summary field {field}")
        require(close(value, expected[field]),
                f"{label}: summary mismatch for {field}: {value} != {expected[field]}")


def c0_tree_hash() -> tuple[str, int]:
    files = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(C0_OUT.rglob("*")) if path.is_file()
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), len(files)


def audit_protocol(protocol: dict) -> dict:
    require(protocol["status"] == "FROZEN_BEFORE_FORMAL_OPTIMIZER",
            "formal protocol is not frozen")
    for relative, expected in {**protocol["source_hashes"], **protocol["data_hashes"]}.items():
        require(digest(ROOT / relative) == expected, f"frozen hash changed: {relative}")
    tree_hash, file_count = c0_tree_hash()
    historical = protocol["historical_C0"]
    require(tree_hash == historical["output_tree_sha256"], "historical C0 tree changed")
    require(file_count == historical["files"], "historical C0 file count changed")
    return {
        "training_protocol_sha256": digest(PROTOCOL),
        "source_files_verified": len(protocol["source_hashes"]),
        "data_files_verified": len(protocol["data_hashes"]),
        "historical_C0_tree_sha256": tree_hash,
        "historical_C0_files": file_count,
    }


def audit_variant(variant: str, protocol: dict, records: list[dict],
                  records_by_id: dict[str, dict]) -> tuple[dict, list[dict]]:
    protocol_sha = digest(PROTOCOL)
    config = protocol["config"]
    seeds = config["training"]["seeds"]
    epochs = int(config["training"]["epochs"])
    combined_predictions = []
    seed_summaries = []
    run_hashes = []
    for seed in seeds:
        predictions = []
        for fold in range(int(config["training"]["folds"])):
            directory = OUT / "variants" / variant / f"seed_{seed}" / f"fold_{fold}"
            checkpoint = directory / "model.pt"
            prediction_path = directory / "predictions.json.gz"
            run_path = directory / "run.json"
            require(all(path.is_file() for path in (checkpoint, prediction_path, run_path)),
                    f"missing {variant}/{seed}/fold_{fold} artifact")
            run = load_json(run_path)
            value = load_gzip_json(prediction_path)
            expected_held = {
                row["state_id"] for row in records if int(row["fold"]) == fold}
            expected_held_instances = sorted({
                row["instance_id"] for row in records if int(row["fold"]) == fold})
            expected_train_instances = sorted({
                row["instance_id"] for row in records if int(row["fold"]) != fold})
            observed_ids = {row["state_id"] for row in value["predictions"]}
            require(run["training_protocol_sha256"] == protocol_sha,
                    "run protocol hash mismatch")
            require(value["training_protocol_sha256"] == protocol_sha,
                    "prediction protocol hash mismatch")
            require((run["variant"], run["seed"], run["held_fold"]) ==
                    (variant, seed, fold), "run identity mismatch")
            require((value["variant"], value["seed"], value["held_fold"]) ==
                    (variant, seed, fold), "prediction identity mismatch")
            require(run["checkpoint_sha256"] == digest(checkpoint),
                    "checkpoint hash mismatch")
            require(run["predictions_sha256"] == digest(prediction_path),
                    "prediction hash mismatch")
            require(run["epochs"] == epochs and len(run["history"]) == epochs,
                    "fixed epoch count mismatch")
            require([row["epoch"] for row in run["history"]] == list(range(1, epochs + 1)),
                    "epoch history is incomplete")
            require(run["checkpoint_selection"] == "fixed final epoch",
                    "checkpoint selection mismatch")
            require(run["instance_overlap"] == [], "instance leakage reported")
            require(run["held_instances"] == expected_held_instances,
                    "held instance set mismatch")
            require(run["train_instances"] == expected_train_instances,
                    "training instance set mismatch")
            require(observed_ids == expected_held and len(value["predictions"]) == 24,
                    "held prediction coverage mismatch")
            require(run["r13"] == run["r14"] == "LOCKED" and not run["gurobi_run"],
                    "forbidden scope access in run")
            predictions.extend(value["predictions"])
            run_hashes.append({
                "variant": variant, "seed": seed, "fold": fold,
                "run_sha256": digest(run_path),
                "checkpoint_sha256": digest(checkpoint),
                "predictions_sha256": digest(prediction_path),
            })
        require(len(predictions) == 72, "seed OOF coverage mismatch")
        require(len({row["state_id"] for row in predictions}) == 72,
                "duplicate state in seed OOF predictions")
        rows = [
            state_metrics(records_by_id[row["state_id"]], row, config["objective"])
            for row in sorted(predictions, key=lambda item: item["state_id"])
        ]
        seed_summaries.append(summarize(rows))
        combined_predictions.extend({"seed": seed, **row} for row in predictions)

    result_path = OUT / "variants" / variant / "oof_result.json"
    combined_path = OUT / "variants" / variant / "oof_predictions.json.gz"
    saved = load_json(result_path)
    combined = load_gzip_json(combined_path)
    require(combined["training_protocol_sha256"] == protocol_sha,
            "combined prediction protocol hash mismatch")
    require(combined["variant"] == variant, "combined prediction variant mismatch")
    key = lambda row: (row["seed"], row["state_id"])
    require(sorted(combined["predictions"], key=key) ==
            sorted(combined_predictions, key=key), "combined predictions differ from fold files")
    require(len(combined_predictions) == 216 and
            len({key(row) for row in combined_predictions}) == 216,
            "combined OOF coverage mismatch")
    for index, summary in enumerate(seed_summaries):
        require(saved["seed_results"][index]["seed"] == seeds[index],
                "saved seed order mismatch")
        compare_summary(summary, saved["seed_results"][index]["summary"],
                        f"{variant}/seed_{seeds[index]}")
    aggregate = {
        field: statistics.fmean(summary[field] for summary in seed_summaries)
        for field in seed_summaries[0]
    }
    compare_summary(aggregate, saved["mean_across_seeds"], f"{variant}/mean")
    return {
        "runs_verified": len(run_hashes),
        "oof_predictions": len(combined_predictions),
        "unique_seed_state_pairs": len({key(row) for row in combined_predictions}),
        "mean_across_seeds": aggregate,
        "gate_pass": saved["gate_pass"],
        "seeds_passing": saved["seeds_passing"],
        "oof_result_sha256": digest(result_path),
        "oof_predictions_sha256": digest(combined_path),
        "run_artifacts": run_hashes,
    }, combined_predictions


def audit_c0(protocol: dict, records_by_id: dict[str, dict], comparison: dict) -> dict:
    values = load_gzip_json(C0_OUT / "oof_predictions.json.gz")["predictions"]
    objective = protocol["config"]["objective"]
    summaries = []
    for seed in protocol["config"]["training"]["seeds"]:
        predictions = [row for row in values if row["seed"] == seed]
        require(len(predictions) == 72 and len({row["state_id"] for row in predictions}) == 72,
                "historical C0 OOF coverage mismatch")
        rows = [
            state_metrics(records_by_id[row["state_id"]], row, objective)
            for row in sorted(predictions, key=lambda item: item["state_id"])
        ]
        summaries.append(summarize(rows))
    aggregate = {
        field: statistics.fmean(summary[field] for summary in summaries)
        for field in summaries[0]
    }
    compare_summary(aggregate, comparison["C0"]["mean_across_seeds"], "C0/mean")
    return {
        "oof_predictions": len(values),
        "mean_across_seeds": aggregate,
        "oof_predictions_sha256": digest(C0_OUT / "oof_predictions.json.gz"),
    }


def audit_selection(protocol: dict, comparison: dict, profile: dict) -> dict:
    selection = protocol["config"]["selection"]
    c1 = comparison["C1"]["mean_across_seeds"]
    r1 = comparison["R1"]["mean_across_seeds"]
    margins = selection["compact_matching_absolute_margins"]
    thresholds = selection["rthgt_material_improvement_any"]
    deltas = {
        "mean_state_spearman": r1["mean_state_spearman"] - c1["mean_state_spearman"],
        "material_pair_accuracy":
            r1["material_pair_accuracy"] - c1["material_pair_accuracy"],
        "mean_selected_advantage":
            r1["mean_selected_advantage"] - c1["mean_selected_advantage"],
        "mean_top1_regret_reduction":
            c1["mean_top1_regret"] - r1["mean_top1_regret"],
    }
    material = any(deltas[field] >= threshold for field, threshold in thresholds.items())
    no_degradation = (
        deltas["mean_selected_advantage"] >= -margins["mean_selected_advantage"]
        and r1["mean_top1_regret"] <=
        c1["mean_top1_regret"] + margins["mean_top1_regret"])
    matching = (
        abs(deltas["mean_state_spearman"]) <= margins["mean_state_spearman"]
        and abs(deltas["material_pair_accuracy"]) <= margins["material_pair_accuracy"]
        and abs(deltas["mean_selected_advantage"]) <= margins["mean_selected_advantage"]
        and abs(r1["mean_top1_regret"] - c1["mean_top1_regret"])
        <= margins["mean_top1_regret"])
    for variant in ("C1", "R1"):
        observed = profile["variants"][variant]
        require(close(observed["worst_representative_p90_ms"],
                      max(row["p90_ms"] for row in observed["states"].values())),
                f"{variant} worst p90 aggregation mismatch")
    c1_latency = profile["variants"]["C1"]["worst_representative_p90_ms"]
    r1_latency = profile["variants"]["R1"]["worst_representative_p90_ms"]
    latency_budget = selection["latency_budget_p90_ms"]
    compact_faster = c1_latency <= (
        1 - selection["compact_substantially_faster_fraction"]) * r1_latency
    checks = {
        "R1_material_improvement": material,
        "R1_no_material_value_degradation": no_degradation,
        "R1_gate_pass": comparison["R1"]["gate_pass"],
        "R1_latency_pass": r1_latency <= latency_budget,
        "C1_matches_R1": matching,
        "C1_substantially_faster": compact_faster,
        "C1_gate_pass": comparison["C1"]["gate_pass"],
        "C1_latency_pass": c1_latency <= latency_budget,
    }
    selected = ""
    if all(checks[key] for key in (
            "R1_material_improvement", "R1_no_material_value_degradation",
            "R1_gate_pass", "R1_latency_pass")):
        selected = "R1"
    elif all(checks[key] for key in (
            "C1_matches_R1", "C1_substantially_faster",
            "C1_gate_pass", "C1_latency_pass")):
        selected = "C1"
    require(selected == comparison["selected_variant"] == "C1",
            "terminal representation selection mismatch")
    require(checks == comparison["selection_evidence"]["checks"],
            "saved selection checks mismatch")
    for field, value in deltas.items():
        require(close(value, comparison["selection_evidence"]
                      ["quality_deltas_R1_minus_C1"][field]),
                f"selection delta mismatch: {field}")
    return {
        "quality_deltas_R1_minus_C1": deltas,
        "checks": checks,
        "selected_variant": selected,
        "C1_worst_p90_ms": c1_latency,
        "R1_worst_p90_ms": r1_latency,
        "latency_budget_p90_ms": latency_budget,
    }


def audit_production(protocol: dict, records: list[dict], final: dict) -> dict:
    checkpoint_path = ROOT / final["production"]["checkpoint_path"]
    require(checkpoint_path.is_file(), "production checkpoint missing")
    require(digest(checkpoint_path) == final["production"]["checkpoint_sha256"],
            "production checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["training_protocol_sha256"] == digest(PROTOCOL),
            "production checkpoint protocol hash mismatch")
    require(checkpoint["selected_variant"] == "C1",
            "production checkpoint variant mismatch")
    require(checkpoint["epochs"] == protocol["config"]["training"]["epochs"],
            "production checkpoint epoch mismatch")
    model = model_for(
        protocol["config"], "C1", int(checkpoint["seed"]), "cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    largest = max(records, key=lambda row: (
        len(row["state_features"]["node_features"]), len(row["actions"]), row["state_id"]))
    batch, _ = prepare(largest, "cpu")
    with torch.inference_mode():
        first = model(batch)
        second = model(batch)
    require(all(torch.isfinite(value).all() for value in first.values()),
            "production replay produced non-finite output")
    require(all(torch.equal(first[key], second[key]) for key in first),
            "production repeated replay differs")
    actions = len(largest["actions"])
    require(all(value.shape[0] == actions for value in first.values()),
            "production replay did not score every action")
    replay = {
        "schema": "ngas-a13r-production-replay-v1",
        "status": "PASS",
        "checkpoint_sha256": digest(checkpoint_path),
        "training_protocol_sha256": digest(PROTOCOL),
        "device": "cpu",
        "state_id": largest["state_id"],
        "nodes": len(largest["state_features"]["node_features"]),
        "edges_with_reverse": len(largest["state_features"]["edge_index"]),
        "joint_actions_scored": actions,
        "finite_outputs": True,
        "repeated_inference_bitwise_equal": True,
        "r13": "LOCKED", "r14": "LOCKED", "gurobi_run": False,
    }
    atomic_json(REPLAY, replay)
    return replay


def audit_regression_and_scope(final: dict) -> dict:
    xml_path = OUT / "audit/final_regression.xml"
    text_path = OUT / "audit/final_regression.txt"
    tree = ET.parse(xml_path)
    suite = tree.getroot()
    while suite.tag != "testsuite" and len(suite):
        suite = suite[0]
    tests = int(suite.attrib["tests"])
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    require((tests, failures, errors, skipped) == (480, 0, 0, 0),
            "final regression result mismatch")
    log = (OUT / "formal_training.log").read_text()
    forbidden_log_tokens = [
        token for token in ("Traceback", "ERROR", "FAILED", "NaN", "Inf")
        if token in log]
    require(not forbidden_log_tokens, "formal log contains failure marker")
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, text=True).splitlines()
    paths = [line[3:] for line in status]
    forbidden_paths = [
        path for path in paths
        if path.startswith("rcias_clgri/") or path.startswith("outputs/phase")]
    require(not forbidden_paths, "frozen Phase 6 path changed")
    require(final["r13"] == final["r14"] == "LOCKED" and not final["gurobi_run"],
            "forbidden scope access in terminal decision")
    return {
        "pytest": {"tests": tests, "failures": failures,
                   "errors": errors, "skipped": skipped},
        "pytest_text_sha256": digest(text_path),
        "pytest_xml_sha256": digest(xml_path),
        "formal_log_failure_markers": forbidden_log_tokens,
        "frozen_phase6_modified_paths": forbidden_paths,
        "r13": final["r13"], "r14": final["r14"],
        "gurobi_run": final["gurobi_run"],
    }


def main() -> None:
    protocol = load_json(PROTOCOL)
    records = load_cache(CACHE, CACHE_MANIFEST)
    require(len(records) == 72, "training cache state count mismatch")
    records_by_id = {row["state_id"]: row for row in records}
    require(len(records_by_id) == len(records), "duplicate cache state ID")
    comparison = load_json(OUT / "representation_comparison.json")
    profile = load_json(OUT / "gpu_profile.json")
    final = load_json(OUT / "final_decision.json")
    progress = load_json(OUT / "progress.json")
    require(progress["status"] == "COMPLETE" and progress["completed_runs"] == 18,
            "formal progress is incomplete")
    require(final["decision"] == comparison["decision"] ==
            "NGAS_A1_3R_PASS_COMPACT", "terminal decision mismatch")
    require(final["a1_4_preparation_authorized"], "A1.4 preparation is not authorized")

    protocol_audit = audit_protocol(protocol)
    variants = {}
    for variant in ("C1", "R1"):
        variants[variant], _ = audit_variant(
            variant, protocol, records, records_by_id)
    c0 = audit_c0(protocol, records_by_id, comparison)
    selection = audit_selection(protocol, comparison, profile)
    production = audit_production(protocol, records, final)
    scope = audit_regression_and_scope(final)
    require(final["gpu_profile_sha256"] == digest(OUT / "gpu_profile.json"),
            "final GPU profile hash mismatch")
    require(final["critical_sync_audit_sha256"] ==
            digest(OUT / "audit/critical_sync.json"), "critical audit hash mismatch")
    require(final["label_identity_audit_sha256"] ==
            digest(OUT / "audit/label_identity.json"), "identity audit hash mismatch")

    result = {
        "schema": "ngas-a13r-completion-audit-v1",
        "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": final["decision"],
        "selected_variant": "C1",
        "protocol": protocol_audit,
        "cache": {
            "states": len(records),
            "sha256": digest(CACHE),
            "manifest_sha256": digest(CACHE_MANIFEST),
        },
        "C0": c0,
        "C1": variants["C1"],
        "R1": variants["R1"],
        "selection": selection,
        "production_replay": production,
        "regression_and_scope": scope,
        "a1_4_preparation_authorized": True,
        "r13": "LOCKED", "r14": "LOCKED", "gurobi_run": False,
    }
    atomic_json(AUDIT, result)
    print(json.dumps({
        "status": result["status"], "decision": result["decision"],
        "selected_variant": result["selected_variant"],
        "runs_verified": variants["C1"]["runs_verified"] + variants["R1"]["runs_verified"],
        "pytest": scope["pytest"],
        "completion_audit_sha256": digest(AUDIT),
    }, indent=2))


if __name__ == "__main__":
    main()
