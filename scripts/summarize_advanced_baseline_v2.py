#!/usr/bin/env python3
"""Audit and summarize the matched three-seed DABC/LG_HGA-v2 Core matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DABC_ROOT = ROOT / "outputs/baselines/comparison_advanced/formal/dabc_riacrsp"
LGHGA_ROOT = (
    ROOT
    / "outputs/baselines/comparison_advanced_v2/formal/lg_hga_riacrsp_v2_n4m"
)
OUTPUT_ROOT = ROOT / "outputs/baselines/comparison_advanced_v2/summary"
FORMAL_MANIFEST = ROOT / "configs/baselines/advanced_formal_manifest.json"
DABC_IMPLEMENTATION = ROOT / "configs/baselines/advanced_implementation_manifest.json"
LGHGA_IMPLEMENTATION = ROOT / "configs/baselines/lghga_v2_implementation_manifest.json"
KNOWLEDGE_MANIFEST = ROOT / "outputs/baselines/lghga_kb_v2/knowledge_manifest.json"
CORE_MANIFEST = ROOT / "instances/controlled/RCIAS-CB1/manifests/core_manifest.csv"
SEEDS = {530101, 530102, 530103}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _wilcoxon(differences: list[float]) -> dict[str, float | int]:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return {"nonzero_pairs": 0, "statistic": 0.0, "p_value_two_sided": 1.0}
    absolute = [abs(value) for value in nonzero]
    ranks = _average_ranks(absolute)
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    count = len(nonzero)
    tie_counts: dict[float, int] = {}
    for value in absolute:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    variance = count * (count + 1) * (2 * count + 1) / 24
    variance -= sum(size**3 - size for size in tie_counts.values()) / 48
    if variance == 0:
        p_value = 1.0
    else:
        mean = count * (count + 1) / 4
        z_score = (abs(min(positive, negative) - mean) - 0.5) / math.sqrt(variance)
        p_value = math.erfc(z_score / math.sqrt(2))
    return {
        "nonzero_pairs": count,
        "statistic": min(positive, negative),
        "p_value_two_sided": p_value,
    }


def _load_results(root: Path, method: str) -> dict[tuple[str, int], dict[str, object]]:
    records: dict[tuple[str, int], dict[str, object]] = {}
    for path in sorted(root.rglob("seed_*.json")):
        payload = json.loads(path.read_text())
        key = (str(payload["instance_id"]), int(payload["seed"]))
        if key[1] not in SEEDS:
            continue
        if key in records:
            raise RuntimeError(f"duplicate result key for {method}: {key}")
        if payload["method"] != method:
            raise RuntimeError(f"wrong method in {path}")
        audit = payload["independent_feasibility_audit"]
        if not payload["feasible"] or not audit["feasible"] or audit["violations"]:
            raise RuntimeError(f"infeasible result in {path}")
        diagnostics = payload.get("diagnostics") or {}
        models = payload.get("lghga_models") or {}
        records[key] = {
            "formal_manifest_sha256": payload["formal_manifest_sha256"],
            "implementation_manifest_sha256": payload["implementation_manifest_sha256"],
            "best_makespan": payload["best_makespan"],
            "runtime": payload["runtime"],
            "decoder_evaluations": payload["decoder_evaluations"],
            "local_search_gate_passes": diagnostics.get("local_search_gate_passes", 0),
            "local_decoder_evaluations": diagnostics.get("local_decoder_evaluations", 0),
            "knowledge_manifest_hash": models.get("knowledge_manifest_hash"),
            "result_path": str(path.relative_to(ROOT)),
        }
    return records


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> None:
    formal_hash = _sha256(FORMAL_MANIFEST)
    dabc_implementation_hash = _sha256(DABC_IMPLEMENTATION)
    lghga_implementation_hash = _sha256(LGHGA_IMPLEMENTATION)
    knowledge = json.loads(KNOWLEDGE_MANIFEST.read_text())
    if knowledge["status"] != "FROZEN_V2_NO_FORMAL_TEST_LEAKAGE":
        raise RuntimeError("LG_HGA-v2 knowledge is not frozen")

    with CORE_MANIFEST.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    cells = {
        row["instance_id"]: (row["scale"], row["CF_level"])
        for row in manifest_rows
    }
    expected = {(instance_id, seed) for instance_id in cells for seed in SEEDS}
    dabc = _load_results(DABC_ROOT, "DABC-RIACRSP")
    lghga = _load_results(LGHGA_ROOT, "LG_HGA-RIACRSP-v2-N4M")
    if set(dabc) != expected or set(lghga) != expected:
        raise RuntimeError(
            f"incomplete matched matrix: DABC={len(dabc)}, LG_HGA-v2={len(lghga)}, "
            f"expected={len(expected)}"
        )

    paired: list[dict[str, object]] = []
    for instance_id, seed in sorted(expected):
        left, right = dabc[(instance_id, seed)], lghga[(instance_id, seed)]
        if left["formal_manifest_sha256"] != formal_hash or right["formal_manifest_sha256"] != formal_hash:
            raise RuntimeError(f"formal manifest mismatch: {(instance_id, seed)}")
        if left["implementation_manifest_sha256"] != dabc_implementation_hash:
            raise RuntimeError(f"DABC implementation mismatch: {(instance_id, seed)}")
        if right["implementation_manifest_sha256"] != lghga_implementation_hash:
            raise RuntimeError(f"LG_HGA-v2 implementation mismatch: {(instance_id, seed)}")
        if right["knowledge_manifest_hash"] != knowledge["knowledge_manifest_hash"]:
            raise RuntimeError(f"knowledge hash mismatch: {(instance_id, seed)}")
        dabc_makespan = float(left["best_makespan"])
        lghga_makespan = float(right["best_makespan"])
        scale, cf_level = cells[instance_id]
        paired.append({
            "instance_id": instance_id,
            "scale": scale,
            "CF_level": cf_level,
            "seed": seed,
            "dabc_makespan": dabc_makespan,
            "lghga_v2_makespan": lghga_makespan,
            "lghga_minus_dabc": lghga_makespan - dabc_makespan,
            "lghga_improvement_over_dabc_pct": 100 * (dabc_makespan - lghga_makespan) / dabc_makespan,
            "dabc_runtime_seconds": float(left["runtime"]),
            "lghga_v2_runtime_seconds": float(right["runtime"]),
            "dabc_decoder_evaluations": int(left["decoder_evaluations"]),
            "lghga_v2_decoder_evaluations": int(right["decoder_evaluations"]),
            "lghga_v2_gate_passes": int(right["local_search_gate_passes"]),
            "lghga_v2_local_evaluations": int(right["local_decoder_evaluations"]),
            "dabc_result_path": left["result_path"],
            "lghga_v2_result_path": right["result_path"],
        })

    by_instance: list[dict[str, object]] = []
    for instance_id in sorted(cells):
        selected = [row for row in paired if row["instance_id"] == instance_id]
        dabc_mean = _mean([float(row["dabc_makespan"]) for row in selected])
        lghga_mean = _mean([float(row["lghga_v2_makespan"]) for row in selected])
        scale, cf_level = cells[instance_id]
        by_instance.append({
            "instance_id": instance_id,
            "scale": scale,
            "CF_level": cf_level,
            "seeds": len(selected),
            "dabc_mean_makespan": dabc_mean,
            "lghga_v2_mean_makespan": lghga_mean,
            "lghga_minus_dabc": lghga_mean - dabc_mean,
            "lghga_improvement_over_dabc_pct": 100 * (dabc_mean - lghga_mean) / dabc_mean,
        })

    cell_summary: list[dict[str, object]] = []
    for scale in ("S", "M", "L"):
        for cf_level in ("CF1", "CF2", "CF3"):
            selected = [
                row for row in paired
                if row["scale"] == scale and row["CF_level"] == cf_level
            ]
            cell_summary.append({
                "scale": scale,
                "CF_level": cf_level,
                "runs": len(selected),
                "dabc_mean_makespan": _mean([float(row["dabc_makespan"]) for row in selected]),
                "lghga_v2_mean_makespan": _mean([float(row["lghga_v2_makespan"]) for row in selected]),
                "mean_paired_lghga_improvement_pct": _mean([
                    float(row["lghga_improvement_over_dabc_pct"]) for row in selected
                ]),
                "lghga_wins": sum(float(row["lghga_minus_dabc"]) < 0 for row in selected),
                "ties": sum(float(row["lghga_minus_dabc"]) == 0 for row in selected),
                "lghga_losses": sum(float(row["lghga_minus_dabc"]) > 0 for row in selected),
                "gate_passes": sum(int(row["lghga_v2_gate_passes"]) for row in selected),
            })

    instance_differences = [float(row["lghga_minus_dabc"]) for row in by_instance]
    audit = {
        "schema": "advanced-baseline-v2-three-seed-summary-v1",
        "status": "PASS",
        "run_pairs": len(paired),
        "instance_pairs": len(by_instance),
        "seeds": sorted(SEEDS),
        "formal_manifest_sha256": formal_hash,
        "dabc_implementation_manifest_sha256": dabc_implementation_hash,
        "lghga_v2_implementation_manifest_sha256": lghga_implementation_hash,
        "knowledge_manifest_hash": knowledge["knowledge_manifest_hash"],
        "overall": {
            "dabc_mean_makespan": _mean([float(row["dabc_makespan"]) for row in paired]),
            "lghga_v2_mean_makespan": _mean([float(row["lghga_v2_makespan"]) for row in paired]),
            "mean_paired_lghga_improvement_pct": _mean([
                float(row["lghga_improvement_over_dabc_pct"]) for row in paired
            ]),
            "lghga_run_wins": sum(float(row["lghga_minus_dabc"]) < 0 for row in paired),
            "run_ties": sum(float(row["lghga_minus_dabc"]) == 0 for row in paired),
            "lghga_run_losses": sum(float(row["lghga_minus_dabc"]) > 0 for row in paired),
            "lghga_instance_wins": sum(value < 0 for value in instance_differences),
            "instance_ties": sum(value == 0 for value in instance_differences),
            "lghga_instance_losses": sum(value > 0 for value in instance_differences),
            "mean_dabc_runtime_seconds": _mean([float(row["dabc_runtime_seconds"]) for row in paired]),
            "mean_lghga_v2_runtime_seconds": _mean([float(row["lghga_v2_runtime_seconds"]) for row in paired]),
            "mean_dabc_decoder_evaluations": _mean([float(row["dabc_decoder_evaluations"]) for row in paired]),
            "mean_lghga_v2_decoder_evaluations": _mean([float(row["lghga_v2_decoder_evaluations"]) for row in paired]),
            "total_lghga_gate_passes": sum(int(row["lghga_v2_gate_passes"]) for row in paired),
            "total_lghga_local_evaluations": sum(int(row["lghga_v2_local_evaluations"]) for row in paired),
        },
        "wilcoxon_on_45_instance_mean_differences": _wilcoxon(instance_differences),
        "cell_summary": cell_summary,
    }
    _atomic_csv(OUTPUT_ROOT / "paired_runs.csv", paired)
    _atomic_csv(OUTPUT_ROOT / "instance_summary.csv", by_instance)
    _atomic_csv(OUTPUT_ROOT / "cell_summary.csv", cell_summary)
    _atomic_json(OUTPUT_ROOT / "summary.json", audit)
    print(
        f"ADVANCED_BASELINE_V2_SUMMARY_COMPLETE pairs={len(paired)} "
        f"instances={len(by_instance)} status={audit['status']}"
    )


if __name__ == "__main__":
    main()
