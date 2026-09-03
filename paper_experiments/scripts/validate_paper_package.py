#!/usr/bin/env python3
"""Validate the initial-manuscript experiment package and report open gates."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
OUTPUT_PATH = PAPER_ROOT / "audit/package_validation.json"


def load_json(relative: str) -> dict[str, Any] | None:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def csv_rows(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check(name: str, passed: bool, evidence: object) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "evidence": evidence}


def main() -> int:
    checks: list[dict[str, object]] = []
    benchmark = load_json(
        "paper_experiments/processed_data/exact_validation/exact_benchmark_audit.json"
    )
    references = csv_rows(
        "paper_experiments/benchmarks/exact_validation_10_final_gurobi/gurobi_results.csv"
    )
    checks.append(check(
        "exact_benchmark_gate",
        bool(benchmark and benchmark.get("all_primary_gates_passed")),
        None if benchmark is None else benchmark.get("all_primary_gates_passed"),
    ))
    checks.append(check(
        "ten_small_exact_instances",
        len(references) == 10 and all(int(row["operation_count"]) <= 12 for row in references),
        {"instances": len(references), "maximum_operations": max(
            (int(row["operation_count"]) for row in references), default=None
        )},
    ))
    checks.append(check(
        "all_gurobi_references_proven_optimal",
        len(references) == 10 and all(
            row["status"] == "OPTIMAL"
            and row["optimality_proven"].lower() == "true"
            and float(row["reported_gap"]) == 0.0
            and row["replay_feasible"].lower() == "true"
            for row in references
        ),
        {"references": len(references)},
    ))

    exact = load_json(
        "paper_experiments/processed_data/exact_validation/exact_result_inventory.json"
    )
    checks.append(check(
        "exact_five_method_matrix",
        bool(exact and exact.get("status") == "PASS_COMPLETE"),
        None if exact is None else {
            "status": exact.get("status"),
            "valid": exact.get("valid_unique_runs"),
            "expected": exact.get("expected_total_runs"),
        },
    ))
    core = load_json("paper_experiments/processed_data/core/source_inventory.json")
    checks.append(check(
        "core_five_method_matrix",
        bool(core and core.get("status") == "PASS_COMPLETE"),
        None if core is None else {
            "status": core.get("status"),
            "valid": core.get("observed_valid_records"),
            "expected": 1125,
        },
    ))
    completion = csv_rows(
        "paper_experiments/processed_data/main/main_completion_matrix.csv"
    )
    seed_manifest = load_json(
        "paper_experiments/processed_data/main/main_seed_manifest.json"
    )
    completion_pass = (
        len(completion) == 225
        and all(
            row["complete"].lower() == "true"
            and int(row["expected_seed_count"]) == 5
            and int(row["observed_seed_count"]) == 5
            and json.loads(row["missing_seeds"]) == []
            for row in completion
        )
        and bool(
            seed_manifest
            and seed_manifest.get("status") == "FROZEN_FIVE_MATCHED_SEEDS"
            and len(seed_manifest.get("primary_seeds", [])) == 5
        )
    )
    checks.append(check(
        "core_completion_and_seed_manifests",
        completion_pass,
        {
            "completion_rows": len(completion),
            "seed_status": None if seed_manifest is None else seed_manifest.get("status"),
        },
    ))
    main = load_json("paper_experiments/processed_data/main/analysis_manifest.json")
    statistics = load_json("paper_experiments/processed_data/main/statistical_analysis.json")
    checks.append(check(
        "core_bks_rpd",
        bool(main and main.get("status") == "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H"),
        None if main is None else main.get("status"),
    ))
    checks.append(check(
        "paired_statistics",
        bool(statistics and statistics.get("status") == "PASS"),
        None if statistics is None else statistics.get("status"),
    ))
    efficiency = load_json(
        "paper_experiments/processed_data/efficiency/efficiency_inventory.json"
    )
    checks.append(check(
        "phase6h_alns_efficiency",
        bool(efficiency and efficiency.get("status") == "PASS_REUSED_PHASE6H_CAL_HOLDOUT"),
        None if efficiency is None else efficiency.get("status"),
    ))

    required_tables = [
        "table1_exact_validation.csv", "table1_exact_validation.tex",
        "table2_core_performance.csv", "table2_core_performance.tex",
        "table3_core_statistics.csv", "table3_core_statistics.tex",
        "table4_efficiency.csv", "table4_efficiency.tex",
        "supplementary_table1_runtime_utilization.csv",
        "supplementary_table1_runtime_utilization.tex",
    ]
    missing_tables = [name for name in required_tables if not (PAPER_ROOT / "tables" / name).is_file()]
    checks.append(check("paper_tables", not missing_tables, {"missing": missing_tables}))
    supplementary = load_json(
        "paper_experiments/processed_data/supplementary/supplementary_analysis.json"
    )
    checks.append(check(
        "core_supplementary_analysis",
        bool(supplementary and supplementary.get("status") == "PASS_DESCRIPTIVE_EXPLORATORY"),
        None if supplementary is None else supplementary.get("status"),
    ))
    figure_bases = (
        "figure1_quality_distribution",
        "figure2_anytime_efficiency",
        "supplementary_figure1_core_heterogeneity",
        "supplementary_figure2_seed_stability",
    )
    required_figure_exports = [
        f"{base}.{extension}"
        for base in figure_bases
        for extension in ("svg", "pdf", "eps", "tiff", "png")
    ]
    required_figure_qa = [
        f"{base}.{suffix}"
        for base in figure_bases
        for suffix in (
            "alignment.json",
            "pdf-text-audit.json",
            "collision-audit.json",
            "qa.json",
        )
    ]
    missing_figures = [
        name
        for name in required_figure_exports + required_figure_qa
        if not (PAPER_ROOT / "figures" / name).is_file()
    ]
    checks.append(check("paper_figures", not missing_figures, {"missing": missing_figures}))
    figure_qa_evidence: dict[str, object] = {}
    figure_qa_pass = not missing_figures
    for figure_base in figure_bases:
        base = f"paper_experiments/figures/{figure_base}"
        qa = load_json(f"{base}.qa.json")
        alignment = load_json(f"{base}.alignment.json")
        collision = load_json(f"{base}.collision-audit.json")
        text_audit = load_json(f"{base}.pdf-text-audit.json")
        passed = bool(
            qa
            and qa.get("status") == "AUTOMATED_QA_PASS"
            and qa.get("font") == "Times New Roman"
            and float(qa.get("final_width_mm", 0)) == 180.0
            and alignment
            and alignment.get("verdict") == "PASS"
            and collision
            and collision.get("verdict") == "PASS"
            and text_audit
            and text_audit.get("auditable") is True
            and int(text_audit.get("below_minimum_count", -1)) == 0
        )
        figure_qa_pass = figure_qa_pass and passed
        figure_qa_evidence[figure_base] = {
            "passed": passed,
            "status": None if qa is None else qa.get("status"),
            "alignment": None if alignment is None else alignment.get("verdict"),
            "collision": None if collision is None else collision.get("verdict"),
            "minimum_glyph_pt": (
                None if text_audit is None else text_audit.get("minimum_found_pt")
            ),
        }
    checks.append(check("paper_figure_qa", figure_qa_pass, figure_qa_evidence))

    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "schema": "initial-manuscript-package-validation-v1",
        "status": "PASS" if passed else "INCOMPLETE",
        "checks_passed": sum(bool(item["passed"]) for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
