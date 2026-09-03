#!/usr/bin/env python3
"""Build manuscript-ready tables whose upstream integrity gates have passed."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
EFFICIENCY_ROOT = PAPER_ROOT / "processed_data/efficiency"
EXACT_ROOT = PAPER_ROOT / "processed_data/exact_validation"
MAIN_ROOT = PAPER_ROOT / "processed_data/main"
SUPPLEMENTARY_ROOT = PAPER_ROOT / "processed_data/supplementary"
TABLE_ROOT = PAPER_ROOT / "tables"
METHOD_LABELS = {
    "GA": "GA",
    "Adapted DCGA": "DCGA",
    "DABC-RIACRSP": "DABC",
    "LG_HGA-RIACRSP-v2-N4M": "LG_HGA",
    "CSG-NI Phase6H provisional": "CSG-NI",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def latex_escape(value: object) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"),
    ):
        text = text.replace(source, target)
    return text


def atomic_latex(path: Path, rows: list[dict[str, object]]) -> None:
    headers = list(rows[0])
    lines = [
        r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_escape(row[header]) for header in headers) + r" \\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_efficiency_table() -> list[dict[str, object]]:
    inventory = read_json(EFFICIENCY_ROOT / "efficiency_inventory.json")
    if inventory.get("status") != "PASS_REUSED_PHASE6H_CAL_HOLDOUT":
        raise RuntimeError("efficiency data integrity gate is not PASS")
    method_rows = {
        row["method"]: row for row in read_csv(
            EFFICIENCY_ROOT / "efficiency_method_summary.csv"
        )
    }
    anytime_rows = {
        row["method"]: row for row in read_csv(EFFICIENCY_ROOT / "anytime_summary.csv")
    }
    output = []
    for raw, display in (
        ("ALNS", "ALNS"),
        ("PHASE6H_CSGNI", "CSG-NI (Phase6H provisional)"),
    ):
        method = method_rows[raw]
        anytime = anytime_rows[raw]
        improvement = method["mean_improvement_vs_alns"]
        output.append({
            "Method": display,
            "Mean final makespan": f"{float(method['mean_of_instance_means']):.2f}",
            "Improvement vs ALNS (%)": f"{100.0 * float(improvement or 0.0):.2f}",
            "Mean decoder evaluations": f"{float(method['mean_decoder_evaluations']):.1f}",
            "Median time-to-best (s)": f"{float(anytime['median_time_to_best']):.2f}",
            "Mean normalized-gap AUC": f"{float(anytime['mean_normalized_gap_auc']):.4f}",
            "Feasibility rate": f"{float(method['feasibility_rate']):.3f}",
        })
    return output


def build_runtime_diagnostics_table() -> list[dict[str, object]]:
    analysis = read_json(SUPPLEMENTARY_ROOT / "supplementary_analysis.json")
    if analysis.get("status") != "PASS_DESCRIPTIVE_EXPLORATORY":
        raise RuntimeError("supplementary Core45 analysis gate is not PASS")
    rows = read_csv(SUPPLEMENTARY_ROOT / "core_runtime_utilization.csv")
    if len(rows) != 5:
        raise RuntimeError("runtime diagnostics require five complete methods")
    return [
        {
            "Method": row["display_method"],
            "Runs": int(row["run_count"]),
            "Median budget used (%)": f"{100 * float(row['median_runtime_budget_fraction']):.1f}",
            "Median decoder evaluations": f"{float(row['median_decoder_evaluations']):.0f}",
            "Median time-to-best (% budget)": f"{100 * float(row['median_time_to_best_budget_fraction']):.1f}",
            "Median iterations": f"{float(row['median_iterations']):.0f}",
            "Termination": row["termination_note"],
        }
        for row in rows
    ]


def build_exact_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    inventory_path = EXACT_ROOT / "exact_result_inventory.json"
    if not inventory_path.is_file():
        return None
    inventory = read_json(inventory_path)
    if inventory.get("status") != "PASS_COMPLETE" or not inventory.get(
        "final_exact_table_authorized", False
    ):
        return None
    summaries = read_csv(EXACT_ROOT / "exact_validation_summary.csv")
    references = {
        row["instance_id"]: row for row in read_csv(EXACT_ROOT / "exact_reference.csv")
    }
    if len(summaries) != 50 or len(references) != 10:
        raise RuntimeError("exact table requires 5 methods x 10 instances")
    by_key = {(row["algorithm"], row["instance_id"]): row for row in summaries}
    compact = []
    for instance_id in sorted(references):
        reference = references[instance_id]
        output: dict[str, object] = {
            "Instance": instance_id,
            "|O|": int(reference["operation_count"]),
            "Gurobi optimum": f"{float(reference['gurobi_optimum']):.0f}",
            "Gurobi runtime (s)": f"{float(reference['gurobi_runtime_seconds']):.2f}",
        }
        for method, label in METHOD_LABELS.items():
            row = by_key[(method, instance_id)]
            hit_time = row["median_time_to_optimum_seconds"]
            time_text = "--" if hit_time == "" else f"{float(hit_time):.2f}s"
            output[label] = (
                f"{float(row['best_gap_percent']):.2f}% / "
                f"{100.0 * float(row['optimum_hit_rate']):.0f}% / {time_text}"
            )
        compact.append(output)
    appendix = [
        {
            "Method": METHOD_LABELS[row["algorithm"]],
            "Instance": row["instance_id"],
            "|O|": int(row["operation_count"]),
            "Exact optimum": f"{float(row['exact_optimum']):.0f}",
            "Best gap (%)": f"{float(row['best_gap_percent']):.4f}",
            "Mean gap (%)": f"{float(row['mean_gap_percent']):.4f}",
            "Optimum hits": f"{int(row['optimum_hit_count'])}/{int(row['expected_runs'])}",
            "Hit rate": f"{float(row['optimum_hit_rate']):.3f}",
            "Median time-to-optimum (s)": (
                "--" if row["median_time_to_optimum_seconds"] == ""
                else f"{float(row['median_time_to_optimum_seconds']):.3f}"
            ),
            "Feasibility rate": f"{float(row['feasibility_rate']):.3f}",
        }
        for row in summaries
    ]
    return compact, appendix


def build_core_tables() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
    manifest_path = MAIN_ROOT / "analysis_manifest.json"
    statistics_path = MAIN_ROOT / "statistical_analysis.json"
    if not manifest_path.is_file() or not statistics_path.is_file():
        return None
    manifest = read_json(manifest_path)
    statistics = read_json(statistics_path)
    if (
        manifest.get("status") != "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H"
        or statistics.get("status") != "PASS"
    ):
        return None
    scale_rows = read_csv(MAIN_ROOT / "main_scale_summary.csv")
    instance_rows = read_csv(MAIN_ROOT / "main_instance_summary.csv")
    test_rows = read_csv(MAIN_ROOT / "statistical_tests.csv")
    if len(scale_rows) != 20 or len(instance_rows) != 225 or len(test_rows) != 4:
        raise RuntimeError("Core table inputs are incomplete")
    by_key = {(row["scale"], row["method"]): row for row in scale_rows}
    table2 = []
    metrics = (
        ("Mean RPD (%)", "mean_of_instance_mean_rpd_percent", ".3f"),
        ("Median RPD (%)", "median_of_instance_median_rpd_percent", ".3f"),
        ("Best count", "bks_attainment_count", ".0f"),
        ("Average rank", "average_rank", ".3f"),
    )
    for scale in ("S", "M", "L", "Overall"):
        for metric_label, field, precision in metrics:
            output: dict[str, object] = {"Scale": scale, "Metric": metric_label}
            for method, label in METHOD_LABELS.items():
                output[label] = format(float(by_key[(scale, method)][field]), precision)
            table2.append(output)
    appendix = [
        {
            "Method": METHOD_LABELS[row["method"]],
            "Instance": row["instance_id"],
            "Scale": row["scale"],
            "CF": row["CF_level"],
            "Best": f"{float(row['best_makespan']):.0f}",
            "Mean": f"{float(row['mean_makespan']):.2f}",
            "Median": f"{float(row['median_makespan']):.2f}",
            "Std": f"{float(row['std_makespan']):.2f}",
            "Mean RPD (%)": f"{float(row['mean_rpd_percent']):.3f}",
            "Median RPD (%)": f"{float(row['median_rpd_percent']):.3f}",
            "Rank": f"{float(row['median_objective_rank']):.2f}",
            "Attains BKS": row["attains_bks"],
        }
        for row in instance_rows
    ]
    table3 = [
        {
            "Competitor": METHOD_LABELS[row["competitor"]],
            "Wins": int(row["wins"]),
            "Ties": int(row["ties"]),
            "Losses": int(row["losses"]),
            "Wilcoxon p": f"{float(row['wilcoxon_p_raw']):.6g}",
            "Holm p": f"{float(row['holm_adjusted_p']):.6g}",
            "Rank-biserial": f"{float(row['paired_rank_biserial_positive_favors_csg']):.3f}",
        }
        for row in test_rows
    ]
    return table2, appendix, table3


def main() -> int:
    generated = []
    blocked: dict[str, str] = {}
    table4 = build_efficiency_table()
    atomic_csv(TABLE_ROOT / "table4_efficiency.csv", table4)
    atomic_latex(TABLE_ROOT / "table4_efficiency.tex", table4)
    generated.extend(["table4_efficiency.csv", "table4_efficiency.tex"])
    runtime_diagnostics = build_runtime_diagnostics_table()
    for name in (
        "supplementary_table1_runtime_utilization.csv",
        "supplementary_table1_runtime_utilization.tex",
    ):
        if name.endswith(".tex"):
            atomic_latex(TABLE_ROOT / name, runtime_diagnostics)
        else:
            atomic_csv(TABLE_ROOT / name, runtime_diagnostics)
        generated.append(name)

    exact = build_exact_tables()
    if exact is None:
        blocked["table1_exact_validation"] = "waiting for 50 Phase6H CSG-NI exact runs"
    else:
        compact, appendix = exact
        for name, rows in (
            ("table1_exact_validation.csv", compact),
            ("table1_exact_validation.tex", compact),
            ("appendix_exact_validation.csv", appendix),
        ):
            if name.endswith(".tex"):
                atomic_latex(TABLE_ROOT / name, rows)
            else:
                atomic_csv(TABLE_ROOT / name, rows)
            generated.append(name)

    core = build_core_tables()
    if core is None:
        blocked["table2_core_performance"] = "waiting for complete 5-method Core matrix and BKS/RPD"
        blocked["table3_core_statistics"] = "waiting for Table 2 inputs and paired statistics"
    else:
        table2, appendix, table3 = core
        for name, rows in (
            ("table2_core_performance.csv", table2),
            ("table2_core_performance.tex", table2),
            ("appendix_core_instances.csv", appendix),
            ("table3_core_statistics.csv", table3),
            ("table3_core_statistics.tex", table3),
        ):
            if name.endswith(".tex"):
                atomic_latex(TABLE_ROOT / name, rows)
            else:
                atomic_csv(TABLE_ROOT / name, rows)
            generated.append(name)
    status = {
        "schema": "initial-manuscript-table-build-status-v1",
        "generated": generated,
        "blocked": blocked,
    }
    temporary = TABLE_ROOT / f"build_status.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(TABLE_ROOT / "build_status.json")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
