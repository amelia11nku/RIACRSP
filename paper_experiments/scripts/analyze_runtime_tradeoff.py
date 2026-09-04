#!/usr/bin/env python3
"""Build P3 quality--runtime summaries from the complete existing Core45 matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
MAIN_ROOT = PAPER_ROOT / "processed_data/main"
OUTPUT_ROOT = PAPER_ROOT / "processed_data/runtime"
SNIPPET_ROOT = PAPER_ROOT / "reports/snippets"
METHODS = (
    "GA",
    "Adapted DCGA",
    "DABC-RIACRSP",
    "LG_HGA-RIACRSP-v2-N4M",
    "CSG-NI Phase6H provisional",
)
DISPLAY = {
    "GA": "GA",
    "Adapted DCGA": "DCGA",
    "DABC-RIACRSP": "DABC",
    "LG_HGA-RIACRSP-v2-N4M": "LG_HGA",
    "CSG-NI Phase6H provisional": "CSG-NI",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def summarize(rows: list[dict[str, object]], method: str, scale: str) -> dict[str, object]:
    runtime = quantiles([float(row["median_runtime_seconds"]) for row in rows])
    utilization = quantiles([float(row["median_runtime_budget_fraction"]) for row in rows])
    rpd = quantiles([float(row["median_rpd_percent"]) for row in rows])
    time_to_best = quantiles([float(row["median_time_to_best_seconds"]) for row in rows])
    time_to_best_fraction = quantiles([
        float(row["median_time_to_best_budget_fraction"]) for row in rows
    ])
    return {
        "method": method,
        "display_method": DISPLAY[method],
        "scale": scale,
        "instance_count": len(rows),
        **{f"{key}_runtime_seconds": value for key, value in runtime.items()},
        **{f"{key}_runtime_budget_fraction": value for key, value in utilization.items()},
        **{f"{key}_rpd_percent": value for key, value in rpd.items()},
        **{f"{key}_time_to_best_seconds": value for key, value in time_to_best.items()},
        **{
            f"{key}_time_to_best_budget_fraction": value
            for key, value in time_to_best_fraction.items()
        },
        "median_decoder_evaluations": float(np.median([
            float(row["median_decoder_evaluations"]) for row in rows
        ])),
        "termination_note": (
            "source-compatible MAXGEN=100 cap or common wall-clock ceiling"
            if method == "LG_HGA-RIACRSP-v2-N4M"
            else "common wall-clock ceiling"
        ),
    }


def main() -> int:
    manifest_path = MAIN_ROOT / "analysis_manifest.json"
    source_path = MAIN_ROOT / "main_runs.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H":
        raise RuntimeError("P3 requires the complete promoted Core45 analysis gate")
    runs = read_csv(source_path)
    if len(runs) != 1125 or {row["method"] for row in runs} != set(METHODS):
        raise RuntimeError("P3 requires 1,125 runs in a complete five-method matrix")
    if any(row["feasible"] != "True" for row in runs):
        raise RuntimeError("P3 refuses infeasible Core45 schedules")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        grouped[(row["instance_id"], row["method"])].append(row)
    if len(grouped) != 225:
        raise RuntimeError("P3 requires 45 instances x five methods")

    per_instance: list[dict[str, object]] = []
    for (instance_id, method), group in sorted(grouped.items()):
        if len(group) != 5 or len({row["seed"] for row in group}) != 5:
            raise RuntimeError(f"incomplete matched-seed cell: {(instance_id, method)}")
        runtime = [float(row["runtime_seconds"]) for row in group]
        limits = [float(row["time_limit_seconds"]) for row in group]
        time_to_best = [float(row["best_found_time_seconds"]) for row in group]
        per_instance.append({
            "instance_id": instance_id,
            "scale": group[0]["scale"],
            "CF_level": group[0]["CF_level"],
            "operation_count": int(group[0]["operation_count"]),
            "method": method,
            "display_method": DISPLAY[method],
            "seed_count": len(group),
            "median_rpd_percent": float(np.median([
                float(row["rpd_percent"]) for row in group
            ])),
            "median_runtime_seconds": float(np.median(runtime)),
            "median_runtime_budget_fraction": float(np.median([
                value / limit for value, limit in zip(runtime, limits)
            ])),
            "median_time_to_best_seconds": float(np.median(time_to_best)),
            "median_time_to_best_budget_fraction": float(np.median([
                value / limit for value, limit in zip(time_to_best, limits)
            ])),
            "median_decoder_evaluations": float(np.median([
                float(row["decoder_evaluations"]) for row in group
            ])),
        })

    overall: list[dict[str, object]] = []
    by_scale: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = [row for row in per_instance if row["method"] == method]
        if len(method_rows) != 45:
            raise RuntimeError(f"expected 45 instance summaries for {method}")
        overall.append(summarize(method_rows, method, "ALL"))
        for scale in ("S", "M", "L"):
            scale_rows = [row for row in method_rows if row["scale"] == scale]
            if len(scale_rows) != 15:
                raise RuntimeError(f"expected 15 instance summaries for {(method, scale)}")
            by_scale.append(summarize(scale_rows, method, scale))

    scale_summary = read_csv(MAIN_ROOT / "main_scale_summary.csv")
    average_rank = defaultdict(list)
    for row in scale_summary:
        average_rank[row["method"]].append(float(row["average_rank"]))
    for row in overall:
        row["average_rank"] = float(np.mean(average_rank[str(row["method"])]))

    atomic_csv(OUTPUT_ROOT / "core_quality_runtime_per_instance.csv", per_instance)
    atomic_csv(OUTPUT_ROOT / "core_quality_runtime_overall.csv", overall)
    atomic_csv(OUTPUT_ROOT / "core_quality_runtime_by_scale.csv", by_scale)

    csg = next(row for row in overall if row["display_method"] == "CSG-NI")
    ga = next(row for row in overall if row["display_method"] == "GA")
    dabc = next(row for row in overall if row["display_method"] == "DABC")
    lghga = next(row for row in overall if row["display_method"] == "LG_HGA")
    summary = {
        "schema": "initial-manuscript-p3-quality-runtime-v1",
        "status": "PASS_DESCRIPTIVE_EXISTING_CORE45",
        "source": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256(source_path),
        "independent_instances": 45,
        "matched_seeds_per_method_instance": 5,
        "time_to_best_scope": "logged consistently for all five methods",
        "decoder_evaluation_boundary": (
            "Recorded but excluded from the shared figure because one evaluation does not represent "
            "an equivalent computational operation across all algorithm families."
        ),
        "supported_findings": {
            "budget_use": (
                f"Median utilization is CSG-NI {float(csg['median_runtime_budget_fraction']):.3f}, "
                f"GA {float(ga['median_runtime_budget_fraction']):.3f}, DABC "
                f"{float(dabc['median_runtime_budget_fraction']):.3f}, and LG_HGA "
                f"{float(lghga['median_runtime_budget_fraction']):.3f}."
            ),
            "time_to_best": (
                f"Median time-to-best fraction is CSG-NI "
                f"{float(csg['median_time_to_best_budget_fraction']):.3f}, GA "
                f"{float(ga['median_time_to_best_budget_fraction']):.3f}, and DABC "
                f"{float(dabc['median_time_to_best_budget_fraction']):.3f}."
            ),
            "termination_disclosure": (
                "LG_HGA retains its source-compatible MAXGEN=100 stopping rule, so its shorter "
                "runtime is expected and is not evidence of missing runs."
            ),
        },
        "pareto_claim": "not made",
    }
    atomic_text(OUTPUT_ROOT / "p3_runtime_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    paragraph = (
        "\\paragraph{Quality and realized runtime.} On Core45, CSG-NI, GA, and DABC used "
        f"median fractions of {float(csg['median_runtime_budget_fraction']):.3f}, "
        f"{float(ga['median_runtime_budget_fraction']):.3f}, and "
        f"{float(dabc['median_runtime_budget_fraction']):.3f} of the common $2|\\mathcal{{O}}|$ "
        "wall-clock ceiling, respectively. LG\\_HGA used a median fraction of "
        f"{float(lghga['median_runtime_budget_fraction']):.3f} because its original "
        "\\texttt{MAXGEN=100} termination rule was retained; this is source-compatible early "
        "termination, not missing computation. CSG-NI reached its final incumbent at a median "
        f"{float(csg['median_time_to_best_budget_fraction']):.3f} of budget, compared with "
        f"{float(ga['median_time_to_best_budget_fraction']):.3f} for GA and "
        f"{float(dabc['median_time_to_best_budget_fraction']):.3f} for DABC. Quality and runtime "
        "are therefore reported jointly; no Pareto-optimality claim is made.\n"
    )
    caption = (
        "\\caption{Solution quality and realized computational cost on Core45. "
        "Points summarize independent instance-level medians over five matched seeds; error bars "
        "show interquartile ranges. Panel (a) compares overall median RPD with normalized budget "
        "utilization, and panel (b) reports scale-specific median RPD against actual runtime. "
        "LG\\_HGA retains its source-compatible \\texttt{MAXGEN=100} stopping rule and therefore "
        "often terminates before the common wall-clock ceiling.}\n"
    )
    atomic_text(SNIPPET_ROOT / "p3_quality_runtime_results.tex", paragraph)
    atomic_text(SNIPPET_ROOT / "p3_figure_caption.tex", caption)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
