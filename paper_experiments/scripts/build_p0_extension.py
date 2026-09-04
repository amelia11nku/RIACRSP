#!/usr/bin/env python3
"""Freeze P0 descriptive conclusions and manuscript insertion text."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
DATA_PATH = PAPER_ROOT / "processed_data/supplementary/core_paired_advantage_scale_cf.csv"
SUMMARY_PATH = PAPER_ROOT / "processed_data/supplementary/p0_scale_cf_summary.json"
SNIPPET_ROOT = PAPER_ROOT / "reports/snippets"
SCALES = ("S", "M", "L")
CF_LEVELS = ("CF1", "CF2", "CF3")
COMPETITORS = ("GA", "DCGA", "DABC", "LG_HGA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        (row["scale"], row["display_competitor"], row["CF_level"]): row
        for row in rows
    }
    expected = {
        (scale, competitor, cf)
        for scale in SCALES
        for competitor in COMPETITORS
        for cf in CF_LEVELS
    }
    if len(rows) != 36 or set(lookup) != expected:
        raise RuntimeError("P0 requires a complete 3 x 4 x 3 cell table")
    if any(int(row["instance_count"]) != 5 for row in rows):
        raise RuntimeError("P0 requires five independent instances in every cell")

    cell_medians = {
        f"{scale}_{competitor}_{cf}": float(
            lookup[(scale, competitor, cf)]["median_advantage_percent_points"]
        )
        for scale in SCALES
        for competitor in COMPETITORS
        for cf in CF_LEVELS
    }
    nonpositive = {
        scale: [
            {"competitor": competitor, "CF_level": cf, "median_advantage": value}
            for competitor in COMPETITORS
            for cf in CF_LEVELS
            if (value := cell_medians[f"{scale}_{competitor}_{cf}"]) <= 0
        ]
        for scale in SCALES
    }
    ga_small = [cell_medians[f"S_GA_{cf}"] for cf in CF_LEVELS]
    lghga_small = [cell_medians[f"S_LG_HGA_{cf}"] for cf in CF_LEVELS]
    summary = {
        "schema": "initial-manuscript-p0-scale-cf-summary-v1",
        "status": "PASS_DESCRIPTIVE_EXPLORATORY",
        "source": str(DATA_PATH.relative_to(ROOT)),
        "source_sha256": sha256(DATA_PATH),
        "independent_instances_per_cell": 5,
        "matched_seeds_per_method_instance": 5,
        "advantage_definition": "competitor median RPD minus CSG-NI median RPD; positive favors CSG-NI",
        "cell_medians_percent_points": cell_medians,
        "nonpositive_cell_medians": nonpositive,
        "supported_findings": {
            "small": (
                "The Small-scale weakness against GA occurs in all three CF strata "
                f"({ga_small[0]:+.3f}, {ga_small[1]:+.3f}, {ga_small[2]:+.3f} percentage points); "
                "against LG_HGA, the cell median reverses only in CF3 "
                f"({lghga_small[2]:+.3f} percentage points)."
            ),
            "medium_large": (
                "All 24 competitor-by-CF cell medians at Medium and Large scale are positive, "
                "although positive medians do not imply wins on every instance."
            ),
        },
        "inference_boundary": (
            "Post-hoc descriptive heterogeneity analysis; no cellwise significance tests and "
            "no seed-level pseudo-replication."
        ),
    }
    atomic_text(SUMMARY_PATH, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    paragraph = (
        "\\paragraph{Scale--CF heterogeneity.} "
        "Exploratory stratification of Core45 showed that the Small-instance boundary was not "
        "confined to one CF level: relative to GA, the median paired RPD advantages of CSG-NI "
        f"were {ga_small[0]:+.2f}, {ga_small[1]:+.2f}, and {ga_small[2]:+.2f} percentage points "
        "for CF1--CF3, respectively. Relative to LG\\_HGA, the Small-scale cell median was "
        f"negative only for CF3 ({lghga_small[2]:+.2f} points). By contrast, every competitor-by-CF "
        "cell median was positive at Medium and Large scale (24/24 cells), while some cells still "
        "contained individual losses. These post-hoc summaries use five independent instances "
        "per cell and are descriptive rather than cellwise significance tests or causal evidence.\n"
    )
    caption = (
        "\\caption{Scale- and CF-dependent paired quality advantage of CSG-NI on Core45. "
        "Each cell reports the median, across five independent instances, of the paired difference "
        "between a competitor's and CSG-NI's median RPD over five matched seeds; positive values "
        "favor CSG-NI. Parentheses report CSG-NI wins/losses. No cellwise hypothesis tests were "
        "performed.}\n"
    )
    atomic_text(SNIPPET_ROOT / "p0_scale_cf_results.tex", paragraph)
    atomic_text(SNIPPET_ROOT / "p0_figure_caption.tex", caption)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
