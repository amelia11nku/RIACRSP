#!/usr/bin/env python3
"""Build the P1 report and integrated P0/P1/P3 manuscript handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
ABLATION_ROOT = PAPER_ROOT / "ablation"
REPORT_PATH = PAPER_ROOT / "reports/P0_P1_P3_EXTENSION_REPORT.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    p0 = read_json(PAPER_ROOT / "processed_data/supplementary/p0_scale_cf_summary.json")
    p3 = read_json(PAPER_ROOT / "processed_data/runtime/p3_runtime_summary.json")
    stats = read_json(ABLATION_ROOT / "processed_data/ablation_statistics.json")
    methods = read_csv(ABLATION_ROOT / "processed_data/ablation_method_summary.csv")
    config = read_json(ABLATION_ROOT / "configs/p1_ablation_protocol.json")
    provenance = read_json(ABLATION_ROOT / "audit/environment_provenance.json")
    if not (
        p0.get("status") == "PASS_DESCRIPTIVE_EXPLORATORY"
        and p3.get("status") == "PASS_DESCRIPTIVE_EXISTING_CORE45"
        and stats.get("status") == "PASS_COMPLETE_REPLAY_VALIDATED"
    ):
        raise RuntimeError("extension analysis gates are not all PASS")
    for qa_path in (
        PAPER_ROOT / "figures/supplementary_figure1_core_heterogeneity.qa.json",
        PAPER_ROOT / "figures/figure3_quality_runtime_tradeoff.qa.json",
        ABLATION_ROOT / "figures/figure_p1_ablation_effect_by_scale.qa.json",
        ABLATION_ROOT / "figures/supplementary_p1_mechanism_diagnostics.qa.json",
    ):
        qa = read_json(qa_path)
        if not str(qa.get("status", "")).startswith("AUTOMATED_QA_PASS"):
            raise RuntimeError(f"figure QA did not pass: {qa_path}")

    lookup = {row["arm"]: row for row in methods}
    full = lookup["CSG-NI Full"]
    random_arm = lookup["Uniform full-bank selection"]
    no_ni = lookup["No NI (ALNS-H1)"]
    post = stats["post_hoc"]
    p1_conclusion = (
        f"Full CSG-NI attained mean/median instance-level RPD "
        f"{float(full['mean_instance_median_rpd_percent']):.2f}%/"
        f"{float(full['median_instance_median_rpd_percent']):.2f}%, versus "
        f"{float(random_arm['mean_instance_median_rpd_percent']):.2f}%/"
        f"{float(random_arm['median_instance_median_rpd_percent']):.2f}% for uniform full-bank "
        f"selection and {float(no_ni['mean_instance_median_rpd_percent']):.2f}%/"
        f"{float(no_ni['median_instance_median_rpd_percent']):.2f}% without NI. The Friedman test "
        f"gave chi-square={float(stats['friedman']['chi_square']):.3f}, "
        f"p={float(stats['friedman']['p_value']):.3g}. Holm-adjusted Full comparisons were "
        f"p={float(post['Uniform full-bank selection']['holm_adjusted_p']):.3g} "
        f"(rank-biserial {float(post['Uniform full-bank selection']['paired_rank_biserial_positive_favors_full']):+.2f}) "
        f"and p={float(post['No NI (ALNS-H1)']['holm_adjusted_p']):.3g} "
        f"(rank-biserial {float(post['No NI (ALNS-H1)']['paired_rank_biserial_positive_favors_full']):+.2f}), respectively."
    )

    ablation_report = f"""# P1 controlled CSG-NI ablation report

## Result

{p1_conclusion}

All 270 schedules replayed exactly from their stored constructive actions and
passed the independent feasibility checker. Inference uses 18 independent
instance medians; the 90 seed runs per arm are not treated as independent units.

## Operational contrasts

- Full: unchanged frozen Phase6H policy and production search.
- Uniform full-bank: unchanged frozen graph, complete 24-rule bank,
  deduplication, scoring and gate; only the executed target is selected
  uniformly from the unique full bank. The deterministic rebuild overhead is
  included in wall time.
- No NI: existing ALNS-H1 results, operationally equivalent to
  `CSGNIConfig(intervention_rate=0)`.
- No-graph was excluded because graph encoding, support gating and structured
  proposal semantics cannot be neutralized independently without creating an
  out-of-distribution input.

## Artifacts

- Design: `paper_experiments/ablation/ABLATION_DESIGN.md`
- Instance manifest: `paper_experiments/ablation/ablation_instance_manifest.csv`
- Processed runs: `paper_experiments/ablation/processed_data/ablation_runs.csv`
- Statistics: `paper_experiments/ablation/processed_data/ablation_statistics.json`
- Table: `paper_experiments/ablation/tables/table_p1_ablation.tex`
- Main figure: `paper_experiments/ablation/figures/figure_p1_ablation_effect_by_scale.pdf`
- Mechanism diagnostic: `paper_experiments/ablation/figures/supplementary_p1_mechanism_diagnostics.pdf`

## Boundary

This is a small controlled mechanism study, separate from the confirmatory
Core45 comparison. It supports only contrasts that were cleanly separable in
the current implementation.
"""
    atomic_text(ABLATION_ROOT / "ABLATION_REPORT.md", ablation_report)

    p0_paragraph = (PAPER_ROOT / "reports/snippets/p0_scale_cf_results.tex").read_text(encoding="utf-8").strip()
    p1_paragraph = (PAPER_ROOT / "reports/snippets/p1_ablation_results.tex").read_text(encoding="utf-8").strip()
    p3_paragraph = (PAPER_ROOT / "reports/snippets/p3_quality_runtime_results.tex").read_text(encoding="utf-8").strip()
    p1_caption = (
        "\\caption{Instance-paired P1 ablation effects by scale. Each point is one independent "
        "Core instance and compares five-seed median RPD with Full CSG-NI; positive values favor "
        "Full. Boxes show the median and interquartile range across six instances per scale, with "
        "1.5-IQR whiskers.}"
    )
    p1_mechanism_caption = (
        "\\caption{Descriptive P1 mechanism diagnostics. (a) Pooled provenance composition of "
        "executed target sets; state-level counts are not inferential replicates. (b) Frozen-gate "
        "intervention coverage and (c) NI overhead as a fraction of runtime, summarized over 30 "
        "runs per arm and scale by the median, interquartile range and 1.5-IQR whiskers.}"
    )
    atomic_text(PAPER_ROOT / "reports/snippets/p1_figure_caption.tex", p1_caption + "\n")
    atomic_text(PAPER_ROOT / "reports/snippets/p1_mechanism_caption.tex", p1_mechanism_caption + "\n")
    captions = []
    for name in (
        "p0_figure_caption.tex",
        "p1_figure_caption.tex",
        "p1_mechanism_caption.tex",
        "p3_figure_caption.tex",
    ):
        captions.append((PAPER_ROOT / "reports/snippets" / name).read_text(encoding="utf-8").strip())
    source_commit = provenance["git_commit"]
    hashes = {
        path: digest(ROOT / path)
        for path in (
            "paper_experiments/processed_data/supplementary/p0_scale_cf_summary.json",
            "paper_experiments/processed_data/runtime/core_quality_runtime_overall.csv",
            "paper_experiments/ablation/processed_data/ablation_runs.csv",
            "paper_experiments/ablation/processed_data/ablation_statistics.json",
        )
    }
    changed = subprocess.run(
        ["git", "status", "--short", "--", "paper_experiments"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip() or "paper_experiments is ignored/untracked in this checkout; artifact inventory below is authoritative"
    report = f"""# P0 + P1 + P3 experimental extension report

## A. Executive result

P0 locates the known Small-instance limitation across CF strata rather than
hiding it in an aggregate. P1 provides a controlled, replay-validated contrast
for learned target prioritization and NI removal. P3 shows final quality jointly
with realized wall-clock cost and explicitly identifies LG_HGA's retained
generation-cap termination.

## B. P0 findings

{p0['supported_findings']['small']} {p0['supported_findings']['medium_large']}
The analysis is exploratory and descriptive (five independent instances per
cell); it does not assign cellwise significance or causality.

- Figure: `paper_experiments/figures/supplementary_figure1_core_heterogeneity.pdf`
- Processed cells: `paper_experiments/processed_data/supplementary/core_paired_advantage_scale_cf.csv`

## C. P1 findings

{p1_conclusion}

The benchmark is the lexicographically first two canonical IDs in every Scale ×
CF cell, fixed before execution, with seeds 530101--530105. Full and No-NI raw
results were reused and replay-audited; only the uniform full-bank arm was newly
executed. The complete code-level definition and limitation are in
`paper_experiments/ablation/ABLATION_DESIGN.md`.

- Table: `paper_experiments/ablation/tables/table_p1_ablation.tex`
- Figure: `paper_experiments/ablation/figures/figure_p1_ablation_effect_by_scale.pdf`
- Additional process diagnostic: `paper_experiments/ablation/figures/supplementary_p1_mechanism_diagnostics.pdf`

## D. P3 findings

{p3['supported_findings']['budget_use']} {p3['supported_findings']['time_to_best']}
{p3['supported_findings']['termination_disclosure']} Decoder-evaluation counts
remain in processed data but are excluded from the common plot because their
computational semantics differ across algorithm families. No Pareto claim is
made.

- Figure: `paper_experiments/figures/figure3_quality_runtime_tradeoff.pdf`
- Overall data: `paper_experiments/processed_data/runtime/core_quality_runtime_overall.csv`
- Scale data: `paper_experiments/processed_data/runtime/core_quality_runtime_by_scale.csv`

## E. Manuscript insertion package

### P0 paragraph

```latex
{p0_paragraph}
```

### P1 subsection

```latex
{p1_paragraph}
```

### P3 paragraph

```latex
{p3_paragraph}
```

### Figure captions

```latex
{chr(10).join(captions)}
```

The P1 table caption is embedded in
`paper_experiments/ablation/tables/table_p1_ablation.tex`.

## F. Reproducibility record

- Source commit: `{source_commit}`
- Python: `{provenance['python']}`
- Packages: `{json.dumps(provenance['packages'], sort_keys=True)}`
- New-arm config: `paper_experiments/ablation/configs/p1_ablation_protocol.json`
- Seeds: `{config['seeds']}`
- Instance manifest: `paper_experiments/ablation/ablation_instance_manifest.csv`
- Full production bank: 24 requested rules, with deterministic target-set deduplication.
- No Gurobi job was run.
- No GitHub push was performed.
- Commands: `prepare_p1_ablation.py`; two persistent `run_p1_ablation.py`
  shards; `analyze_p1_ablation.py`; `plot_p1_ablation.py`;
  `plot_p1_mechanism_diagnostics.py`; `build_extension_report.py`.

Key output hashes:

```json
{json.dumps(hashes, indent=2, sort_keys=True)}
```

Paper-scoped git status at report generation:

```text
{changed}
```

## G. Recommendation

The P0/P1/P3 package is sufficient for the initial manuscript if the P1 paired
contrasts are interpreted according to their paired effect sizes and adjusted
p-values. No additional experiment is essential merely for completeness; a
future Phase6I replacement remains conditional on its separately frozen R11
promotion decision.
"""
    atomic_text(REPORT_PATH, report)
    print(json.dumps({"status": "PASS", "report": str(REPORT_PATH.relative_to(ROOT)), "hashes": hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
