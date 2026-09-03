#!/usr/bin/env python3
"""Build a manuscript-ready provisional experiment report from audited outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
MAIN_ROOT = PAPER_ROOT / "processed_data/main"
EXACT_ROOT = PAPER_ROOT / "processed_data/exact_validation"
EFFICIENCY_ROOT = PAPER_ROOT / "processed_data/efficiency"
SUPPLEMENTARY_ROOT = PAPER_ROOT / "processed_data/supplementary"
REPORT_PATH = PAPER_ROOT / "reports/initial_experiment_results.md"
DISPLAY = {
    "GA": "GA",
    "Adapted DCGA": "Adapted DCGA",
    "DABC-RIACRSP": "DABC",
    "LG_HGA-RIACRSP-v2-N4M": "LG_HGA",
    "CSG-NI Phase6H provisional": "CSG-NI (Phase6H provisional)",
}
CSG = "CSG-NI Phase6H provisional"
SCIPY_VERSION = version("scipy")
MATPLOTLIB_VERSION = version("matplotlib")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_gates() -> None:
    gates = {
        MAIN_ROOT / "analysis_manifest.json": (
            "status", "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H"
        ),
        MAIN_ROOT / "statistical_analysis.json": ("status", "PASS"),
        EXACT_ROOT / "exact_result_inventory.json": ("status", "PASS_COMPLETE"),
        EFFICIENCY_ROOT / "efficiency_inventory.json": (
            "status", "PASS_REUSED_PHASE6H_CAL_HOLDOUT"
        ),
        PAPER_ROOT / "audit/package_validation.json": ("status", "PASS"),
    }
    for path, (field, expected) in gates.items():
        observed = read_json(path).get(field)
        if observed != expected:
            raise RuntimeError(f"report gate failed for {path}: {observed!r} != {expected!r}")


def main() -> int:
    require_gates()
    scale_rows = read_csv(MAIN_ROOT / "main_scale_summary.csv")
    tests = read_csv(MAIN_ROOT / "statistical_tests.csv")
    statistics = read_json(MAIN_ROOT / "statistical_analysis.json")
    exact_rows = read_csv(EXACT_ROOT / "exact_validation_summary.csv")
    efficiency_methods = {
        row["method"]: row
        for row in read_csv(EFFICIENCY_ROOT / "efficiency_method_summary.csv")
    }
    efficiency_pairs = read_csv(EFFICIENCY_ROOT / "efficiency_pairwise.csv")
    anytime = {
        row["method"]: row
        for row in read_csv(EFFICIENCY_ROOT / "anytime_summary.csv")
    }
    supplementary = read_json(SUPPLEMENTARY_ROOT / "supplementary_analysis.json")
    best_baseline_rows = read_csv(SUPPLEMENTARY_ROOT / "core_best_baseline_advantage.csv")
    variability_rows = read_csv(SUPPLEMENTARY_ROOT / "core_seed_variability_summary.csv")
    runtime_rows = {
        row["method"]: row
        for row in read_csv(SUPPLEMENTARY_ROOT / "core_runtime_utilization.csv")
    }
    by_scale = {(row["scale"], row["method"]): row for row in scale_rows}
    if len(by_scale) != 20 or len(tests) != 4 or len(exact_rows) != 50:
        raise RuntimeError("report inputs do not have the expected complete dimensions")

    exact_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in exact_rows:
        exact_hits[row["algorithm"]][0] += int(row["optimum_hit_count"])
        exact_hits[row["algorithm"]][1] += int(row["expected_runs"])

    overall = by_scale[("Overall", CSG)]
    small_ga = by_scale[("S", "GA")]
    small_csg = by_scale[("S", CSG)]
    medium_csg = by_scale[("M", CSG)]
    large_csg = by_scale[("L", CSG)]
    friedman = statistics["friedman"]
    paired = {row["competitor"]: row for row in tests}
    eff_csg = efficiency_methods["PHASE6H_CSGNI"]
    eff_alns = efficiency_methods["ALNS"]
    eff_pair = efficiency_pairs[0]
    anytime_csg = anytime["PHASE6H_CSGNI"]
    anytime_alns = anytime["ALNS"]
    if (
        supplementary.get("status") != "PASS_DESCRIPTIVE_EXPLORATORY"
        or len(best_baseline_rows) != 45
        or len(variability_rows) != 15
        or len(runtime_rows) != 5
    ):
        raise RuntimeError("supplementary Core45 analysis is incomplete")
    best_by_scale = {}
    for scale in ("S", "M", "L"):
        advantages = [
            float(row["advantage_percent_points_positive_favors_csgni"])
            for row in best_baseline_rows
            if row["scale"] == scale
        ]
        best_by_scale[scale] = {
            "median": median(advantages),
            "wins": sum(value > 0 for value in advantages),
            "losses": sum(value < 0 for value in advantages),
        }
    variability = {
        (row["scale"], row["method"]): float(row["median_seed_rpd_sd_percent_points"])
        for row in variability_rows
    }

    hit_text = ", ".join(
        f"{DISPLAY[method]} {hits}/{total}"
        for method, (hits, total) in sorted(
            exact_hits.items(), key=lambda item: list(DISPLAY).index(item[0])
        )
    )
    comparison_text = "; ".join(
        (
            f"versus {DISPLAY[method]}: {row['wins']}/{row['ties']}/{row['losses']} "
            f"wins/ties/losses, Holm-adjusted p={float(row['holm_adjusted_p']):.3g}, "
            f"rank-biserial={float(row['paired_rank_biserial_positive_favors_csg']):.3f}"
        )
        for method, row in sorted(
            paired.items(), key=lambda item: list(DISPLAY).index(item[0])
        )
    )

    source_files = [
        MAIN_ROOT / "main_scale_summary.csv",
        MAIN_ROOT / "statistical_tests.csv",
        EXACT_ROOT / "exact_validation_summary.csv",
        EFFICIENCY_ROOT / "efficiency_method_summary.csv",
        EFFICIENCY_ROOT / "efficiency_pairwise.csv",
        EFFICIENCY_ROOT / "anytime_summary.csv",
        SUPPLEMENTARY_ROOT / "core_best_baseline_advantage.csv",
        SUPPLEMENTARY_ROOT / "core_paired_advantage_scale_cf.csv",
        SUPPLEMENTARY_ROOT / "core_seed_variability.csv",
        SUPPLEMENTARY_ROOT / "core_runtime_utilization.csv",
        SUPPLEMENTARY_ROOT / "supplementary_analysis.json",
    ]
    source_hashes = "\n".join(
        f"  {path.relative_to(ROOT)}: {sha256(path)}" for path in source_files
    )
    report = f"""---
schema: initial-manuscript-experiment-report-v1
status: PROVISIONAL_PHASE6H_COMPLETE
analysis_python: {platform.python_version()}
analysis_scipy: {SCIPY_VERSION}
figure_matplotlib: {MATPLOTLIB_VERSION}
source_sha256:
{source_hashes}
---

# Provisional initial-manuscript experiment report

## One-sentence argument

Across the frozen five-seed Core45 protocol, CSG-NI (Phase6H provisional) provided the strongest overall solution quality, particularly on medium and large instances, while exact-set and efficiency analyses established feasibility and exposed the limits of the current provisional evidence.

## Terminology ledger

| Canonical term | Definition and manuscript use |
|---|---|
| RCIAS-CB1 Core (Core45) | The frozen 45-instance primary comparison benchmark, with 15 Small, 15 Medium and 15 Large instances. |
| CSG-NI (Phase6H provisional) | The current proposed-method column; retain the provisional label until the preregistered Phase6I-MR decision. |
| Adapted DCGA | A faithful RI-ACRSP adaptation; do not describe it as an exact source implementation. |
| LG_HGA | Manuscript display name following the original paper; the longer implementation identifier is retained only in provenance data. |
| Draft BKS | The minimum makespan over five methods and five matched seeds for each Core45 instance. |
| RPD | Relative percentage deviation from the draft BKS; lower is better. |

## Draft: Experimental setup

We evaluated GA, Adapted DCGA, DABC, LG_HGA and CSG-NI on the frozen RCIAS-CB1 Core benchmark. Core45 contains 45 instances equally divided among Small, Medium and Large scales. Each method was run with the same five matched stochastic seeds (530101–530105) and a common maximum wall-clock budget of 2 s per operation, giving 225 runs per method and 1,125 runs in total. All returned schedules were independently replayed and feasible. LG_HGA also retained the original-paper MAXGEN=100 termination rule and therefore could stop before exhausting the common wall-clock ceiling. For each instance, the draft best-known solution (BKS) was defined as the minimum makespan observed across the five methods and five matched seeds, and RPD was computed relative to this value. The main descriptive unit was the instance-level median across five seeds; the 45 instances, rather than the 225 seed-runs per method, were the independent paired units for inference.

The dedicated exact-validation benchmark comprised ten resized and scale-renamed instances with 6–9 operations. Gurobi was executed on an external licensed workstation and proved the optimum for every instance. Each heuristic was evaluated for five matched seeds under the same 2 s-per-operation budget. Optimum-hit time was measured from the first incumbent equal to the proven optimum; runs that did not reach the optimum were retained as right-censored observations rather than assigned an artificial time.

Statistical analysis used a Friedman test across the five methods on per-instance median RPD. Because the global test was significant, CSG-NI was compared with each competitor using two-sided paired Wilcoxon signed-rank tests, Holm correction over the four comparisons and paired rank-biserial effect sizes. The family-wise significance threshold was 0.05. Boxplots show the median, interquartile range and 1.5-IQR whiskers, with every Core instance displayed as one point. Analysis and figure generation used Python {platform.python_version()}, SciPy {SCIPY_VERSION} and Matplotlib {MATPLOTLIB_VERSION}.

AUTHOR_INPUT_NEEDED: provide the CPU, GPU, memory and operating-system specifications for the heuristic runs, and the hardware/software specifications of the external Gurobi workstation.

## Draft: Exact validation

All 250 heuristic runs on the exact-validation set returned feasible schedules. Aggregate optimum hits were {hit_text}. Thus, the exact set primarily supports implementation correctness and optimum-recovery capability rather than a claim that the proposed method dominates on very small instances. The E10 reference optimum remains 129 and its native Gurobi schedule replay is feasible; the separately decoded Gurobi action-sequence diagnostic yields a feasible makespan of 131 and is retained as a decoder-representation diagnostic, not as a replacement for the proven reference objective.

## Draft: Comparative performance on Core45

CSG-NI achieved the lowest overall mean RPD ({float(overall['mean_of_instance_mean_rpd_percent']):.3f}%), the lowest median RPD ({float(overall['median_of_instance_median_rpd_percent']):.3f}%) and the best average rank ({float(overall['average_rank']):.3f}) across Core45, while attaining the draft BKS on {int(overall['bks_attainment_count'])} of 45 instances (Table 2 and Fig. 1). The advantage was scale dependent. On Medium instances, CSG-NI reached a mean RPD of {float(medium_csg['mean_of_instance_mean_rpd_percent']):.3f}% and an average rank of {float(medium_csg['average_rank']):.3f}; on Large instances, these values improved to {float(large_csg['mean_of_instance_mean_rpd_percent']):.3f}% and {float(large_csg['average_rank']):.3f}, with the draft BKS attained on all 15 instances. In contrast, GA remained strongest on the Small subset (mean RPD {float(small_ga['mean_of_instance_mean_rpd_percent']):.3f}% and average rank {float(small_ga['average_rank']):.3f}, compared with {float(small_csg['mean_of_instance_mean_rpd_percent']):.3f}% and {float(small_csg['average_rank']):.3f} for CSG-NI). The results therefore support an overall and scale-dependent advantage, not uniform dominance at every problem size.

## Draft: Statistical analysis

The Friedman test rejected equal method performance across the 45 paired instances (χ²={float(friedman['statistic']):.3f}, p={float(friedman['p_value']):.3e}). In the prespecified post-hoc comparisons, CSG-NI had lower median RPD on 33–45 of 45 instances depending on the competitor, and every Holm-adjusted comparison remained below 0.05. Specifically, {comparison_text}. These effects support the overall Core45 ranking while retaining the Small-scale reversal as a substantive boundary.

## Draft: Exploratory heterogeneity, stability and budget diagnostics

The paired advantage over the best-performing baseline selected separately on each instance changed from a median of {best_by_scale['S']['median']:.3f} percentage points on Small instances ({best_by_scale['S']['wins']} wins, {best_by_scale['S']['losses']} losses) to {best_by_scale['M']['median']:.3f} on Medium instances ({best_by_scale['M']['wins']} wins, {best_by_scale['M']['losses']} losses) and {best_by_scale['L']['median']:.3f} on Large instances ({best_by_scale['L']['wins']} wins, {best_by_scale['L']['losses']} loss; Supplementary Fig. 1). Operation count was positively associated with this post-hoc advantage (Spearman ρ={float(supplementary['exploratory_association']['spearman_rho']):.3f}, unadjusted two-sided p={float(supplementary['exploratory_association']['two_sided_p_value_unadjusted']):.3e}, n=45). Because operation count is confounded with the predefined scale classes and other instance characteristics, this association is descriptive and neither causal nor predictive.

Across Medium and Large instances, the median within-instance seed RPD s.d. for CSG-NI was {variability[('M', CSG)]:.3f} and {variability[('L', CSG)]:.3f} percentage points, respectively, below GA ({variability[('M', 'GA')]:.3f}, {variability[('L', 'GA')]:.3f}), DABC ({variability[('M', 'DABC-RIACRSP')]:.3f}, {variability[('L', 'DABC-RIACRSP')]:.3f}) and LG_HGA ({variability[('M', 'LG_HGA-RIACRSP-v2-N4M')]:.3f}, {variability[('L', 'LG_HGA-RIACRSP-v2-N4M')]:.3f}); Adapted DCGA was still less variable despite its poorer objective quality (Supplementary Fig. 2). On Small instances, CSG-NI's median seed RPD s.d. was {variability[('S', CSG)]:.3f}. These are descriptive stability estimates from only five seeds and do not convert low variability into evidence of superior quality.

The median observed runtime divided by the allowed maximum was {float(runtime_rows['LG_HGA-RIACRSP-v2-N4M']['median_runtime_budget_fraction']):.3f} for LG_HGA, which reached the preregistered 100-generation cap, while GA, DABC and CSG-NI were each approximately 1.000 (Supplementary Table 1). LG_HGA's shorter runtime is therefore a disclosed consequence of retaining its source termination rule, not a missing-run or feasibility failure. CSG-NI found its final incumbent after a median {100 * float(runtime_rows[CSG]['median_time_to_best_budget_fraction']):.1f}% of the allowed budget, compared with {100 * float(runtime_rows['GA']['median_time_to_best_budget_fraction']):.1f}% for GA and {100 * float(runtime_rows['DABC-RIACRSP']['median_time_to_best_budget_fraction']):.1f}% for DABC. These timing diagnostics are descriptive because the algorithms have different internal stopping and evaluation mechanics.

## Draft: Search quality and computational efficiency

On the nine-instance CAL-HOLDOUT efficiency set, CSG-NI reduced mean final makespan from {float(eff_alns['mean_of_instance_means']):.2f} for ALNS to {float(eff_csg['mean_of_instance_means']):.2f}, corresponding to a mean paired improvement of {100 * float(eff_pair['mean_relative_improvement']):.2f}% (95% bootstrap interval {100 * float(eff_pair['bootstrap_95_low']):.2f}% to {100 * float(eff_pair['bootstrap_95_high']):.2f}%; one-sided Wilcoxon p={float(eff_pair['wilcoxon_one_sided_p']):.3f}; 6 wins and 3 losses across instances). CSG-NI also used fewer mean decoder evaluations ({float(eff_csg['mean_decoder_evaluations']):.1f} versus {float(eff_alns['mean_decoder_evaluations']):.1f}), reached its final best solution earlier in median wall-clock time ({float(anytime_csg['median_time_to_best']):.2f} s versus {float(anytime_alns['median_time_to_best']):.2f} s) and had a lower mean normalized-gap AUC ({float(anytime_csg['mean_normalized_gap_auc']):.4f} versus {float(anytime_alns['mean_normalized_gap_auc']):.4f}; Table 4 and Fig. 2). Because the confidence interval for final improvement includes zero and the paired test did not cross 0.05, these results are descriptive evidence of improved search efficiency, not a confirmatory superiority claim.

## Draft: Discussion

Taken together, the experiments indicate that the provisional Phase6H policy is most useful as problem scale increases. Its overall Core45 rank and complete BKS attainment on the Large subset are consistent with a search policy that allocates effort more effectively in larger solution spaces, but the present design does not isolate a causal mechanism for that pattern. The Small-scale advantage of GA is equally informative: CSG-NI should not be presented as uniformly superior, and low-overhead search remains competitive when instances are small.

The exact-validation study provides a separate correctness envelope. Every method produced feasible schedules, and all methods recovered proven optima on most or all of the ten small instances. This guards against interpreting Core45 differences as consequences of invalid schedules, but the small exact set is not sufficiently difficult to establish comparative dominance. Similarly, the CAL-HOLDOUT analysis suggests that CSG-NI can achieve better anytime quality with fewer decoder evaluations than ALNS, although the uncertainty interval and paired test leave the final-quality advantage inconclusive at the current nine-instance sample size.

Three boundaries remain. First, the Core45 BKS and proposed-method column are provisional and must be recomputed if Phase6I-MR replaces Phase6H. Second, ALNS is an efficiency comparator on CAL-HOLDOUT and is not part of the five-method Core45 ranking. Third, the E10 action-sequence discrepancy shows that native schedule replay and action-sequence re-decoding are not interchangeable validation paths; both diagnostics should remain visible, while the proven native optimum remains the exact reference. A decisive next analysis is therefore the preregistered Phase6I-MR promotion decision, followed—only if the artifact changes—by an automatic rebuild of BKS, RPD, statistics, tables and figures.

## Figure legends

**Figure 1 | Distribution of solution-quality gaps on Core45.** Panels show Small (a), Medium (b) and Large (c) instances. Each point is one independent Core instance and represents the median RPD across five matched seeds (n=15 instances per panel and method). Boxes show the median and interquartile range; whiskers extend to 1.5 times the interquartile range. RPD is calculated against the draft pooled BKS over all five methods and five seeds. No observations were excluded. CSG-NI denotes the provisional Phase6H artifact.

**Figure 2 | Anytime quality and search effort of CSG-NI relative to ALNS.** (a) Median gap to the pooled BKS at six normalized runtime checkpoints; shaded bands show the interquartile range. (b) Median decoder evaluations against median gap at the same checkpoints on a logarithmic evaluation axis; horizontal and vertical error bars show interquartile ranges. Each checkpoint summarizes 45 matched runs (nine CAL-HOLDOUT instances × five seeds) per method; no runs were excluded. The comparison is descriptive efficiency evidence, ALNS is not included in the Core45 ranking, and CSG-NI retains its provisional Phase6H identity.

**Supplementary Figure 1 | Scale- and configuration-dependent paired quality differences on Core45.** Panels show Small (a), Medium (b) and Large (c) strata. Each cell is the median over five independent Core instances of the competitor's per-instance median RPD minus the corresponding CSG-NI value; positive values therefore favor CSG-NI. Parentheses give CSG-NI wins/losses among the five instances. Each per-instance value summarizes five matched seeds. No observations were excluded. The stratification is post hoc and descriptive; no cellwise hypothesis tests were performed.

**Supplementary Figure 2 | Seed-to-seed solution-quality stability on Core45.** Panels show Small (a), Medium (b) and Large (c) instances. Each point is one independent Core instance and represents the sample standard deviation of RPD across five matched seeds (n=15 instances per panel and method). Boxes show the median and interquartile range; whiskers extend to 1.5 times the interquartile range. No observations were excluded; separate outlier glyphs are suppressed because all points are shown. Lower variability does not imply better objective quality.

## Main-text discipline audit

| Result | Class | Destination |
|---|---|---|
| Overall and scale-specific Core45 quality | Core discovery | Main text; Table 2 and Fig. 1 carry the full summaries. |
| Friedman and four corrected paired comparisons | Necessary support | Global result and effect-size range in main text; full comparisons in Table 3. |
| Exact optimum recovery and feasibility | Necessary support | Main text and compact Table 1; seed-level detail in the appendix. |
| CAL-HOLDOUT anytime/efficiency | Qualification | Main text with non-significant final-quality boundary; Table 4 and Fig. 2. |
| Scale × CF paired differences | Exploratory heterogeneity | Supplementary Fig. 1; concise scale-dependent boundary in Results/Discussion. |
| Five-seed within-instance variability | Exploratory robustness | Supplementary Fig. 2; do not equate stability with quality. |
| Runtime-budget utilization and termination | Fairness disclosure | Methods and Supplementary Table 1; state that LG_HGA retains MAXGEN=100. |
| E10 action replay 131 versus native optimum 129 | Edge case | Main-text boundary or Methods note; retain full diagnostic in the audit record. |
| Hashes, completion matrices and per-instance rows | Provenance detail | Source data, audit package and appendices. |

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| CSG-NI is strongest overall on Core45 under the frozen five-seed protocol. | Mean/median RPD, average rank, BKS count and paired tests over 45 instances. | Supported, provisional artifact. |
| The advantage increases on Medium and Large instances. | Scale-specific summaries; Large BKS count 15/15; Small-scale GA reversal. | Supported within Core45. |
| The scale trend persists against the best baseline selected per instance. | Median paired advantage shifts from negative on Small to strongly positive on Large; exploratory operation-count association. | Descriptively supported; post hoc and confounded. |
| CSG-NI is comparatively stable over seeds on Medium and Large instances. | Per-instance sample s.d. of RPD over five matched seeds. | Descriptively supported against GA, DABC and LG_HGA; not against Adapted DCGA. |
| CSG-NI is more search efficient than ALNS. | Lower evaluations, time-to-best and gap AUC on nine CAL-HOLDOUT instances. | Descriptively supported; confirmatory final-quality superiority not established. |
| Heuristic results are feasible and can recover exact optima. | 250 feasible exact-set runs and optimum-hit records against ten proven optima. | Supported on the dedicated small exact set. |
| CSG-NI's mechanism causes the scale-dependent gain. | No causal ablation in the current main package. | Not established; do not claim. |
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_name(f"{REPORT_PATH.name}.tmp.{os.getpid()}")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(REPORT_PATH)
    print(json.dumps({
        "status": "PASS",
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "source_files": len(source_files),
        "sha256": sha256(REPORT_PATH),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
