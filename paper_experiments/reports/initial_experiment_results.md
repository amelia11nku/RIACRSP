---
schema: initial-manuscript-experiment-report-v1
status: PROVISIONAL_PHASE6H_COMPLETE
analysis_python: 3.11.15
analysis_scipy: 1.17.1
figure_matplotlib: 3.10.9
source_sha256:
  paper_experiments/processed_data/main/main_scale_summary.csv: f57f2f177a7bb56edce4bbe0b906e1adc4fa1f2189a50ff7727ed6dd5624fcc9
  paper_experiments/processed_data/main/statistical_tests.csv: 438de1401c737d70f73ac90620a7fb6149b716164aeda36a07b1d01ee85d6e1e
  paper_experiments/processed_data/exact_validation/exact_validation_summary.csv: 6c2bcb26eb406c527315c78656d90123f92290e855e5557e01cb756e15bb1d5f
  paper_experiments/processed_data/efficiency/efficiency_method_summary.csv: 0f0fbc30c98298c5af3e875af738b368a2ab0562dfeda201251b6239fcd82126
  paper_experiments/processed_data/efficiency/efficiency_pairwise.csv: 0f0fec2db4211887900dfdf442c2ef51b208a2806c6dc35a3060bdedc130ca30
  paper_experiments/processed_data/efficiency/anytime_summary.csv: c8b69e98c563a05846ad13da6ab4554253f8a5e36068493f786e04a2ef83656a
  paper_experiments/processed_data/supplementary/core_best_baseline_advantage.csv: f3f4ea6444c077c35c2a249dd29dfb6748ab8a0590d0e7f1036dadd9f3bdb98a
  paper_experiments/processed_data/supplementary/core_paired_advantage_scale_cf.csv: 1782a49abcedb210dca09c690d1c17a036e8de5f95196f53edd32ad8efca2a0b
  paper_experiments/processed_data/supplementary/core_seed_variability.csv: ef7a80070599540a373610de67c98062bfe7c26077bec1361f2762f25aa1c386
  paper_experiments/processed_data/supplementary/core_runtime_utilization.csv: c1aee988724064d23e887c8e7abbad9990aa63cc23d88dc0356d6ea89d832cfd
  paper_experiments/processed_data/supplementary/supplementary_analysis.json: bc0066c0a7a72061e5aeefd568358058fb39bf1481e3873c78a948808b176138
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

Statistical analysis used a Friedman test across the five methods on per-instance median RPD. Because the global test was significant, CSG-NI was compared with each competitor using two-sided paired Wilcoxon signed-rank tests, Holm correction over the four comparisons and paired rank-biserial effect sizes. The family-wise significance threshold was 0.05. Boxplots show the median, interquartile range and 1.5-IQR whiskers, with every Core instance displayed as one point. Analysis and figure generation used Python 3.11.15, SciPy 1.17.1 and Matplotlib 3.10.9.

AUTHOR_INPUT_NEEDED: provide the CPU, GPU, memory and operating-system specifications for the heuristic runs, and the hardware/software specifications of the external Gurobi workstation.

## Draft: Exact validation

All 250 heuristic runs on the exact-validation set returned feasible schedules. Aggregate optimum hits were GA 45/50, Adapted DCGA 50/50, DABC 50/50, LG_HGA 43/50, CSG-NI (Phase6H provisional) 44/50. Thus, the exact set primarily supports implementation correctness and optimum-recovery capability rather than a claim that the proposed method dominates on very small instances. The E10 reference optimum remains 129 and its native Gurobi schedule replay is feasible; the separately decoded Gurobi action-sequence diagnostic yields a feasible makespan of 131 and is retained as a decoder-representation diagnostic, not as a replacement for the proven reference objective.

## Draft: Comparative performance on Core45

CSG-NI achieved the lowest overall mean RPD (3.567%), the lowest median RPD (3.127%) and the best average rank (1.578) across Core45, while attaining the draft BKS on 23 of 45 instances (Table 2 and Fig. 1). The advantage was scale dependent. On Medium instances, CSG-NI reached a mean RPD of 2.996% and an average rank of 1.267; on Large instances, these values improved to 2.186% and 1.067, with the draft BKS attained on all 15 instances. In contrast, GA remained strongest on the Small subset (mean RPD 3.852% and average rank 1.733, compared with 5.518% and 2.400 for CSG-NI). The results therefore support an overall and scale-dependent advantage, not uniform dominance at every problem size.

## Draft: Statistical analysis

The Friedman test rejected equal method performance across the 45 paired instances (χ²=129.369, p=5.313e-27). In the prespecified post-hoc comparisons, CSG-NI had lower median RPD on 33–45 of 45 instances depending on the competitor, and every Holm-adjusted comparison remained below 0.05. Specifically, versus GA: 33/0/12 wins/ties/losses, Holm-adjusted p=1.04e-05, rank-biserial=0.712; versus Adapted DCGA: 45/0/0 wins/ties/losses, Holm-adjusted p=2.27e-13, rank-biserial=1.000; versus DABC: 40/0/5 wins/ties/losses, Holm-adjusted p=9.79e-10, rank-biserial=0.929; versus LG_HGA: 36/0/9 wins/ties/losses, Holm-adjusted p=9.83e-07, rank-biserial=0.793. These effects support the overall Core45 ranking while retaining the Small-scale reversal as a substantive boundary.

## Draft: Exploratory heterogeneity, stability and budget diagnostics

The paired advantage over the best-performing baseline selected separately on each instance changed from a median of -2.138 percentage points on Small instances (3 wins, 12 losses) to 2.118 on Medium instances (12 wins, 3 losses) and 16.549 on Large instances (14 wins, 1 loss; Supplementary Fig. 1). Operation count was positively associated with this post-hoc advantage (Spearman ρ=0.831, unadjusted two-sided p=1.699e-12, n=45). Because operation count is confounded with the predefined scale classes and other instance characteristics, this association is descriptive and neither causal nor predictive.

Across Medium and Large instances, the median within-instance seed RPD s.d. for CSG-NI was 1.253 and 1.608 percentage points, respectively, below GA (4.290, 3.922), DABC (3.547, 2.724) and LG_HGA (3.823, 2.889); Adapted DCGA was still less variable despite its poorer objective quality (Supplementary Fig. 2). On Small instances, CSG-NI's median seed RPD s.d. was 2.096. These are descriptive stability estimates from only five seeds and do not convert low variability into evidence of superior quality.

The median observed runtime divided by the allowed maximum was 0.132 for LG_HGA, which reached the preregistered 100-generation cap, while GA, DABC and CSG-NI were each approximately 1.000 (Supplementary Table 1). LG_HGA's shorter runtime is therefore a disclosed consequence of retaining its source termination rule, not a missing-run or feasibility failure. CSG-NI found its final incumbent after a median 27.8% of the allowed budget, compared with 93.2% for GA and 94.0% for DABC. These timing diagnostics are descriptive because the algorithms have different internal stopping and evaluation mechanics.

## Draft: Search quality and computational efficiency

On the nine-instance CAL-HOLDOUT efficiency set, CSG-NI reduced mean final makespan from 3086.33 for ALNS to 3042.47, corresponding to a mean paired improvement of 1.11% (95% bootstrap interval -0.04% to 2.25%; one-sided Wilcoxon p=0.064; 6 wins and 3 losses across instances). CSG-NI also used fewer mean decoder evaluations (19008.1 versus 32711.7), reached its final best solution earlier in median wall-clock time (68.87 s versus 94.72 s) and had a lower mean normalized-gap AUC (0.0618 versus 0.0783; Table 4 and Fig. 2). Because the confidence interval for final improvement includes zero and the paired test did not cross 0.05, these results are descriptive evidence of improved search efficiency, not a confirmatory superiority claim.

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
