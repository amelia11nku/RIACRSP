# P0 + P1 + P3 experimental extension report

## A. Executive result

P0 locates the known Small-instance limitation across CF strata rather than
hiding it in an aggregate. P1 provides a controlled, replay-validated contrast
for learned target prioritization and NI removal. P3 shows final quality jointly
with realized wall-clock cost and explicitly identifies LG_HGA's retained
generation-cap termination.

## B. P0 findings

The Small-scale weakness against GA occurs in all three CF strata (-0.094, -1.519, -3.212 percentage points); against LG_HGA, the cell median reverses only in CF3 (-0.235 percentage points). All 24 competitor-by-CF cell medians at Medium and Large scale are positive, although positive medians do not imply wins on every instance.
The analysis is exploratory and descriptive (five independent instances per
cell); it does not assign cellwise significance or causality.

- Figure: `paper_experiments/figures/supplementary_figure1_core_heterogeneity.pdf`
- Processed cells: `paper_experiments/processed_data/supplementary/core_paired_advantage_scale_cf.csv`

## C. P1 findings

Full CSG-NI attained mean/median instance-level RPD 2.38%/2.39%, versus 3.19%/3.00% for uniform full-bank selection and 4.79%/4.61% without NI. The Friedman test gave chi-square=24.875, p=3.97e-06. Holm-adjusted Full comparisons were p=0.00836 (rank-biserial +0.75) and p=0.000876 (rank-biserial +1.00), respectively.

The benchmark is the lexicographically first two canonical IDs in every Scale ×
CF cell, fixed before execution, with seeds 530101--530105. Full and No-NI raw
results were reused and replay-audited; only the uniform full-bank arm was newly
executed. The complete code-level definition and limitation are in
`paper_experiments/ablation/ABLATION_DESIGN.md`.

- Table: `paper_experiments/ablation/tables/table_p1_ablation.tex`
- Figure: `paper_experiments/ablation/figures/figure_p1_ablation_effect_by_scale.pdf`
- Additional process diagnostic: `paper_experiments/ablation/figures/supplementary_p1_mechanism_diagnostics.pdf`

## D. P3 findings

Median utilization is CSG-NI 1.000, GA 1.000, DABC 1.000, and LG_HGA 0.130. Median time-to-best fraction is CSG-NI 0.314, GA 0.938, and DABC 0.940.
LG_HGA retains its source-compatible MAXGEN=100 stopping rule, so its shorter runtime is expected and is not evidence of missing runs. Decoder-evaluation counts
remain in processed data but are excluded from the common plot because their
computational semantics differ across algorithm families. No Pareto claim is
made.

- Figure: `paper_experiments/figures/figure3_quality_runtime_tradeoff.pdf`
- Overall data: `paper_experiments/processed_data/runtime/core_quality_runtime_overall.csv`
- Scale data: `paper_experiments/processed_data/runtime/core_quality_runtime_by_scale.csv`

## E. Manuscript insertion package

### P0 paragraph

```latex
\paragraph{Scale--CF heterogeneity.} Exploratory stratification of Core45 showed that the Small-instance boundary was not confined to one CF level: relative to GA, the median paired RPD advantages of CSG-NI were -0.09, -1.52, and -3.21 percentage points for CF1--CF3, respectively. Relative to LG\_HGA, the Small-scale cell median was negative only for CF3 (-0.23 points). By contrast, every competitor-by-CF cell median was positive at Medium and Large scale (24/24 cells), while some cells still contained individual losses. These post-hoc summaries use five independent instances per cell and are descriptive rather than cellwise significance tests or causal evidence.
```

### P1 subsection

```latex
\subsection{Mechanism ablation} On the 18-instance balanced Core subset, Full CSG-NI achieved a mean instance-median RPD of 2.38\%, compared with 3.19\% after replacing learned target prioritization by uniform full-bank selection and 4.79\% without NI. The three-arm Friedman test gave $\chi^2=24.87$ ($p=3.97\times 10^{-6}$). Holm-adjusted paired results and effect sizes are reported in Table~\ref{tab:p1_ablation}; mechanism claims are restricted to this ablation subset and to components that were cleanly separable.
```

### P3 paragraph

```latex
\paragraph{Quality and realized runtime.} On Core45, CSG-NI, GA, and DABC used median fractions of 1.000, 1.000, and 1.000 of the common $2|\mathcal{O}|$ wall-clock ceiling, respectively. LG\_HGA used a median fraction of 0.130 because its original \texttt{MAXGEN=100} termination rule was retained; this is source-compatible early termination, not missing computation. CSG-NI reached its final incumbent at a median 0.314 of budget, compared with 0.938 for GA and 0.940 for DABC. Quality and runtime are therefore reported jointly; no Pareto-optimality claim is made.
```

### Figure captions

```latex
\caption{Scale- and CF-dependent paired quality advantage of CSG-NI on Core45. Each cell reports the median, across five independent instances, of the paired difference between a competitor's and CSG-NI's median RPD over five matched seeds; positive values favor CSG-NI. Parentheses report CSG-NI wins/losses. No cellwise hypothesis tests were performed.}
\caption{Instance-paired P1 ablation effects by scale. Each point is one independent Core instance and compares five-seed median RPD with Full CSG-NI; positive values favor Full. Boxes show the median and interquartile range across six instances per scale, with 1.5-IQR whiskers.}
\caption{Descriptive P1 mechanism diagnostics. (a) Pooled provenance composition of executed target sets; state-level counts are not inferential replicates. (b) Frozen-gate intervention coverage and (c) NI overhead as a fraction of runtime, summarized over 30 runs per arm and scale by the median, interquartile range and 1.5-IQR whiskers.}
\caption{Solution quality and realized computational cost on Core45. Points summarize independent instance-level medians over five matched seeds; error bars show interquartile ranges. Panel (a) compares overall median RPD with normalized budget utilization, and panel (b) reports scale-specific median RPD against actual runtime. LG\_HGA retains its source-compatible \texttt{MAXGEN=100} stopping rule and therefore often terminates before the common wall-clock ceiling.}
```

The P1 table caption is embedded in
`paper_experiments/ablation/tables/table_p1_ablation.tex`.

## F. Reproducibility record

- Source commit: `6a8ac523adc2186c7cb359c2902c207d90184b5b`
- Python: `3.11.15`
- Packages: `{"matplotlib": "3.10.9", "numpy": "1.26.4", "pandas": "2.0.3", "pyarrow": "25.0.1", "scikit-learn": "1.9.0", "scipy": "1.17.1", "torch": "2.11.0+cu128"}`
- New-arm config: `paper_experiments/ablation/configs/p1_ablation_protocol.json`
- Seeds: `[530101, 530102, 530103, 530104, 530105]`
- Instance manifest: `paper_experiments/ablation/ablation_instance_manifest.csv`
- Full production bank: 24 requested rules, with deterministic target-set deduplication.
- No Gurobi job was run.
- No GitHub push was performed.
- Commands: `prepare_p1_ablation.py`; two persistent `run_p1_ablation.py`
  shards; `analyze_p1_ablation.py`; `plot_p1_ablation.py`;
  `plot_p1_mechanism_diagnostics.py`; `build_extension_report.py`.

Key output hashes:

```json
{
  "paper_experiments/ablation/processed_data/ablation_runs.csv": "a180d9260d021cd12bb4d63bf4962b7a60fc170693aba724d8854b3615d9181b",
  "paper_experiments/ablation/processed_data/ablation_statistics.json": "6f7e95d3dd52dc2f33f3cfdca713030e901cc67964b74fe35ac630ccc189c5fb",
  "paper_experiments/processed_data/runtime/core_quality_runtime_overall.csv": "bda366438653eb8c8696b51b48fe290258c343158482282c4b304ebca832d5f6",
  "paper_experiments/processed_data/supplementary/p0_scale_cf_summary.json": "3424d4104cf53c1d5a1033f410eb048febe1d0821cbaf8655e23a84b58fe5777"
}
```

Paper-scoped git status at report generation:

```text
M paper_experiments/README.md
 M paper_experiments/figures/FIGURE_CONTRACTS.md
 M paper_experiments/figures/supplementary_figure1_core_heterogeneity.alignment.json
 M paper_experiments/figures/supplementary_figure1_core_heterogeneity.alignment.svg
 M paper_experiments/figures/supplementary_figure1_core_heterogeneity.eps
 M paper_experiments/figures/supplementary_figure1_core_heterogeneity.qa.json
 M paper_experiments/figures/supplementary_figure1_core_heterogeneity.svg
 M paper_experiments/figures/supplementary_figure1_core_heterogeneity.tiff
 M paper_experiments/figures/supplementary_figure2_seed_stability.eps
 M paper_experiments/figures/supplementary_figure2_seed_stability.qa.json
 M paper_experiments/figures/supplementary_figure2_seed_stability.svg
 M paper_experiments/processed_data/supplementary/core_paired_advantage_by_instance.csv
 M paper_experiments/processed_data/supplementary/core_paired_advantage_scale_cf.csv
 M paper_experiments/processed_data/supplementary/core_runtime_utilization.csv
 M paper_experiments/processed_data/supplementary/core_seed_variability.csv
 M paper_experiments/processed_data/supplementary/core_seed_variability_summary.csv
 M paper_experiments/reports/initial_experiment_results.md
 M paper_experiments/scripts/analyze_core_supplementary.py
 M paper_experiments/scripts/build_results_report.py
 M paper_experiments/scripts/paper_figure_style.py
 M paper_experiments/scripts/plot_supplementary_core_heterogeneity.py
 M paper_experiments/scripts/plot_supplementary_seed_stability.py
 M paper_experiments/tables/supplementary_table1_runtime_utilization.csv
 M paper_experiments/tables/supplementary_table1_runtime_utilization.tex
?? paper_experiments/ablation/
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.alignment.json
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.alignment.svg
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.collision-audit.json
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.eps
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.pdf-text-audit.json
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.qa.json
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.svg
?? paper_experiments/figures/figure3_quality_runtime_tradeoff.tiff
?? paper_experiments/processed_data/runtime/
?? paper_experiments/processed_data/supplementary/p0_scale_cf_summary.json
?? paper_experiments/reports/P0_P1_P3_EXTENSION_REPORT.md
?? paper_experiments/reports/snippets/
?? paper_experiments/scripts/analyze_runtime_tradeoff.py
?? paper_experiments/scripts/build_p0_extension.py
?? paper_experiments/scripts/plot_quality_runtime_tradeoff.py
```

## G. Recommendation

The P0/P1/P3 package is sufficient for the initial manuscript if the P1 paired
contrasts are interpreted according to their paired effect sizes and adjusted
p-values. No additional experiment is essential merely for completeness; a
future Phase6I replacement remains conditional on its separately frozen R11
promotion decision.
