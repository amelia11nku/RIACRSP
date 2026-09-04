# Initial-manuscript experiments

This directory is the reproducible assembly layer for the initial manuscript.
Existing formal outputs remain in their original locations; they are referenced
and audited rather than copied.

Current boundaries:

- The primary Core comparison uses the frozen 45-instance RCIAS-CB1 Core set.
- The common stochastic seeds are `530101` through `530105`, selected as the
  first five preregistered seeds and matched across all five methods. Existing
  results for `530106` through `530110` are preserved but excluded from this
  manuscript analysis.
- The five primary manuscript method names are GA, DCGA, DABC,
  LG_HGA, and CSG-NI. The longer LG_HGA implementation identifier is retained
  only in raw and provenance records. The internal baseline identifier
  `Adapted DCGA` is likewise retained only in raw data and code-level mappings.
- ALNS-H1 is not part of the five-method ranking; it is reserved for CSG-NI
  efficiency and anytime comparisons.
- BKS, RPD, rankings, and significance tests must not be generated until all
  five primary matrices pass the collector audit.
- The provisional CSG-NI column uses the frozen Phase 6H artifact and retains
  the raw identity `CSG_NI_PROVISIONAL_PHASE6H`. Phase 6I-MR may replace only
  that column after the complete preregistered R11 decision.

Run the read-only inventory collector with:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/collect_core_results.py
```

It writes only derived inventory files under `processed_data/core/`; source
results and frozen benchmark files are never modified.

The exact-result collector and the Core downstream analysis are:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/collect_exact_results.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/compute_core_bks_rpd.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/statistical_analysis.py
```

The last two commands intentionally fail while the five-method Core matrix is
incomplete. There is no partial-data override.

Reusable Phase 6H CSG-NI versus ALNS efficiency evidence is assembled with:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/collect_efficiency_results.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/build_tables.py
```

This currently produces manuscript Table 4 in CSV and LaTeX form. ALNS is an
efficiency comparator only, and the CSG-NI identity remains explicitly
provisional Phase 6H.

The two plotting scripts use the `nature-figure` contract in
`figures/FIGURE_CONTRACTS.md`. They enforce the required Times New Roman font,
retain editable SVG text, run a 1.5-pt multi-panel alignment gate, export
SVG/PDF/EPS/TIFF/600-dpi PNG, and run final-PDF glyph and collision audits:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/plot_quality_distribution.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/plot_anytime_efficiency.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/analyze_core_supplementary.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/plot_supplementary_core_heterogeneity.py
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/plot_supplementary_seed_stability.py
```

Figure 1 and both supplementary figures additionally require the complete Core
BKS/RPD gate. No plotting script permits an implicit fallback font or an
unaudited multi-panel export. Supplementary scale-by-CF and stability results
are explicitly post-hoc descriptive analyses and do not alter the primary
Friedman/Wilcoxon family.

The P0/P1/P3 extension package is built after the 90-run P1 uniform-selection
arm reaches `COMPLETE`:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/ablation/finalize_p1_extension.py
```

Its integrated report is `reports/P0_P1_P3_EXTENSION_REPORT.md`. P1 raw,
processed, table and figure artifacts are indexed in
`ablation/ABLATION_REPORT.md`; the primary scale-stratified ablation figure and
the descriptive mechanism diagnostic are stored in `ablation/figures/`.

Run the final package validator at any time to obtain a machine-readable list
of passed and open gates:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/validate_paper_package.py
```

It exits nonzero until every required experiment, table, and figure is present.

After the package validator passes, build the manuscript-ready provisional
experiment narrative, statistical wording, figure legends, evidence-allocation
audit, and discussion boundaries directly from the audited CSV/JSON outputs:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python \
  paper_experiments/scripts/build_results_report.py
```

The generated report is `reports/initial_experiment_results.md`. Hardware fields
that cannot be recovered from the result artifacts remain explicitly marked as
author input rather than being inferred.
