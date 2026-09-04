# P1 controlled CSG-NI ablation report

## Result

Full CSG-NI attained mean/median instance-level RPD 2.38%/2.39%, versus 3.19%/3.00% for uniform full-bank selection and 4.79%/4.61% without NI. The Friedman test gave chi-square=24.875, p=3.97e-06. Holm-adjusted Full comparisons were p=0.00836 (rank-biserial +0.75) and p=0.000876 (rank-biserial +1.00), respectively.

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
