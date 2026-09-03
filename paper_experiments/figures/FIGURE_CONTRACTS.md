# Initial-manuscript figure contracts

These contracts define the scientific purpose and integrity boundary before
styling or rendering. The target is a 180 mm double-column, Python/matplotlib
figure bundle with editable SVG/PDF text, Times New Roman as required by the
experiment manual, and SVG/PDF/EPS/TIFF/PNG exports.

## Figure 1 — Core45 solution-quality distribution

- Core conclusion: relative solution-quality differences among GA, Adapted
  DCGA, DABC, LG_HGA and provisional Phase6H CSG-NI are assessed across
  Small, Medium and Large Core45 instances.
- Results question: does the comparative quality pattern persist or change
  with problem scale?
- Archetype: quantitative grid.
- Panel roles: `a`, Small stratification; `b`, Medium stratification; `c`,
  Large stratification. Equal geometry is intentional because the panels invite
  direct scale comparison.
- Evidence unit: one Core instance. Each visible point is the median RPD over
  the five matched manuscript seeds; seeds are not treated as 225 independent
  inferential units.
- Center/spread: median, interquartile range and 1.5-IQR whiskers, with all 15
  per-scale instance summaries shown.
- Exclusions: none.
- Reviewer risks: BKS must be calculated only after the complete five-method
  matrix; Phase6H is provisional; formal paired inference belongs to Table 3,
  not to visual overlap between marginal boxplots.

## Figure 2 — Anytime performance and search effort

- Core conclusion: provisional Phase6H CSG-NI is evaluated against ALNS using
  complementary wall-clock convergence and decoder-effort evidence.
- Results question: does CSG-NI change both convergence quality and the amount
  of decoded search needed to reach it?
- Archetype: quantitative grid.
- Panel roles: `a`, primary anytime convergence over normalized wall-clock
  budget; `b`, supporting efficiency view over decoder evaluations.
- Evidence unit: 45 matched runs per method from nine CAL-HOLDOUT instances and
  five seeds.
- Center/spread: median and interquartile range at all six normalized-budget
  checkpoints. Panel `b` reports both horizontal and vertical IQR.
- Transformations: relative gap is displayed in percent; decoder evaluations
  use a positive logarithmic axis; no interpolation, downsampling or simulated
  values.
- Exclusions: none.
- Reviewer risks: pooled BKS is a descriptive common reference; ALNS is an
  efficiency comparator only and must not enter the five-method Core ranking;
  CAL-HOLDOUT evidence must not be described as Core45 evidence.

## Supplementary Figure 1 — Scale and CF heterogeneity

- Core conclusion: the paired quality advantage of provisional Phase6H CSG-NI
  is strongly scale dependent and is not uniform on Small instances.
- Results question: within each scale and CF stratum, how large is the median
  paired RPD difference from each baseline and how consistent is its direction?
- Archetype: quantitative grid.
- Panel roles: `a`, Small boundary; `b`, Medium transition; `c`, Large advantage.
- Evidence unit: one Core instance. Each cell summarizes five independent
  instances whose method values are medians over the five matched seeds.
- Center/spread: color and signed number show the median paired RPD difference;
  parenthetical annotations show CSG-NI wins/losses among the five instances.
- Exclusions: none.
- Reviewer risks: this is a post-hoc descriptive stratification with only five
  instances per cell; it carries no cellwise significance or causal claim.

## Supplementary Figure 2 — Seed-to-seed stability

- Core conclusion: stochastic variability differs by method and scale and
  should qualify, rather than replace, the solution-quality comparison.
- Results question: how variable is RPD over the five matched seeds within each
  method-instance cell?
- Archetype: quantitative grid.
- Evidence unit: one Core instance. Each point is the sample standard deviation
  of five seed-level RPD values; there are 15 points per scale and method.
- Center/spread: median, interquartile range and 1.5-IQR whiskers, with every
  instance shown.
- Exclusions: none; duplicate boxplot outlier glyphs are suppressed because all
  raw instance-level points are overlaid.
- Reviewer risks: low variance is not evidence of good objective quality, and
  five seeds give only a coarse stability estimate.

## Blocking delivery gates

All main and supplementary figures require a fresh 1.5 pt rendered panel-alignment report, PDF glyph
audit with a 5 pt floor, PDF collision audit, and panel-by-panel inspection at
final physical size. Figure 1 and both supplementary figures additionally
require the complete Core BKS/RPD gate. Export remains blocked until Times New
Roman and the collision-audit runtime are available in the plotting environment.

The `nature-figure` static validator recognizes only its default publication
sans-serif families. Its `FONT-FAMILY` result is therefore an expected protocol
exception here: the experiment manual explicitly requires Times New Roman, and
the plotting runtime verifies that exact installed family with fallback
disabled. The shared rendering helper passes every other static check (20/20).
Entry-point-only preflight does not follow imports and consequently reports the
helper-owned export/alignment settings as absent; the executable helper remains
the authoritative gate and cannot export before alignment and PDF QA run.
All fills and error bars use opaque, pre-mixed publication colors so the EPS
render preserves the same evidence hierarchy as the PDF/PNG outputs; no
PostScript transparency fallback is accepted.
