# P1 figure contracts

## Main ablation effect figure

- Core conclusion: the quality cost of neutralizing learned target selection or
  disabling NI is measured as a paired contrast to frozen Full CSG-NI.
- Evidence unit: one of 18 independently selected Core instances; every plotted
  value is the difference between five-seed arm medians.
- Panels: Small, Medium and Large, with six independently selected instances
  per panel.
- Center/spread: median, IQR, 1.5-IQR whiskers and all instance values.
- Direction: positive values favor Full CSG-NI.
- Reviewer risks: this is a distinct ablation family, not an enlargement of the
  main Core45 confirmatory test; marginal box overlap is not a significance
  test.

## Supplementary mechanism diagnostics

- Core conclusion: interpret the learned-selection contrast alongside executed
  target provenance, frozen-gate intervention coverage, and measured NI
  overhead.
- Panel `a`: pooled executed target-family composition, explicitly descriptive
  at the state level and not an inferential replicate count.
- Panels `b` and `c`: run-level distributions over 30 runs per arm and scale;
  seeds remain repeated stochastic runs rather than independent problem units.
- Candidate bank: both arms request all 24 production rules and deduplicate only
  identical target sets.
- Reviewer risks: process composition and overhead explain operational behavior
  but do not independently establish a quality mechanism.

Both figures inherit the manuscript 180 mm, Times New Roman, editable vector
text, PDF/SVG/EPS/TIFF/PNG export, alignment, glyph and collision QA contract.
