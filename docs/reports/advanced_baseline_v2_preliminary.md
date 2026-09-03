# Advanced baseline v2 three-seed preliminary gate

## Gate result

The matched three-seed CB1-Core preliminary matrix is complete and passes its
artifact audit. It contains 45 instances, seeds 530101--530103, and 135 results
for each of DABC-RIACRSP and LG_HGA-RIACRSP-v2-N4M. All keys are unique and all
stored schedules passed a fresh independent feasibility check.

The frozen evidence identifiers are:

- formal manifest SHA-256: `1895f418f15f9bf01f29d50891cdef4c13eae083445140b4a0c615492887868e`;
- DABC implementation manifest SHA-256: `d27206cf25fa74abd0a7d7767a277ff2ced0713f1876f639471b36c9d2840c8a`;
- LG_HGA-v2 implementation manifest SHA-256: `9dbf3c5e468d749831d2cef05aec10023373bdedc437bf3584175f7d67975223`;
- LG_HGA-v2 knowledge manifest hash: `8f316ca918d3709f237205edf7ea9466da54006689b056c3866cf35d8e4cc515`.

No training instance ID or content hash overlaps the 45 Core instances. The
knowledge freeze contains 180 runs, 72,000 one-step Eq. (11) rows, and nine
separate `scale x CF_level` DTR bundles.

## Preliminary result

Across the 135 matched runs, mean makespan was 3437.96 for DABC and 3201.35 for
LG_HGA-v2. The mean of paired percentage improvements was 5.55% in favor of
LG_HGA-v2. LG_HGA-v2 recorded 109 wins, 2 ties, and 24 losses. At the primary
45-instance unit, using each algorithm's mean over three seeds, it recorded 37
wins and 8 losses. A tie-corrected normal approximation to the paired Wilcoxon
test gives statistic 64.0 and two-sided p approximately 3.17e-7; SciPy's
approximation gives 3.07e-7.

The effect is scale-dependent. Mean paired improvements were near zero for the
small instances (-1.42%, -0.07%, and +1.74% over CF1--CF3), about 7.09--8.92%
for medium instances, and about 8.66--8.92% for large instances.

## Local-search gate behavior

The strict paper threshold remains `predicted R > 50`. The frozen models pass
it only at generation 1 in three regimes:

- `S_CF1`: N4_MMIT, one gate pass and 100 local evaluations per run;
- `L_CF1`: N3_TOPO, one gate pass and 100 local evaluations per run;
- `L_CF2`: N3_TOPO, one gate pass and 100 local evaluations per run.

The other six regimes never pass the threshold. The completed matrix therefore
contains 45 gate passes and 4,500 local decoder evaluations in total. This is a
partial correction of the v1 zero-trigger outcome, not evidence that the
source's local-search behavior has been recovered for every regime.

## Compute-accounting limitation

The methods shared the same `2 x number_of_operations` wall-clock ceiling, but
LG_HGA-v2 also reached its preregistered 100-generation cap. Consequently it
used a mean of 4,073 decoder evaluations and 34.61 seconds, compared with 20,489
evaluations and 238.36 seconds for DABC. The preliminary result supports
continuation to the remaining seven frozen seeds, but it must not be described
as an equal-consumed-compute comparison. Final reporting must include both
wall-clock and decoder-evaluation views.

## Continuation

The remaining seeds 530104--530110 are being collected without changing the
algorithm, threshold, models, or manifests. DABC and LG_HGA-v2 write to separate
resumable output roots. The 10-seed summary must not be generated until both
methods have all 450 expected Core results and pass the same full audit.

Machine-readable preliminary outputs are under
`outputs/baselines/comparison_advanced_v2/summary/`.
