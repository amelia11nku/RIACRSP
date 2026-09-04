# P1 controlled CSG-NI mechanism ablation

## Frozen reference and causal contrast

The reference arm is the completed provisional Phase6H CSG-NI Core artifact.
Its checkpoint, calibration policy, graph construction, 24-rule production
candidate bank, deduplication, H1 initialization, decoder, intervention rate,
random-number namespaces, and `2|O|` wall-clock ceiling are not retrained or
retuned. Existing reference results are reused rather than rerun.

The smallest clean design has three arms:

1. **CSG-NI Full** — the frozen Phase6H reference.
2. **Uniform full-bank selection at the frozen gate** — the full frozen
   inference call still builds the CSG, generates and deduplicates all 24
   production rules, scores the unique targets, and applies the frozen
   calibration/support gate. Whenever that gate elects to intervene, learned
   top-target selection is replaced by a deterministic uniform draw over all
   unique targets in the same full bank. The draw uses an isolated preregistered
   RNG namespace. A second deterministic bank construction is used to avoid
   editing frozen inference internals; that overhead is included in wall time
   and logged separately. This contrast isolates learned target prioritization
   conditional on the unchanged frozen intervention gate.
3. **No NI intervention (ALNS-H1 equivalence)** — `CSGNIConfig(intervention_rate=0)`
   delegates directly to the frozen ALNS implementation. Completed ALNS-H1
   Core runs use the same H1 initialization, ALNS parameters, five seeds,
   decoder, and `2|O|` ceiling, so those schedules are replay-validated and
   reused. This is an equivalence arm, not a newly proposed method.

## Implementation audit

- **H1 initialization:** `solve_csgni` always constructs H1 before search; it is
  held fixed in all three arms.
- **CSG/graph guidance:** `FrozenLiveInference.decide` constructs the critical
  synchronization graph and graph-derived state features on every eligible
  decision. Those features feed the learned state encoder and support guard.
- **Candidate generation:** `generate_revised_target_arms` produces exactly 24
  rule proposals: original destroy operators, related/random variants, local
  perturbations, and structured near-neighbours.
- **Deduplication:** identical operation target sets are merged after rule
  generation; live states normally expose approximately 23--24 unique targets.
- **Neural actor/scoring and critic/utility:** all unique targets are tensorized,
  embedded, scored, and assigned utility estimates. The Full arm takes the
  highest raw-score target. The random-selection arm retains these computations
  only for the frozen gate and neutralizes the final target ranking.
- **Feasibility decoder:** every neighbour in every arm passes through the same
  insertion decoder; final schedules are also replayed from stored actions and
  checked by the independent schedule validator.
- **Intervention schedule:** Phase6H uses deterministic 100% eligibility; the
  calibrated gate decides whether an eligible state is actually intervened.
- **Stopping/budget:** elapsed wall time, including graph, proposal, inference,
  random-selection rebuild, repair, and decoding, is bounded by `2|O|` seconds.
- **Caching/batching:** one state is tensorized per live decision; the checkpoint
  is loaded once per persistent worker. Model-loading time is recorded but lies
  outside each run budget in both the original Full runner and this ablation.

## Excluded pseudo-ablation

“No graph guidance” is not included because graph state encoding, structured
proposal generation, support guarding, and the checkpoint input distribution
are operationally entangled. Zeroing or replacing graph inputs would create an
out-of-distribution model input while also changing candidate semantics, so it
would not isolate one mechanism. A reduced top-8 layer is likewise unnecessary
and is not described as the production bank.

## Benchmark and inference unit

The benchmark contains the lexicographically first two canonical IDs in each
Scale × CF stratum (18 independent instances). The matched seeds are 530101--
530105. Performance-based selection and post-hoc exclusions are prohibited.
Statistical inference uses the 18 instance-level medians, never 90 seed runs as
independent observations. The analysis family is explicitly separate from the
main Core45 confirmatory comparison.
