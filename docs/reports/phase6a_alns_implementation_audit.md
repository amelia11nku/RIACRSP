# Phase 6A ALNS Implementation Audit

## Scope and frozen paths

The implementation audited at commit `e9aa48418bf91143f10537ac5256003a06b2ff60` is
`rcias_clgri/search/alns.py`. Frozen instances are located at
`instances/canonical/RCIAS-2.0`, `instances/controlled/RCIAS-CB1/core`, and
`instances/controlled/RCIAS-CB1/sensitivity`; the diagnostic DEV set is at
`instances/controlled/RCIAS-CB1/dev`. The decoder and checker are
`rcias_clgri/search/common.py:decode_candidate` and
`rcias_clgri/env/feasibility.py:check_schedule`.

## Exact implementation

- Initialization: frozen deterministic H1 through `solve_dispatching(instance, "H1")`, converted to a `Candidate`, then decoded by the common decoder.
- Destroy operators, exact names: `random`, `critical`, `overloaded_island`, `high_reconfiguration`, `w_bottleneck`, `f_bottleneck`, `related`.
- Repair operators, exact names: `greedy`, `regret2`, `regret3`, `reconfiguration_aware`, `transport_aware`.
- Destroy size: `max(2, round(num_operations * destroy_fraction))`; the frozen fraction is 0.15. The count is constant within an instance.
- Adaptive selection: separate roulette selection over destroy and repair families. Both selected weights are updated each iteration with reaction factor 0.2. Scores are 5 for a global-best candidate, 1 for an accepted candidate, and 0 otherwise, floored to 0.1 during the update.
- Acceptance: improvement/equality is unconditional; worsening candidates use simulated annealing probability `exp(-delta/temperature)`.
- Temperature: initial value is `0.05 * max(1, H1 makespan)` and geometric cooling is 0.995 per iteration.
- Candidate generation: the selected removed set is repaired eight times; the lowest-makespan decoded candidate is considered.
- Stopping: formal search uses the frozen wall-clock budget `2 * num_operations` seconds. An optional fixed iteration limit exists solely for instrumentation regression tests and is unset in formal runs.
- Best handling: a candidate strictly below the historical best becomes the new global best regardless of whether the acceptance branch was needed; best time, decoder count, and convergence trace are recorded.
- Seeds: one local `random.Random(seed)` drives roulette, stochastic destroy/repair, and annealing. No global RNG is used.
- Feasibility: every candidate passes through the common deterministic construction decoder, which independently calls `check_schedule` and raises on infeasibility.
- Existing logging: Phase 5C records aggregate selections, acceptances, global-best counts, final weights, and global-best trace. It does not contain rejected transition details or destroyed targets.

## Instrumentation boundary

Phase 6A adds an optional post-decision observer. It is invoked only after candidate generation, acceptance, best handling, weight update, and cooling. It performs no RNG calls and cannot change search objects. The formal config is otherwise unchanged. A fixed-iteration S/M/L regression verifies exact operator, acceptance, best, convergence, and decoder-count sequences.
