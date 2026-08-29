# Phase 6B Counterfactual Interface Audit

The audited ALNS remains `rcias_clgri/search/alns.py`. A search state is represented by an immutable `Candidate` plus a `DecodedCandidate`; the schedule has a deep-copy `clone()` method, while the common decoder constructs a fresh environment and schedule for every candidate.

Relevant properties:

- `_destroy` reads the instance and decoded schedule and returns a new target set. It mutates neither input. Its stochastic branches accept an explicit RNG.
- `_neighbor` constructs new tuple-backed decision layers and does not mutate the base candidate. It accepts an explicit RNG.
- `decode_candidate` creates a fresh `RCIASConstructionEnv`, returns a new decoded schedule, and runs the independent checker.
- Adaptive weights, SA acceptance, temperature, historical best, and the live ALNS RNG exist only inside `solve_alns`; the counterfactual API does not receive or update them.
- A current state is reconstructable from the instance path and four candidate layers. Historical best, temperature, progress, and weights are retained as analysis metadata but are not needed for decoding.

Phase 6B therefore implements counterfactual evaluation as a separate pure composition of `_neighbor` and `decode_candidate`, with a local `random.Random(repair_seed)`. The primary API rejects repair operators other than `transport_aware`. Integrity tests fingerprint the input candidate/schedule, global RNG, and subsequent fixed-iteration ALNS trajectory.
