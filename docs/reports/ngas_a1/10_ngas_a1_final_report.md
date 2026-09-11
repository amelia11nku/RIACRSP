# NGAS-A1 stage delivery and continuation boundary

The current terminal state is `NGAS_A15R_RUNTIME_PASS`. A1.4 completed and passed
its completion audit. A1.5 completed under its frozen boundary and failed the latency
gate with `NGAS_A1_REVISE_RUNTIME`; those historical outputs remain unchanged.
A1.5R then revised only the runtime architecture and passed its separately frozen gate.

The actual neural NGAS search and the latency harness now call the same
`ProductionRefreshRuntime`. The frozen C1 RT-HGT checkpoint, complete three-size
candidate bank, target provenance/deduplication/order, five repairs, action identity,
prior, RNG namespaces, and search policy were preserved. The runtime uses compact
indexed CSG/event/critical arrays, shared per-refresh features, reusable workspaces,
and direct tensor views.

Formal complete-refresh p90 values are S 11.579 ms / M 19.269 ms / L 24.449 ms / L_MAX 26.512 ms; pooled p90 is
`24.938 ms`, all within the frozen 30 ms
criterion. A 40-state oracle equivalence matrix, deterministic real-search replay,
80-distinct-state transition trace, zero workspace resizes, and 492 passing tests
complete the gate evidence. See `11_a15r_runtime_revision.md` and
`outputs/ngas_a1/runtime_revision_v1/`.

A1.6 is eligible for a new explicit frozen stage but was not started. R13 and R14
remain locked. Gurobi was not run. No remote push was performed.
