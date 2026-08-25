# Phase 2E — Behavior Cloning Validation

## Demonstrations

Both frozen Tiny instances were replayed from four sources: exact, H1, H2,
and H3. Each pre-action record contains the typed graph state, all hard masks,
the selected O/M/W/F action, and all four levels of candidate features. The
complete eight-episode dataset is stored at
`outputs/bc_validation/run_1/demonstrations.json`.

Because an optimal expert was available, training selected the six-step
`tiny_01` exact episode rather than a heuristic episode.

## Training result

The deterministic validation run used a 64-dimensional, four-head, two-layer
RT-HGT to make the Tiny overfit check fast; the project model default remains
128 dimensions, four heads, and two layers. With seed 2026, training stopped
after 26 epochs:

- total mean cross-entropy: **0.11791679**;
- operation loss: **0.11682138**;
- island loss: **0.00109541**;
- operation/island/W/F reproduction: **100% / 100% / 100% / 100%**;
- joint expert action reproduction: **100%**.

The nonzero operation-loss floor is consistent with ID-invariant symmetric
operation candidates in the Tiny state. The policy retains relabeling
equivariance and nevertheless selects the entire expert sequence exactly.

## Independent rollout validation

The greedy policy was rolled out through the production decoder and then
checked independently. It reproduced all six exact actions, was 100% feasible,
and obtained makespan **157.0**, equal to the proven exact optimum.

Formal metrics and diagnostic figures are under
`outputs/bc_validation/run_1/`. No PPO, multiobjective preference policy,
critical synchronization graph, or neural destroy/repair was used.

Full regression before the final named-test split: **45 passed**. The final
repository regression contains 46 tests.

`BC_VALIDATED = TRUE`
