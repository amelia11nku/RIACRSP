# NGAS A1.7A-R R12 exposure audit

R12 is an architecture-development benchmark with complete instance-level exposure to the final frozen C1 fitting process and therefore is not an independent or unseen test set.

## Reconstructed C1 exposure

- Instances: **18 R12 instances**.
- States: **72**.
- Joint-action labels: **6,465**.
- `no_continuation_rollout`: **`true`**.
- Each instance contributes one H1 state and one state from each of `NATIVE16_746101`, `NATIVE16_746102`, and `NATIVE16_746103`.
- Every listed state contributed to the final all-development production C1 fit.

## Reconstructed solver-comparison exposure

- A1.6 preserved invalid-concurrency campaign: 54 R12 runs.
- A1.6R revalidated NGAS campaign: 54 R12 runs.
- Frozen A1.6R comparators: 216 runs across four algorithms.
- A1.6R uses 18 instances and seeds `[746101, 746102, 746103]`.

## Overlap

- C1 final-fit instance overlap with A1.6R: **18/18 = 100.0%**.
- C1 NATIVE16 seed overlap with A1.6R: **3/3 = 100.0%**.
- The instance-content hashes and all 54 instance×seed identities also overlap completely.

A1.6R remains valid for integration, budget accounting, concurrency, feasibility, reproducibility, runtime qualification, and R12 development-benchmark performance. It is not evidence of unseen-instance generalization, a leakage-free held-out test, or an unbiased final comparison.
