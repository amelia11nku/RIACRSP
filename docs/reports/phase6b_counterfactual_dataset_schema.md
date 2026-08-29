# Phase 6B Counterfactual Dataset and Reconstruction Contract

## Training distribution

`instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv` identifies 405 training-only instances and their Scale/CF/RI/TI cell, structural replicate, split, generation seed, trajectory seed, state-sampling seed, arm namespace, repair namespace, path, and SHA-256. R01–R03 are TRAIN, R04 is TRAIN_VALIDATION, and R05 is TRAIN_INTERNAL_HOLDOUT.

## State reconstruction

`pilot_state_manifest.parquet` stores one row per distinct pre-action state. To reconstruct:

1. resolve `instance_id` through the training manifest;
2. parse `current_candidate`, a compact JSON object containing operation order and island/W/F assignment layers;
3. construct `Candidate` from those four tuples;
4. call the frozen common `decode_candidate`;
5. verify decoded makespan equals `current_makespan`.

The historical-best candidate is separately reconstructable. Search progress, stage, temperature, weights, and bottleneck proxy are analysis metadata. No future trajectory field is required.

## Counterfactual arms

`counterfactual_arm_results.parquet` contains identity and regime fields, current makespan, target-set identity, deduplication origins, fixed repair details, repair-seed group, eight-trial outcome, signed improvement labels, runtime, within-state rank, best/top-3 flags, regret, and percentile. Positive `absolute_improvement` always means improvement.

`counterfactual_target_rows.parquet` is the normalized state × primary-arm × operation table. It contains `is_targeted` and only pre-action schedule features. Schedule objects are not duplicated.

`marginal_swap_results.parquet` contains paired one-for-one swaps of a predefined related reference set. Every swap uses the reference arm's repair seed. `marginal_swap_gain` is the conditional change relative to that reference, not a universal operation contribution.

The exact model-input boundary is machine-readable in `outputs/phase6b/audit/information_leakage_audit.csv`.
