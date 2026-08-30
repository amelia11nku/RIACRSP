# Phase 6C NI Dataset Contract

Contract version: `phase6c-v1`. This document freezes the data interface for a future Phase 6D CSG definition; it does not define graph topology, tensors, or a neural model.

## 1. Units of supervision

The state unit is one distinct feasible pre-action ALNS state. Its identity includes the frozen training instance, deterministic trajectory run, and iteration. Each trajectory uses the frozen 250-iteration budget, so concurrent wall-clock load cannot alter state identity or content. Repair-seed repetitions never create additional states.

The action unit is one deduplicated fixed-size target set in a state. Several outcome-blind generation rules may map to the same action; the action is evaluated once per repair seed while all origin rules remain metadata.

The conditional operation-pair unit is one reference/perturbation comparison. Its removed and added operations are meaningful only for that state, canonical related reference set, fixed destroy count, `transport_aware` repair, and frozen three-seed aggregation.

## 2. Split semantics

`TRAIN`, `TRAIN_VALIDATION`, and `TRAIN_INTERNAL_HOLDOUT` contain 60,000, 20,000, and 20,000 states respectively. They retain the frozen R01–R03/R04/R05 instance partition. Instances and states cannot cross splits. CB1-DEV, CB1-Core, CB1-Sensitivity, and Legacy-130 are forbidden.

TRAIN_VALIDATION may be used for diagnostic selection. TRAIN_INTERNAL_HOLDOUT is evaluation-only and must never be used for tuning.

## 3. Legal model information

Legal pre-action inputs are grouped as follows:

- instance-static: Scale, CF, RI, TI, processing/configuration requirements, eligibility and precedence counts;
- operation-static: product membership and eligible-island count;
- schedule-dynamic: current makespan, start/end, slack, critical proxies and realized positions;
- island/resource-dynamic: assigned island, relative island load and realized resource-chain positions;
- reconfiguration-dynamic: preceding, following and local reconfiguration contributions;
- W-logistics dynamic: W waiting/delay and realized W-chain position;
- F-logistics dynamic: F waiting/delay and realized F-chain position;
- search-progress: normalized progress, five-stage bucket and natural bottleneck proxy;
- action description: target membership and outcome-blind origin-rule metadata.

The authoritative field-by-field classification is generated from `rcias_clgri.data.phase6c_contract.TABLE_FIELDS` into `outputs/phase6c/audit/leakage_audit.csv`.

## 4. Forbidden information

Counterfactual makespans, improvements, probabilities, ranks, top-k flags, regrets and pair preferences are labels only. Repair seeds, identifiers, trajectory operators, elapsed time, temperature, adaptive weights and historical-best diagnostics are not model inputs. Future best, future-window outcomes, final trajectory makespan, after-state candidates and post-action acceptance are forbidden future information.

## 5. Three-seed aggregation

Every unique state–target-set action has exactly three deterministic independent repair seeds. All raw outcomes are retained. The globally frozen ranking score is mean relative improvement. The dataset also stores median improvement, population standard deviation, improvement probability, and positivity under at least 1/3, 2/3 and 3/3 seeds. The robust positivity convention is at least 2/3 positive seeds.

## 6. Labels and ordering

Improvement uses `current_makespan - counterfactual_makespan`; positive is better. Rank 1 is the greatest mean relative improvement, with target-set ID as the deterministic tie-breaker. Rank percentile decreases from 1. Top-1/3/5 follow this ordering. Regret is best mean relative improvement minus action mean; robust regret uses the best median minus action median.

Supported future tasks are set-level improvement classification, within-state ranking, top-k retrieval, secondary operator selection, and conditional operation-pair preference. Direct node-value or universal scalar operation-importance regression is unsupported.

## 7. Reconstruction API

`rcias_clgri.data.phase6c.reconstruct_state` loads only the frozen training instance referenced by a state record, validates the serialized candidate checksum, decodes through the common deterministic decoder, and checks the stored current makespan. It returns the instance, candidate, current feasible decoded schedule, makespan, normalized progress and stage without consulting any later trajectory row.

## 8. Shard format

Each instance-disjoint Parquet shard contains:

- `states.parquet`;
- `repair_seed_outcomes.parquet`;
- `target_set_aggregates.parquet`;
- `target_membership.parquet`;
- `operation_pairs.parquet`;
- `status.json` written last as the completion marker.

The central shard manifest records split, structural cell, counts, checksum and status. Writes use same-directory partial files followed by atomic replacement. A shard is resumable only when every declared file exists and matches its checksum; otherwise it is atomically regenerated.

## 9. Versioning and freeze rules

State IDs, split membership, target-set IDs, repair-seed namespaces, aggregation, production config, shard checksums and label-generation version are frozen in `dataset_freeze_record.json`. A substantive label or arm-contract change requires a new dataset version. Silent regeneration under `phase6c-v1` is prohibited.
