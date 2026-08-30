# Phase 6C Scaled Counterfactual Dataset Report

## 1. Executive conclusion

Phase 6C generated and froze the leakage-safe, instance-disjoint scaled dataset. Set-level labels and three-seed aggregation were assessed before any future graph definition. `PHASE6D_RECOMMENDATION = PROCEED_TO_CSG_DEFINITION`.

## 2. Frozen Phase 6B evidence

Phase 6B remains byte-frozen under `outputs/phase6c/environment/phase6b_freeze_record.json`. Its 13,500-state result selected `SCALE_WITH_REVISED_ARM_DESIGN`; no Phase 6A/6B search or evidence artifact was modified.

## 3. Dataset objective

The dataset supports set-level improvement classification, within-state ranking, top-k retrieval, secondary operator selection and conditional operation-pair preference. It deliberately excludes direct scalar operation importance, destroy-size learning and repair selection.

## 4. State reservoir

Two deterministic 250-iteration frozen ALNS trajectories were run for every one of 405 training-only instances. Selection used only structural cell, five-stage availability, natural bottleneck proxy and deterministic priority; outcomes were never read. Exactly 100,000 distinct pre-action states were retained from the 202,500-state initial reservoir.

## 5. Structural/split coverage

- TRAIN: states=60000, arms=1417585, positive=0.3558
- TRAIN_INTERNAL_HOLDOUT: states=20000, arms=472452, positive=0.3713
- TRAIN_VALIDATION: states=20000, arms=472685, positive=0.3199

Every split contains all 81 Scale × CF × RI × TI cells. Cell state counts range from 246 to 741; no instance or state crosses splits.

## 6. Revised arm design

Each state requested 24 outcome-blind rules: seven original controls, four related variants, three matched random controls, four local perturbations and six structured near neighbors. Exact target sets were deduplicated without discarding origin metadata.

## 7. Three-seed aggregation

Every unique state–target action has exactly three independent deterministic `transport_aware` repair outcomes. Single-seed versus aggregate rank Spearman is 0.8419, versus Phase 6B's 0.6436; sign agreement is 0.9030. `REPAIR_NOISE_REDUCED_BY_AGGREGATION = TRUE`.

## 8. Dataset statistics

The frozen dataset contains 2,362,722 unique state–target actions. Aggregate positive-arm fraction is 0.3517; robust 2/3 positive fraction is 0.3531; states with at least one positive aggregate arm are 0.7842.

## 9. Arm-family contribution

- LOCAL_PERTURBATION: positive=0.3767, top1=0.0268, mean regret=0.0543
- MATCHED_RANDOM: positive=0.3276, top1=0.0346, mean regret=0.0598
- ORIGINAL_OPERATOR: positive=0.2667, top1=0.0338, mean regret=0.0679
- RELATED_VARIANT: positive=0.4208, top1=0.0732, mean regret=0.0493
- STRUCTURED_NEAR_NEIGHBOR: positive=0.3939, top1=0.0478, mean regret=0.0525

Low-performing controls remain in the dataset by design.

## 10. Label stability

Top-1 agreement is 0.4943, top-3 overlap is 0.6263, and conditional pair preference stability is 0.7643. Rankings use mean relative improvement globally.

## 11. Operation-pair analysis

The dataset retains 400,000 conditional comparisons. `OPERATION_PAIR_SIGNAL = WEAK`. These labels remain conditional on state, reference set, fixed size and repair; `DIRECT_OPERATION_SCALAR_LABEL_READY = FALSE`.

## 12. Diagnostic predictability

On untouched internal holdout, the leakage-safe set diagnostic obtains ROC-AUC 0.7685, PR-AUC 0.6672, pairwise accuracy 0.5815, Spearman 0.2297 and NDCG 0.9030. `SET_LEVEL_TARGET_SIGNAL_READY = TRUE` and `OPERATOR_SELECTION_SIGNAL_READY = TRUE`.

## 13. Structural-regime generalization

Diagnostics were reported separately for S/M/L, CF1–3, RI1–3 and TI1–3. The weakest internal-holdout ROC regime is TI_level=TI3 at 0.7124. All regimes satisfy the pre-frozen minimum positive-signal gate: True.

## 14. Leakage/integrity audit

Every stored field has exactly one classification. Future trajectory outcomes, after states and acceptance are forbidden. All state IDs, split boundaries, fixed destroy counts, three-seed mappings, repairs and shard checksums passed; frozen evaluation suites are absent.

## 15. Compute/storage profile

The final dataset occupies 1.53 GiB across 405 instance shards. Summed shard runtime is 274.97 CPU-hours. All raw outcomes and reconstruction records remain sharded and resumable.

## 16. Dataset freeze

`DATASET_FROZEN = TRUE`. Freeze hash: `695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437`. State IDs, split membership, target IDs, three-seed aggregation, label version and shard checksums are immutable under `phase6c-v1`.

## 17. CSG information requirements

Future CSG definition must represent slack, W/F delay, island relative load, local reconfiguration, eligibility, precedence neighborhoods, realized island/W/F chains, synchronization and search progress. This is an information requirement only; no topology or tensorizer was implemented.

## 18. NI dataset contract

`docs/reports/phase6c_ni_dataset_contract.md` is the authoritative interface for state, action, conditional pair, labels, legal inputs, reconstruction, shards and versioning.

## 19. Limitations

Counterfactual quality remains tied to fixed 15% destroy size, `transport_aware` repair and eight candidate trials. Diagnostic linear models establish signal, not the value of a future CSG. Operation-pair evidence is conditional and cannot justify node-scalar supervision.

## 20. Phase 6D recommendation

`PHASE6D_RECOMMENDATION = PROCEED_TO_CSG_DEFINITION`. The decision follows the pre-frozen readiness thresholds; this phase does not implement Phase 6D.

## 21. Reproducibility checklist

`SCALED_DATASET_COMPLETE = TRUE`  
`DISTINCT_STATE_COUNT = 100000`  
`TRAIN_STATE_COUNT = 60000`  
`VALIDATION_STATE_COUNT = 20000`  
`INTERNAL_HOLDOUT_STATE_COUNT = 20000`  
`THREE_REPAIR_SEED_AGGREGATION_COMPLETE = TRUE`  
`REPAIR_NOISE_REDUCED_BY_AGGREGATION = TRUE`  
`SET_LEVEL_TARGET_SIGNAL_READY = TRUE`  
`OPERATOR_SELECTION_SIGNAL_READY = TRUE`  
`OPERATION_PAIR_SIGNAL = WEAK`  
`DIRECT_OPERATION_SCALAR_LABEL_READY = FALSE`  
`DESTROY_SIZE_REMAINS_DEFERRED = TRUE`  
`REPAIR_SELECTION_REMAINS_DEFERRED = TRUE`  
`DATASET_FROZEN = TRUE`  
`PHASE6D_RECOMMENDATION = PROCEED_TO_CSG_DEFINITION`
