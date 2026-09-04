```json
{
  "artifact_hashes": {
    "artifact_freeze": "1fad76dddbd4b6c2286e8b00bac2719191933ae3120babc730fa8b1fb3fd24b0",
    "final_decision": "400c3d4c4cb4759463dcfd3d46fbaeaeef0758b7c24ad64357f32ef5074ca235",
    "r11_forced_diagnostics": "445010850fb0cb68fe234ce5d7be4de7a100eae611ad4b80dba6afdbb4951947",
    "r11_result_manifest": "48341eb15b4f467195094058b743e389c67bfce7bc367328e9307431bb3847a8",
    "selected_artifact": "dc59abbe34be21480c8f855e2bd62e5253fdf3f15f947c94bc124c5cbd4ec89a"
  },
  "chosen_model": {
    "candidate_id": "U2_MIXED_OLD_NEW",
    "ensemble_rule": "ARITHMETIC_MEAN_THREE_TRAINING_SEEDS",
    "family": "U2",
    "r10_mean_training_seed_spearman": 0.1847803992648386,
    "r10_training_seed_sd": 0.003683568135939043,
    "r10_worst_training_seed_spearman": 0.18102275827458605,
    "training_seeds": [
      686101,
      686102,
      686103
    ]
  },
  "feasibility_rate": 1.0,
  "gurobi_executed": false,
  "holdout_access_integrity": {
    "completion_audit": "PASS",
    "no_r11_retuning": true,
    "r11_accessed_once": true,
    "split_audit": "PASS"
  },
  "intervention_reliability": {
    "coverage_by_scale": {
      "L": 0.02666666666666667,
      "M": 0.0033333333333333335,
      "S": 0.0
    },
    "fallback_coverage_by_scale": {
      "L": 0.9733333333333334,
      "M": 0.9966666666666667,
      "S": 1.0
    },
    "selected_actions": 9,
    "selected_actions_by_scale": {
      "L": 8,
      "M": 1,
      "S": 0
    },
    "selected_immediate_utility": -0.003656702664162567,
    "selected_lift_over_fallback": -0.003029308969373459
  },
  "phase_decision": "MODEL_REVISION",
  "probability": {
    "brier_score": 0.06702449316052649,
    "expected_calibration_error": 0.03998056522740875,
    "mean_predicted_probability": 0.035248502997208546,
    "negative_log_likelihood": 0.2661383557242695,
    "realized_positive_fraction": 0.07277777777777777
  },
  "r11_instance_count": 18,
  "r11_iterative_seeds_per_method": 5,
  "r11_runs": 288,
  "search_efficiency": {
    "decoder_reduction_vs_alns": 0.5159352510734373,
    "method_summary": [
      {
        "mean_decoder_evals": 38032.03333333333,
        "mean_of_instance_final": 3055.5444444444447,
        "mean_runtime": 245.11521481093234,
        "median_evals_to_best": 20629.0,
        "median_time_to_best": 108.58022964101474,
        "method": "ALNS"
      },
      {
        "mean_decoder_evals": 18711.433333333334,
        "mean_of_instance_final": 3026.633333333333,
        "mean_runtime": 245.1262571601798,
        "median_evals_to_best": 8237.0,
        "median_time_to_best": 101.95465296499606,
        "method": "PHASE6H_CSGNI"
      },
      {
        "mean_decoder_evals": 18409.966666666664,
        "mean_of_instance_final": 3062.222222222222,
        "mean_runtime": 245.1281709307708,
        "median_evals_to_best": 12125.0,
        "median_time_to_best": 125.54713074850224,
        "method": "PHASE6I_MR_CSGNI"
      }
    ],
    "normalized_gap_auc_improvement_vs_alns": -0.3247633241825188,
    "runtime_components": [
      {
        "calibration_gate_seconds": 0.0,
        "candidate_proposal_seconds": 0.0,
        "csg_seconds": 0.0,
        "decoder_seconds": 242.61369009749296,
        "method": "ALNS",
        "neural_seconds": 0.0,
        "repair_seconds": 1.2154519007373488
      },
      {
        "calibration_gate_seconds": 0.0,
        "candidate_proposal_seconds": 0.0,
        "csg_seconds": 0.0,
        "decoder_seconds": 0.0,
        "method": "H1",
        "neural_seconds": 0.0,
        "repair_seconds": 0.0
      },
      {
        "calibration_gate_seconds": 0.3101470538902099,
        "candidate_proposal_seconds": 5.533745468894348,
        "csg_seconds": 37.13030709171661,
        "decoder_seconds": 140.2476361605075,
        "method": "PHASE6H_CSGNI",
        "neural_seconds": 49.573337152158395,
        "repair_seconds": 0.7638210928435001
      },
      {
        "calibration_gate_seconds": 0.2349801008379371,
        "candidate_proposal_seconds": 5.44061687624065,
        "csg_seconds": 39.590374321059265,
        "decoder_seconds": 137.83208524423972,
        "method": "PHASE6I_MR_CSGNI",
        "neural_seconds": 49.63802028928848,
        "repair_seconds": 0.6923343689425464
      }
    ],
    "target_summary": [
      {
        "hit_rate": 0.9,
        "hits": 81,
        "median_conditional_decoder_evaluations": 4289.0,
        "median_conditional_time_seconds": 31.08050800700221,
        "method": "ALNS",
        "right_censored_runs": 9,
        "runs": 90,
        "target_gap": "5%"
      },
      {
        "hit_rate": 0.5555555555555556,
        "hits": 50,
        "median_conditional_decoder_evaluations": 14189.0,
        "median_conditional_time_seconds": 80.56798821651682,
        "method": "ALNS",
        "right_censored_runs": 40,
        "runs": 90,
        "target_gap": "2%"
      },
      {
        "hit_rate": 0.4,
        "hits": 36,
        "median_conditional_decoder_evaluations": 20561.0,
        "median_conditional_time_seconds": 87.55068766950353,
        "method": "ALNS",
        "right_censored_runs": 54,
        "runs": 90,
        "target_gap": "1%"
      },
      {
        "hit_rate": 0.35555555555555557,
        "hits": 32,
        "median_conditional_decoder_evaluations": 24765.0,
        "median_conditional_time_seconds": 100.44266814200272,
        "method": "ALNS",
        "right_censored_runs": 58,
        "runs": 90,
        "target_gap": "0.5%"
      },
      {
        "hit_rate": 0.9333333333333333,
        "hits": 84,
        "median_conditional_decoder_evaluations": 249.0,
        "median_conditional_time_seconds": 4.153456925014325,
        "method": "PHASE6H_CSGNI",
        "right_censored_runs": 6,
        "runs": 90,
        "target_gap": "5%"
      },
      {
        "hit_rate": 0.7333333333333333,
        "hits": 66,
        "median_conditional_decoder_evaluations": 3545.0,
        "median_conditional_time_seconds": 42.23006700049882,
        "method": "PHASE6H_CSGNI",
        "right_censored_runs": 24,
        "runs": 90,
        "target_gap": "2%"
      },
      {
        "hit_rate": 0.5888888888888889,
        "hits": 53,
        "median_conditional_decoder_evaluations": 6537.0,
        "median_conditional_time_seconds": 71.93217972599814,
        "method": "PHASE6H_CSGNI",
        "right_censored_runs": 37,
        "runs": 90,
        "target_gap": "1%"
      },
      {
        "hit_rate": 0.5222222222222223,
        "hits": 47,
        "median_conditional_decoder_evaluations": 7745.0,
        "median_conditional_time_seconds": 79.01313505800135,
        "method": "PHASE6H_CSGNI",
        "right_censored_runs": 43,
        "runs": 90,
        "target_gap": "0.5%"
      },
      {
        "hit_rate": 0.8666666666666667,
        "hits": 78,
        "median_conditional_decoder_evaluations": 4341.0,
        "median_conditional_time_seconds": 59.573712783010706,
        "method": "PHASE6I_MR_CSGNI",
        "right_censored_runs": 12,
        "runs": 90,
        "target_gap": "5%"
      },
      {
        "hit_rate": 0.4777777777777778,
        "hits": 43,
        "median_conditional_decoder_evaluations": 6609.0,
        "median_conditional_time_seconds": 82.48851397600083,
        "method": "PHASE6I_MR_CSGNI",
        "right_censored_runs": 47,
        "runs": 90,
        "target_gap": "2%"
      },
      {
        "hit_rate": 0.3333333333333333,
        "hits": 30,
        "median_conditional_decoder_evaluations": 12473.0,
        "median_conditional_time_seconds": 135.39690849700128,
        "method": "PHASE6I_MR_CSGNI",
        "right_censored_runs": 60,
        "runs": 90,
        "target_gap": "1%"
      },
      {
        "hit_rate": 0.2777777777777778,
        "hits": 25,
        "median_conditional_decoder_evaluations": 12553.0,
        "median_conditional_time_seconds": 128.71214462502394,
        "method": "PHASE6I_MR_CSGNI",
        "right_censored_runs": 65,
        "runs": 90,
        "target_gap": "0.5%"
      }
    ]
  },
  "solver_quality": {
    "bootstrap_95_percent": {
      "high": 0.0013383651779790237,
      "low": -0.011517042734702367
    },
    "improvement_vs_alns": -0.005023572933870545,
    "relative_worse_than_alns_by_scale": {
      "L": -0.0030042883625625984,
      "M": 0.002243531262028722,
      "S": 0.015831475902145512
    },
    "wilcoxon": {
      "alternative": "revised improvement greater than zero",
      "nonzero_pairs": 18,
      "p_value": 0.9406471252441406,
      "paired_unit": "18 instance means over five matched seeds",
      "statistic": 50.0,
      "zero_handling": "discard exact-zero paired differences before exact test"
    },
    "wins_ties_losses_vs_alns": {
      "losses": 12,
      "ties": 0,
      "wins": 6
    },
    "wins_ties_losses_vs_phase6h": {
      "losses": 14,
      "ties": 1,
      "wins": 3
    }
  },
  "utility_ranking": {
    "kendall_overall": 0.109065639101871,
    "ndcg_at_1": 0.6708333333333334,
    "ndcg_at_2": 0.731118622217165,
    "pairwise_accuracy": 0.5546296296296296,
    "spearman_by_scale": {
      "L": 0.14073879007118595,
      "M": 0.15425243191581747,
      "S": 0.08905409255338946
    },
    "spearman_overall": 0.12801510484679765,
    "top1_agreement": 0.32555555555555554
  },
  "v1_frozen": false
}
```

# Phase 6I-MR live utility revision report

## 1. Final decision

The immutable R11 gate returns **`MODEL_REVISION`**. Correctness and solver
non-inferiority remain intact, but the selected-action utility and support-aware
coverage gates fail. The selected artifact is therefore not frozen as CSG-NI
v1, and Phase 6I-MR does not authorize Core/Sensitivity/Legacy replacement or
the v2 architecture backlog.

Failed mandatory checks: `selected_utility_positive_and_above_fallback, support_aware_coverage`.

## 2. Data and leakage boundary

R09, R10 and R11 contain 18 disjoint instances each, with two instances in
every Scale × CF cell. The split audit reports unique hashes, no historical
ID/hash/seed overlap, R11 content withheld before artifact freeze, and no R10
refit. R11 was opened once after freezing artifact `dc59abbe34be21480c8f855e2bd62e5253fdf3f15f947c94bc124c5cbd4ec89a`.
The final matrix contains 18 H1 runs and 90 runs for each iterative method.
All 288 action sequences replay to their recorded makespans, all schedules are
feasible, all traces are monotone, and all individual live/forced-log hashes
match.

## 3. Why Phase 6H failed

The R09 pilot localized the dominant problems to gate-selection bias (40/54,
74.1%), within-state inversion (37/54, 68.5%), candidate-source bias (23/54,
42.6%), and cross-state miscalibration/sign error (22/54, 40.7%). Inversions
were present at every scale and increased from 61.1% early to 72.2% in middle
and late search. This is not explained by low-support extrapolation alone.

The continuation diagnostic also found a target mismatch: immediate utility
and 12-iteration continuation value had median within-state Spearman
-0.738 and top-1 agreement
7.41%, activating `TARGET_MISMATCH`.

## 4. Candidate truncation and representation

The audited bank averaged 23.33 unique targets.
The true full-bank best was absent from broad-4 in
66.7% of states and
from top-8 in 33.3%.
Thus broad forced labeling has material truncation bias, but the frozen protocol
did not permit candidate-bank redesign. The frozen-embedding probe achieved
only 0.549 pairwise accuracy; U3 was activated under the preregistered underfit
rule, but R10 ultimately selected U2 Mixed Old/New.

## 5. Selected model and training-seed stability

R10 selected `U2_MIXED_OLD_NEW`: a three-seed arithmetic ensemble
with training seeds `[686101, 686102, 686103]`.
Its seed-level mean/worst Spearman was
0.185/
0.181, SD
0.0037. The one-time R10 solver
translation remained within the preregistered non-inferiority envelope
(0.85% worse than ALNS), but
did not show directional improvement.

## 6. R11 ranking and calibration

R11 utility Spearman is 0.128 overall
(S 0.089, M
0.154, L
0.141); Kendall is
0.109, NDCG@1/2 is
0.671/
0.731. Pairwise accuracy and NDCG@1
improve over U0 by 0.014 and
0.021, respectively. Ranking therefore
improves modestly and stays non-negative by scale, but remains below the
preferred 0.20 Spearman target.

Probability reliability passes: ECE
0.040, Brier
0.067, and NLL
0.266.

## 7. Cross-state intervention failure

Only 9/900 forced diagnostic states pass the
frozen gate: coverage is S 0.00%, M
0.33%, and L
2.67%. Their realized immediate utility is
-0.37%, with lift over fallback
-0.30%. Both are negative. Moreover,
predicted-abstained states still contain mean best-forced lift of 3.02–3.64%,
and their grouped-bootstrap upper bounds are about 4.1–4.5%, so low coverage
cannot be justified as safe abstention. This is the decisive Phase 6I-MR
failure: weak cross-state gating, not feasibility.

## 8. Final solver quality

Phase 6I-MR is 0.50% worse
than ALNS on the mean of 18 paired instance means, while remaining inside the
+1% non-inferiority margin. It records
6/0/12
wins/ties/losses. The paired bootstrap interval for improvement is
[-1.15%,
0.13%], and the exact
one-sided Wilcoxon statistic is 50
(`p=0.940647`). Small instances are
1.58% worse than ALNS on average;
there is no catastrophic subgroup collapse under the frozen numerical rule.

Relative to the Phase 6H reference, Phase 6I-MR records
3/1/14
wins/ties/losses across the 18 instance means. Ranking improvements therefore
did not translate into better final search behavior.

## 9. Search efficiency and runtime fairness

Decoder evaluations fall by 51.59%
versus ALNS, satisfying the disjunctive efficiency gate without a final-quality
collapse. However, normalized-gap AUC is
32.48% worse, and
median time to final best is 125.55 s versus 108.58 s for ALNS and 101.95 s for
Phase 6H. Raw wall-clock results apply only to the recorded RTX 4060 Ti / i7-14700
single-worker platform; checkpoint loading is reported separately.

| method | mean_of_instance_final | mean_runtime | mean_decoder_evals | median_time_to_best | median_evals_to_best |
| --- | --- | --- | --- | --- | --- |
| ALNS | 3055.544 | 245.115 | 38032.033 | 108.580 | 20629.000 |
| PHASE6H_CSGNI | 3026.633 | 245.126 | 18711.433 | 101.955 | 8237.000 |
| PHASE6I_MR_CSGNI | 3062.222 | 245.128 | 18409.967 | 125.547 | 12125.000 |

## 10. Time-to-target evidence

Hit times and decoder counts are conditional on reaching the target; misses are
right-censored and retained in the hit-rate column.

| method | target_gap | runs | hits | hit_rate | median_conditional_time_seconds | median_conditional_decoder_evaluations | right_censored_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALNS | 5% | 90 | 81 | 90.0% | 31.1 | 4289 | 9 |
| ALNS | 2% | 90 | 50 | 55.6% | 80.6 | 14189 | 40 |
| ALNS | 1% | 90 | 36 | 40.0% | 87.6 | 20561 | 54 |
| ALNS | 0.5% | 90 | 32 | 35.6% | 100.4 | 24765 | 58 |
| PHASE6H_CSGNI | 5% | 90 | 84 | 93.3% | 4.2 | 249 | 6 |
| PHASE6H_CSGNI | 2% | 90 | 66 | 73.3% | 42.2 | 3545 | 24 |
| PHASE6H_CSGNI | 1% | 90 | 53 | 58.9% | 71.9 | 6537 | 37 |
| PHASE6H_CSGNI | 0.5% | 90 | 47 | 52.2% | 79.0 | 7745 | 43 |
| PHASE6I_MR_CSGNI | 5% | 90 | 78 | 86.7% | 59.6 | 4341 | 12 |
| PHASE6I_MR_CSGNI | 2% | 90 | 43 | 47.8% | 82.5 | 6609 | 47 |
| PHASE6I_MR_CSGNI | 1% | 90 | 30 | 33.3% | 135.4 | 12473 | 60 |
| PHASE6I_MR_CSGNI | 0.5% | 90 | 25 | 27.8% | 128.7 | 12553 | 65 |

## 11. Evidence boundary and next phase

No Gurobi work was run. R11 may not be revisited for tuning, threshold changes,
model selection, or rescue analysis. The failed artifact remains a frozen
experimental candidate, not CSG-NI v1. Any further model revision requires a
new preregistered development/selection/holdout boundary. Until a future fresh
holdout yields `PROCEED_FREEZE_V1`, the v2 mechanisms in the Phase 6I-MR backlog
remain design-only and the manuscript's Phase 6H CSG-NI results remain the
operative Core45 evidence.

## 12. Reproducibility artifacts

- Completion audit: `outputs/phase6i_mr/r11_validation/completion_integrity_audit.json`
- Summary compatibility audit: `outputs/phase6i_mr/r11_validation/summary_compatibility_audit.json`
- Gate decision: `outputs/phase6i_mr/r11_validation/final_decision.json`
- Ranking/calibration: `outputs/phase6i_mr/r11_validation/r11_ranking_calibration.json`
- Anytime/runtime: `outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json`
- Pairwise quality: `outputs/phase6i_mr/final/r11_pairwise_quality.csv`
- Target summary: `outputs/phase6i_mr/final/r11_target_summary.csv`
- Machine-readable status: `outputs/phase6i_mr/final/final_status.json`
