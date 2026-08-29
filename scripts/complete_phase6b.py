#!/usr/bin/env python3
"""Validate and write the Phase 6B completion gate."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/phase6b"
def main():
    manifest=pd.read_csv(ROOT/"instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv")
    states=pd.read_parquet(OUT/"trajectory_reservoir/pilot_state_manifest.parquet",columns=["state_id"])
    arms=pd.read_parquet(OUT/"counterfactual/counterfactual_arm_results.parquet",columns=["state_id","repair_seed_group"])
    swaps=pd.read_parquet(OUT/"marginal_target/marginal_swap_results.parquet",columns=["state_id"])
    environment=json.loads((OUT/"environment/environment.json").read_text())
    train_audit=json.loads((OUT/"audit/train_distribution_audit.json").read_text())
    integrity=json.loads((OUT/"audit/counterfactual_integrity_audit.json").read_text())
    recommendation=json.loads((OUT/"diagnostics/phase6c_recommendation.json").read_text())
    repository_audit=json.loads((OUT/"audit/repository_audit.json").read_text())
    validation=environment["final_validation"]
    primary=arms[arms.repair_seed_group==0]
    immutable_fields=("adaptive_weights_immutable","current_candidate_immutable","current_schedule_immutable",
                      "historical_best_immutable","temperature_immutable")
    gates={"PHASE6A_FROZEN":(OUT/"environment/phase6a_freeze_record.json").exists(),
           "RCIAS_CB1_TRAIN_CREATED":len(manifest)==405,"TRAIN_SPLITS_VALIDATED":manifest.training_split.value_counts().to_dict()=={"TRAIN":243,"TRAIN_VALIDATION":81,"TRAIN_INTERNAL_HOLDOUT":81},
           "ALL_TRAIN_SEEDS_DISJOINT_FROM_TEST":train_audit["all_phase6b_seeds_disjoint_from_frozen_sets"],
           "COUNTERFACTUAL_EVALUATOR_IMPLEMENTED":(ROOT/"rcias_clgri/search/counterfactual.py").exists(),
           "COUNTERFACTUAL_EVALUATOR_IMMUTABLE":all(integrity[field] for field in immutable_fields),
           "COUNTERFACTUAL_RNG_ISOLATED":integrity["live_rng_immutable"] and integrity["arm_order_invariant"],
           "PILOT_STATE_RESERVOIR_CREATED":states.state_id.nunique()>=10000,
           "COUNTERFACTUAL_ARMS_EVALUATED":primary.state_id.nunique()==states.state_id.nunique() and primary.groupby("state_id").size().min()>=2,
           "REPAIR_SEED_STABILITY_ANALYZED":(OUT/"summaries/repair_seed_stability.csv").exists(),
           "OPERATION_MARGINAL_PILOT_COMPLETE":swaps.state_id.nunique()>=1000,
           "COUNTERFACTUAL_SAMPLE_BALANCE_ANALYZED":(OUT/"summaries/counterfactual_sample_balance.csv").exists(),
           "COUNTERFACTUAL_PREDICTABILITY_ANALYZED":(OUT/"summaries/target_predictability.csv").exists(),
           "ALL_FACTOR_GROUPS_ANALYZED":(OUT/"summaries/counterfactual_group_summary.csv").exists(),
           "LEAKAGE_AUDIT_PASSED":(OUT/"audit/information_leakage_audit.csv").exists() and not recommendation["FROZEN_TEST_LEAKAGE"],
           "FULL_TEST_SUITE_PASSED":validation["full_test_suite"]=="PASS_128_OF_128",
           "FROZEN_BENCHMARK_CHECKSUMS_UNCHANGED":validation["canonical_regeneration"]=="PASS_130_OF_130" and validation["cb1_checksums"]=="PASS_113_OF_113",
           "PHASE6B_REPORT_COMPLETE":(ROOT/"docs/reports/phase6b_counterfactual_data_pilot_report.md").exists(),
           "REPOSITORY_AUDITED":repository_audit["audit_status"]=="PASS" and not repository_audit["scope"]["historical_tracked_outputs_modified"]}
    names=["train_distribution_factorial_coverage","counterfactual_positive_rate_by_scale_cf","counterfactual_positive_rate_by_ri_ti",
           "arm_improvement_distributions","within_state_best_arm_frequency","related_variants_vs_matched_random","repair_seed_rank_stability",
           "counterfactual_predictability","operation_marginal_swap_effects","bottleneck_proxy_coverage"]
    gates["REQUIRED_FIGURES_PRESENT"]=all((OUT/"figures"/f"Fig{i:02d}_{name}.png").exists() and (OUT/"figures"/f"Fig{i:02d}_{name}.pdf").exists() for i,name in enumerate(names,1))
    gates={name:bool(passed) for name,passed in gates.items()}
    gates["PHASE6B_COMPLETE"]=all(gates.values())
    path=OUT/"audit/completion_gate.json";path.write_text(json.dumps(gates,indent=2,sort_keys=True)+"\n")
    if not gates["PHASE6B_COMPLETE"]:raise RuntimeError(gates)
    print("PHASE6B_COMPLETE = TRUE")
if __name__=="__main__":main()
