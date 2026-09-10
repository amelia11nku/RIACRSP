# A1.2-v2 label revision results

Decision: **NGAS_A1_LABEL_PILOT_PASS**, determined only by preregistered T8_R9.
Control T2_R9 is diagnostic; there is no rescue or replacement of the failed V1 gate.
Completed 900 action/variant rows and 8,100 paired replicates on the same six states.
Actual label work: 123,120 decoder calls, 931.83 compute seconds.
Equivalent uncached work: 243,000 decoder calls.
Actual costs count fallback cache creation once, amortized equally over each state's
75 actions for the per-action cost gate. Source replay and file I/O are additional
worker overhead, not silently counted as decoder work.

Primary checks: `{"all_feasible": true, "all_six_states": true, "balanced_coverage_each_state": true, "cost_usable": true, "noise_usable": true, "positive_opportunities": true}`.
The .001 material gap, twice paired SE, >=10% informative pairs in at least three
states, feasibility, coverage, positive opportunities and 30-second cost cap remain.
Nine independent repeats are derived from three canonical root seeds; V1 samples
are not pooled. Both variants share the same new per-step CRN keys and joint actions.
Encoded repair identity is preserved in every label and checked before publishing.

Detailed first-three/all-nine, root-block, matched T8/T2 and repair-interaction
diagnostics are in `outputs/ngas_a1/label_pilot_v2/gate.json`. These reused-state
development diagnostics are not untouched-test evidence or subgroup performance.
V1 files and frozen comparator hashes were reverified after completion.

No model training or expanded labels were launched. On PASS, preregister expanded
development labeling and instance-separated critic training. Otherwise remain in
A1.2 label revision. R13/R14 remain locked; Gurobi was not run.
