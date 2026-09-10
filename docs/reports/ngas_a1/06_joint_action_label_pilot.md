# Joint-action label pilot

Decision: **NGAS_A1_REVISE_LABELS**. 450 joint actions, 1350
paired CRN replicates across six states. R12 DEVELOPMENT only; R13/R14 remain locked.
No training or expanded labeling was launched.

Action = (size, exact target, repair), with the encoded repair executed in every
label. Each first move uses two repair trials and simulated-annealing acceptance;
two subsequent native iterations use the same versioned policy in both branches.
Both branches retain the source state's incumbent. Advantage is
(fallback best - action best) / source makespan. Independent component namespaces
share the same per-step keys between branches. Feasibility, executed actions,
acceptance, immediate improvement, continuation mean/variance and beats-fallback
frequency are stored per action with three paired replicates.

Every size includes critical-sync priority, related variant, matched random control,
local perturbation and structured neighbor targets, crossed with all five repairs.
Exact target collisions are deduplicated. No model exists yet, so neural-priority
sampling is unavailable and no neural prediction is claimed. This short two-trial
pilot policy must not be relabeled as the eventual eight-trial online policy.

Noise-separated pairs require abs(mean paired difference) > max(.001, 2*paired SE).
Gate thresholds were frozen before outcomes in `configs/ngas_a1_label_pilot.json`.
Checks: `{"all_feasible": true, "all_six_states": true, "balanced_coverage_each_state": true, "cost_usable": true, "noise_usable": false, "positive_opportunities": true}`.
Mean cost: 0.262 s per joint action (three paired replicates).
State diagnostics and target/repair rank reversals are in `label_pilot/gate.json`.
Residual interaction RMS is descriptive, not a significance result. The diagonal
S/CF1, M/CF2, L/CF3 subset confounds scale and CF; no subgroup claims are valid.

Repair stays in the architecture regardless of this pilot's measured interaction.
If PASS, the next step is a separately frozen expanded development labeling/training
design with instance-level separation and multiple training seeds. If REVISE_LABELS,
stop before training and diagnose noise, opportunity coverage and policy horizon.
Raw rows and their hash manifest remain immutable and resumable.
