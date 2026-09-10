# Expanded development collection

Decision: **READY_FOR_JOINT_CRITIC_TRAINING**. 72 planned R12 DEVELOPMENT states completed;
6465 deduplicated joint actions. Labels use nine paired CRN replicates,
eight repair trials, two continuation moves, and the frozen V2 primary semantics.
Retained label computation: 1,411,992 decoder calls, 10328.2 seconds.
Native source generation: 6,912 explicit decoder calls; H1 initialization,
source replays, CSG construction, storage and interrupted unpersisted work are additional.

Checks: `{"all_feasible": true, "all_rules": true, "all_states": true, "balanced_coverage": true, "informative_each_fold": true, "positive_opportunities": true}`. Complete per-state/fold
diagnostics and file hashes are under `outputs/ngas_a1/development_v1/`.
All cell replicates and source states stay within the same structural fold.
No training was launched automatically. On READY_FOR_JOINT_CRITIC_TRAINING, verify
the completed artifact contract and execute the preregistered 3-seed, 3-fold fixed-
epoch training. Otherwise diagnose the data gate before training. R13/R14 stay locked.
