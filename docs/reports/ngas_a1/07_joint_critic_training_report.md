# A1.3 joint critic training report

Status: **READY_FOR_A1_4_SEARCH_INTEGRATION**. The frozen grouped OOF experiment completed three seeds
and three held-instance folds. 3/3 seeds passed the preregistered gate;
the gate required at least 2. R12 remains development evidence.

The OOF gate and all per-state, fold, scale and CF diagnostics are in
`outputs/ngas_a1/critic_training_gpu_v1/oof_gate.json`. Fixed final epoch checkpoints
were used; held-fold labels did not select epochs or checkpoints. The single
production critic was fit only after the OOF gate passed.

R13/R14 stayed locked. No Gurobi or frozen comparator run was executed.
