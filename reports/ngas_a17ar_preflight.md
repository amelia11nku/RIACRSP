# NGAS A1.7A-R preflight

- Stage-entry commit: `2f1e27bef9e8a935bcfc8d7d263a17a4c418d8b3` on `main`; working tree was clean before A1.7A-R modifications.
- Python: `3.11.15` at `/home/liulei/miniconda3/envs/gnn311/bin/python`.
- PyTorch/CUDA: `2.11.0+cu128` / `12.8`; CUDA available = `true`.
- Frozen production C1: `outputs/ngas_a1/critic_training_rthgt_v2/production/revised_joint_critic.pt`, SHA256 `448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`.
- A1.6R manifest: `outputs/ngas_a1/solver_comparison_a16r_v1/result_manifest.json`, SHA256 `06e937a4726690317d510d43d4582c0aad3ea6bd3d0a1886333fdeafe25a747f`; terminal decision remains `NGAS_A1_6R_PASS_REVALIDATED`.
- R12 authority: `outputs/frozen_2o_baselines/instance_manifest.json`; its 18 IDs and hashes agree with the Phase 6J source manifest.
- R13: `LOCKED_FINAL_EVAL`; R14: `LOCKED_GENERALIZATION_EVAL`. Only manifest metadata and byte hashes were inspected. No solver, model, or performance access occurred.
- RCIAS-CB1-CORE45: `EXTERNAL_BASELINE_ONLY`; it is reserved from training and model selection.
- Dataset registry: `configs/dataset_role_registry.json`, covering 853 instance files.

All mandatory preflight identities are unique and internally consistent. Formal R13/R14 evaluation remains locked.
