# NGAS A1.7A-R CUDA feasibility audit

Protocol revision 3 passed the pre-formal CUDA smoke on an NVIDIA GeForce RTX
4060 Ti with the frozen C1 checkpoint and production `PERSISTENT_FIXED_REFRESH`
semantics.

- Instance: `CB1_TRAIN_S_CF2_RI1_TI2_R03` (`TRAIN`), content SHA256
  `e6837a7979c890b1c80dca4a6f5423281017a0ef71b98037facc0cd7a9d9d9a8`.
- Search budget/elapsed: 8.0 / 8.003835 seconds.
- Throughput: 419 iterations (52.35/s) and 21 critic refreshes (2.62/s).
- Coverage: all five frozen capture fractions; 345–355 joint actions and
  191–199 compact relational nodes per captured state.
- Numerical audit: all critic outputs were finite and every neural prior was
  normalized within the collector tolerance.
- Peak CUDA memory: 41,540,096 bytes allocated and 60,817,408 bytes reserved.
- R13 and R14 remained locked; RCIAS-CB1-CORE45 remained excluded.

The complete local smoke payload is
`outputs/ngas_a1/trajectory_utility_a17ar_v1/smoke/cuda_feasibility.json`, SHA256
`0dc067c6ec660c32edf6f7e76d9851b19696794e4e999e9db4e5aa19f35eb411`.
Its compact tracked audit is `artifacts/ngas_a17ar/cuda_smoke_revision3.json`.
