# RCIAS-CLGRI

Reproducible research implementation for the **Reconfigurable
Capability-based Island Assembly Scheduling** problem. The repository provides
a frozen 130-instance public FJSP extension suite, a deterministic insertion
decoder with independent feasibility checks, exact Tiny validation, a
semantics-hardened heterogeneous graph interface, RT-HGT, an autoregressive
O→M→W→F policy, synthetic BC warm start, and constructive PPO.

This revision deliberately keeps the existing F-kit semantics: there is no
early material release/staging constraint and no modified F-AGV formulation.
Phase 3 optimizes makespan only. Multiobjective preference policies, the
Critical Synchronization Graph, and neural destroy/repair remain intentionally
out of scope.

## Installation

Python 3.11 is used for the validated environment.

```bash
python -m pip install -r requirements.txt
```

## Reproducible validation

Run from the repository root:

```bash
python -m compileall .
pytest -q
python scripts/generate_canonical_benchmarks.py --verify-only
python scripts/run_small_validation.py
python scripts/profile_graph_builder.py
python scripts/run_bc_validation.py
python scripts/run_ppo_sanity.py --device cuda
python scripts/train_bc_pretrain.py --device cuda
python scripts/train_ppo.py --seed 31001 --device cuda
python scripts/evaluate_ppo.py --device cuda
python scripts/profile_training.py --device cuda --episodes 300
python scripts/plot_phase3_results.py
python scripts/audit_repo_structure.py
```

The native solver comparison requires a working local Gurobi license:

```bash
python scripts/run_native_tiny_validation.py
```

On this machine the Gurobi license is bound to the Windows user `ASUS`, so the
command must run in that user context rather than an alternate sandbox account.

Canonical generation itself is a separate, deterministic operation:

```bash
python scripts/generate_canonical_benchmarks.py
```

It reuses the untouched public sources under `FJSP-benchmark-main/`, validates
all outputs, and byte-verifies the 130 frozen JSON files. Training and
profiling read checked-in instances; they never regenerate benchmark data.

## Repository map

- `instances/canonical/RCIAS-2.0/`: 10 Brandimarte and 120 Hurink E/R/V
  extensions, manifests, validation rows, generation config, and SHA256 file.
- `instances/tiny/`: three frozen exact/edge validation instances, including
  the four-island/two-W/two-F native-solver comparison case.
- `rcias_clgri/data/`: strict RCIAS-2.0 loading, validation, and canonical
  generation support.
- `rcias_clgri/env/`: insertion timelines, decoder, schedule records,
  objective, exporter, and independent checker.
- `rcias_clgri/graph/`: normalized graph construction and hierarchical
  ready-only candidate interfaces.
- `rcias_clgri/nn/`: tensorizer, RT-HGT, hard-masked autoregressive policy,
  and value head.
- `rcias_clgri/learning/`: demonstration replay, rollout buffer,
  telescope-preserving reward, GAE, clipped PPO, trainer, and evaluation.
- `rcias_clgri/training/`: deterministic synthetic instance factory and
  validation-gated S/M/L curriculum with old-level replay.
- `configs/phase3_training.json`: frozen seeds, distribution, model, PPO, BC,
  curriculum, and evaluation settings.
- `outputs/validation/tiny_01/`: exact solution, CSVs, feasibility report, and
  Gantt PNG/PDF derived from one schedule object.
- `outputs/profiling/`: graph complexity measurements.
- `outputs/bc_validation/run_1/`: demonstrations, training history, figures,
  and final BC metrics.
- `outputs/phase3/`: BC warm start, three independent PPO seeds, held-out
  synthetic/canonical evaluation, profiling, and five PNG/PDF figures.
- `docs/reports/`: phase-by-phase and final validation reports.

## Current validated state

- Canonical suite: 130/130 valid and byte-reproducible.
- `tiny_01`: exhaustive active-schedule BnB status `OPTIMAL`, makespan 157;
  decoder replay and 13-category independent checker pass.
- `tiny_03`: Gurobi MILP 13.0.3 and OR-Tools CP-SAT 9.11.4210 both prove
  makespan 36 with bound 36 and gap zero; both decoder replays are feasible.
- Graph: ID-independent numeric semantics, permutation-equivariant actions,
  and O(|W|+|F|) probing per ready operation-island pair.
- BC: 100% expert action reproduction, 100% feasible greedy rollout, and
  makespan 157.
- Constructive PPO: three independent synthetic-only training seeds, joint
  autoregressive PPO ratios, 100% feasible training rollouts, fixed held-out
  validation, and checkpoint-frozen canonical evaluation.
- Frozen evaluation: 130/130 canonical instances and all 910 method-runs are
  feasible; pooled PPO gap is 41.54% versus 48.22% for BC. PPO improves BC in
  every benchmark family but does not outperform H1 (0.87% gap).
- Profiling: 300 detached CUDA episodes, 6200 steps, 7.90 steps/s, stable CPU
  RAM and GPU reserved memory.

The mathematical model is in
`reconfigurable_island_assembly_mathematical_model.md`; the execution design is
in `codex_algorithm_execution_spec.md`.
