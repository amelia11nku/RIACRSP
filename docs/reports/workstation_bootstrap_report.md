# New Workstation Bootstrap Report

## Environment

The project was restored in the user-requested `gnn311` Conda environment
using Python 3.11.15. The workstation has an Intel Core i7-14700, 31.1 GiB RAM,
and an NVIDIA GeForce RTX 4060 Ti with 8,188 MiB VRAM. Driver 575.64.03 reports
CUDA compatibility 12.9. PyTorch 2.11.0+cu128 passes a real CUDA tensor test.
Gurobi 13.0.3 and its local restricted license are operational.

Full dependency details are in `workstation_environment_report.md`.

## Historical regression

| Check | Result |
|---|---|
| Focused Phase 4 tests | 5 passed |
| Full pytest suite | 74 passed |
| Python compileall | PASS |
| Canonical verify-only | 130/130 checksum and byte regeneration PASS |
| Tiny exact validation | makespan 157 / 57 / 36, all feasible |
| Native Tiny validation | Gurobi = CP-SAT = exhaustive = 36, gap 0 |
| Phase 2 BC validation | joint reproduction 1.0, feasible, makespan 157 |
| CUDA PPO sanity | `TINY_PPO_SANITY_VALIDATED = TRUE` |

## Checkpoint compatibility

The formal checkpoints from `outputs/phase3/bc_pretrain` and all three
`outputs/phase3/ppo_seed_*` directories loaded without missing or unexpected
state-dict keys (291 entries each). Each completed two deterministic CUDA
replays on the same held-out S-level instance with identical action sequences
and makespan; all replays were feasible.

## Short workstation profile

The required 30-episode CUDA profile completed 631 environment steps at 86.34
steps/s and 4.10 episodes/s. CPU and GPU memory were stable. Measured time was
39.82% RT-HGT forward, 40.67% PPO update, 12.43% graph construction, 6.90%
policy scoring, and 0.18% decoder. This confirms that batched conditional PPO
reevaluation is the relevant performance target; decoder changes are not.

Artifacts are under `outputs/phase4/environment/ppo_sanity/` and
`outputs/phase4/profiling/bootstrap_30/`.

```text
ENVIRONMENT_READY = TRUE
PHASE1_REGRESSION_PASS = TRUE
PHASE2_REGRESSION_PASS = TRUE
PHASE3_REGRESSION_PASS = TRUE
READY_TO_CONTINUE_PHASE4 = TRUE
```
