# Phase 5A Stable PPO Validation Report

## Executive conclusion

Phase 5A substantially improved checkpoint quality and seed consistency, but it did not pass the stability gate. The three frozen `best_mean` checkpoints improve the BC policy by 7.07% on the independent 30-instance development set, versus Phase 4 PPO being 4.33% worse than BC. Phase 5A seed-level development gaps are -8.53%, -6.67%, and -6.02%, with a standard deviation of 1.06 percentage points, compared with 13.87 for Phase 4.

The canonical gate nevertheless fails for two independent reasons. All three training trajectories regress by 15.32–17.44% from their best development checkpoint to the final checkpoint, and historical-validation level L is 13.02% worse than BC. Therefore the canonical 130 suite remains frozen and no canonical evaluation was executed.

## Experimental boundary

- Training environment: `gnn311`, CUDA.
- Frozen Phase 5A config: `configs/phase5a_final.json`.
- Independent training seeds: 510101, 510102, 510103.
- Development set: 30 synthetic instances, ten each at S/M/L.
- Historical validation: the ten previously fixed S/M/L seeds.
- Structural generalization: three seeds each for fleet scarcity, high reconfiguration, and high travel.
- Canonical instances used for training, screening, gradients, or selection: 0.
- Canonical full evaluations: 0.
- CSG, neural destroy selection, new GNN architecture, potential shaping, and multi-objective PPO remain disabled.

## Frozen checkpoint results

All policies and heuristics were 100% feasible.

| Split | Phase 4 PPO gap to BC | Phase 5A PPO gap to BC | Interpretation |
|---|---:|---:|---|
| Development | +4.33% | -7.07% | Phase 5A passes the BC-quality gate |
| Historical validation | +10.16% | +3.76% | Improved over Phase 4, still worse than BC |
| Structural generalization | +7.22% | +2.56% | Improved over Phase 4, still worse than BC |

Development results by level are -1.99% (S), -7.15% (M), and -12.08% (L). Historical results are +2.75% (S), -0.74% (M), and +13.02% (L). The historical L result crosses the predeclared 10% severe-regression threshold.

Structural gaps to BC are:

| Scenario | Phase 5A gap to BC |
|---|---:|
| Fleet scarcity | -0.77% |
| High reconfiguration | +2.05% |
| High travel | +6.42% |

The high-reconfiguration and high-travel means remain below the 10% severe-regression threshold, although high travel is the weakest structural group and seed 510102 reaches +13.24% there.

`best_mean` and `best_robust` select the same model parameters for every seed (updates 25, 28, and 33 respectively). Their serialized SHA256 values differ because checkpoint metadata differs; tensor-by-tensor model states are identical.

## Training stability

| Seed | Best update | Best normalized makespan | Final normalized makespan | Final regression from best |
|---:|---:|---:|---:|---:|
| 510101 | 25 | 0.15251 | 0.17911 | 17.44% |
| 510102 | 28 | 0.15566 | 0.18088 | 16.20% |
| 510103 | 33 | 0.15671 | 0.18071 | 15.32% |

Early stopping and robust checkpoint retention prevent these late trajectories from replacing the selected models, but they do not make the underlying optimization stable. Consequently `STABLE_PPO` remains false.

## Q1 — Why did Phase 4 PPO damage BC?

The evidence points to representation and operation-selection drift rather than infeasibility.

- KL: the one-update sanity run already showed operation KL (0.00698) dominating island/W/F KL. In formal runs, teacher KL continued rising during late validation deterioration.
- Entropy: normalized per-stage entropy remains measurable and non-collapsed, so the failure is not a simple deterministic-policy entropy collapse.
- Gradient interference: actor/critic encoder-gradient cosine is negative on average for S, M, and L (-0.006, -0.078, and -0.048); 25% of sampled L cases are below -0.2.
- Rollout diversity: the complete-episode collector supplies at least eight unique completed instances per update and all levels in mixed windows. The Phase 4 failure therefore cannot be attributed solely to repeatedly updating on one partial episode.
- Stagewise oracle: keeping BC operation choice and using PPO downstream produces mean makespan 1571.77, better than both BC (1882.20) and Phase 4 PPO (1629.93). Replacing only the operation-side choice with PPO makes the hybrid 1833.13. The operation stage is the primary harmful drift location, while downstream PPO decisions contain useful signal.

## Q2 — Which stabilization mechanism was most effective?

The conservative A1 optimizer settings plus mixed-scale curriculum and best-checkpoint retention were most effective. Shared actor/critic encoder gradients (C0) outperformed critic stop-gradient (C1), despite the measured negative gradient cosine. Teacher anchoring was implemented and validated, but B1/B2 reduced drift at the cost of development quality and was not selected. Mixed curriculum D1 slightly outperformed staged D0 and maintained S/M/L exposure in every sampling window.

## Q3 — Is PPO genuinely better than BC?

Only on the independent development distribution and fleet-scarcity structural cases. The frozen checkpoints beat BC consistently across all three development seeds, but lose on historical validation overall and show a severe historical-L regression. The defensible conclusion is “promising but not globally superior,” not “PPO has replaced BC.”

## Q4 — Where does PPO improve?

The largest development improvement occurs on L, followed by M. Fleet scarcity also improves slightly. The stagewise decomposition shows that gains are concentrated in island/vehicle downstream decisions when the operation decision remains anchored to BC. High travel and historical L remain the principal weaknesses.

## Q5 — Should PPO remain the final constructive component?

Not yet as the sole final component. The selected checkpoints are materially better and more reproducible than Phase 4, so PPO should remain an experimental candidate. BC should remain the safe constructive reference until operation-stage drift and late-training collapse are controlled on historical L and high-travel cases.

## Canonical gate

The machine-readable gate is `outputs/phase5a/canonical_gate.json`. Passed checks are development PPO not worse than BC, 100% feasibility, seed variance below Phase 4, and no severe mean high-reconfiguration/high-travel regression. Failed checks are no late catastrophic collapse and no severe per-level collapse.

Result: `PHASE5A_CANONICAL_GATE = FAIL`; `canonical_instances = 0`.

## Reproducibility and validation

- Full test suite: 94 passed.
- Canonical byte-level regeneration/checksum verification: 130/130 passed; this was verification only, not policy evaluation.
- Nine required diagnostic figures were produced as both PNG and PDF under `outputs/phase5a/figures/`.
- Repository audit removed only safe Python/test caches. `repository_clean=false` remains because the audit identifies duplicate files inside frozen Phase 4 capacity-study outputs; those frozen artifacts were intentionally not altered. It also lists the new Phase 5A scripts as review candidates, but each is a distinct required entry point rather than a duplicate implementation.

## Final state

```text
PHASE4_DEGRADATION_REPRODUCED = TRUE
NORMALIZED_ENTROPY_VALIDATED = TRUE
TEACHER_ANCHOR_VALIDATED = TRUE
ACTOR_CRITIC_INTERFERENCE_DIAGNOSED = TRUE
ROLLOUT_DIVERSITY_VALIDATED = TRUE
STAGEWISE_ORACLE_VALIDATED = TRUE
STABLE_PPO = FALSE
PPO_NOT_WORSE_THAN_BC = TRUE
PPO_IMPROVES_BC = TRUE
CANONICAL_EVALUATION_EXECUTED = FALSE
REPOSITORY_CLEAN = FALSE
READY_FOR_CSG = FALSE
```

