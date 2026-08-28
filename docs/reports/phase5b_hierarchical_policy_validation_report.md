# Phase 5B Operation-Anchored Hierarchical Policy Validation Report

## Executive conclusion

Phase 5B confirms that the four autoregressive action stages should not be optimized as one undifferentiated PPO policy. Freezing the complete BC mapping from graph state to greedy operation choice, while refining island/W/F decisions with PPO, produces a stable and reproducible constructive policy on the expanded synthetic evaluation distribution.

The selected architecture is:

```text
F1 = Frozen BC operation encoder/head + trainable PPO island/W/F encoder/heads
```

Across the 60-instance independent holdout, the three formal F1 seeds improve BC by 7.98%, 6.49%, and 3.12%. Their mean improvement is 5.86%, with a seed standard deviation of 2.03 percentage points and 100% feasibility. The one-time canonical result is more cautious: the three-seed mean is 0.16% worse than BC overall, with improvements on Brandimarte and Hurink E/R but a regression on Hurink V. The canonical result was not used for tuning.

## Frozen experimental boundary

- Phase 3, Phase 4, and Phase 5A configs and outputs were not overwritten.
- Phase 5B training config: `configs/phase5b_training.json`.
- Final config: `configs/phase5b_final.json`.
- Frozen final-config SHA256: `2476d52ea82ee21461ca9128c84ab80743e391bfe346c0b87008763d4f244e42`.
- Expanded holdout: 20 S, 20 M, and 20 L instances.
- Structural stress: ten instances each for high reconfiguration, high travel, and fleet scarcity.
- All Phase 5B evaluation seeds are disjoint from training, Phase 5A development, historical validation, earlier structural seeds, and canonical data.
- Canonical gradient, training, screening, and checkpoint-selection instances: 0.
- Reward, D128-L3 RT-HGT architecture, hard masks, decoder, and scheduling semantics were unchanged.

## Phase 5A oracle revalidation

The sequential hybrid diagnosis used all three frozen Phase 5A `best_mean` checkpoints over 129 instances per checkpoint. All 387 policy/instance records and every hybrid schedule were feasible. Every decision step reconstructed the current graph from the decoder state; no action sequences were spliced.

On the 60-instance holdout:

| Policy | Mean gap to BC |
|---|---:|
| Full Phase 5A PPO | -2.49% |
| BC-O + PPO-MWF | -3.08% |
| PPO-O + BC-MWF | -1.20% |
| BC-OM + PPO-WF | -3.07% |
| PPO-OM + BC-WF | -1.30% |
| BC-OMW + PPO-F | -2.95% |
| PPO-OMW + BC-F | -1.47% |

BC-O + PPO-MWF beats full PPO for two of the three Phase 5A seeds, beats BC in aggregate, and remains stable on high-reconfiguration and high-travel stress. Therefore `OPERATION_DRIFT_CONFIRMED = TRUE`.

## Frozen-operation implementation

The Phase 5B model contains two independent branches:

```text
GraphTensor
  ├─ frozen BC RT-HGT + frozen operation scorer → greedy operation
  └─ trainable BC-initialized RT-HGT + M/W/F heads + value head
```

The frozen branch runs under `torch.no_grad()`, remains in evaluation mode, and has `requires_grad=False` for every parameter. Its operation distribution was checked before and after a downstream PPO update on fixed S/M/L states; the maximum elementwise difference was exactly 0.0.

The PPO joint log probability, likelihood ratio, and entropy loss contain only M/W/F. Frozen operation entropy is recorded but excluded from the loss. Peak pilot GPU allocated/reserved memory was approximately 3.86/6.93 GiB, within the 8GB device limit.

## Best-policy reference and rollback

The trainer keeps an in-memory best downstream-policy snapshot. Three consecutive validations more than 4% worse than the best reference trigger restoration, optimizer-state reset, and a 0.5 learning-rate reduction, with at most two rollbacks.

No formal F1 trajectory met the three-consecutive-validation trigger. Seed 1 and seed 3 last checkpoints were approximately 2.95% and 3.00% worse than their best checkpoints; seed 2 ended at its best. This is materially below the 15–17% Phase 5A late collapse. Best-policy KL was measured but not added to the loss.

## Formal F1 results

| Seed | Best update | Development best | Holdout gap to BC |
|---:|---:|---:|---:|
| 520101 | 18 | 0.14553 | -7.98% |
| 520102 | 25 | 0.14952 | -6.49% |
| 520103 | 11 | 0.15735 | -3.12% |

The three-seed holdout summary is:

| Group | Mean gap | Seed std | Best seed | Worst seed |
|---|---:|---:|---:|---:|
| S | -2.00% | 0.67 pp | -2.84% | -1.20% |
| M | -5.24% | 2.89 pp | -7.50% | -1.17% |
| L | -10.35% | 3.11 pp | -14.48% | -6.99% |
| Overall | -5.86% | 2.03 pp | -7.98% | -3.12% |

Structural results are:

| Scenario | Mean gap to BC | Worst seed gap |
|---|---:|---:|
| Fleet scarcity | +0.87% | +1.33% |
| High reconfiguration | -8.11% | +2.01% |
| High travel | -10.57% | -0.65% |

The formal downstream-PPO gate passes: mean performance is better than BC, seed standard deviation is at most three percentage points overall, and feasibility is 100%.

## Freeze-boundary experiment

F2 freezes both BC operation and island decisions and trains only W/F. It used the same seed and 25-update budget as the F1 pilot.

| Boundary | Holdout gap to BC | Structural overall gap |
|---|---:|---:|
| F1 pilot: Frozen O + PPO-MWF | -2.42% | -0.41% |
| F2 pilot: Frozen OM + PPO-WF | -1.54% | -0.74% |
| Formal F1 three-seed mean | -5.86% | -5.93% |

F2 is viable but weaker on holdout overall, S, M, and high travel. F1 is selected because allowing PPO to learn island assignment provides useful incremental value while the operation sequence remains protected.

## One-time canonical evaluation

The canonical gate was evaluated and frozen before any canonical policy rollout. The config and three checkpoint SHA256 values are recorded in `outputs/phase5b/canonical_gate.json`. The protected evaluation script refuses to overwrite an existing canonical result.

All 130 canonical instances and all 780 method/seed runs were feasible. Runtime was 1243.5 seconds.

| Method | Overall gap to BC |
|---|---:|
| H1 | -14.02% |
| H2 | -7.66% |
| BC | 0.00% |
| Phase 5B F1, three-seed mean | +0.16% |

Phase 5B seed gaps are +0.22%, +1.76%, and -1.51%, with a standard deviation of 1.33 percentage points. Family means are:

| Family | Phase 5B gap to BC |
|---|---:|
| Brandimarte | -3.68% |
| Hurink E | -5.09% |
| Hurink R | -3.79% |
| Hurink V | +10.32% |

The synthetic expanded-holdout success therefore transfers to three canonical families but not Hurink V. This is reported as external-distribution evidence, not used to modify Phase 5B.

## Required questions

### Q1. Does operation drift remain in all three Phase 5A best checkpoints?

Yes in aggregate. For two of three seeds, replacing the Phase 5A operation branch with BC improves the full policy, and the aggregate BC-O hybrid is better than both BC and full PPO. The third seed benefits from its learned operation branch, so the conclusion is a robust architecture-level risk rather than a claim that every learned operation action is harmful.

### Q2. Was historical-L failure general or two-instance noise?

It was predominantly sampling noise. On 20 independent L instances, full Phase 5A PPO improves BC by 2.95% and the BC-O hybrid by 3.34%. Formal Phase 5B F1 improves L by 10.35% on average. The earlier +13.02% result from two L instances is not representative of the expanded holdout.

### Q3. Does frozen-operation M/W/F PPO provide stable incremental value?

Yes on the expanded synthetic distribution. All three independent seeds improve BC, overall seed standard deviation is 2.03 percentage points, and all schedules are feasible. Canonical evidence is mixed: three families improve while Hurink V regresses.

### Q4. Can rollback manage late collapse?

The mechanism is implemented and unit-tested, including parameter restoration, learning-rate reduction, and optimizer-state reset. No F1 formal trajectory required a rollback because none crossed the configured trigger. Best checkpoint retention and the rollback threshold constrained last-vs-best regression to about 3%, substantially below Phase 5A.

### Q5. Is PPO worth retaining?

Yes, as the downstream M/W/F refinement component behind a frozen operation prior. It is not supported as an unconstrained full O/M/W/F policy. For canonical deployment, BC or a family-aware safeguard remains prudent for Hurink V until a future phase addresses cross-domain selection without tuning on this canonical holdout.

### Q6. Is Elite Distillation more reliable?

Not tested, by design. The Phase 5B instruction makes Elite Distillation a fallback only if reasonable F1/F2 pilots cannot stably reach BC. F1 passed the pilot and three-seed formal gates, so invoking the fallback would spend additional search budget without satisfying its trigger condition.

## Verification and repository audit

- Full tests: 108 passed.
- Canonical byte-level regeneration/checksum verification: 130/130 passed.
- Frozen-operation distribution invariance: exact on fixed S/M/L states.
- Required figures: eight PNG and eight PDF files under `outputs/phase5b/figures/`.
- Repository audit removed only rebuildable caches.
- `repository_clean=false` remains because the audit identifies duplicate reproducibility configs and frozen Phase 4 study outputs. These are intentional historical/frozen artifacts and were not deleted.

## Final state

```text
PHASE5A_ORACLE_REVALIDATED = TRUE
OPERATION_DRIFT_CONFIRMED = TRUE
EXPANDED_HOLDOUT_VALIDATED = TRUE
FROZEN_OPERATION_INVARIANT = TRUE
DOWNSTREAM_PPO_STABLE = TRUE
DOWNSTREAM_PPO_IMPROVES_BC = TRUE
ELITE_DISTILLATION_TESTED = FALSE
ELITE_DISTILLATION_IMPROVES_BC = FALSE
CONSTRUCTIVE_POLICY_READY = TRUE
CANONICAL_EVALUATION_EXECUTED = TRUE
REPOSITORY_CLEAN = FALSE
READY_FOR_CSG = TRUE
```

