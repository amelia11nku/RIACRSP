# Phase 5C Controlled Benchmark Report: RCIAS-CB1

## 1. Executive conclusion

`NEW_CONTROLLED_BENCHMARK_REQUIRED = TRUE`. RCIAS-CB1 contains exactly 18 development, 45 Core test, and 45 paired sensitivity instances. All 108 load, produce a feasible H1 schedule, and pass the independent checker. Core structural gates and sensitivity pairing pass. The frozen benchmark hash is `3d97e6a3f84d930213c424a633fa9e8192b16ec4eb9c2f169f65c0b48e691f0a`.

## 2. Motivation from legacy audit

Legacy-130 has mean capability coverage 0.9788, full-island ratio 0.9002, processing-time CV 0.0254, and RI 0.1270. It therefore emphasizes near-universal capability and weak processing/reconfiguration heterogeneity. It remains valuable as public-derived external and high-flexibility validation, but is not the sole primary test set.

## 3. Native RCIAS benchmark design

CB1 is generated directly under RCIAS-2.0 semantics. Capability sets are generated before operation requirements and routing eligibility. Eligibility is always a subset of islands supporting the required configuration. Processing, reconfiguration, W logistics, and F logistics are explicit native structures.

## 4. DEV/TEST isolation

DEV uses a disjoint 551xxx base-seed range and contains 18 calibration-only instances. CORE uses 552xxx and SENS uses 553xxx. DEV is excluded from formal statistics. CORE and SENS became immutable at the recorded freeze boundary.

## 5. Scale design

Each of Small, Medium, and Large contributes 15 Core instances. Operation gates are 55–65, 110–130, and 165–195. Product counts are 8, 15, and 22. Small uses seven islands: six would make the mandatory two-eligible-island gate imply a minimum routing flexibility of 0.333, contradicting the CF1 upper bound 0.30; seven is the nearest feasible interpretation of “approximately six.”

## 6. Capability flexibility

CF1, CF2, and CF3 target capability intervals 0.30–0.40, 0.45–0.55, and 0.60–0.70. Each contributes 15 Core instances. Realized overall Core mean is 0.5016, every configuration is supported by at least two islands, and no island supports every configuration.

## 7. Routing flexibility

CF routing intervals are 0.20–0.30, 0.33–0.43, and 0.47–0.57. Every operation has at least two eligible islands, `R_full_op=0`, and eligibility-capability consistency holds. Overall Core mean is 0.3701.

## 8. Processing heterogeneity

Each operation receives an independently sampled target CV and standardized, shuffled island-efficiency factors. Core processing CV ranges inside 0.20–0.45 and averages 0.3248. A pre-freeze generator version showed unwanted CV correlations with scale and CF; it was rejected, corrected, and fully regenerated before checksums were created. Final Core correlations are 0.026 with size and −0.038 with routing flexibility.

## 9. DAG generation

Products use connected branching/converging DAGs with incomparable branches and multiple topological orders. The mean Core precedence density is 0.4060 and mean precedence-only ready-set size is 9.19. Density decreases with product size partly because its denominator counts all possible ordered pairs; this expected scale effect is reported rather than treated as a factorial variable.

## 10. Reconfiguration design

Diagonals are zero and off-diagonal transitions are positive, heterogeneous, island-specific values. Core RI averages 0.4199. Nonzero transition mean, median, standard deviation, CV, and maximum are saved in the structural metrics.

## 11. Transportation design

Coordinates are generated relative to one warehouse and Manhattan geometry determines base distances and travel times. Common scalar normalization preserves spatial ordering. Core mean W and F intensities are 0.5500 and 1.0001.

## 12. Core composition

CORE is exactly `3 scales × 3 CF levels × 5 replicates = 45`. Every Scale×CF cell contains five independently seeded instances. The five replicates vary product sizes, DAGs, configuration demand, capability matrices, eligibility, processing efficiencies, and geometry.

## 13. Sensitivity factorial design

Five independent Medium-CF2 base structures each produce a 3×3 RI×TI factorial. Across cells, mean realized RI is 0.2092, 0.4190, and 0.8380. Mean W intensity is 0.2748, 0.5500, and 1.1000; mean F intensity is 0.4989, 1.0011, and 2.0022.

## 14. Structural acceptance gates

Deterministic rejection checks scale bounds, capability/routing intervals, eligibility, configuration support, processing CV, RI, W/F intensity, configuration entropy, positive processing, zero reconfiguration diagonals, and branching DAGs. All accepted structures required at most two attempts except one Small-CF3 Core structure requiring two; no target interval was widened.

## 15. Legacy vs controlled comparison

| Metric | Legacy-130 mean | CB1-Core mean |
|---|---:|---:|
| Routing flexibility | 0.3014 | 0.3701 |
| Capability coverage | 0.9788 | 0.5016 |
| Full-island ratio | 0.9002 | 0.0000 |
| Processing CV | 0.0254 | 0.3248 |
| Reconfiguration intensity | 0.1270 | 0.4199 |
| W transport intensity | 0.3623 | 0.5500 |
| F transport intensity | 1.0371 | 1.0001 |
| Configuration entropy | 0.9998 | 0.9429 |

The 12 structural figures and machine-readable comparison tables are under `outputs/phase5c/controlled_benchmark_audit/`.

## 16. Feasibility validation

All 108 JSON files load with the frozen loader. H1 constructs one complete schedule for every instance, and all 108 schedules pass the independent feasibility checker. H1 makespan is not used for generation or acceptance.

## 17. Benchmark freeze

`checksums.sha256` covers 108 instances plus generation spec and four manifests (113 files). `freeze_record.json` records counts, legacy manifest hash, coverage-audit hash, checksum-file hash, and the benchmark freeze hash. CORE and SENS contents may only change in a future CB2 version.

## 18. Limitations

CB1 is synthetic and static. It excludes path planning, dynamic arrivals, inventory constraints, and multiobjective optimization. Sensitivity scales integer travel/reconfiguration times, so realized ratios are close to rather than exactly the nominal factors. DAG structure varies but is not factorially controlled.

## 19. Recommended experimental usage

Use DEV only for calibration, CORE for primary paired algorithm comparisons, SENS for within-base causal RI×TI analysis, and Legacy-130 for separate public-derived/OOD validation. Never combine these suites into one grand mean.

## 20. Reproducibility checklist

- `DEV_COUNT = 18`
- `CORE_COUNT = 45`
- `SENS_COUNT = 45`
- `ALL_108_FEASIBLE = TRUE`
- `CORE_STRUCTURAL_GATES_PASSED = TRUE`
- `SENSITIVITY_PAIRING_PASSED = TRUE`
- `TEST_BENCHMARK_FROZEN = TRUE`
- deterministic seeds and accepted attempts recorded
- manifests and generation history retained
- 12 benchmark figures saved as PNG and PDF
- Legacy-130 checksum unchanged
