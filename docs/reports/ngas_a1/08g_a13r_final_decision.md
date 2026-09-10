# NGAS-A1.3R final decision

Decision: **NGAS_A1_3R_PASS_COMPACT**. Selected representation: **C1**.
A1.4 search-integration preparation is authorized.

The pre-task HEAD was `2fec6f365095385bd443f7e6481499a716e49049`; the
starting-boundary evidence commit is `7376c12b2e2c8fc615b478fc995dd2e61faf01ca`.
The new formal protocol was frozen before optimization at SHA-256
`0cdd5de321066c1f301afda7da86b91854d1d4b55fa2ee4df76a2dfa518d6f28`.
The 41-file historical C0 output tree remains unchanged at
`4239e9236cb5a5d9837def6acc70baa9155b32cb2282a85efb604a57a4b3a9f9`.

The critical-sync audit passed standard CPM over the full realized DAG and the
union of all tied makespan-critical chains. All 8,453 critical event nodes
satisfied the independent longest-path identity. The 144 events without a path
to the makespan sink retained undefined slack and were not labeled critical.
Deterministic OP/W/F/reconfiguration mapping and all seven reason categories
passed.

The A1.2 identity audit covered 6,465 joint actions, 58,185 replicate labels
and 174,555 paired continuation-step pairs. Encoded and executed `(k,D,R)`,
repair IDs, common-random-number streams and fallback identities matched with
zero mismatches. The revised 72-state cache reused the exact historical labels;
no continuation rollout was rerun.

| Variant | Spearman | Pair accuracy | NDCG@1 | NDCG@3 | NDCG@5 | Selected advantage | Top-1 regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.357974 | 0.812006 | 0.494255 | 0.523178 | 0.545011 | 0.018811 | 0.022403 |
| C1 | 0.350615 | 0.808400 | 0.459704 | 0.498847 | 0.520820 | 0.016320 | 0.024894 |
| R1 | 0.360814 | 0.812919 | 0.478086 | 0.521825 | 0.553702 | 0.017100 | 0.024114 |

C1 and R1 both passed 3/3 seed gates. R1 did not reach any predeclared
material-improvement threshold over C1. C1 matches R1 within every practical
margin and its worst representative p90 is 2.056765 ms. R1's corresponding
p90 is 49.825778 ms, above the frozen 30 ms limit. C1 therefore remains the
efficient production representation; R1 is retained as the richer ablation.
The selected production checkpoint SHA-256 is
`448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`.

The terminal audit verified 18/18 formal runs, 432 revised OOF predictions,
all checkpoint/prediction hashes, instance-disjoint folds, fixed final epochs,
raw-metric recomputation and deterministic maximum-state production replay.
The full regression suite passed 480 tests with no failures, errors or skips.
No frozen Phase 6 path changed. R13 and R14 remained locked, and Gurobi was not
invoked. The training service exited successfully and no long-running job
remains active.
