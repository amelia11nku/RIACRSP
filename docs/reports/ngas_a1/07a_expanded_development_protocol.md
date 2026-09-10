# Expanded R12 development data and training preregistration

The V2 completion audit verified 9,108 raw files, 900 action/variant rows and 8,100
paired replicates. It replayed six source states and twelve candidate continuations,
and reproduced the entire saved gate. The sole primary T8_R9 passed; V1 remains
failed evidence. Its compact verified V2 archive is `outputs/ngas_a1/archive/label_pilot_v2.tar.gz`.

## Data scope and sampling

Use every canonical R12 CAUR-FIT instance: 3 scales × 3 CF levels × 2 cell replicates.
For each of 18 instances collect H1 once and NATIVE16 at seeds 746101–746103, for
72 source states. NATIVE16 uses the already versioned native policy with eight
repair trials. Each state is the current feasible candidate, not a selected best
state chosen after inspecting labels. All planned states are retained, with no
favorable state selection. Exact duplicate states, if any, remain in the same fold.

Build the entire 24-rule bank for each of three sizes before sampling. Per size,
include critical-sync, a rotating second original operator, a related variant,
a matched random rule, a local perturbation and a structured-neighbor rule. Cyclic
selection depends only on preregistered state ordinal and size, not outcome. All
24 rules are covered across the plan. Cross each target with all five repairs;
deduplicate exact joint actions and retain every origin. Maximum: 90 actions/state.

Each joint action has nine fresh paired repeats, eight repair trials and two native
continuation moves, with the accepted T8_R9 semantics. New state IDs generate disjoint
CRN keys. Pilot labels are not pooled for training. Exact fallback caching counts
generation once per state/repeat, preserving identical candidate/fallback CRN.

Upper bounds: 6,480 actions, 1,399,680 candidate and 15,552 fallback decoder calls;
1,415,232 label calls in total. Native source generation uses 6,912 additional
explicit decoder calls. H1 initialization, source replay, graph construction and
I/O are additional overhead. Raw per-action results are atomically published and
hash-validated on resume; only an unpersisted interrupted action may be recomputed.
Retained-work counters exclude interrupted unpersisted work and must not be called
total historical consumed compute across restarts.

## Separation and gates

Fold = (scale index + CF index) mod 3, with each axis ordered S/M/L and CF1/2/3.
Both cell replicates and every source state of an instance remain in the same fold.
Each fold contains six instances, all three scales and all three CF levels.
This is grouped development OOF evaluation, not a newly untouched test set.

Before training require all planned states, feasibility, all size/repair/family
coverage and all 24 rules. In each fold at least half the states must have >=10%
noise-separated action pairs, and the overall positive-advantage action fraction
must be >=5%. Pair separation retains max(.001, twice paired SE).
Failed data gates stop model fitting and are recorded without changing thresholds.

The architecture, loss, fixed training epochs, three seeds and OOF gate are frozen
in `configs/ngas_a1_development_v1.json`. No labels are used to fit preprocessing.
The production seed is fixed before OOF outcomes. No training starts automatically
after collection: audit the raw/feature/action contract first, then implement and
run the prescribed training/OOF runner.

## Operation and resume

Active artifacts: `outputs/ngas_a1/development_v1/progress.json`, `launch_record.json`,
`states/`, and eventual `data_gate.json` / `result_hash_manifest.json`.
Check the worker PID/lock before resuming with:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/launch_ngas_development.py
```

The protocol hashes all existing NGAS/shared source, instances and previous V2
evidence. Changing those files invalidates resume; future implementations can add
new files while preserving this boundary. R13/R14 stay locked, with no Gurobi or push.
