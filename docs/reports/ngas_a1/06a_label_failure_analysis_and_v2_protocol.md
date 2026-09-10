# A1.2 completion audit and bounded label revision

Continuation starting commit: `13404cc0c6b44ed244a0d1c88b4806beff541b5a`.
V1 remains **NGAS_A1_REVISE_LABELS**. A1.3 training stays gated.

The independent audit verified 462 result files, 450 joint actions and 1,350 paired
replicates, replayed all six source schedules and one complete label per state,
and reproduced the saved gate exactly. Fallback trajectories are identical across
actions for the same state/CRN. Action repair identity, counters and advantages agree
with raw trajectories. No source, checkpoint or historical comparator drift was found.

Only S/CF1 H1 clears the preregistered 10% noise-separated pair fraction; three of
six were required. Five other fractions are 0%, 4.97%, 6.95%, 7.53% and 7.57%.
There are two distinct limitations: many action pairs are below the unchanged .001
material-gap floor, and many remaining pairs are obscured by three-replicate noise.
The L H1 labels are especially sparse: 92.89% of candidate continuations retain the
source best, and none improves after the initial move. In M NATIVE4, 1,801/2,775
pairs have a material empirical mean gap but fail the noise criterion.
These are descriptive diagnoses, not evidence that more replication must succeed.

Repair is material in a subset of this pilot: 53/900 same-size/same-target repair
pairs separate above the noise threshold, and two target-rank reversals across
repairs are observed. Repair remains in the joint architecture.

## Pre-outcome V2 design

Create a separate `outputs/ngas_a1/label_pilot_v2` boundary. Keep all six source
states, all 450 actions, the two-step continuation horizon, the acceptance rule,
the primary advantage, and the .001 / twice-SE ranking criterion unchanged.
Do not pool old replicates or change the failed V1 decision.

Use nine fresh independent CRN replicates: three named repeats under each of the
three canonical seed roots. Compare two preregistered variants on matched keys:

- T2_R9 is the replication control, retaining two repair trials.
- T8_R9 is the sole primary, using eight repair trials to align label execution
  with the intended eight-trial online repair and test whether sparse opportunities
  become more observable. It is not selected after outcomes.

First-three versus all-nine diagnostics isolate the replication effect within the
same new draws; T8 versus T2 diagnostics isolate repair effort at matched CRN.
The first-three and root-block views are nested diagnostics, not separate test sets.
There is no control-arm rescue if T8_R9 fails. No gate is relaxed: at least three
states must have >=10% informative pairs, all feasibility/coverage checks must pass,
and positive opportunities and the 30-second action cost cap still apply (the cap
now includes all nine replicates and amortized fallback construction).

Exact fallback trajectories may be cached once per state, variant and replicate;
the cache key includes state/config/protocol hashes and CRN identity. This changes
consumed computation, not label semantics. Actual cost counts cache creation once;
an equivalent uncached decoder count is reported separately. V2 has 121,500 candidate
and 1,620 fallback decoder calls, totaling 123,120, plus six source replays.
Every action is retained regardless of early outcomes; no outcome-based early stop.

The diagonal S/CF1, M/CF2, L/CF3 subset still cannot support scale/CF claims.
V2 is a bounded cost/noise pilot, not a large training-label campaign or solver
comparison. If the primary passes, a separately frozen expanded development-data
and training protocol is needed. Otherwise retain `NGAS_A1_REVISE_LABELS` and diagnose
the failed mechanism without training. R13/R14 remain locked; no Gurobi or push.

Evidence: `outputs/ngas_a1/audit/label_pilot_completion.json` and
`configs/ngas_a1_label_pilot_v2.json`. The V2 protocol will record code, config, V1
raw-file and instance hashes before new labels are generated.

The 466-file V1 archive `outputs/ngas_a1/archive/label_pilot_v1.tar.gz` was verified
entry by entry against the original bytes. Its adjacent hash manifest supports
transfer and restoration; all original raw files remain in place. New unit tests
check cached/uncached label equivalence, matched trial prefixes, fresh CRN identities,
atomic recovery/corruption rejection, and refusal to rescue a failed primary with
a passing control. Full regression is required by V2 freeze before job launch.
