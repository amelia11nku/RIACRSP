# NGAS-A1 starting-state audit

Starting commit: `156b017b1651e803f5a6bd4894e2713cfec1cc9d`. Origin: `https://github.com/amelia11nku/RIACRSP.git`.
The working tree was clean before this audit script was added.

Environment: `/home/liulei/miniconda3/envs/gnn311/bin/python`, Python 3.11.15, PyTorch 2.11.0+cu128,
CUDA build 12.8, CUDA available: False.
CPU is sufficient for infrastructure and the bounded label pilot; GPU training is not qualified.

Canonical manifest SHA256: `5fb1d78201fd62dcc64fb1db4bbb5de9cffa11bed07bbd5c890a71c8e130b8af`. 18 R12 DEVELOPMENT instances;
seeds [746101, 746102, 746103]. All four comparators have 54 frozen raw runs.
Verified 355 files; mismatches: [].
Exact implementation commits, source/config/checkpoint identifiers, raw run manifests,
and retained hashes are in `outputs/ngas_a1/audit/starting_state.json`.

Phase 6P is terminal `ADAPTIVE_PORTFOLIO_PILOT_NO_GO`. No historical implementation
or raw result is to be changed. R13/R14 remain locked; no Gurobi work is authorized.

The historical BKS was computed from the 270 Phase 6P development/comparator raw runs.
NGAS will materialize that exact scope as immutable BKS v001 and separate raw provenance
from derived RPD. Historical anytime decoder counts come from incumbent events;
the makespan curve remains useful but the work-count column is not a budget counter.

Execution order: A1.0 infrastructure tests; A1.1 critical-sync/bank validation;
A1.2 preregistered bounded continuation pilot; training only after the pilot gate passes.
No solver-quality comparison before the RNG gate, and no large labeling before the pilot.

Qualification supplement: the actual Phase 6H/Phase 6F checkpoint was also verified
against both its experiment freeze and the Phase 6H policy. Its SHA256 is
`f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7`;
the exact checkpoint path and freeze hash are retained in
`outputs/ngas_a1/audit/infrastructure_bank_gate.json`.
