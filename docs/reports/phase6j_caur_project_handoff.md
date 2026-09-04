# Phase 6J CAUR Project Handoff

## Current state

- Substage decision: `R12_PILOT_PASS`; no Phase 6J scientific final decision
  has been made.
- Selected model: none.
- CSG-NI v1 frozen: no.
- Phase 6H replacement authorized: none; Phase 6H remains the reference.
- R12 pilot accessed: yes, through the preregistered 27-state pilot only.
- R12 pilot completeness: `PASS` with 27/27 states, 640 unique
  state-candidate pairs, 3,840 paired-seed/horizon rows, and every integrity
  check true.
- Frozen continuation horizon: H=4. It is the shortest horizon satisfying the
  preregistered agreement rule against H=12 and was selected without solver
  outcome evidence.
- R13 content accessed: no; locked.
- R14 content accessed: no; locked.
- The full R12 collection implementation uses all 18 fit instances, two source
  trajectories per instance, eight states per trajectory, the true full bank,
  both CRN seeds, and the frozen horizon: 288 state lists in total.
- Phase 6J setup tests cover the access locks, CRN continuation target,
  H4/H8/H12 prefix equivalence, full-bank label completeness, outcome-blind
  candidate-source features, grouped OOF assignment, deterministic LCB gate,
  R12 collection authorization hashes, and the 36-trajectory task boundary.

## Frozen files

- `configs/phase6j_caur.json`
- `configs/phase6j_caur_command_manifest.json`
- `configs/phase6j_caur_phase6i_mr_evidence_manifest.json`
- `docs/reports/phase6j_caur_preregistered_protocol.md`
- `instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_instance_manifest.csv`

The authoritative hashes are recorded at the top of the preregistered
protocol. R11 evidence remains immutable and may not be used to adjust Phase
6J.

## Current gate and authorized next action

Host-level GPU verification passes on the RTX 4060 Ti using the `gnn311`
environment; the restricted command sandbox alone cannot see the driver. The
pilot projects approximately 0.49 million decoder evaluations and about 1.35
hours for the full H4 collection. Even the conservative H12 projection remains
below the preregistered eight-hour cap, so the cost fallback is not activated.

The exact authorized stage is the resumable full-bank R12 collection. First
freeze the pilot hashes, H4 decision, cost projection, and collection-script
hash; then run one host-GPU state as a smoke check and launch or resume exactly
one persistent worker. Runtime status is authoritative in
`outputs/phase6j_caur/r12_collection/progress.json`. Training begins only after
the 288-state integrity audit passes. R13 and R14 remain unauthorized.
