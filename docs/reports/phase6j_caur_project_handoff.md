# Phase 6J CAUR Project Handoff

## Current state

- Substage decision: `BOUNDED_SETUP_IN_PROGRESS`; no Phase 6J scientific final
  decision has been made.
- Selected model: none.
- CSG-NI v1 frozen: no.
- Phase 6H replacement authorized: none; Phase 6H remains the reference.
- R12 pilot accessed: no.
- R13 content accessed: no; locked.
- R14 content accessed: no; locked.
- Starting boundary, historical evidence hashes, fresh split generation, and
  split-integrity checks pass.
- Phase 6J setup tests cover the access locks, CRN continuation target,
  H4/H8/H12 prefix equivalence, full-bank label completeness, outcome-blind
  candidate-source features, grouped OOF assignment, and deterministic LCB gate.

## Frozen files

- `configs/phase6j_caur.json`
- `configs/phase6j_caur_command_manifest.json`
- `configs/phase6j_caur_phase6i_mr_evidence_manifest.json`
- `docs/reports/phase6j_caur_preregistered_protocol.md`
- `instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_instance_manifest.csv`

The authoritative hashes are recorded at the top of the preregistered
protocol. R11 evidence remains immutable and may not be used to adjust Phase
6J.

## Current blocker and authorized next action

Host-level GPU verification now passes on the RTX 4060 Ti using the `gnn311`
environment; the restricted command sandbox alone cannot see the driver. The
R12 pilot collector, supervisor, and persistent launcher are implemented and
must pass the final full regression plus a one-state host-GPU smoke before the
remaining worker is launched.

The exact authorized next action is to implement and validate only the R12
horizon/full-bank pilot collector. After a device choice passes runtime
preflight, launch the single R12 pilot worker, verify the first state and
replace the 20–35 minute provisional ETA with measured throughput. R13/R14 and
full R12 collection remain unauthorized.
