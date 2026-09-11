# NGAS A1.7A-R state-overlap audit

The A1.6R artifacts retain 18 replayable trajectory snapshots: three progress points for each of six R12 instances, all under seed `746102`. Each has one matching C1 `NATIVE16_746102` training state by instance and seed.

- Exact state matches supported by the full fingerprint contract: **0**.
- Same candidate/schedule structure with incomplete runtime metadata: **0**.
- Same instance and seed but non-identical candidate state: **18**.
- No recoverable pair among the retained snapshots: **0**.

All 18 recoverable comparisons have different candidate-assignment fingerprints. The evidence therefore supports same-instance and same-seed trajectory provenance, but does not support an exact state-duplication claim. This conservative state-level result does not alter the instance-level conclusion: all 18 R12 instances were exposed during final C1 fitting.
