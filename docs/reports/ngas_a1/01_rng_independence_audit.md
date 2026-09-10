# RNG independence audit

The new `RNGStreams` API derives seeds by SHA256 over the version, instance ID,
run seed, namespace, state ID and event index. Eight explicit namespaces isolate
size, target, repair, portfolio, neighbor, acceptance, fallback and continuation CRN.
Streams are fresh objects: callers retain them locally or use distinct event indices.
Selection mode and joint action ID are excluded from downstream CRN keys.

Tests cover unrelated draws, changed namespace keys, top-1/sampled matched actions,
repeated actual decode trajectories, and independent Python processes with different
PYTHONHASHSEED values. The shared repair primitive receives a canonical tuple in NGAS;
historical set-based comparator execution remains byte-for-byte unchanged.

Reproducibility means the same trajectory at a fixed sequence of iteration/evaluation
decisions. A wall-clock stop can complete different numbers of iterations under OS
scheduling noise; identical wall-clock trajectory lengths are not promised.
