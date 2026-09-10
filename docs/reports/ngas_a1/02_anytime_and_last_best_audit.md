# Anytime and last-best audit

Confirmed in `scripts/finalize_phase6p_development.py` lines 387–405: each checkpoint
selects the last incumbent trace event and copies its decoder count. The historical
best curve is valid after initialization, but the work-count field is a last-best
count. Before initialization, the old fallback to trace[0] also backdates a future
initial solution. NGAS represents the not-yet-initialized solution as null.

The old derived table and all raw evidence are preserved. A deprecation entry for
its `mean_decoder_evaluations` column is recorded in the new raw registry.

NGAS telemetry separates best events, budget checkpoints, and termination. Callers
must observe every completed decoder/critic call and every state transition. When a
call straddles a checkpoint, only work completed at that checkpoint is counted.
No later best or work count is backdated. Exact-deadline events are included.
An overrun's termination count remains separate from the 100% budget count.
An early stop does not fabricate later checkpoints. Last-best time/evals/iteration
are retained independently of the end-of-run counters.

Regression: the 75% and 100% makespans remain 90, decoder counts increase 30→70,
and last-best decoder count stays 30. Separate tests cover initialization and
atomic-call overrun boundaries.
