# NGAS A1.7A-R trajectory protocol revision 2 rejection

Protocol revision 2 had SHA256 `aa79611757f037394763755f15a56c8d0fbf87089e1134c414ee4f6e2e83d1ce`.
The pre-instance CUDA smoke again stopped before checkpoint loading, dataset access,
ledger append, or solver execution. PyTorch's peak-memory reset requires its CUDA
caching allocator to be initialized first in this environment.

No formal trajectory run started. The rejected manifest is preserved at
`artifacts/ngas_a17ar/trajectory_protocol_manifest_rejected_v2.json`. Revision 3
initializes the allocator with a one-element CUDA tensor before resetting peak
statistics, as verified by an isolated CUDA call.
