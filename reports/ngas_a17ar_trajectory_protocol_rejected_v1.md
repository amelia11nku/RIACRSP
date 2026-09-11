# NGAS A1.7A-R trajectory protocol revision 1 rejection

Protocol revision 1 had SHA256 `441b613936fc7f0ef59b3410dfbcd5135af3481e8b5ce43f8ce596766b1b0a73`.
The pre-instance CUDA smoke stopped before checkpoint loading, dataset access, ledger
append, or solver execution because PyTorch 2.11 rejected a string device argument
in `torch.cuda.reset_peak_memory_stats`.

No formal trajectory run started. The rejected manifest is preserved at
`artifacts/ngas_a17ar/trajectory_protocol_manifest_rejected_v1.json`. Revision 2
changes only the smoke-test device argument and records this supersession before a
new protocol freeze.
