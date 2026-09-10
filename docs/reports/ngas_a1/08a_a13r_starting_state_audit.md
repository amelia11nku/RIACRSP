# NGAS-A1.3R starting-state and provenance audit

Starting commit: `2fec6f365095385bd443f7e6481499a716e49049` with a clean worktree.
The prior compact A1.3 run is frozen historical R12 development evidence, not an
authorization for A1.4 under the A1.3R supplement. Its independent completion audit
passed after exact replay of all 216 seed/held-fold states; 3/3 training seeds passed
the old minimum gate. All 41 files under `outputs/ngas_a1/critic_training_gpu_v1/`
are preserved unchanged with tree-manifest SHA-256
`4239e9236cb5a5d9837def6acc70baa9155b32cb2282a85efb604a57a4b3a9f9`.

The revision reuses exactly the frozen V2 T8_R9 and expanded R12 DEVELOPMENT raw
labels. It may regenerate current-schedule CSG features but may not rerun continuation
rollouts. The V2 result manifest SHA-256 is
`a274642c9a62a2ff153a095f15d9bacb04b04a5609aff49774445cce1c4c3616`;
the expanded result manifest SHA-256 is
`44e3a3deb826fb38a0d81f58957efba05fb0ee78d77cefcab07c67cec14992c1`.
The old feature cache SHA-256 is
`b1673d86b44589735b7d01cfcae99096ff05206235c785475ccbeba28c111541`
and is reference-only because A1.3R changes the feature schema.

The current compact model source SHA-256 is
`a9b170b2b25b58ce565159e3c621df30e64a516b23544d62669816c9e54949ef`;
it must remain available as named C0. The current NGAS critical-sync source SHA-256
is `0e4abc70210cebf4203ab7734498a3609134ca682fe43e2fbf88ed271c75cb1b`.
The shared generalized CHDG source SHA-256 is
`dc535e424a28cacdae6075e8765f42d1aa2394c07e6816d9c5a3339db5560555`
and remains frozen comparator code; A1.3R will audit it read-only and port/wrap new
semantics inside `rcias_ngas` where necessary.

No NGAS R13/R14 access ledger exists. R13 and R14 remain locked. No Gurobi process
or result is part of A1.3R. Frozen Phase 6, ALNS and LG_HGA_2O semantics/results will
not be modified. The new namespace is
`outputs/ngas_a1/critic_training_rthgt_v2/`; all new reports use distinct filenames.
Only `NGAS_A1_3R_PASS_RTHGT` or `NGAS_A1_3R_PASS_COMPACT` can reopen preparation for
A1.4.
