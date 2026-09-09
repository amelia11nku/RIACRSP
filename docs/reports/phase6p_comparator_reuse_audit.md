# Phase 6P comparator reuse audit

The audit completed before any development comparator launch. No canonical `outputs/frozen_2o_baselines/` registry existed beforehand.

| Comparator | Action | Blocking mismatch |
| --- | --- | --- |
| ALNS | `RERUN_REQUIRED` | no canonical frozen_2o registry result exists; historical 2|O| ALNS validation used CAL R08 instances and seeds 671301-671305; historical R12 collection used different seeds and a 15.25 s budget for 61 operations, not 2|O|=122 s |
| PHASE6H | `RERUN_REQUIRED` | no canonical frozen_2o registry result exists; Phase 6H validation used CAL R08 instances and seeds 671301-671305; Phase 6I R11 evidence used different LIVE_REV instances and seeds 681401-681405 |
| PHASE6N_TOP1 | `RERUN_REQUIRED` | no prior live deterministic top-1 solver result exists; the comparator was first implemented after Phase 6P preregistration |
| LG_HGA_2O | `RERUN_REQUIRED` | no canonical frozen_2o registry result exists; the only 2|O| evidence is a 12 s tiny_01 smoke; the smoke used seed 696101 rather than the Phase 6P development seeds |

The existing ALNS R12 collection used different seeds and a non-2|O| collection budget. Phase 6H evidence uses CAL R08 or LIVE_REV R11 instances. Phase 6N top-1 has no prior live solver run. LG_HGA-2O has only a `tiny_01` real-clock smoke. None can be promoted by relabeling.

All four comparators may now run once under the canonical development protocol. Their existing evidence remains preserved and will not be overwritten.
