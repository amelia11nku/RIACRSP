# Frozen 2|O| baseline registry

The canonical registry has been initialized for the 18 frozen R12 CAUR-FIT instances, development seeds 746101–746103, and a wall-clock budget of `2 * num_operations` seconds with initialization and decoder work included. The machine, dependency, instance, config, checkpoint, and relevant source hashes are frozen under `outputs/frozen_2o_baselines/`.

No completed historical run satisfies the full canonical contract. All four Phase 6P comparators are currently `RERUN_REQUIRED`; entries remain pending until their complete three-seed results pass feasibility and integrity checks. Future phases must reuse a resulting `FROZEN_CANONICAL` entry when all ten compatibility conditions match, and may extend only missing seeds when the existing seed list is a strict valid subset.

Valid results may not be rerun based on whether a new method performs well or poorly. Superseded entries must remain preserved with an explicit reason and link.
