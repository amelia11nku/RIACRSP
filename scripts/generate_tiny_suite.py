#!/usr/bin/env python3
"""Freeze the two deterministic Phase 2 tiny validation instances."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_automotive_semantic import build_tiny_instance as build_automotive_tiny
from generate_fjsp_reconfigurable import build_tiny_instance as build_fjsp_tiny
from rcias_clgri.data.generation import finalize_instance, write_json


def build_multi_vehicle_tiny(seed: int = 23) -> dict:
    """Build the four-island/two-fleet native-solver validation instance.

    Each operation has one fixed eligible island and each island processes one
    operation. This isolates W/F routing and synchronization so the Gurobi and
    CP-SAT formulations can be independently audited against decoder replay.
    """

    products = {
        "J1": {"operations": ["o11", "o12"], "precedence": [["o11", "o12"]]},
        "J2": {"operations": ["o21", "o22"], "precedence": [["o21", "o22"]]},
    }
    operations = {
        "o11": {
            "product": "J1", "required_configuration": "C1",
            "eligible_islands": ["M1"], "processing_time": {"M1": 8},
        },
        "o12": {
            "product": "J1", "required_configuration": "C2",
            "eligible_islands": ["M2"], "processing_time": {"M2": 10},
        },
        "o21": {
            "product": "J2", "required_configuration": "C1",
            "eligible_islands": ["M3"], "processing_time": {"M3": 7},
        },
        "o22": {
            "product": "J2", "required_configuration": "C3",
            "eligible_islands": ["M4"], "processing_time": {"M4": 9},
        },
    }
    islands = {
        "M1": {"supported_configurations": ["C1", "C2", "C3"], "initial_configuration": "C2"},
        "M2": {"supported_configurations": ["C1", "C2", "C3"], "initial_configuration": "C1"},
        "M3": {"supported_configurations": ["C1", "C2", "C3"], "initial_configuration": "C1"},
        "M4": {"supported_configurations": ["C1", "C2", "C3"], "initial_configuration": "C2"},
    }
    coordinates = {
        "WH": (0, 0), "M1": (4, 3), "M2": (10, 3),
        "M3": (4, 10), "M4": (10, 10),
    }
    return finalize_instance(
        instance_id="tiny_03",
        generator="hand-auditable multi-vehicle RCIAS-2.0",
        seed=seed,
        products=products,
        operations=operations,
        islands=islands,
        configurations=("C1", "C2", "C3"),
        agvs_w=("W1", "W2"),
        agvs_f=("F1", "F2"),
        coordinates=coordinates,
        rng=random.Random(seed),
        extra_meta={
            "purpose": "Gurobi MILP versus CP-SAT exact validation",
            "native_solver_profile": "fixed-operation-island assignment",
        },
    )


def main() -> None:
    target = ROOT / "instances" / "tiny"
    target.mkdir(parents=True, exist_ok=True)
    tiny_01 = build_automotive_tiny(seed=11)
    tiny_01["meta"]["instance_id"] = "tiny_01"
    tiny_01["meta"]["purpose"] = "minimum exact and resource-timeline validation"
    tiny_02 = build_fjsp_tiny(seed=7)
    tiny_02["meta"]["instance_id"] = "tiny_02"
    tiny_02["meta"]["purpose"] = "boundary logic and alternate generator validation"
    tiny_03 = build_multi_vehicle_tiny(seed=23)
    write_json(tiny_01, target / "tiny_01.json")
    write_json(tiny_02, target / "tiny_02.json")
    write_json(tiny_03, target / "tiny_03.json")
    (target / "README.md").write_text(
        "# Frozen tiny validation suite\n\n"
        "- `tiny_01.json`: two 3-operation automotive products, nonlinear DAGs, one W and one F vehicle. "
        "This is the exact/replay/CSV/Gantt reference instance.\n"
        "- `tiny_02.json`: five-operation FJSP-family boundary instance with two islands and one W/F vehicle.\n\n"
        "- `tiny_03.json`: four operations on four fixed islands with two W-AGVs and two F-AGVs; "
        "used for native Gurobi MILP versus CP-SAT exact/replay comparison.\n\n"
        "Both use fixed seeds and the unchanged RCIAS F-kit semantics. Regenerate only with "
        "`python scripts/generate_tiny_suite.py`.\n",
        encoding="utf-8",
    )
    print("Frozen instances/tiny/tiny_01.json, tiny_02.json, and tiny_03.json")


if __name__ == "__main__":
    main()
