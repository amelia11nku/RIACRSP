#!/usr/bin/env python3
"""Strictly load every checked-in RCIAS JSON instance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance


DEFAULT_PATHS = (
    Path("instances/tiny/tiny_01.json"),
    Path("instances/tiny/tiny_02.json"),
    Path("instances/tiny/tiny_03.json"),
    Path("instances/canonical/RCIAS-2.0/brandimarte/BR_Mk01.json"),
    Path("instances/canonical/RCIAS-2.0/hurink/edata/HU_E_la01.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RCIAS-2.0 JSON instances")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = tuple(args.paths) or DEFAULT_PATHS
    print("This run verifies schema, IDs, DAGs, configurations, processing domains, and travel matrices.")
    for path in paths:
        instance = load_instance(path)
        print(
            f"OK {path}: products={len(instance.products)} operations={len(instance.operations)} "
            f"islands={len(instance.islands)} W={len(instance.agvs_w)} F={len(instance.agvs_f)}"
        )


if __name__ == "__main__":
    main()
