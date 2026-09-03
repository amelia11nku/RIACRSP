#!/usr/bin/env python3
"""Wait for Phase6H Core GPU collection, then run final-exact CSG-NI alone."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CORE_PROGRESS = (
    ROOT / "paper_experiments/raw_results/core45"
    / "CSG_NI_PROVISIONAL_PHASE6H/progress.json"
)
CORE_UNITS = (
    "paper-phase6h-core-5seed-s0.service",
    "paper-phase6h-core-5seed-s1.service",
)
EXACT_RUNNER = ROOT / "paper_experiments/scripts/run_final_exact_heuristics.py"


def units_active() -> bool:
    return any(
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit], check=False
        ).returncode == 0
        for unit in CORE_UNITS
    )


def main() -> None:
    while True:
        progress = json.loads(CORE_PROGRESS.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_runs", 0))
        total = int(progress.get("total_runs", 225))
        print(
            f"WAIT_PHASE6H_CORE status={progress.get('status')} "
            f"completed={completed}/{total}",
            flush=True,
        )
        if progress.get("status") == "COMPLETE" and completed == total == 225:
            break
        if not units_active():
            raise RuntimeError(
                "Phase6H Core stopped before its 225-run completion gate"
            )
        time.sleep(60)
    command = [
        sys.executable,
        str(EXACT_RUNNER),
        "--methods",
        "CSG-NI Phase6H provisional",
        "--workers",
        "1",
        "--device",
        "cuda",
    ]
    print("PHASE6H_CORE_COMPLETE_START_FINAL_EXACT_CSGNI", flush=True)
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
