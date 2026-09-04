#!/usr/bin/env python3
"""Run the complete P1 analysis/figure/report chain after 90/90 completion."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = ROOT / "paper_experiments/ablation"
PYTHON = Path(sys.executable)


def main() -> int:
    progress_path = ABLATION_ROOT / "raw_results/random_full_bank_frozen_gate/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") != "COMPLETE" or progress.get("completed_runs") != 90:
        raise RuntimeError(f"P1 is not complete: {progress.get('completed_runs')}/90")
    scripts = (
        ABLATION_ROOT / "analyze_p1_ablation.py",
        ABLATION_ROOT / "plot_p1_ablation.py",
        ABLATION_ROOT / "plot_p1_mechanism_diagnostics.py",
        ABLATION_ROOT / "build_extension_report.py",
    )
    for script in scripts:
        subprocess.run([str(PYTHON), str(script)], cwd=ROOT, check=True)
    for qa_name in (
        "figure_p1_ablation_effect_by_scale.qa.json",
        "supplementary_p1_mechanism_diagnostics.qa.json",
    ):
        qa = json.loads((ABLATION_ROOT / "figures" / qa_name).read_text(encoding="utf-8"))
        if not str(qa.get("status", "")).startswith("AUTOMATED_QA_PASS"):
            raise RuntimeError(f"P1 figure QA failed: {qa_name}")
    print("P1_EXTENSION_FINALIZATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
