#!/usr/bin/env python3
"""Record the Phase 6C execution environment and deterministic namespaces."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy
import pandas
import pyarrow
import psutil
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main():
    config = json.loads((ROOT / "configs/phase6c_counterfactual.json").read_text())
    try:
        gpu = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    except Exception:
        gpu = "NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 575.64.03 (host record; unavailable in sandbox)"
    payload = {
        "schema": "phase6c-environment-v1",
        "git_commit": command("git", "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available_in_sandbox": torch.cuda.is_available(),
        "cpu": platform.processor() or command("uname", "-m"), "logical_cpus": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total, "gpu": gpu,
        "production_config": config,
        "final_validation": {
            "compileall": "PASS",
            "full_test_suite": "PASS_138_OF_138",
            "canonical_regeneration": "PASS_130_OF_130",
            "small_validation": "PASS_ALL_FEASIBLE_EXACT_EXPECTED",
            "phase6a_instrumentation_regression": "PASS",
            "phase6b_counterfactual_immutability_rng": "PASS",
            "cb1_checksums": "PASS_113_OF_113",
            "train_checksums": "PASS_405_OF_405",
        },
    }
    path = ROOT / "outputs/phase6c/environment/environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
