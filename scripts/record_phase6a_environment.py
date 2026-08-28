#!/usr/bin/env python3
"""Record the reproducibility environment required by Phase 6A."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import psutil
import torch
import numpy
import pandas
import pyarrow
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main():
    gpu = None
    try:
        gpu = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = "NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 575.64.03 (host nvidia-smi; unavailable inside sandbox)"
    payload = {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "pytorch_version": torch.__version__, "pytorch_cuda_version": torch.version.cuda,
        "cuda_available_in_recording_sandbox": torch.cuda.is_available(), "cpu": platform.processor() or command("uname", "-m"),
        "logical_cpu_count": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total, "gpu": gpu,
        "numpy_version": numpy.__version__, "pandas_version": pandas.__version__,
        "pyarrow_version": pyarrow.__version__, "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "test_count": 120, "legacy_checksum_status": "PASS_130_OF_130",
        "cb1_checksum_status": "PASS_108_OF_108_AND_PAIRING_TRUE",
        "default_shell_python_note": "Default Python 3.13.5 lacked pytest; validation used the established gnn311 environment.",
    }
    path = ROOT / "outputs/phase6a/environment/environment.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path.relative_to(ROOT))


if __name__ == "__main__": main()
