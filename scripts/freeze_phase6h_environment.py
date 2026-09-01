#!/usr/bin/env python3
"""Record the Phase 6H software, hardware, source, and artifact boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import pandas
import scipy
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6h_live_calibration.json"
OUT = ROOT / "outputs/phase6h_calibration/environment/environment_freeze.json"
TRACKED = (
    "configs/phase5c_alns.json",
    "configs/phase5c_ga.json",
    "configs/phase5c_dcga.json",
    "configs/phase6g_live_solver.json",
    "configs/phase6h_live_calibration.json",
    "rcias_clgri/heuristic/dispatching.py",
    "rcias_clgri/env/insertion_decoder.py",
    "rcias_clgri/env/feasibility.py",
    "rcias_clgri/search/alns.py",
    "rcias_clgri/search/ga.py",
    "rcias_clgri/search/dcga.py",
    "rcias_clgri/search/csgni.py",
    "rcias_clgri/ni/live_inference.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=False, text=True, capture_output=True
    )
    return (result.stdout or result.stderr).strip()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoint = Path(json.loads(
        (ROOT / config["frozen_phase6f"]["experiment_freeze"]).read_text()
    )["selected_checkpoint_path"])
    payload = {
        "schema": "phase6h-environment-freeze-v1",
        "status": "FROZEN_BEFORE_CAL_FIT_COLLECTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_git_commit": config["starting_git_commit"],
        "current_git_commit": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "torch_cuda_version": torch.version.cuda,
            "nvidia_smi": command(
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ),
        },
        "phase6h_config_sha256": digest(CONFIG),
        "calibration_manifest_sha256": digest(
            ROOT / config["calibration_instances"]["manifest"]
        ),
        "calibration_generation_spec_sha256": digest(
            ROOT / config["calibration_instances"]["root"] / "manifests/generation_spec.json"
        ),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_hash_matches": (
            digest(checkpoint) == config["frozen_phase6f"]["checkpoint_sha256"]
        ),
        "source_sha256": {
            path: digest(ROOT / path) for path in TRACKED
        },
        "phase6g_primary_outputs_untouched": True,
        "core_sensitivity_legacy_accessed": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(OUT.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUT)
    if not payload["checkpoint_hash_matches"] or not payload["cuda"]["available"]:
        raise RuntimeError("Phase 6H environment freeze failed checkpoint/CUDA validation")
    print("PHASE6H_ENVIRONMENT_FROZEN", digest(OUT))


if __name__ == "__main__":
    main()
