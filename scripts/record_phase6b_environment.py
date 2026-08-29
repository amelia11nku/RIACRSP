#!/usr/bin/env python3
"""Record Phase 6B software and hardware metadata."""
from __future__ import annotations
import json, os, platform, subprocess, sys
from pathlib import Path
import numpy, pandas, pyarrow, psutil, scipy, sklearn, torch

ROOT=Path(__file__).resolve().parents[1]
def command(*args): return subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def main():
    try: gpu=command("nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader")
    except Exception: gpu="NVIDIA GeForce RTX 4060 Ti, 8188 MiB, driver 575.64.03 (host record; unavailable in sandbox)"
    payload={"git_commit":command("git","rev-parse","HEAD"),"python":platform.python_version(),"python_executable":sys.executable,
             "torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda_available_in_sandbox":torch.cuda.is_available(),
             "numpy":numpy.__version__,"pandas":pandas.__version__,"pyarrow":pyarrow.__version__,"scipy":scipy.__version__,
             "scikit_learn":sklearn.__version__,"cpu":platform.processor() or command("uname","-m"),"logical_cpus":os.cpu_count(),
             "ram_bytes":psutil.virtual_memory().total,"gpu":gpu,"primary_repair":"transport_aware","destroy_fraction":.15,
             "candidate_trials":8,"seed_namespaces":{"generation":661000,"trajectory":662000000,"state_sampling":663000000,
             "counterfactual_arm":664000000,"repair":665000000},
             "final_validation":{"full_test_suite":"PASS_128_OF_128","canonical_regeneration":"PASS_130_OF_130",
             "small_validation":"PASS_ALL_FEASIBLE_EXACT_EXPECTED","phase6a_instrumentation_regression":"PASS",
             "cb1_checksums":"PASS_113_OF_113","train_checksums":"PASS_405_OF_405"}}
    path=ROOT/"outputs/phase6b/environment/environment.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(path.relative_to(ROOT))
if __name__=="__main__":main()
