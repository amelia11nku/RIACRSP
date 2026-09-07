#!/usr/bin/env python3
"""Real-clock tiny smoke of the budget adapter with audited frozen v2 DTRs."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.search.lghga import LGHGAConfig
from rcias_clgri.search.lghga_2o import operation_budget, solve_lghga_2o
from rcias_clgri.search.lghga_learning import load_dtr_bundle
from scripts.audit_phase6k_start import OUT, digest, write_once
from scripts.run_advanced_baseline_v2 import CONFIG_PATH, _validate_models, _verify_implementation
from scripts.run_advanced_baselines import _read_dataclass_config


def main():
    path = OUT / "lghga_2o_real_clock_smoke.json"
    if path.exists():
        raise RuntimeError("smoke exists; inspect it rather than replace it")
    model_root = ROOT / "outputs/baselines/lghga_kb_v2/models"
    implementation_hash = _verify_implementation()
    _, manifest = _validate_models(model_root)
    model_dir = model_root / "S_CF1"
    assert digest(model_dir / "model_manifest.json") == manifest["regimes"]["S_CF1"]["model_manifest_sha256"]
    bundle = load_dtr_bundle(model_dir)
    instance = load_instance(ROOT / "instances/tiny/tiny_01.json")
    config = _read_dataclass_config(CONFIG_PATH, LGHGAConfig)
    budget = operation_budget(instance)
    start = time.perf_counter()
    result = solve_lghga_2o(instance, budget, 696101, bundle, config)
    outer_seconds = time.perf_counter() - start
    feasible = check_schedule(instance, result.best.schedule)["feasible"]
    trace = result.convergence_trace
    monotone = all(a.elapsed_time <= b.elapsed_time and a.decoder_evaluations <= b.decoder_evaluations
                   and a.current_best_makespan >= b.current_best_makespan for a, b in zip(trace, trace[1:]))
    assert feasible and monotone and budget <= result.runtime <= outer_seconds
    record = {"status": "PASS", "scope": "TINY_SMOKE_ONLY_NOT_SOLVER_COMPARISON",
              "method": result.method, "budget_seconds": budget, "runtime_seconds": result.runtime,
              "outer_seconds": outer_seconds, "overshoot_seconds": result.runtime - budget,
              "generations": result.iterations, "decoder_evaluations": result.decoder_evaluations,
              "feasible": feasible, "monotone_trace": monotone, "actions": [asdict(x) for x in result.best.actions],
              "config": asdict(config), "implementation_manifest_sha256": implementation_hash,
              "adapter_sha256": digest(ROOT / "rcias_clgri/search/lghga_2o.py"),
              "model_hashes": dict(bundle.model_hashes), "r13_accessed": False, "r14_accessed": False}
    write_once(path, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("actions", "config", "model_hashes")}))


if __name__ == "__main__":
    main()
