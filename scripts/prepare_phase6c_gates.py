#!/usr/bin/env python3
"""Create outcome-blind 1k and 5k production-protocol gate manifests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6c import candidate_sha256
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json, sha256_file
from rcias_clgri.search.counterfactual import stable_seed

SOURCE = ROOT / "outputs/phase6b/trajectory_reservoir/pilot_state_manifest.parquet"
TRAIN = ROOT / "instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv"
OUT = ROOT / "outputs/phase6c/gates"
CELL = ["scale", "CF_level", "RI_level", "TI_level"]
STAGES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")


def allocate(total: int, values: list[tuple]) -> dict:
    base, remainder = divmod(total, len(values))
    return {value: base + (index < remainder) for index, value in enumerate(values)}


def select(states: pd.DataFrame, total: int) -> pd.DataFrame:
    cells = sorted(map(tuple, states[CELL].drop_duplicates().to_numpy()))
    selected = []
    for cell, quota in allocate(total, cells).items():
        mask = True
        for column, value in zip(CELL, cell):
            mask &= states[column] == value
        part = states[mask]
        stage_quotas = allocate(quota, list(STAGES))
        chosen_ids = set()
        pieces = []
        for stage, stage_quota in stage_quotas.items():
            available = part[part.search_stage == stage].copy()
            available["_priority"] = available.state_id.map(
                lambda value: stable_seed(value, f"phase6c_gate_{total}", namespace=666100000)
            )
            chosen = available.sort_values(["_priority", "state_id"]).head(stage_quota)
            pieces.append(chosen.drop(columns="_priority"))
            chosen_ids.update(chosen.state_id)
        cell_rows = pd.concat(pieces, ignore_index=True)
        if len(cell_rows) < quota:
            remaining = part[~part.state_id.isin(chosen_ids)].copy()
            remaining["_priority"] = remaining.state_id.map(
                lambda value: stable_seed(value, f"phase6c_gate_{total}_fill", namespace=666100000)
            )
            cell_rows = pd.concat([
                cell_rows,
                remaining.sort_values(["_priority", "state_id"]).head(quota - len(cell_rows)).drop(columns="_priority"),
            ], ignore_index=True)
        if len(cell_rows) != quota:
            raise RuntimeError(f"insufficient Phase 6B states for cell {cell}")
        selected.append(cell_rows)
    result = pd.concat(selected, ignore_index=True).sort_values("state_id").reset_index(drop=True)
    if len(result) != total or result.state_id.nunique() != total:
        raise RuntimeError("gate state selection failed")
    return result


def main():
    states = pd.read_parquet(SOURCE)
    manifest = pd.read_csv(TRAIN).set_index("instance_id")
    states["instance_relative_path"] = states.instance_id.map(manifest.relative_path)
    states["replicate"] = states.instance_id.map(manifest.replicate)
    states["trajectory_run"] = 0
    states["trajectory_seed"] = states.seed
    states["state_sampling_seed"] = states.instance_id.map(manifest.state_sampling_seed)
    states["candidate_sha256"] = states.current_candidate.map(candidate_sha256)
    for name, total in (("gate_a", 1000), ("gate_b", 5000)):
        target = OUT / name
        selected = select(states, total)
        atomic_write_csv(selected, target / "state_manifest.csv")
        record = {
            "schema": "phase6c-scale-gate-state-selection-v1",
            "gate": name,
            "state_count": total,
            "state_manifest_sha256": sha256_file(target / "state_manifest.csv"),
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": sha256_file(SOURCE),
            "selection_is_outcome_blind": True,
            "production_protocol_config": "configs/phase6c_counterfactual.json",
        }
        atomic_write_json(record, target / "selection_manifest.json")
        print(name, total, selected[CELL].drop_duplicates().shape[0])


if __name__ == "__main__":
    main()
