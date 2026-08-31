"""Phase 6E state/action records and deterministic Phase 6C loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch

from rcias_clgri.csg import build_csg, project_target_set
from rcias_clgri.data.phase6c import reconstruct_state

from .tensorize import CSGTensorizer, NITensorGraph


PHASE6C_DATASET_SPLIT_DIRECTORIES = {
    "TRAIN": "train",
    "TRAIN_VALIDATION": "validation",
    "TRAIN_INTERNAL_HOLDOUT": "internal_holdout",
    "REVISION_HOLDOUT": "revision_holdout",
}
LABEL_COLUMNS = (
    "mean_relative_improvement",
    "rank_within_state",
    "rank_percentile",
    "regret_to_best",
    "top1",
    "top3",
)
ACTION_METADATA_COLUMNS = (
    "arm_family",
    "origin_destroy_operator",
    "origin_rules",
    "origin_families",
)
STRUCTURAL_METADATA_COLUMNS = (
    "training_split",
    "scale",
    "CF_level",
    "RI_level",
    "TI_level",
    "search_stage",
    "bottleneck_proxy",
)


@dataclass(frozen=True)
class NIActionSet:
    """All frozen candidate target sets for one state.

    Target IDs and origin metadata are retained for audit/baselines only. They are
    never represented as predictive tensors.
    """

    target_set_ids: tuple[str, ...]
    target_operation_indices: torch.Tensor
    target_action_index: torch.Tensor
    action_ptr: torch.Tensor
    utility: torch.Tensor
    positive: torch.Tensor
    rank_within_state: torch.Tensor
    rank_percentile: torch.Tensor
    regret_to_best: torch.Tensor
    top1: torch.Tensor
    top3: torch.Tensor
    arm_family: tuple[str, ...]
    origin_destroy_operator: tuple[str, ...]
    origin_rules: tuple[str, ...]
    origin_families: tuple[str, ...]

    @property
    def action_count(self) -> int:
        return len(self.target_set_ids)

    @property
    def membership_count(self) -> int:
        return int(self.target_operation_indices.numel())

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "NIActionSet":
        values = {}
        for name in (
            "target_operation_indices", "target_action_index", "action_ptr", "utility",
            "positive", "rank_within_state", "rank_percentile", "regret_to_best",
            "top1", "top3",
        ):
            values[name] = getattr(self, name).to(device, non_blocking=non_blocking)
        return NIActionSet(
            target_set_ids=self.target_set_ids,
            arm_family=self.arm_family,
            origin_destroy_operator=self.origin_destroy_operator,
            origin_rules=self.origin_rules,
            origin_families=self.origin_families,
            **values,
        )


@dataclass(frozen=True)
class NIStateSample:
    graph: NITensorGraph
    actions: NIActionSet
    structural_metadata: Mapping[str, str]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "NIStateSample":
        return NIStateSample(
            self.graph.to(device, non_blocking=non_blocking),
            self.actions.to(device, non_blocking=non_blocking),
            self.structural_metadata,
        )


def _records(values: pd.DataFrame | Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    if isinstance(values, pd.DataFrame):
        return values.to_dict("records")
    return list(values)


def tensorize_action_records(
    graph,
    values: pd.DataFrame | Iterable[Mapping[str, object]],
) -> NIActionSet:
    """Project all actions without using outcome fields as model inputs."""
    rows = _records(values)
    if not rows:
        raise ValueError(f"state {graph.state_id} has no candidate target sets")
    if len({str(row["target_set_id"]) for row in rows}) != len(rows):
        raise ValueError(f"state {graph.state_id} contains duplicate target_set_id values")

    target_indices: list[int] = []
    target_action_index: list[int] = []
    action_ptr = [0]
    for action_index, row in enumerate(rows):
        if str(row["state_id"]) != graph.state_id:
            raise ValueError("candidate target set belongs to a different state")
        raw_operations = json.loads(str(row["destroyed_operation_ids"]))
        view = project_target_set(
            graph,
            str(row["target_set_id"]),
            raw_operations,
            {name: row[name] for name in ACTION_METADATA_COLUMNS if name in row},
        )
        target_indices.extend(view.target_operation_node_indices)
        target_action_index.extend([action_index] * len(view.target_operation_node_indices))
        action_ptr.append(len(target_indices))

    utility = torch.tensor(
        [float(row["mean_relative_improvement"]) for row in rows], dtype=torch.float32
    )
    return NIActionSet(
        target_set_ids=tuple(str(row["target_set_id"]) for row in rows),
        target_operation_indices=torch.tensor(target_indices, dtype=torch.long),
        target_action_index=torch.tensor(target_action_index, dtype=torch.long),
        action_ptr=torch.tensor(action_ptr, dtype=torch.long),
        utility=utility,
        positive=(utility > 0).to(torch.float32),
        rank_within_state=torch.tensor(
            [float(row["rank_within_state"]) for row in rows], dtype=torch.float32
        ),
        rank_percentile=torch.tensor(
            [float(row["rank_percentile"]) for row in rows], dtype=torch.float32
        ),
        regret_to_best=torch.tensor(
            [float(row["regret_to_best"]) for row in rows], dtype=torch.float32
        ),
        top1=torch.tensor([bool(row["top1"]) for row in rows], dtype=torch.bool),
        top3=torch.tensor([bool(row["top3"]) for row in rows], dtype=torch.bool),
        arm_family=tuple(str(row.get("arm_family", "")) for row in rows),
        origin_destroy_operator=tuple(
            str(row.get("origin_destroy_operator", "")) for row in rows
        ),
        origin_rules=tuple(str(row.get("origin_rules", "")) for row in rows),
        origin_families=tuple(str(row.get("origin_families", "")) for row in rows),
    )


def build_state_sample(
    state_record: Mapping[str, object],
    action_records: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    train_root: Path,
    tensorizer: CSGTensorizer,
) -> NIStateSample:
    reconstructed = reconstruct_state(state_record, train_root)
    graph = build_csg(reconstructed, state_record)
    actions = tensorize_action_records(graph, action_records)
    structural_metadata = {
        name: str(state_record.get(name, "")) for name in STRUCTURAL_METADATA_COLUMNS
    }
    return NIStateSample(tensorizer.tensorize(graph), actions, structural_metadata)


def shard_source_paths(
    dataset_root: Path,
    instance_id: str,
    training_split: str,
) -> tuple[Path, Path]:
    try:
        split_directory = PHASE6C_DATASET_SPLIT_DIRECTORIES[training_split]
    except KeyError as error:
        raise ValueError(f"unknown frozen training split: {training_split}") from error
    directory = dataset_root / split_directory / instance_id
    return directory / "states.parquet", directory / "target_set_aggregates.parquet"


def load_shard_frames(
    dataset_root: Path,
    instance_id: str,
    training_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states_path, actions_path = shard_source_paths(dataset_root, instance_id, training_split)
    states = pd.read_parquet(states_path)
    actions = pd.read_parquet(actions_path)
    if set(states["state_id"]) != set(actions["state_id"]):
        raise ValueError(f"state/action coverage mismatch in {instance_id}")
    if states["state_id"].duplicated().any():
        raise ValueError(f"duplicate state_id in {instance_id}")
    return states, actions


def iter_shard_samples(
    states: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    train_root: Path,
    tensorizer: CSGTensorizer,
) -> Iterable[NIStateSample]:
    grouped = {state_id: frame for state_id, frame in actions.groupby("state_id", sort=False)}
    for state_record in states.to_dict("records"):
        state_id = str(state_record["state_id"])
        if state_id not in grouped:
            raise ValueError(f"missing target sets for {state_id}")
        yield build_state_sample(
            state_record,
            grouped[state_id],
            train_root=train_root,
            tensorizer=tensorizer,
        )
