"""JSON instance loader for SOFJSPT benchmark data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
INST_DIR = DATA_DIR / "instances"


class Instance:
    """Container for one scheduling benchmark instance.

    The lowercase attributes are used by the evolutionary algorithms. Uppercase
    aliases are kept for older MILP/CPLEX scripts that used module constants.
    """

    def __init__(
        self,
        name: str,
        num_jobs: int,
        num_machines: int,
        num_operations: int,
        num_agvs_w: int,
        num_agvs_f: int,
        job_operations: dict[int, int],
        processing_times: dict[str, dict[int, float]],
        agv_w_transport_times: dict[str, dict[str, float]],
        agv_f_transport_times: dict[str, dict[str, float]],
        priority_dict: dict[str, list[str]],
        metadata: dict[str, Any] | None = None,
        distances: dict[str, dict[str, float]] | None = None,
    ):
        self.name = name
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = num_operations
        self.num_agvs_w = num_agvs_w
        self.num_agvs_f = num_agvs_f
        self.job_operations = job_operations
        self.processing_times = processing_times
        self.agv_w_transport_times = agv_w_transport_times
        self.agv_f_transport_times = agv_f_transport_times
        self.priority_dict = priority_dict
        self.metadata = metadata or {}
        self.distances = distances or {}

        self.NUM_JOBS = self.num_jobs
        self.NUM_MACHINES = self.num_machines
        self.NUM_OPERATIONS = self.num_operations
        self.NUM_AGVS_W = self.num_agvs_w
        self.NUM_AGVS_F = self.num_agvs_f
        self.JOB_OPERATIONS = self.job_operations
        self.PROCESSING_TIMES = self.processing_times
        self.AGV_W_TRANSPORT_TIMES = self.agv_w_transport_times
        self.AGV_F_TRANSPORT_TIMES = self.agv_f_transport_times
        self.PRIORITY_DICT = self.priority_dict


def _to_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _nested_numeric_dict(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(src): {str(dst): _to_number(val) for dst, val in dsts.items()}
        for src, dsts in data.items()
    }


def _resolve_instance_path(name: str) -> Path:
    requested = Path(name)
    if requested.suffix.lower() == ".json" and requested.exists():
        return requested

    stem = requested.stem if requested.suffix else str(name)
    direct = INST_DIR / f"{stem}.json"
    if direct.exists():
        return direct

    lowered = stem.lower()
    for path in INST_DIR.glob("*.json"):
        if path.stem.lower() == lowered:
            return path

    available = ", ".join(list_instances())
    raise FileNotFoundError(
        f"Instance '{name}' was not found in {INST_DIR}. Available instances: {available}"
    )


def _instance_from_json(path: Path) -> Instance:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        base = raw["base_info"]
        metadata = raw.get("metadata", {})
        name = metadata.get("instance_name", path.stem)
        job_operations = {int(job): int(count) for job, count in raw["product_operations"].items()}
        processing_times = {
            str(op): {int(machine): _to_number(time) for machine, time in machines.items()}
            for op, machines in raw["processing_times"].items()
        }
        agv_w_transport_times = _nested_numeric_dict(raw["agv_w_transport_times"])
        agv_f_transport_times = _nested_numeric_dict(raw["agv_f_transport_times"])
        priority_dict = {
            str(op): [str(pred) for pred in preds]
            for op, preds in raw.get("priority_dict", {}).items()
        }
    except KeyError as exc:
        raise KeyError(f"Missing required key {exc!s} in JSON instance {path}") from exc

    return Instance(
        name=name,
        num_jobs=int(base["num_products"]),
        num_machines=int(base["num_units"]),
        num_operations=int(base["num_operations"]),
        num_agvs_w=int(base["num_agvs_w"]),
        num_agvs_f=int(base["num_agvs_f"]),
        job_operations=job_operations,
        processing_times=processing_times,
        agv_w_transport_times=agv_w_transport_times,
        agv_f_transport_times=agv_f_transport_times,
        priority_dict=priority_dict,
        metadata=metadata,
        distances=_nested_numeric_dict(raw.get("distances", {})),
    )


def load_instance(name: str) -> Instance:
    """Load one instance from data/inst/*.json.

    The lookup is case-insensitive, so both 'behnke05' and 'Behnke05' resolve
    to the same JSON file.
    """

    return _instance_from_json(_resolve_instance_path(name))


def list_instances() -> list[str]:
    """Return all JSON instance names available under data/instances."""

    if not INST_DIR.exists():
        return []
    return sorted(path.stem for path in INST_DIR.glob("*.json"))


INSTANCES = list_instances()
