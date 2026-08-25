"""Single-source exporters for operation and resource-level schedule inspection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rcias_clgri.data.instance import Instance

from .schedule import FTask, Schedule, WTask


OPERATION_FIELDS = (
    "product", "operation", "actual_sequence_position", "technological_predecessors",
    "actual_predecessor", "island", "required_configuration", "previous_island_configuration",
    "reconfiguration_start", "reconfiguration_end", "process_start", "process_end",
    "product_ready_time", "island_ready_time", "config_ready_time", "w_ready_time",
    "f_ready_time", "W_required", "W_AGV", "W_pickup", "W_destination",
    "W_empty_start", "W_empty_end", "W_loaded_start", "W_loaded_end", "W_arrival",
    "F_AGV", "F_departure_WH", "F_arrival_island", "F_return_WH", "binding_resource",
)

RESOURCE_FIELDS = (
    "resource_type", "resource_id", "activity_type", "product", "operation",
    "start", "end", "duration", "origin", "destination",
    "from_configuration", "to_configuration",
)

W_FIELDS = (
    "vehicle", "product", "operation", "actual_predecessor", "vehicle_previous_location",
    "pickup", "destination", "empty_start", "empty_end", "loaded_start", "loaded_end", "arrival",
)

F_FIELDS = (
    "vehicle", "product", "operation", "destination", "departure_WH", "arrival_island", "return_WH",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class ResourceTimelineExporter:
    """Derive all human-inspection tables from one validated schedule object."""

    instance: Instance
    schedule: Schedule

    def _w_by_operation(self) -> dict[str, WTask]:
        return {
            task.operation_id: task
            for tasks in self.schedule.w_timelines.values()
            for task in tasks
        }

    def _f_by_operation(self) -> dict[str, FTask]:
        return {
            task.operation_id: task
            for tasks in self.schedule.f_timelines.values()
            for task in tasks
        }

    def operation_rows(self) -> list[dict[str, Any]]:
        w_by_operation = self._w_by_operation()
        f_by_operation = self._f_by_operation()
        product_positions = {
            op_id: position
            for sequence in self.schedule.product_sequences.values()
            for position, op_id in enumerate(sequence, start=1)
        }
        previous_config: dict[str, str] = {}
        for island_id, sequence in self.schedule.island_timelines.items():
            config = self.instance.island_data[island_id].initial_config
            for op_id in sequence:
                previous_config[op_id] = config
                config = self.schedule.operation_schedules[op_id].config_id
        rows: list[dict[str, Any]] = []
        for product_id in self.instance.products:
            for op_id in self.schedule.product_sequences[product_id]:
                record = self.schedule.operation_schedules[op_id]
                w_task = w_by_operation.get(op_id)
                f_task = f_by_operation[op_id]
                rows.append({
                    "product": product_id,
                    "operation": op_id,
                    "actual_sequence_position": product_positions[op_id],
                    "technological_predecessors": ";".join(sorted(self.instance.predecessors[op_id])),
                    "actual_predecessor": record.product_predecessor or "NONE",
                    "island": record.island_id,
                    "required_configuration": record.config_id,
                    "previous_island_configuration": previous_config[op_id],
                    "reconfiguration_start": record.reconfiguration_start,
                    "reconfiguration_end": record.reconfiguration_end,
                    "process_start": record.start_time,
                    "process_end": record.completion_time,
                    "product_ready_time": record.product_ready_time,
                    "island_ready_time": record.island_ready_time,
                    "config_ready_time": record.config_ready_time,
                    "w_ready_time": record.w_ready_time,
                    "f_ready_time": record.f_ready_time,
                    "W_required": w_task is not None,
                    "W_AGV": "NONE" if w_task is None else w_task.vehicle_id,
                    "W_pickup": "NA" if w_task is None else w_task.pickup,
                    "W_destination": "NA" if w_task is None else w_task.destination,
                    "W_empty_start": "NA" if w_task is None else w_task.empty_start,
                    "W_empty_end": "NA" if w_task is None else w_task.empty_arrival,
                    "W_loaded_start": "NA" if w_task is None else w_task.loaded_start,
                    "W_loaded_end": "NA" if w_task is None else w_task.arrival_time,
                    "W_arrival": "NA" if w_task is None else w_task.arrival_time,
                    "F_AGV": f_task.vehicle_id,
                    "F_departure_WH": f_task.departure_wh,
                    "F_arrival_island": f_task.arrival_island,
                    "F_return_WH": f_task.return_wh,
                    "binding_resource": ";".join(record.binding_resource),
                })
        return rows

    def resource_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for island_id, sequence in self.schedule.island_timelines.items():
            previous_config = self.instance.island_data[island_id].initial_config
            for op_id in sequence:
                record = self.schedule.operation_schedules[op_id]
                if record.reconfiguration_end > record.reconfiguration_start:
                    rows.append({
                        "resource_type": "ASSEMBLY_ISLAND", "resource_id": island_id,
                        "activity_type": "RECONFIGURATION", "product": record.product_id,
                        "operation": op_id, "start": record.reconfiguration_start,
                        "end": record.reconfiguration_end,
                        "duration": record.reconfiguration_end - record.reconfiguration_start,
                        "origin": "", "destination": "",
                        "from_configuration": previous_config, "to_configuration": record.config_id,
                    })
                rows.append({
                    "resource_type": "ASSEMBLY_ISLAND", "resource_id": island_id,
                    "activity_type": "PROCESSING", "product": record.product_id,
                    "operation": op_id, "start": record.start_time, "end": record.completion_time,
                    "duration": record.completion_time - record.start_time,
                    "origin": "", "destination": "", "from_configuration": "",
                    "to_configuration": record.config_id,
                })
                previous_config = record.config_id
        for vehicle_id, tasks in self.schedule.w_timelines.items():
            for task in tasks:
                if task.empty_arrival > task.empty_start:
                    rows.append({
                        "resource_type": "W_AGV", "resource_id": vehicle_id,
                        "activity_type": "EMPTY_REPOSITION", "product": task.product_id,
                        "operation": task.operation_id, "start": task.empty_start,
                        "end": task.empty_arrival, "duration": task.empty_arrival - task.empty_start,
                        "origin": task.empty_origin, "destination": task.pickup,
                        "from_configuration": "", "to_configuration": "",
                    })
                rows.append({
                    "resource_type": "W_AGV", "resource_id": vehicle_id,
                    "activity_type": "LOADED_TRANSPORT", "product": task.product_id,
                    "operation": task.operation_id, "start": task.loaded_start,
                    "end": task.arrival_time, "duration": task.arrival_time - task.loaded_start,
                    "origin": task.pickup, "destination": task.destination,
                    "from_configuration": "", "to_configuration": "",
                })
        for vehicle_id, tasks in self.schedule.f_timelines.items():
            for task in tasks:
                product_id = self.instance.product_of[task.operation_id]
                rows.append({
                    "resource_type": "F_AGV", "resource_id": vehicle_id,
                    "activity_type": "OUTBOUND_DELIVERY", "product": product_id,
                    "operation": task.operation_id, "start": task.departure_wh,
                    "end": task.arrival_island, "duration": task.arrival_island - task.departure_wh,
                    "origin": "WH", "destination": task.island_id,
                    "from_configuration": "", "to_configuration": "",
                })
                rows.append({
                    "resource_type": "F_AGV", "resource_id": vehicle_id,
                    "activity_type": "RETURN_TO_WH", "product": product_id,
                    "operation": task.operation_id, "start": task.arrival_island,
                    "end": task.return_wh, "duration": task.return_wh - task.arrival_island,
                    "origin": task.island_id, "destination": "WH",
                    "from_configuration": "", "to_configuration": "",
                })
        type_order = {"ASSEMBLY_ISLAND": 0, "W_AGV": 1, "F_AGV": 2}
        rows.sort(key=lambda row: (type_order[row["resource_type"]], row["resource_id"], float(row["start"]), row["activity_type"]))
        return rows

    def w_rows(self) -> list[dict[str, Any]]:
        return [{
            "vehicle": task.vehicle_id,
            "product": task.product_id,
            "operation": task.operation_id,
            "actual_predecessor": task.predecessor_op or "NONE",
            "vehicle_previous_location": task.empty_origin,
            "pickup": task.pickup,
            "destination": task.destination,
            "empty_start": task.empty_start,
            "empty_end": task.empty_arrival,
            "loaded_start": task.loaded_start,
            "loaded_end": task.arrival_time,
            "arrival": task.arrival_time,
        } for tasks in self.schedule.w_timelines.values() for task in tasks]

    def f_rows(self) -> list[dict[str, Any]]:
        return [{
            "vehicle": task.vehicle_id,
            "product": self.instance.product_of[task.operation_id],
            "operation": task.operation_id,
            "destination": task.island_id,
            "departure_WH": task.departure_wh,
            "arrival_island": task.arrival_island,
            "return_WH": task.return_wh,
        } for tasks in self.schedule.f_timelines.values() for task in tasks]

    def export(self, output_directory: str | Path) -> dict[str, Path]:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "operation": output / "operation_schedule.csv",
            "resource": output / "resource_timeline.csv",
            "w": output / "w_agv_tasks.csv",
            "f": output / "f_agv_tasks.csv",
        }
        _write_csv(paths["operation"], OPERATION_FIELDS, self.operation_rows())
        _write_csv(paths["resource"], RESOURCE_FIELDS, self.resource_rows())
        _write_csv(paths["w"], W_FIELDS, self.w_rows())
        _write_csv(paths["f"], F_FIELDS, self.f_rows())
        return paths
