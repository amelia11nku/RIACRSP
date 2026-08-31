"""Phase 6F utility-aware extension of the frozen Phase 6E trainer."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Sequence

import torch

from .batching import NIBatch
from .scorer import CSGTargetSetScorer
from .trainer import NITrainer
from .utility_losses import Phase6FLossConfig, phase6f_loss


class Phase6FTrainer(NITrainer):
    def __init__(
        self,
        *args,
        loss_config: Phase6FLossConfig,
        teacher_models: Sequence[CSGTargetSetScorer] = (),
        **kwargs,
    ) -> None:
        super().__init__(*args, loss_config=loss_config, **kwargs)
        self.loss_config = loss_config
        self.teacher_models = tuple(model.to(self.device).eval() for model in teacher_models)
        if bool(self.teacher_models) != bool(loss_config.distillation_weight):
            raise ValueError("teacher models and distillation weight must be enabled together")
        for model in self.teacher_models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _compute_losses(self, output, batch: NIBatch) -> dict[str, torch.Tensor]:
        teacher_scores = None
        if self.teacher_models:
            with torch.no_grad():
                teacher_scores = torch.stack([
                    teacher(batch).scores.float() for teacher in self.teacher_models
                ]).mean(dim=0)
        return phase6f_loss(
            output.scores,
            output.utility_predictions,
            batch,
            self.loss_config,
            teacher_scores=teacher_scores,
        )

    def _validation_objective(self, metrics) -> float:
        # Early stopping is aligned with Phase 6F's primary deployment criterion.
        return float(metrics["mean_selected_utility"])

    def _checkpoint_payload(self, **kwargs):
        payload = super()._checkpoint_payload(**kwargs)
        payload["schema"] = "phase6f-training-checkpoint-v1"
        payload["objective"] = self.loss_config.objective
        payload["distillation_weight"] = self.loss_config.distillation_weight
        payload["teacher_count"] = len(self.teacher_models)
        payload["loss_config"] = asdict(self.loss_config)
        return payload

    def fit(self, *args, **kwargs):
        summary = super().fit(*args, **kwargs)
        summary.update({
            "schema": "phase6f-training-summary-v1",
            "objective": self.loss_config.objective,
            "distillation_weight": self.loss_config.distillation_weight,
            "teacher_count": len(self.teacher_models),
        })
        (self.output_directory / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
