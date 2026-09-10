"""Frozen A1.3R checkpoint inference for live joint-action scoring."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch

from rcias_ngas.csg.revised_features import action_features, state_features, tensorize
from rcias_ngas.critic.revised_critic import RevisedJointCritic


def file_sha256(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FrozenJointCritic:
    """Load one immutable C1/R1 checkpoint and score a complete action bank."""

    def __init__(self, checkpoint_path: str | Path, device: str = 'cuda',
                 expected_sha256: str | None = None,
                 expected_variant: str | None = None) -> None:
        self.path = Path(checkpoint_path)
        self.sha256 = file_sha256(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('Frozen critic checkpoint hash mismatch')
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        if not str(payload.get('schema', '')).startswith('ngas-a13r-'):
            raise ValueError('Checkpoint is not an A1.3R artifact')
        self.variant = payload.get('selected_variant', payload.get('variant'))
        if self.variant is None:
            raise ValueError('Checkpoint has no representation variant')
        if expected_variant is not None and self.variant != expected_variant:
            raise ValueError('Frozen critic variant mismatch')
        config = payload['model_config']
        self.device = torch.device(device)
        self.model = RevisedJointCritic(
            encoder_type=config['encoder_type'],
            hidden=int(config['hidden_dim']),
            layers=int(config['message_passing_layers']),
            heads=int(config.get('heads', 4)),
        ).to(self.device)
        self.model.load_state_dict(payload['model_state'], strict=True)
        self.model.eval()
        self.training_protocol_sha256 = payload['training_protocol_sha256']

    def score(self, instance, current, state_id: str, actions: tuple) -> tuple[dict, dict]:
        if not actions:
            raise ValueError('Cannot score an empty joint-action bank')
        started = time.perf_counter()
        state = state_features(instance, current, state_id)
        feature_seconds = time.perf_counter() - started
        started = time.perf_counter()
        features = action_features(state, actions)
        action_feature_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batch = tensorize(state, features, device=self.device)
        tensor_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(batch)
        advantages = output['advantage'].detach().cpu().tolist()
        logits = output['beats_fallback_logit'].detach().cpu()
        probabilities = torch.sigmoid(logits).tolist()
        model_seconds = time.perf_counter() - started
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise FloatingPointError('Non-finite frozen critic output')
        return ({
            'advantage': [float(value) for value in advantages],
            'beats_fallback_probability': [float(value) for value in probabilities],
            'state_feature_hash': state['feature_hash'],
            'graph_hash': state['graph_hash'],
        }, {
            'state_feature_seconds': feature_seconds,
            'action_feature_seconds': action_feature_seconds,
            'tensor_transfer_seconds': tensor_seconds,
            'model_forward_seconds': model_seconds,
        })
