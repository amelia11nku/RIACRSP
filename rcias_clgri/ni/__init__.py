"""Offline supervised Neural Improvement components for CSG-1.0."""

from .tensorize import (
    CSGTensorizer,
    NIEdgeTensor,
    NIRelationSpec,
    NITensorGraph,
)
from .batching import NIBatch, batch_state_samples
from .cache import CACHE_SCHEMA, load_shard_cache, write_shard_cache
from .dataset import NIActionSet, NIStateSample, build_state_sample, tensorize_action_records
from .encoder import NIModelConfig
from .losses import NILossConfig, phase6e_loss
from .scorer import CSGTargetSetScorer, NIModelOutput

__all__ = [
    "CSGTensorizer",
    "NIEdgeTensor",
    "NIRelationSpec",
    "NITensorGraph",
    "NIActionSet",
    "NIStateSample",
    "NIBatch",
    "batch_state_samples",
    "build_state_sample",
    "CACHE_SCHEMA",
    "load_shard_cache",
    "NIModelConfig",
    "NILossConfig",
    "NIModelOutput",
    "CSGTargetSetScorer",
    "phase6e_loss",
    "tensorize_action_records",
    "write_shard_cache",
]
