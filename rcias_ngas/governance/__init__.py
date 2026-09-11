"""Dataset-role governance for NGAS formal pipelines."""

from .dataset_roles import (
    AccessDenied,
    DatasetRegistry,
    append_exposure,
    validate_manifest,
    validate_training_records,
    verify_exposure_ledger,
)

__all__ = [
    'AccessDenied',
    'DatasetRegistry',
    'append_exposure',
    'validate_manifest',
    'validate_training_records',
    'verify_exposure_ledger',
]
