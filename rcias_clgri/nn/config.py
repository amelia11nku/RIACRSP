"""Central neural architecture configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 128
    heads: int = 4
    layers: int = 2
    feedforward_multiplier: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.layers < 1:
            raise ValueError("layers must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
