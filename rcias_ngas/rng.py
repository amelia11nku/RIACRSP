"""Stable, explicitly named independent random streams (never Python hash())."""
from dataclasses import dataclass
import hashlib
import json
import random

NAMESPACES = (
    "destroy_size", "target", "repair", "portfolio", "neighbor",
    "acceptance", "fallback", "continuation_crn", "neural_prior",
    "online_exploration", "diagnostics",
)


@dataclass(frozen=True)
class RNGStreams:
    instance_id: str
    run_seed: int

    def seed(self, namespace: str, state_id: str = "", index: int = 0) -> int:
        if namespace not in NAMESPACES:
            raise ValueError(f"Unknown RNG namespace: {namespace}")
        key = ["ngas-rng-v1", self.instance_id, self.run_seed, namespace, state_id, index]
        payload = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")

    def stream(self, namespace: str, state_id: str = "", index: int = 0) -> random.Random:
        """Return a fresh stream; keep it locally or use a distinct event index.

        Downstream keys deliberately exclude action ID and selection mode, so
        matched actions receive matched draws under top-1 and sampling policies.
        """
        return random.Random(self.seed(namespace, state_id, index))
