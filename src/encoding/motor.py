"""Motor neuron spikes → MuJoCo joint torques.

Decoding path:
  motor neuron spike counts (10 ms window)
    → per-muscle activation level
    → joint torque via NeuroMechFly actuator mapping

Motor neuron → muscle assignments come from Phelps et al. 2021 (FANC)
plus the Sept 2026 male CNS release's updated motor neuron pools.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class MotorDecoder:
    def __init__(
        self,
        motor_ids_per_muscle: dict[str, list[int]],   # muscle name → body IDs
        window_ms: int = 10,
        gain: float = 1.0,
    ):
        self.motor_ids = motor_ids_per_muscle
        self.window = window_ms
        self.gain = gain
        self._spike_buffer: deque[torch.Tensor] = deque(maxlen=window_ms)

    def ingest(self, spikes_this_ms: torch.Tensor) -> None:
        self._spike_buffer.append(spikes_this_ms.detach().cpu())

    def decode(self) -> dict[str, float]:
        """Return per-muscle activation (0..1-ish) based on windowed spike rates."""
        if not self._spike_buffer:
            return {m: 0.0 for m in self.motor_ids}

        counts = torch.stack(list(self._spike_buffer)).sum(dim=0)
        activations: dict[str, float] = {}
        for muscle, ids in self.motor_ids.items():
            # ids are body IDs — real impl needs id → tensor-index map.
            # For now return a placeholder of 0.0 so the sim can run end-to-end.
            activations[muscle] = 0.0
        return activations
