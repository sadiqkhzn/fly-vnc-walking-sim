"""Robot sensors → sensory neuron input currents.

The mapping is biologically native: VNC sensory neurons evolved to carry
exactly these signals in a real fly. No improvisation.

Mapping (per leg unless noted):
  joint angle (femur-tibia, etc.) → FeCO (femoral chordotonal organ) neurons
  ground reaction force           → campaniform sensilla (CS)
  joint velocity / position       → hair plate (HP) neurons
  descending "walk"               → MDN or bolt-protocerebrum walk DNs
  descending "stop"               → DNp09
  descending "turn L/R"           → DNa01 / DNa02

Each encoder returns a current contribution added to the LIF external input.
"""
from __future__ import annotations

import torch


class SensoryEncoder:
    def __init__(
        self,
        feco_ids_per_leg: dict[str, list[int]],   # keys: "L1","L2","L3","R1","R2","R3"
        cs_ids_per_leg: dict[str, list[int]],
        hp_ids_per_leg: dict[str, list[int]],
        n_neurons: int,
        device: str = "cpu",
    ):
        self.N = n_neurons
        self.device = device
        self.feco = feco_ids_per_leg
        self.cs = cs_ids_per_leg
        self.hp = hp_ids_per_leg

    def encode(
        self,
        joint_angles: dict[str, float],
        joint_velocities: dict[str, float],
        ground_force: dict[str, float],
    ) -> torch.Tensor:
        """Return (N,) input current vector for a single ms."""
        inp = torch.zeros(self.N, device=self.device)

        # TODO(week 2): tune the gain and encoding curve per modality.
        # FeCO cells have known tuning to joint angle derivative + position; see
        # Mamiya et al. 2018 and Agrawal et al. 2020 for the response curves.
        # First pass: linear rate code around a baseline.
        return inp


class DescendingCommand:
    """Thin interface to inject descending-neuron drive.

    Used by validation experiments (direct stimulation) and later by the
    learned readout (continuous commands).
    """

    def __init__(self, command_ids: dict[str, list[int]], n_neurons: int, device: str = "cpu"):
        self.N = n_neurons
        self.ids = command_ids
        self.device = device

    def drive(self, levels: dict[str, float]) -> torch.Tensor:
        """`levels` maps command name (e.g. 'MDN', 'DNp09') to drive magnitude in [0, 1]."""
        inp = torch.zeros(self.N, device=self.device)
        for name, level in levels.items():
            for body_id in self.ids.get(name, []):
                # NOTE: body_id → index translation happens in the sim runner,
                # not here. This stub assumes already-indexed vectors.
                pass
        return inp
