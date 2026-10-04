"""Closed-loop runner: fly biomech ⇄ LIF VNC ⇄ descending commands.

One tick:
  1. FlyBridge gives joint angles + ground forces (every ms, or N:1 subsampled).
  2. SensoryEncoder maps those to sensory neuron input currents.
  3. DescendingCommand adds any top-down drive (from experiment script or readout).
  4. LIFBrain.step() produces spikes.
  5. MotorDecoder ingests spikes.
  6. Every W ms, MotorDecoder.decode() → muscle activations → FlyBridge.step().
  7. Spikes are also pushed to the WebSocket for live viz (see src/server).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.biomech.fly_bridge import FlyBridge
from src.encoding.motor import MotorDecoder
from src.encoding.sensory import DescendingCommand, SensoryEncoder
from src.sim.lif import LIFBrain


@dataclass
class RunnerConfig:
    sim_ms: int = 5_000
    motor_window_ms: int = 10
    fly_step_ms: int = 10


def run_closed_loop(
    brain: LIFBrain,
    fly: FlyBridge,
    sensory: SensoryEncoder,
    descending: DescendingCommand,
    motor: MotorDecoder,
    command_fn: Callable[[int], dict[str, float]],
    cfg: RunnerConfig,
    on_spikes: Callable[[int, torch.Tensor], None] | None = None,
):
    """`command_fn(t_ms) -> {command_name: drive}` lets callers inject experiment protocols."""
    obs = fly.reset()
    syn_current = torch.zeros(brain.N, device=brain.device)

    for t in range(cfg.sim_ms):
        sens_current = sensory.encode(
            joint_angles=obs.joint_angles,
            joint_velocities=obs.joint_velocities,
            ground_force=obs.ground_force,
        )
        desc_current = descending.drive(command_fn(t))

        total_input = sens_current + desc_current + syn_current
        spikes = brain.step(total_input)
        syn_current = brain.propagate(spikes)

        motor.ingest(spikes)
        if on_spikes is not None:
            on_spikes(t, spikes)

        if t % cfg.fly_step_ms == 0:
            activations = motor.decode()
            obs = fly.step(activations)
