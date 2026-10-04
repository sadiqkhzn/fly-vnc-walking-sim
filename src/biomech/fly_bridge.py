"""FlyBridge — thin wrapper around flygym.Simulation for the LIF closed loop.

Keeps the sim loop simulator-agnostic. The LIF side never touches MuJoCo /
flygym APIs directly; it reads an `Observation` dict and writes a `(n_dof,)`
position-target vector.

Conventions:
  - Actuator type   : POSITION (kp control). Target angles are in radians.
  - Timestep        : 0.1 ms (flygym default). Caller usually sub-steps 10x
                      per 1 ms LIF tick.
  - Action baseline : neutral posture angles, captured at reset. The LIF
                      output is interpreted as a DELTA on this baseline, so
                      zero neural activity → stand still.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FOOT_SEGMENTS = tuple(f"{s}{r}_tarsus5" for s in ("l", "r") for r in ("f", "m", "h"))


@dataclass
class Observation:
    joint_angles: np.ndarray      # (n_dof,) rad
    joint_velocities: np.ndarray  # (n_dof,) rad/s
    foot_force: np.ndarray        # (6,) magnitude of ground reaction per leg (lf, lm, lh, rf, rm, rh)
    root_pos: np.ndarray          # (3,) c_thorax position
    sim_time: float


class FlyBridge:
    def __init__(
        self,
        spawn_z: float = 0.7,
        actuator_kp: float = 50.0,
    ):
        # Imports inside __init__ to keep top-of-module cheap for scripts that
        # don't need flygym (e.g. the brain-only tests).
        from flygym.anatomy import (
            ActuatedDOFPreset,
            AxisOrder,
            JointPreset,
            Skeleton,
        )
        from flygym.compose import (
            ActuatorType,
            FlatGroundWorld,
            KinematicPosePreset,
            NeuroMechFly,
        )
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D

        self._ActuatorType = ActuatorType
        self._foot_segments = list(FOOT_SEGMENTS)

        self.fly = NeuroMechFly(name="nmf")
        skeleton = Skeleton(
            joint_preset=JointPreset.ALL_BIOLOGICAL,
            axis_order=AxisOrder.ROLL_PITCH_YAW,
        )
        self.fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        actuated_dofs = skeleton.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        )
        self.fly.add_actuators(
            actuated_dofs,
            actuator_type="position",
            neutral_input=KinematicPosePreset.NEUTRAL,
            kp=actuator_kp,
        )

        world = FlatGroundWorld()
        world.add_fly(
            self.fly,
            spawn_position=np.array([0.0, 0.0, spawn_z]),
            spawn_rotation=Rotation3D(format="quat", values=[1, 0, 0, 0]),
            add_ground_contact_sensors=False,
        )
        self.sim = Simulation(world)
        self.timestep = self.sim.timestep

        # Two different sizes:
        #   n_dof        = 126 (all skeletal DOFs — chordotonal organs sense all of them)
        #   n_actuators  = 42  (actively driven: 6 legs × 7 per leg)
        # The LIF motor layer outputs actions of size n_actuators; observation
        # exposes joint_angles / joint_velocities at size n_dof for sensory encoding.
        self._actuated_dof_order = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self._n_actuators = len(self._actuated_dof_order)
        self._n_dof = len(self.fly.get_jointdofs_order())
        self._neutral_action = np.array(
            [self.fly.jointdof_to_neutralangle[d] for d in self._actuated_dof_order],
            dtype=np.float32,
        )

    # --- lifecycle ---

    @property
    def n_dof(self) -> int:
        """Total skeletal DOFs available for sensory readout (126)."""
        return self._n_dof

    @property
    def n_actuators(self) -> int:
        """Actively driven DOFs (42). This is the motor-output dimensionality."""
        return self._n_actuators

    @property
    def neutral_action(self) -> np.ndarray:
        return self._neutral_action.copy()

    def reset(self) -> Observation:
        self.sim.reset()
        return self.observation()

    # --- step ---

    def step(self, position_targets: np.ndarray) -> Observation:
        assert position_targets.shape == (self._n_actuators,), (
            f"position_targets must be ({self._n_actuators},), got {position_targets.shape}"
        )
        self.sim.set_actuator_inputs(
            self.fly.name, self._ActuatorType.POSITION, position_targets.astype(np.float32)
        )
        self.sim.step()
        return self.observation()

    def observation(self) -> Observation:
        angles = self.sim.get_joint_angles(self.fly.name)
        vels = self.sim.get_joint_velocities(self.fly.name)
        gc = np.asarray(
            self.sim.get_bodysegment_contact_forces(self.fly.name, self._foot_segments)
        )
        foot_force = np.linalg.norm(gc, axis=1)
        pos = self.sim.get_body_positions(self.fly.name)
        return Observation(
            joint_angles=angles,
            joint_velocities=vels,
            foot_force=foot_force,
            root_pos=pos[0],
            sim_time=self.sim.time,
        )

    def close(self) -> None:
        # Nothing to clean up in flygym 2.1 headless, but keep the method for symmetry.
        pass
