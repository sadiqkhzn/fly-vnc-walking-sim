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

# Canonical leg ordering used throughout the project (sensors return in this order).
LEG_ORDER = ("lf", "lm", "lh", "rf", "rm", "rh")


@dataclass
class Observation:
    joint_angles: np.ndarray         # (n_dof,) rad — all 126 DOFs including passive tarsi
    joint_velocities: np.ndarray     # (n_dof,) rad/s
    foot_force_vec: np.ndarray       # (6, 3) full 3D ground reaction force per leg (lf, lm, lh, rf, rm, rh)
    foot_force_mag: np.ndarray       # (6,) magnitude per leg (|foot_force_vec|)
    foot_contact: np.ndarray         # (6,) binary — 1 if leg is in contact
    foot_pos: np.ndarray             # (6, 3) world position of contact point per leg
    root_pos: np.ndarray             # (3,) c_thorax position
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
            ContactBodiesPreset,
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

        # Ground contact biology: only tibias + tarsi touch the floor when a fly
        # walks. Using the LEGS_THORAX_ABDOMEN_HEAD preset would also sensorize
        # non-leg segments which is both biologically wrong (the body doesn't
        # scrape) and triggers a sensor-registration edge case in flygym 2.1.0
        # (coxa listed as both root-segment and ground-contact body). Using the
        # restricted preset is correct AND avoids the edge case.
        world = FlatGroundWorld()
        world.add_fly(
            self.fly,
            spawn_position=np.array([0.0, 0.0, spawn_z]),
            spawn_rotation=Rotation3D(format="quat", values=[1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=True,
        )
        self.sim = Simulation(world)
        self.timestep = self.sim.timestep
        self._has_ground_sensors = (
            self.sim.world.legpos_to_groundcontactsensors_by_fly is not None
            and self.fly.name in self.sim.world.legpos_to_groundcontactsensors_by_fly
        )

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
        pos = self.sim.get_body_positions(self.fly.name)

        # Prefer the native MuJoCo contact sensor (per-leg 3D force + position +
        # normals). Fall back to the body-contact force sum if sensors absent
        # (keeps the API intact if a caller disables sensors for perf).
        if self._has_ground_sensors:
            found, cpos, _torque, force, _normal, _tangent = self.sim.get_ground_contact_info(
                self.fly.name
            )
            foot_force_vec = np.asarray(force, dtype=np.float64)        # (6, 3)
            foot_force_mag = np.linalg.norm(foot_force_vec, axis=1)     # (6,)
            foot_contact = np.asarray(found, dtype=np.float64)          # (6,)
            foot_pos = np.asarray(cpos, dtype=np.float64)               # (6, 3)
        else:
            gc = np.asarray(
                self.sim.get_bodysegment_contact_forces(self.fly.name, self._foot_segments)
            )
            foot_force_vec = gc
            foot_force_mag = np.linalg.norm(gc, axis=1)
            foot_contact = (foot_force_mag > 1e-4).astype(np.float64)
            foot_pos = np.zeros_like(foot_force_vec)

        return Observation(
            joint_angles=angles,
            joint_velocities=vels,
            foot_force_vec=foot_force_vec,
            foot_force_mag=foot_force_mag,
            foot_contact=foot_contact,
            foot_pos=foot_pos,
            root_pos=pos[0],
            sim_time=self.sim.time,
        )

    def close(self) -> None:
        # Nothing to clean up in flygym 2.1 headless, but keep the method for symmetry.
        pass
