"""Pass 7 smoke test: can we build a NeuroMechFly, step it headless, read state?

In flygym 2.1 a bare NeuroMechFly() has no actuated DOFs. The full composition
requires: add joints from a Skeleton preset, then add position actuators for
the leg DOFs. Only then does Simulation populate the per-fly qpos/qvel maps.
"""
from __future__ import annotations

import time

import numpy as np

from flygym.anatomy import ActuatedDOFPreset, AxisOrder, ContactBodiesPreset, JointPreset, Skeleton
from flygym.compose import FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D


def main():
    t0 = time.time()

    fly = NeuroMechFly(name="nmf")
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.ROLL_PITCH_YAW,
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        actuated_dofs,
        actuator_type="position",
        neutral_input=KinematicPosePreset.NEUTRAL,
        kp=50,
    )

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=np.array([0.0, 0.0, 0.7]),
        spawn_rotation=Rotation3D(format="quat", values=[1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
    )
    sim = Simulation(world)
    print(f"construction: {time.time() - t0:.2f}s")
    print(f"timestep    : {sim.timestep*1000:.3f} ms  (so {int(1/sim.timestep)} steps = 1 s)")
    print(f"actuated DOF: {len(fly.jointdof_to_mjcfjoint)}")

    joints0 = sim.get_joint_angles(fly.name)
    pos0 = sim.get_body_positions(fly.name)
    print(f"joint angles length: {len(joints0)}  bodies: {len(pos0)}")

    N = 2000  # 200 ms of sim time (dt=0.1 ms)
    t0 = time.time()
    for _ in range(N):
        sim.step()
    dt_wall = time.time() - t0
    sim_sec = N * sim.timestep
    print(f"\n{N} steps = {sim_sec*1000:.1f} ms sim  in {dt_wall:.2f}s wall  "
          f"({sim_sec/dt_wall:.2f}x real-time)")

    joints1 = sim.get_joint_angles(fly.name)
    pos1 = sim.get_body_positions(fly.name)
    thorax = fly.root_segment  # c_thorax
    print(f"root segment: {thorax}")
    print(f"thorax z: {pos0[0, 2]:.4f} → {pos1[0, 2]:.4f}  (should drop under gravity if feet not supporting)")
    print(f"joint-angle delta range: {(joints1 - joints0).min():.4f} .. {(joints1 - joints0).max():.4f} rad")

    # Contact forces: fully-qualified per-leg distal segments (tarsus5 = foot tip).
    feet = [f"{side}{row}_tarsus5" for side in ("l", "r") for row in ("f", "m", "h")]
    gc = sim.get_bodysegment_contact_forces(fly.name, feet)
    gc = np.asarray(gc)
    print(f"foot contact forces  shape {gc.shape}  per-leg |F|: " +
          "  ".join(f"{feet[i][:2]}={np.linalg.norm(gc[i]):.2f}" for i in range(len(feet))))


if __name__ == "__main__":
    main()
