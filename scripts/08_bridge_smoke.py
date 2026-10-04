"""Pass 8 smoke test: FlyBridge wrapper, no neural sim yet.

Three mini-experiments:
  A) Stand still (neutral action, 500 ms). Fly should settle onto feet with
     6 nonzero contact forces, root should barely move.
  B) Sinusoidal perturbation on the left front coxa for 500 ms. The perturbed
     joint should track the target, other joints move little, foot forces
     for lf may drop during the lifted phase of the cycle.
  C) Verify observation() shapes and dtypes end-to-end.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.biomech.fly_bridge import FlyBridge


def main():
    t0 = time.time()
    bridge = FlyBridge()
    print(f"construct: {time.time() - t0:.2f}s   "
          f"actuators: {bridge.n_actuators}   total DOFs: {bridge.n_dof}   "
          f"timestep: {bridge.timestep*1000:.3f} ms")

    neutral = bridge.neutral_action
    print(f"neutral action range: [{neutral.min():.3f}, {neutral.max():.3f}] rad   mean: {neutral.mean():.3f}")

    # --- A) Stand still ---
    print("\n[A] stand still, 5000 steps (500 ms)")
    bridge.reset()
    t0 = time.time()
    obs = None
    for _ in range(5000):
        obs = bridge.step(neutral)
    print(f"  wall: {time.time()-t0:.2f}s   root pos: {obs.root_pos}  foot |F|: " +
          "  ".join(f"{bridge._foot_segments[i][:2]}={obs.foot_force[i]:.2f}"
                    for i in range(6)))
    assert (obs.foot_force > 0.1).sum() == 6, "all 6 feet should be loaded while standing"

    # --- B) Perturb one leg ---
    print("\n[B] sinusoidal perturbation on first joint (nominally lf coxa pitch), 5000 steps")
    bridge.reset()
    perturbation_idx = 0  # whatever DOF index 0 is — we'll verify by observing it moves
    t0 = time.time()
    joint0_history = []
    for step in range(5000):
        action = neutral.copy()
        action[perturbation_idx] = neutral[perturbation_idx] + 0.5 * np.sin(2 * np.pi * step * bridge.timestep * 3.0)  # 3 Hz
        obs = bridge.step(action)
        joint0_history.append(float(obs.joint_angles[perturbation_idx]))
    j0 = np.array(joint0_history)
    print(f"  wall: {time.time()-t0:.2f}s   joint[0] angle range: [{j0.min():.3f}, {j0.max():.3f}]  "
          f"ptp: {np.ptp(j0):.3f} rad   (should be well above 0 since we drove it)")

    # --- C) Observation sanity ---
    print("\n[C] observation sanity")
    print(f"  joint_angles      shape {obs.joint_angles.shape}  dtype {obs.joint_angles.dtype}")
    print(f"  joint_velocities  shape {obs.joint_velocities.shape}")
    print(f"  foot_force        shape {obs.foot_force.shape}  values {obs.foot_force}")
    print(f"  root_pos          shape {obs.root_pos.shape}  values {obs.root_pos}")
    print(f"  sim_time          {obs.sim_time:.4f} s")


if __name__ == "__main__":
    main()
