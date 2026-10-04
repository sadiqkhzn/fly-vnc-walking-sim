"""Pass 9: full closed loop smoke test. Brain spikes → fly actions.

Pipeline per 1 ms LIF tick:
  1. LIF.step(external_input)      → spikes (24,115,)  (recurrence internal)
  2. Bucket motor spikes into 42 channels (quick-and-dirty mapping — Pass 10
     replaces this with a proper muscle→actuator mapping)
  3. If 10 ms elapsed since last fly update, decode last-10ms spike buckets
     into position deltas and run 10 fly substeps.

This is NOT a validation experiment. It only confirms:
  - Shapes line up end-to-end
  - MDN drive produces non-zero motor spikes within the window
  - Fly produces non-zero movement in response
  - Nothing crashes / explodes / stalls
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.biomech.fly_bridge import FlyBridge
from src.sim.lif import LIFBrain, LIFParams

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "vnc_graph.pt"

LIF_DT_MS = 1.0
FLY_SUBSTEPS_PER_LIF_TICK = 10   # LIF tick 1ms, fly tick 0.1ms
MOTOR_WINDOW_MS = 10              # spike counts aggregated over this
ACTION_GAIN_RAD = 0.3             # max delta applied to neutral per fully-saturated bucket


def build_motor_to_action_map(motor_indices: list[int], n_actuators: int, seed: int = 0):
    """Deterministic hash of motor neuron index → action bucket. Not biology.

    Returns an array `bucket[i]` giving the action bucket for motor neuron index i.
    Pass 10 replaces this with a mapping derived from muscle names.
    """
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, n_actuators, size=len(motor_indices))
    return dict(zip(motor_indices, buckets.tolist()))


def main():
    torch.manual_seed(0)
    print("[loading brain]")
    brain = LIFBrain.from_bundle(GRAPH_PATH, params=LIFParams())
    bundle = brain.bundle
    mdn_idx = bundle["command_idx"]["MDN"]
    motor_indices = [i for ids in bundle["motor_idx_by_type"].values() for i in ids]
    print(f"  N={brain.N:,}  motor cells={len(motor_indices)}  MDN cells={mdn_idx}")

    print("[building fly bridge]")
    t0 = time.time()
    bridge = FlyBridge()
    print(f"  construct {time.time()-t0:.2f}s   actuators={bridge.n_actuators}")

    # Random-hash motor → actuator bucket. Pass 10 will replace this with a
    # muscle-name-based mapping from the Phelps 2021 motor-neuron atlas.
    motor_to_bucket = build_motor_to_action_map(motor_indices, bridge.n_actuators)
    bucket_arr = np.full(brain.N, -1, dtype=np.int64)
    for idx, b in motor_to_bucket.items():
        bucket_arr[idx] = b
    bucket_tensor = torch.from_numpy(bucket_arr)

    neutral = bridge.neutral_action
    bridge.reset()

    # External drive: tonic MDN during t > 100 ms (baseline silence first).
    drive_vec = torch.zeros(brain.N)
    drive_vec[mdn_idx] = 2.0  # suprathreshold

    duration_lif_ms = 400  # total LIF simulation
    stim_start_ms = 100

    recent_motor_spikes = deque(maxlen=MOTOR_WINDOW_MS)  # last N ms of motor spike-count tensors

    # Diagnostics
    total_motor_spikes = 0
    motor_spike_log: list[int] = []
    action_log: list[np.ndarray] = []
    root_log: list[np.ndarray] = []

    t_wall = time.time()
    for t_ms in range(duration_lif_ms):
        # 1) brain step
        inp = drive_vec if t_ms >= stim_start_ms else torch.zeros(brain.N)
        spikes = brain.step(inp)

        # 2) motor spike aggregation
        motor_spikes = spikes[motor_indices]
        total_motor_spikes += int(motor_spikes.sum().item())
        motor_spike_log.append(int(motor_spikes.sum().item()))
        recent_motor_spikes.append(spikes.clone())

        # 3) every 10 ms, decode an action and run fly substeps
        if t_ms > 0 and t_ms % MOTOR_WINDOW_MS == 0:
            window = torch.stack(list(recent_motor_spikes)).sum(dim=0)  # (N,) spike counts
            # Scatter-add into buckets
            bucket_counts = torch.zeros(bridge.n_actuators)
            mask = bucket_arr >= 0
            bucket_counts.scatter_add_(
                0,
                bucket_tensor[mask],
                window[mask],
            )
            # Normalize to 0..1 (saturate at 20 spikes per bucket per window)
            activation = (bucket_counts / 20.0).clamp(0, 1).numpy()
            # Position target = neutral + gain * (activation - 0.5) so negative/positive deltas
            action = neutral + ACTION_GAIN_RAD * (activation - 0.5)
            action_log.append(action.copy())

            for _ in range(FLY_SUBSTEPS_PER_LIF_TICK):
                obs = bridge.step(action.astype(np.float32))
            root_log.append(obs.root_pos.copy())
    wall = time.time() - t_wall

    print(f"\n[run]")
    print(f"  wall time        : {wall:.2f}s for {duration_lif_ms} ms LIF sim ({duration_lif_ms/wall/1000:.2f}x real-time)")
    print(f"  total motor spikes: {total_motor_spikes:,}")
    print(f"  motor spikes/ms before stim onset ({stim_start_ms}ms): "
          f"{np.mean(motor_spike_log[:stim_start_ms]):.2f}")
    print(f"  motor spikes/ms after stim onset: "
          f"{np.mean(motor_spike_log[stim_start_ms:]):.2f}")

    if action_log:
        acts = np.stack(action_log)
        print(f"  action trajectory shape: {acts.shape}")
        print(f"  action deviation from neutral (max per dof): "
              f"{np.max(np.abs(acts - neutral), axis=0)[:6].round(3)}")
    if root_log:
        pos = np.stack(root_log)
        print(f"  thorax displacement (first vs last): "
              f"Δ = {np.linalg.norm(pos[-1] - pos[0]):.4f} mm")

    print("\n[sanity checks]")
    pre = np.mean(motor_spike_log[:stim_start_ms])
    post = np.mean(motor_spike_log[stim_start_ms:])
    assert pre < 1.0, f"baseline motor firing too high ({pre:.2f})"
    assert post > 2 * (pre + 0.1), f"motor firing did not increase after MDN stim ({pre:.2f} → {post:.2f})"
    print(f"  OK: motor firing jumped {pre:.2f} → {post:.2f} spikes/ms after MDN onset")

    if root_log:
        pos = np.stack(root_log)
        disp = np.linalg.norm(pos[-1] - pos[0])
        if disp > 0.05:
            print(f"  OK: fly moved {disp:.3f} mm — closed loop produced motion")
        else:
            print(f"  NOTE: fly barely moved ({disp:.3f} mm). Expected for random motor→actuator map.")

    bridge.close()


if __name__ == "__main__":
    main()
