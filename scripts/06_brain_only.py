"""Brain-only smoke test: no biomechanics, no sensors.

Protocol:
  1. Load data/vnc_graph.pt into LIFBrain.
  2. Baseline: 100 ms with zero drive. Expect near-silent (occasional noise spikes OK).
  3. Stimulate MDN tonically for 500 ms (inject suprathreshold current at each MDN idx).
  4. Record and report:
       - MDN firing rate (should be high — it's being forced)
       - Downstream population activation (should jump above baseline)
       - Spike-rate histogram across the whole population
       - Any sign of runaway excitation (bad: everybody fires at max rate)
       - Any sign of total shutdown by inhibition (bad: nothing fires)

Interpretation guide:
  Baseline mean rate near 0, MDN-driven downstream rate measurably >baseline,
  no neuron pinned at refractory ceiling = healthy. Tune LIFParams.syn_scale
  if the population either explodes or stays silent.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.lif import LIFBrain, LIFParams

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "vnc_graph.pt"


def run(brain: LIFBrain, duration_ms: int, drive_idx: list[int], drive_level: float):
    """Return (spike_counts [N], raster [T, n_drive_idx])."""
    N = brain.N
    spike_counts = torch.zeros(N)
    drive_raster = torch.zeros(duration_ms, len(drive_idx))

    drive_vec = torch.zeros(N, device=brain.device)
    if drive_idx:
        drive_vec[drive_idx] = drive_level

    for t in range(duration_ms):
        spikes = brain.step(drive_vec)
        spike_counts += spikes.cpu()
        if drive_idx:
            drive_raster[t] = spikes.cpu()[drive_idx]
    return spike_counts, drive_raster


def describe_pop(label: str, counts: torch.Tensor, duration_ms: int, mask: torch.Tensor | None = None):
    pop = counts if mask is None else counts[mask]
    n = pop.numel()
    rate_hz = pop.sum().item() / (n * duration_ms / 1000.0) if n else 0.0
    frac_active = (pop > 0).float().mean().item() if n else 0.0
    print(
        f"  {label:28s} n={n:6d}  mean_rate={rate_hz:6.1f} Hz  "
        f"active_frac={frac_active*100:5.1f}%  max_count={int(pop.max().item() if n else 0):4d}"
    )


def main():
    torch.manual_seed(0)
    print("[loading brain]")
    brain = LIFBrain.from_bundle(GRAPH_PATH, params=LIFParams())
    print(f"  N={brain.N:,}  nnz={brain.W.values().numel():,}  dt={brain.p.dt_ms} ms  tau_m={brain.p.tau_m_ms} ms")

    bundle = brain.bundle
    mdn_idx = bundle["command_idx"]["MDN"]
    descending_idx = bundle["descending_idx"]
    motor_idx_all = [i for ids in bundle["motor_idx_by_type"].values() for i in ids]
    print(f"  MDN cells        : {mdn_idx}")
    print(f"  descending cells : {len(descending_idx)}")
    print(f"  motor cells      : {len(motor_idx_all)} across {len(bundle['motor_idx_by_type'])} types")

    N = brain.N
    desc_mask = torch.zeros(N, dtype=torch.bool); desc_mask[descending_idx] = True
    motor_mask = torch.zeros(N, dtype=torch.bool); motor_mask[motor_idx_all] = True
    mdn_mask = torch.zeros(N, dtype=torch.bool); mdn_mask[mdn_idx] = True
    other_mask = ~(desc_mask | motor_mask)

    # --- Baseline: no drive ---
    print("\n[baseline, 200 ms, zero drive]")
    brain.reset()
    t0 = time.time()
    counts_base, _ = run(brain, duration_ms=200, drive_idx=[], drive_level=0.0)
    print(f"  wall time: {time.time()-t0:.2f}s")
    describe_pop("population", counts_base, 200)
    describe_pop("descending", counts_base, 200, desc_mask)
    describe_pop("motor", counts_base, 200, motor_mask)

    # --- MDN tonic drive ---
    print("\n[MDN tonic drive, 500 ms, drive_level=2.0 (suprathreshold)]")
    brain.reset()
    t0 = time.time()
    counts_mdn, mdn_raster = run(brain, duration_ms=500, drive_idx=mdn_idx, drive_level=2.0)
    print(f"  wall time: {time.time()-t0:.2f}s")
    describe_pop("population", counts_mdn, 500)
    describe_pop("MDN (driven)", counts_mdn, 500, mdn_mask)
    describe_pop("other descending", counts_mdn, 500, desc_mask & ~mdn_mask)
    describe_pop("motor", counts_mdn, 500, motor_mask)
    describe_pop("other neurons", counts_mdn, 500, other_mask)

    # MDN firing itself
    mdn_rate_each = mdn_raster.sum(dim=0) * 1000.0 / 500.0
    print(f"\n  MDN per-cell rates (Hz): {mdn_rate_each.tolist()}")

    # Downstream delta
    delta = (counts_mdn / 500.0 - counts_base / 200.0) * 1000.0  # Hz change
    print("\n[top 10 cells with largest rate increase vs baseline]")
    top = delta.topk(10)
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        bid = bundle["body_ids"][i]
        tag = (
            "MOTOR" if motor_mask[i].item()
            else "DN" if desc_mask[i].item()
            else "other"
        )
        print(f"  idx {i:5d}  bodyId {bid:>7d}  Δrate +{v:6.1f} Hz  [{tag}]")

    # Health check summary
    print("\n[health]")
    max_rate = counts_mdn.max().item() * 1000 / 500
    population_mean = counts_mdn.mean().item() * 1000 / 500
    if max_rate > 400:
        print(f"  WARN: a neuron fired at {max_rate:.0f} Hz (near refractory ceiling 1/(2 ms)=500 Hz).")
        print(f"        Consider lowering syn_scale.")
    elif population_mean > 50:
        print(f"  WARN: population mean rate {population_mean:.1f} Hz suggests runaway excitation.")
    elif counts_mdn.sum().item() == counts_mdn[mdn_idx].sum().item():
        print(f"  WARN: only MDN fired; nothing propagated. Consider raising syn_scale.")
    else:
        print(f"  OK: max {max_rate:.0f} Hz, population mean {population_mean:.2f} Hz. Sensible range.")


if __name__ == "__main__":
    main()
