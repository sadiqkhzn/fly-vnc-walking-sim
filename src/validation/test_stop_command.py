"""Validation experiment 2 — DNp09 descending stop command.

Reference: Bidaye et al. 2014 — activating DNp09 halts walking in flies.

Biological note: DNp09 is cholinergic (excitatory), not inhibitory. The stop
mechanism is di-synaptic: DNp09 excites intermediate inhibitory interneurons
in the VNC which in turn suppress the leg motor pools. So this test is a
genuine check of whether the connectome contains sufficient inhibitory
intermediaries to transmit the stop signal.

Protocol:
  phase 0 : 0-200  ms   silence (verify baseline)
  phase 1 : 200-1000 ms MDN tonic (verify "walking" state onset)
  phase 2 : 1000-2000 ms MDN + DNp09 tonic (test: motor pool should drop)

Pass criteria:
  (1) phase 1 motor rate > 5x phase 0 rate (MDN induces motor firing)
  (2) phase 2 motor rate < 0.5x phase 1 rate (DNp09 suppresses it)
  (3) suppression establishes within 150 ms of DNp09 onset (biological latency)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.sim.lif import LIFBrain, LIFParams

GRAPH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "vnc_graph.pt"
ASSIGN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "motor_leg_assignment.parquet"
FIG_PATH = Path(__file__).resolve().parent / "figures" / "02_stop_command.png"
JSON_PATH = Path(__file__).resolve().parent / "figures" / "02_stop_command.json"

PHASE_BASELINE_END = 200
PHASE_MDN_END = 1000
PHASE_END = 2000
BIN_MS = 10


def main():
    torch.manual_seed(0)
    print("[loading]")
    brain = LIFBrain.from_bundle(GRAPH_PATH, params=LIFParams())
    bundle = brain.bundle
    assign = pd.read_parquet(ASSIGN_PATH)

    body_to_leg = dict(zip(assign.index.astype(int), assign["leg"]))
    body_ids = bundle["body_ids"]
    motor_indices = [
        i for i, b in enumerate(body_ids)
        if body_to_leg.get(int(b)) in ("lf", "lm", "lh", "rf", "rm", "rh")
    ]
    print(f"  N={brain.N:,}  motor cells (leg-assigned): {len(motor_indices)}")

    mdn_idx = bundle["command_idx"]["MDN"]
    dnp09_idx = bundle["command_idx"]["DNp09"]
    print(f"  MDN idx:   {mdn_idx}")
    print(f"  DNp09 idx: {dnp09_idx}")

    N = brain.N
    mdn_vec = torch.zeros(N); mdn_vec[mdn_idx] = 2.0
    dnp09_vec = torch.zeros(N); dnp09_vec[dnp09_idx] = 2.0

    def drive_at(t_ms: int) -> torch.Tensor:
        if t_ms < PHASE_BASELINE_END:
            return torch.zeros(N)
        if t_ms < PHASE_MDN_END:
            return mdn_vec
        return mdn_vec + dnp09_vec  # phase 2: both

    print(f"\n[running] {PHASE_END} ms   baseline 0..{PHASE_BASELINE_END} | "
          f"MDN {PHASE_BASELINE_END}..{PHASE_MDN_END} | +DNp09 {PHASE_MDN_END}..{PHASE_END}")
    motor_counts_per_ms = np.zeros(PHASE_END, dtype=np.int32)

    t0 = time.time()
    for t_ms in range(PHASE_END):
        inp = drive_at(t_ms)
        spikes = brain.step(inp)
        motor_counts_per_ms[t_ms] = int(spikes[motor_indices].sum().item())
    print(f"  wall: {time.time()-t0:.1f}s ({PHASE_END/(time.time()-t0)/1000:.2f}x real-time)")

    # Bin into 10ms bins → spikes per cell per second
    nbins = PHASE_END // BIN_MS
    rate = motor_counts_per_ms.reshape(nbins, BIN_MS).sum(axis=1) * (1000.0 / BIN_MS) / len(motor_indices)
    time_axis = (np.arange(nbins) + 0.5) * BIN_MS

    base_mask = time_axis < PHASE_BASELINE_END
    mdn_mask = (time_axis >= PHASE_BASELINE_END) & (time_axis < PHASE_MDN_END)
    stop_mask = time_axis >= PHASE_MDN_END

    r_base = float(rate[base_mask].mean())
    r_mdn = float(rate[mdn_mask].mean())
    r_stop = float(rate[stop_mask].mean())

    # Latency: time after DNp09 onset for rate to drop below 50% of mdn phase.
    threshold = 0.5 * r_mdn
    stop_bins = np.where(time_axis >= PHASE_MDN_END)[0]
    latency_ms = None
    for b in stop_bins:
        if rate[b] <= threshold:
            latency_ms = float(time_axis[b] - PHASE_MDN_END)
            break

    # Pass criteria
    p1 = r_mdn > 5 * (r_base + 0.1)
    p2 = r_stop < 0.5 * r_mdn if r_mdn > 0 else False
    p3 = latency_ms is not None and latency_ms <= 150

    # Figure
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(time_axis, rate, color="tab:blue", linewidth=1.4)
    ax.axvspan(0, PHASE_BASELINE_END, alpha=0.1, color="gray", label="baseline")
    ax.axvspan(PHASE_BASELINE_END, PHASE_MDN_END, alpha=0.1, color="orange", label="MDN")
    ax.axvspan(PHASE_MDN_END, PHASE_END, alpha=0.1, color="red", label="MDN + DNp09")
    ax.axhline(r_mdn, color="orange", linestyle="--", alpha=0.5, label=f"MDN rate ({r_mdn:.1f} Hz/cell)")
    ax.axhline(threshold, color="red", linestyle=":", alpha=0.5, label=f"50% threshold ({threshold:.1f})")
    if latency_ms is not None:
        ax.axvline(PHASE_MDN_END + latency_ms, color="green",
                   label=f"50% drop @ {latency_ms:.0f} ms post-onset")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("motor pool rate (Hz / cell)")
    ax.set_title(
        f"Validation 2 — DNp09 stop  "
        f"[{'PASS' if (p1 and p2 and p3) else 'FAIL'}]  "
        f"base={r_base:.2f}  mdn={r_mdn:.1f}  stop={r_stop:.1f}"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(FIG_PATH, dpi=140)
    plt.close(fig)

    report = {
        "duration_ms": PHASE_END,
        "n_motor_cells": len(motor_indices),
        "n_MDN_cells": len(mdn_idx),
        "n_DNp09_cells": len(dnp09_idx),
        "rate_baseline_hz": r_base,
        "rate_mdn_hz": r_mdn,
        "rate_stop_hz": r_stop,
        "mdn_over_baseline_ratio": r_mdn / max(r_base, 0.01),
        "stop_over_mdn_ratio": r_stop / max(r_mdn, 0.01),
        "stop_latency_ms_to_50pct": latency_ms,
        "pass_mdn_induces_firing": p1,
        "pass_dnp09_suppresses": p2,
        "pass_latency_under_150ms": p3,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2))

    print("\n[results]")
    for k, v in report.items():
        print(f"  {k:30s} {v}")

    print("\n[verdict]")
    for name, ok, detail in [
        ("MDN induces firing", p1, f"ratio = {r_mdn / max(r_base, 0.01):.1f}x"),
        ("DNp09 suppresses", p2, f"ratio = {r_stop / max(r_mdn, 0.01):.2f}  (need < 0.5)"),
        ("latency < 150 ms", p3, f"latency = {latency_ms} ms" if latency_ms else "never dropped"),
    ]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")
    all_pass = p1 and p2 and p3
    print(f"\n{'ALL PASS' if all_pass else 'NOT PASSING'}  — figure: {FIG_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
