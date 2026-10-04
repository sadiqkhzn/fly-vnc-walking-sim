"""Validation experiment 1 — tripod gait from tonic MDN descending drive.

Reference: Bidaye et al. 2020 (MDN as the moonwalker command), Cande et al.
2018 (CPG tripod coordination). Pass criteria:

  (1) Oscillation in per-leg motor firing rate with dominant frequency in 5-15 Hz
  (2) Tripod phasing: group A (lf, rm, lh) anti-phase with group B (rf, lm, rh).
      Measured as Pearson correlation between A-trace and B-trace in [-1, 0];
      we require < -0.3 to pass.
  (3) No pathology: no leg at 500 Hz, no leg silent.

Brain-only — no biomechanics. The CPGs either produce rhythmic motor output
or they don't; if they do, the biomechanics is just interpretation.
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
MOTOR_ASSIGN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "motor_leg_assignment.parquet"
FIG_PATH = Path(__file__).resolve().parent / "figures" / "01_tripod_gait.png"
JSON_PATH = Path(__file__).resolve().parent / "figures" / "01_tripod_gait.json"

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
TRIPOD_A = {"lf", "rm", "lh"}
TRIPOD_B = {"rf", "lm", "rh"}

BIN_MS = 10
STIM_START_MS = 200
DURATION_MS = 2000  # 2 seconds total


def main():
    torch.manual_seed(0)
    print("[loading]")
    brain = LIFBrain.from_bundle(GRAPH_PATH, params=LIFParams())
    bundle = brain.bundle
    assign = pd.read_parquet(MOTOR_ASSIGN_PATH)

    # body_id -> leg
    body_to_leg = dict(zip(assign.index.astype(int), assign["leg"]))

    # tensor-index -> leg (via body_ids list)
    body_ids = bundle["body_ids"]
    leg_to_idx = {leg: [] for leg in LEGS}
    for i, b in enumerate(body_ids):
        leg = body_to_leg.get(int(b))
        if leg in leg_to_idx:
            leg_to_idx[leg].append(i)
    for leg in LEGS:
        print(f"  {leg} motor cells: {len(leg_to_idx[leg])}")

    mdn_idx = bundle["command_idx"]["MDN"]

    # --- run simulation ---
    print(f"\n[running] {DURATION_MS} ms, MDN tonic from t={STIM_START_MS} ms")
    N = brain.N
    drive_vec = torch.zeros(N)
    drive_vec[mdn_idx] = 2.0

    per_leg_counts_per_ms = {leg: np.zeros(DURATION_MS, dtype=np.int32) for leg in LEGS}

    t0 = time.time()
    for t_ms in range(DURATION_MS):
        inp = drive_vec if t_ms >= STIM_START_MS else torch.zeros(N)
        spikes = brain.step(inp)
        for leg in LEGS:
            idxs = leg_to_idx[leg]
            if idxs:
                per_leg_counts_per_ms[leg][t_ms] = int(spikes[idxs].sum().item())
    wall = time.time() - t0
    print(f"  wall: {wall:.1f}s ({DURATION_MS/wall/1000:.2f}x real-time)")

    # --- bin into BIN_MS windows ---
    nbins = DURATION_MS // BIN_MS
    per_leg_rate = {}
    for leg in LEGS:
        counts = per_leg_counts_per_ms[leg].reshape(nbins, BIN_MS).sum(axis=1)
        rate_hz = counts * (1000.0 / BIN_MS) / max(1, len(leg_to_idx[leg]))
        per_leg_rate[leg] = rate_hz
    time_axis_ms = (np.arange(nbins) + 0.5) * BIN_MS

    # Analyse only the post-stim window
    stim_mask = time_axis_ms >= STIM_START_MS
    traces = {leg: per_leg_rate[leg][stim_mask] for leg in LEGS}
    tA = np.sum([traces[l] for l in TRIPOD_A], axis=0)
    tB = np.sum([traces[l] for l in TRIPOD_B], axis=0)

    # --- (1) FFT: dominant frequency ---
    freqs = np.fft.rfftfreq(len(tA), d=BIN_MS / 1000.0)
    psdA = np.abs(np.fft.rfft(tA - tA.mean())) ** 2
    psdB = np.abs(np.fft.rfft(tB - tB.mean())) ** 2
    psd = psdA + psdB
    # Ignore DC
    peak_idx = int(np.argmax(psd[1:]) + 1)
    peak_freq = float(freqs[peak_idx])
    peak_in_range = 5.0 <= peak_freq <= 15.0

    # --- (2) Tripod phasing ---
    if tA.std() > 0 and tB.std() > 0:
        corr = float(np.corrcoef(tA, tB)[0, 1])
    else:
        corr = 0.0
    anti_phase = corr < -0.3

    # --- (3) pathology ---
    max_leg_rate = max(float(traces[l].max()) for l in LEGS)
    any_silent = any(float(traces[l].mean()) < 0.5 for l in LEGS if leg_to_idx[l])
    healthy = max_leg_rate < 400 and not any_silent

    # --- figure ---
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    for leg in LEGS:
        linestyle = "-" if leg in TRIPOD_A else "--"
        axes[0].plot(time_axis_ms, per_leg_rate[leg], label=leg, linestyle=linestyle, linewidth=1)
    axes[0].axvline(STIM_START_MS, color="k", linestyle=":", alpha=0.5, label="MDN onset")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("rate (Hz / cell)")
    axes[0].set_title("Per-leg motor firing rate (tonic MDN drive)")
    axes[0].legend(loc="upper right", ncol=4)

    stim_t = time_axis_ms[stim_mask]
    axes[1].plot(stim_t, tA, label="tripod A (lf, rm, lh)", color="tab:blue")
    axes[1].plot(stim_t, tB, label="tripod B (rf, lm, rh)", color="tab:orange")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("summed group rate (Hz)")
    axes[1].set_title(f"Tripod phasing — Pearson r(A, B) = {corr:+.3f}  "
                      f"[{'PASS' if anti_phase else 'FAIL'} < -0.3]")
    axes[1].legend()

    axes[2].plot(freqs, psd)
    axes[2].axvspan(5, 15, color="green", alpha=0.15, label="expected band 5–15 Hz")
    axes[2].axvline(peak_freq, color="red", linestyle="--",
                    label=f"peak = {peak_freq:.2f} Hz  [{'PASS' if peak_in_range else 'FAIL'}]")
    axes[2].set_xlim(0, 30)
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("PSD")
    axes[2].set_title("Power spectrum (sum of tripod A, B)")
    axes[2].legend()

    fig.suptitle(
        f"Validation 1: tripod gait from MDN  — "
        f"{'ALL PASS' if (peak_in_range and anti_phase and healthy) else 'INCOMPLETE'}"
    )
    fig.savefig(FIG_PATH, dpi=140)
    plt.close(fig)

    # Dump raw numbers alongside the figure for reproducibility.
    report = {
        "duration_ms": DURATION_MS,
        "stim_start_ms": STIM_START_MS,
        "bin_ms": BIN_MS,
        "n_motor_per_leg": {leg: len(leg_to_idx[leg]) for leg in LEGS},
        "peak_freq_hz": peak_freq,
        "peak_in_5_15_hz": peak_in_range,
        "tripod_corr": corr,
        "tripod_anti_phase": anti_phase,
        "max_leg_rate_hz": max_leg_rate,
        "healthy": healthy,
    }
    JSON_PATH.write_text(json.dumps(report, indent=2))

    print("\n[results]")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:6s} {vv}")
        else:
            print(f"  {k:25s} {v}")

    print("\n[verdict]")
    verdict_lines = [
        ("osc 5-15 Hz", peak_in_range, f"peak = {peak_freq:.2f} Hz"),
        ("tripod anti-phase", anti_phase, f"r = {corr:+.3f}"),
        ("healthy (no pathology)", healthy, f"max rate {max_leg_rate:.1f} Hz"),
    ]
    for name, ok, detail in verdict_lines:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")

    all_pass = peak_in_range and anti_phase and healthy
    print(f"\n{'ALL PASS' if all_pass else 'NOT PASSING'}  — figure: {FIG_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
