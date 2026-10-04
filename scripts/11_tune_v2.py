"""Pass 11: quick sweep over (syn_scale_i / syn_scale_e ratio) × (tau_i)
to find the dual-τ regime where DNp09 suppresses the MDN-induced motor pool.

Keeps syn_scale_e fixed at 0.02 and tau_e at 5 ms (fast ACh baseline).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.lif import LIFBrain, LIFParams

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "vnc_graph.pt"
ASSIGN_PATH = Path(__file__).resolve().parent.parent / "data" / "motor_leg_assignment.parquet"

DURATION = 2500           # longer so suppression has time to establish
MDN_START = 200
DNP09_START = 1000


def one(syn_e: float, syn_i: float, tau_i: float, motor_idx: list[int], mdn_idx: list[int], dnp09_idx: list[int]) -> dict:
    torch.manual_seed(0)
    brain = LIFBrain.from_bundle(GRAPH_PATH, params=LIFParams(
        syn_scale_e=syn_e, syn_scale_i=syn_i, tau_e_ms=5.0, tau_i_ms=tau_i,
    ))
    N = brain.N
    mdn_vec = torch.zeros(N); mdn_vec[mdn_idx] = 2.0
    dnp_vec = torch.zeros(N); dnp_vec[dnp09_idx] = 2.0

    rate = np.zeros(DURATION, dtype=np.int32)
    for t in range(DURATION):
        if t < MDN_START:
            inp = torch.zeros(N)
        elif t < DNP09_START:
            inp = mdn_vec
        else:
            inp = mdn_vec + dnp_vec
        spikes = brain.step(inp)
        rate[t] = int(spikes[motor_idx].sum().item())

    mdn_rate = rate[MDN_START:DNP09_START].mean() * 1000.0 / len(motor_idx)
    stop_rate = rate[DNP09_START:].mean() * 1000.0 / len(motor_idx)
    base_rate = rate[:MDN_START].mean() * 1000.0 / len(motor_idx)
    return {
        "syn_e": syn_e, "syn_i": syn_i, "tau_i": tau_i,
        "base": base_rate, "mdn": mdn_rate, "stop": stop_rate,
        "stop_ratio": stop_rate / max(mdn_rate, 0.01),
        "pass_mdn_fires": mdn_rate > 5 * (base_rate + 0.1),
        "pass_dnp09_suppresses": stop_rate < 0.5 * mdn_rate if mdn_rate > 0 else False,
    }


def main():
    assign = pd.read_parquet(ASSIGN_PATH)
    body_to_leg = dict(zip(assign.index.astype(int), assign["leg"]))
    bundle = torch.load(GRAPH_PATH, weights_only=False)
    body_ids = bundle["body_ids"]
    motor_idx = [i for i, b in enumerate(body_ids)
                 if body_to_leg.get(int(b)) in ("lf", "lm", "lh", "rf", "rm", "rh")]
    mdn_idx = bundle["command_idx"]["MDN"]
    dnp09_idx = bundle["command_idx"]["DNp09"]

    results = []
    print(f"{'syn_e':>7s} {'syn_i':>7s} {'tau_i':>6s}  {'base':>5s} {'mdn':>6s} {'stop':>6s} {'ratio':>6s}   verdict  [sec]")
    # Settle around the sweet spot found in earlier sweep (syn_i = syn_e, tau_i ~ 150ms)
    # Narrow grid, longer sim so transient effects dominate less
    for syn_i_mult in (0.9, 1.0, 1.1):
        for tau_i in (120.0, 150.0, 180.0):
            t0 = time.time()
            r = one(0.02, 0.02 * syn_i_mult, tau_i, motor_idx, mdn_idx, dnp09_idx)
            dur = time.time() - t0
            tag = (
                ("P" if r["pass_mdn_fires"] else "-")
                + ("P" if r["pass_dnp09_suppresses"] else "-")
            )
            print(
                f"{r['syn_e']:7.3f} {r['syn_i']:7.3f} {r['tau_i']:6.0f}  "
                f"{r['base']:5.1f} {r['mdn']:6.1f} {r['stop']:6.1f} {r['stop_ratio']:6.2f}   "
                f"{tag}       [{dur:.0f}]"
            )
            results.append(r)

    good = [r for r in results if r["pass_mdn_fires"] and r["pass_dnp09_suppresses"]]
    if good:
        best = min(good, key=lambda r: r["stop_ratio"])
        print(f"\n*** best suppression: syn_e={best['syn_e']} syn_i={best['syn_i']} "
              f"tau_i={best['tau_i']}ms  "
              f"stop_ratio={best['stop_ratio']:.3f} ***")
    else:
        best = min(results, key=lambda r: r["stop_ratio"])
        print(f"\n(no setting fully passes; closest stop_ratio={best['stop_ratio']:.2f}, "
              f"at syn_e={best['syn_e']} syn_i={best['syn_i']} tau_i={best['tau_i']}ms)")


if __name__ == "__main__":
    main()
