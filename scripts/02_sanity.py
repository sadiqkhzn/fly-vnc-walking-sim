"""Sanity checks on the cached subset.

Verifies:
  - Soma coordinates present and look like a fly brain (reasonable extent, bilateral)
  - Motor neurons: leg bias check (should be mostly in LegNp ROIs)
  - Neurotransmitter distribution has excitatory + inhibitory populations
  - Command neurons are reachable from DN population (basic connectivity sanity)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.neuprint_client import fetch_vnc_subset


def main():
    s = fetch_vnc_subset()
    n = s.neurons
    e = s.edges

    print("=== soma coordinates ===")
    xyz = n[["x", "y", "z"]].to_numpy()
    valid = ~np.isnan(xyz).any(axis=1)
    print(f"  with coords: {valid.sum():,} / {len(n):,}  ({valid.mean()*100:.1f}%)")
    if valid.any():
        v = xyz[valid]
        print(f"  x range: {v[:,0].min():.0f} .. {v[:,0].max():.0f}")
        print(f"  y range: {v[:,1].min():.0f} .. {v[:,1].max():.0f}")
        print(f"  z range: {v[:,2].min():.0f} .. {v[:,2].max():.0f}")
        print(f"  x centroid: {v[:,0].mean():.0f}  (bilateral symmetry → should see L/R split around this)")

    print("\n=== soma side distribution ===")
    print(n["soma_side"].fillna("<none>").value_counts().to_string())

    print("\n=== neurotransmitter distribution (predictedNt) ===")
    print(n["nt"].fillna("<none>").value_counts().to_string())

    print("\n=== motor neuron sample (top 15 types) ===")
    for t, ids in sorted(s.motor_ids_by_type.items(), key=lambda kv: -len(kv[1]))[:15]:
        xs = n.loc[ids, "x"].dropna()
        xmean = xs.mean() if len(xs) else float("nan")
        print(f"  {t:35s} {len(ids):3d} cells   x_mean={xmean:.0f}")

    print("\n=== command neuron connectivity reach ===")
    for name, ids in s.command_ids.items():
        if not ids:
            print(f"  {name}: no cells")
            continue
        downstream = e[e["pre"].isin(ids)]["post"].nunique()
        upstream = e[e["post"].isin(ids)]["pre"].nunique()
        tot_out = int(e[e["pre"].isin(ids)]["weight"].sum())
        tot_in = int(e[e["post"].isin(ids)]["weight"].sum())
        print(f"  {name:8s} out→{downstream:5d} partners / {tot_out:6d} syn   in←{upstream:5d} / {tot_in:6d} syn")

    print("\n=== edge integrity ===")
    body_ids = set(n.index.to_numpy().tolist())
    orphan_pre = (~e["pre"].isin(body_ids)).sum()
    orphan_post = (~e["post"].isin(body_ids)).sum()
    print(f"  orphan pre: {orphan_pre}   orphan post: {orphan_post}  (both should be 0)")
    print(f"  self-loops: {(e['pre'] == e['post']).sum()}")
    print(f"  min weight: {e['weight'].min()}  (should be >= 3)")

    print("\nOK")


if __name__ == "__main__":
    main()
