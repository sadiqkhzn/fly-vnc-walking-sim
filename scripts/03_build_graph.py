"""Build the signed sparse weight tensor consumed by the LIF sim.

Produces data/vnc_graph.pt:
  weights_csr        torch sparse CSR, shape (N, N), float32
                     W[i, j] = contribution to postsynaptic neuron i from
                     presynaptic neuron j, signed by j's predicted NT (Dale's law)
  body_ids           list[int], length N, body_ids[i] is the bodyId at index i
  id_to_idx          dict[int, int]
  command_idx        {name: [idx, ...]} for MDN, DNp09, DNa01, DNa02, DNb01
  motor_idx_by_type  {type_str: [idx, ...]}
  descending_idx     [idx, ...] for all DN cells
  meta               dict of dataset, thresholds, sign table, counts

Sign convention (Dale's law):
  acetylcholine → +1      (fast excitatory)
  gaba          → -1      (fast inhibitory)
  glutamate     → -1      (inhibitory in fly VNC via GluCl, standard choice)
  serotonin, dopamine, octopamine, histamine, unclear, None → 0
      (neuromodulators / unresolved — not driving fast LIF membrane in v1;
       can be added later with slower dynamics if validation needs them)

Self-loops are dropped. Deterministic: same cache → same output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.neuprint_client import fetch_vnc_subset

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "vnc_graph.pt"

# Signs = biology (Dale's law). Unit magnitude for E and I keeps the sign
# interpretation unambiguous ("one synapse = one unit of signed drive"). An
# earlier attempt with |I| = 2 improved steady-state rate balance but quenched
# rhythms; see passes/10c and 10d. The honest baseline is 1:1 and the open
# question of CPG phasing (which likely requires sensory feedback) is deferred
# to the closed-loop milestone, not papered over here.
NT_SIGN: dict[str, float] = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "serotonin": 0.0,
    "dopamine": 0.0,
    "octopamine": 0.0,
    "histamine": 0.0,
    "unclear": 0.0,
}


def _sign_for(nt) -> float:
    if nt is None or (isinstance(nt, float) and np.isnan(nt)):
        return 0.0
    return NT_SIGN.get(str(nt).lower(), 0.0)


def main():
    s = fetch_vnc_subset()

    body_ids = s.neurons.index.to_numpy(dtype=np.int64)
    N = len(body_ids)
    id_to_idx = {int(b): i for i, b in enumerate(body_ids)}

    # Drop self-loops.
    self_loops = int((s.edges["pre"] == s.edges["post"]).sum())
    edges = s.edges[s.edges["pre"] != s.edges["post"]].copy()

    # Sanity: every endpoint maps to an index. (Fetcher already enforces this; we
    # re-check here because this script may run against an older cache.)
    missing_pre = ~edges["pre"].isin(id_to_idx)
    missing_post = ~edges["post"].isin(id_to_idx)
    assert not missing_pre.any() and not missing_post.any(), (
        f"orphan endpoints: pre={missing_pre.sum()} post={missing_post.sum()}"
    )

    pre_idx = edges["pre"].map(id_to_idx).to_numpy(dtype=np.int64)
    post_idx = edges["post"].map(id_to_idx).to_numpy(dtype=np.int64)
    raw_w = edges["weight"].to_numpy(dtype=np.float32)

    # Dale's law: sign of each edge is determined by the PREsynaptic cell's NT.
    pre_nts = s.neurons.loc[edges["pre"].values, "nt"].to_numpy()
    signs = np.array([_sign_for(nt) for nt in pre_nts], dtype=np.float32)
    signed_w = raw_w * signs

    # COO -> CSR. Rows = post (destination), cols = pre (source).
    # coalesce() sums any duplicate (post, pre) pairs — rare but defensive.
    coo = torch.sparse_coo_tensor(
        indices=torch.from_numpy(np.vstack([post_idx, pre_idx])),
        values=torch.from_numpy(signed_w),
        size=(N, N),
    ).coalesce()
    csr = coo.to_sparse_csr()

    command_idx = {
        name: [id_to_idx[int(b)] for b in ids] for name, ids in s.command_ids.items()
    }
    motor_idx_by_type = {
        t: [id_to_idx[int(b)] for b in ids] for t, ids in s.motor_ids_by_type.items()
    }
    descending_idx = [id_to_idx[int(b)] for b in s.descending_ids]

    nnz = csr.values().numel()
    frac_exc = float((signed_w > 0).sum()) / max(1, len(signed_w))
    frac_inh = float((signed_w < 0).sum()) / max(1, len(signed_w))
    frac_zero = float((signed_w == 0).sum()) / max(1, len(signed_w))

    bundle = {
        "weights_csr": csr,
        "body_ids": body_ids.tolist(),
        "id_to_idx": id_to_idx,
        "command_idx": command_idx,
        "motor_idx_by_type": motor_idx_by_type,
        "descending_idx": descending_idx,
        "meta": {
            "dataset": s.dataset,
            "weight_threshold": s.weight_threshold,
            "n_neurons": N,
            "n_edges_coalesced": nnz,
            "n_edges_raw": len(edges),
            "self_loops_dropped": self_loops,
            "frac_excitatory": frac_exc,
            "frac_inhibitory": frac_inh,
            "frac_zero_sign": frac_zero,
            "nt_sign_table": NT_SIGN,
            "sign_convention": "Dale's law (presynaptic NT determines sign of all outputs)",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, OUT_PATH)

    print(f"wrote {OUT_PATH.relative_to(Path.cwd())}")
    print(f"  N neurons   : {N:,}")
    print(f"  edges in    : {len(edges):,}   coalesced nnz: {nnz:,}")
    print(f"  self-loops dropped: {self_loops}")
    print(f"  excitatory  : {frac_exc * 100:5.1f}%")
    print(f"  inhibitory  : {frac_inh * 100:5.1f}%")
    print(f"  zero sign   : {frac_zero * 100:5.1f}%  (modulators / unclear)")
    print(f"  command neuron tensor indices:")
    for name, idxs in command_idx.items():
        print(f"    {name:8s} {idxs}")


if __name__ == "__main__":
    main()
