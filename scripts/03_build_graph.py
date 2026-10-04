"""Build split W_exc, W_inh sparse weight tensors for the dual-τ LIF.

Produces data/vnc_graph.pt with:
  w_exc_csr          torch sparse CSR, shape (N, N), float32
                     W[i, j] = synapse count from j→i if j releases ACh, else 0
  w_inh_csr          torch sparse CSR, shape (N, N), float32  (magnitude, not signed)
                     W[i, j] = synapse count from j→i if j releases GABA/Glu, else 0
  body_ids           list[int], length N, body_ids[i] is the bodyId at index i
  id_to_idx          dict[int, int]
  command_idx        {name: [idx, ...]} for MDN, DNp09, DNa01, DNa02, DNb01
  motor_idx_by_type  {type_str: [idx, ...]}
  descending_idx     [idx, ...] for all DN cells
  meta               dict of dataset, thresholds, nt table, counts

Sign classification (Dale's law):
  acetylcholine → excitatory
  gaba          → inhibitory
  glutamate     → inhibitory (fly VNC via GluCl, standard choice)
  serotonin, dopamine, octopamine, histamine, unclear, None → modulatory → neither

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

NT_CLASS: dict[str, str] = {
    "acetylcholine": "exc",
    "gaba": "inh",
    "glutamate": "inh",
    "serotonin": "mod",
    "dopamine": "mod",
    "octopamine": "mod",
    "histamine": "mod",
    "unclear": "mod",
}


def _class_for(nt) -> str:
    if nt is None or (isinstance(nt, float) and np.isnan(nt)):
        return "mod"
    return NT_CLASS.get(str(nt).lower(), "mod")


def main():
    s = fetch_vnc_subset()

    body_ids = s.neurons.index.to_numpy(dtype=np.int64)
    N = len(body_ids)
    id_to_idx = {int(b): i for i, b in enumerate(body_ids)}

    self_loops = int((s.edges["pre"] == s.edges["post"]).sum())
    edges = s.edges[s.edges["pre"] != s.edges["post"]].copy()

    missing_pre = ~edges["pre"].isin(id_to_idx)
    missing_post = ~edges["post"].isin(id_to_idx)
    assert not missing_pre.any() and not missing_post.any(), (
        f"orphan endpoints: pre={missing_pre.sum()} post={missing_post.sum()}"
    )

    pre_idx = edges["pre"].map(id_to_idx).to_numpy(dtype=np.int64)
    post_idx = edges["post"].map(id_to_idx).to_numpy(dtype=np.int64)
    raw_w = edges["weight"].to_numpy(dtype=np.float32)

    # Classify each edge by its PRESYNAPTIC neuron's NT (Dale's law).
    pre_nts = s.neurons.loc[edges["pre"].values, "nt"].to_numpy()
    classes = np.array([_class_for(nt) for nt in pre_nts])

    exc_mask = classes == "exc"
    inh_mask = classes == "inh"

    def build_csr(mask):
        if not mask.any():
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.int64),
                torch.zeros(0, dtype=torch.float32),
                (N, N),
            ).coalesce().to_sparse_csr()
        coo = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack([post_idx[mask], pre_idx[mask]])),
            torch.from_numpy(raw_w[mask]),
            (N, N),
        ).coalesce()
        return coo.to_sparse_csr()

    W_exc = build_csr(exc_mask)
    W_inh = build_csr(inh_mask)

    command_idx = {
        name: [id_to_idx[int(b)] for b in ids] for name, ids in s.command_ids.items()
    }
    motor_idx_by_type = {
        t: [id_to_idx[int(b)] for b in ids] for t, ids in s.motor_ids_by_type.items()
    }
    descending_idx = [id_to_idx[int(b)] for b in s.descending_ids]

    nnz_e = W_exc.values().numel()
    nnz_i = W_inh.values().numel()
    frac_exc = float(exc_mask.sum()) / max(1, len(edges))
    frac_inh = float(inh_mask.sum()) / max(1, len(edges))
    frac_mod = 1.0 - frac_exc - frac_inh

    bundle = {
        "w_exc_csr": W_exc,
        "w_inh_csr": W_inh,
        "body_ids": body_ids.tolist(),
        "id_to_idx": id_to_idx,
        "command_idx": command_idx,
        "motor_idx_by_type": motor_idx_by_type,
        "descending_idx": descending_idx,
        "meta": {
            "dataset": s.dataset,
            "weight_threshold": s.weight_threshold,
            "n_neurons": N,
            "n_edges_exc": nnz_e,
            "n_edges_inh": nnz_i,
            "n_edges_raw": len(edges),
            "self_loops_dropped": self_loops,
            "frac_excitatory": frac_exc,
            "frac_inhibitory": frac_inh,
            "frac_modulatory": frac_mod,
            "nt_class_table": NT_CLASS,
            "sign_convention": "Dale's law; magnitudes stored, sign applied by LIF via dual conductances",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, OUT_PATH)

    print(f"wrote {OUT_PATH.relative_to(Path.cwd())}")
    print(f"  N neurons   : {N:,}")
    print(f"  self-loops dropped: {self_loops}")
    print(f"  edges exc   : {nnz_e:>10,}   ({frac_exc * 100:5.1f}%)")
    print(f"  edges inh   : {nnz_i:>10,}   ({frac_inh * 100:5.1f}%)")
    print(f"  edges mod   : {int(frac_mod * len(edges)):>10,}   ({frac_mod * 100:5.1f}%)   (dropped from LIF)")
    print(f"  command neuron tensor indices:")
    for name, idxs in command_idx.items():
        print(f"    {name:8s} {idxs}")


if __name__ == "__main__":
    main()
