"""Leaky integrate-and-fire simulator over a fixed connectome.

Design:
  - Membrane potentials as a dense (N,) tensor on CPU.
    (MPS sparse CSR support is incomplete as of torch 2.14; CPU is the safe default.)
  - Synaptic weights as a sparse CSR tensor built once from the connectome.
    W[i, j] = post i ← pre j, signed by presynaptic NT (Dale's law).
  - 1 ms timestep. One sparse matmul per step for recurrent current.
  - Weights are NEVER modified. Only external inputs can change between runs.

The `syn_scale` global is applied AT PROPAGATE TIME (dense op) rather than
baked into the sparse tensor, because scalar * sparse_csr is beta-state in
torch 2.14 and we want to be able to retune without rebuilding the tensor.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LIFParams:
    tau_m_ms: float = 20.0        # membrane time constant
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    refractory_ms: float = 2.0
    dt_ms: float = 1.0
    syn_scale: float = 0.01       # global multiplier on recurrent current


class LIFBrain:
    """Fixed-weight LIF network. Call step() each ms with external input."""

    def __init__(
        self,
        n_neurons: int,
        weights_csr: torch.Tensor,
        params: LIFParams,
        device: str = "cpu",
    ):
        assert weights_csr.is_sparse_csr, "weights must be sparse CSR"
        assert weights_csr.shape == (n_neurons, n_neurons)
        self.N = n_neurons
        self.W = weights_csr.to(device)
        self.p = params
        self.device = device

        self.v = torch.full((n_neurons,), params.v_rest, device=device)
        self.refrac = torch.zeros(n_neurons, device=device)
        self.decay = float(torch.exp(torch.tensor(-params.dt_ms / params.tau_m_ms)))

        # Pre-materialize scalar reset/refrac tensors to avoid per-step allocs.
        self._v_reset_t = torch.tensor(params.v_reset, device=device)
        self._refrac_t = torch.tensor(params.refractory_ms, device=device)

        # Side-channel: anything the loader wants to stash (bundle metadata, etc.)
        self.bundle: dict[str, Any] | None = None

    # --- lifecycle ---

    def reset(self) -> None:
        self.v.fill_(self.p.v_rest)
        self.refrac.zero_()

    # --- step ---

    def step(self, external_input: torch.Tensor) -> torch.Tensor:
        """Advance 1 ms. Returns float32 binary spike vector (N,)."""
        self.v = self.p.v_rest + (self.v - self.p.v_rest) * self.decay
        self.v = self.v + external_input

        active = self.refrac <= 0
        self.refrac = torch.clamp(self.refrac - self.p.dt_ms, min=0.0)

        spikes = (self.v >= self.p.v_thresh) & active
        self.v = torch.where(spikes, self._v_reset_t, self.v)
        self.refrac = torch.where(spikes, self._refrac_t, self.refrac)
        return spikes.to(torch.float32)

    def propagate(self, spikes: torch.Tensor) -> torch.Tensor:
        """Synaptic current induced by `spikes`, scaled by syn_scale."""
        out = torch.sparse.mm(self.W, spikes.unsqueeze(1)).squeeze(1)
        return out * self.p.syn_scale

    # --- construction from on-disk bundle ---

    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        params: LIFParams | None = None,
        device: str = "cpu",
    ) -> "LIFBrain":
        """Load a `scripts/03_build_graph.py` output and return a ready brain."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # beta-state CSR warnings
            bundle = torch.load(path, weights_only=False)
        brain = cls(
            n_neurons=bundle["meta"]["n_neurons"],
            weights_csr=bundle["weights_csr"],
            params=params or LIFParams(),
            device=device,
        )
        brain.bundle = bundle
        return brain
