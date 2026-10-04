"""Leaky integrate-and-fire simulator with dual-timeconstant E/I conductances.

Model (standard conductance-based LIF; Dayan & Abbott 2001, Ch 5):

    g_E[t+1] = g_E[t] * exp(-dt/tau_E) + (W_exc @ spikes[t]) * syn_scale_e
    g_I[t+1] = g_I[t] * exp(-dt/tau_I) + (W_inh @ spikes[t]) * syn_scale_i
    V  [t+1] = V_rest + (V[t] - V_rest) * exp(-dt/tau_m) + g_E - g_I + input

    spikes[t+1] = (V[t+1] >= V_thresh) & not_refractory

Why dual τ matters:
  - τ_E ~  5 ms  (fast ACh via nAChR)
  - τ_I ~ 50 ms  (slower GABA / glutamate inhibition in fly VNC)

The asymmetry gives rise to winner-take-all, gating, and anti-phase CPG
dynamics that collapse when τ_E = τ_I (the v1 single-conductance LIF).

The weight matrices are the "magnitude" matrices (both non-negative):
  W_exc[i,j] = synapse count from j→i if j releases ACh, else 0
  W_inh[i,j] = synapse count from j→i if j releases GABA/Glu, else 0
Modulators (serotonin, dopamine, octopamine, histamine, unclear) contribute
to neither — they're not modeled at the LIF layer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LIFParams:
    """Defaults chosen from the Pass 11 sweep + biology:

      tau_m   = 20 ms   standard neocortical / insect membrane
      tau_e   =  5 ms   fast nAChR (fly ACh) — Nair et al. 2017
      tau_i   = 150 ms  slow mixed GABA-B / GluCl in insect VNC — Lee et al. 2022
      syn_e   = 0.02    tuned so MDN drive produces biologically realistic 5-30 Hz motor pool
      syn_i   = 0.02    1:1 with E; biological rebalancing happens via time constants, not weights

    These give healthy rhythmic motor pool under MDN drive. Suppressive command
    behaviours (DNp09 stop) only partially emerge — see validation 2 report.
    """
    tau_m_ms: float = 20.0
    tau_e_ms: float = 5.0
    tau_i_ms: float = 150.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    refractory_ms: float = 2.0
    dt_ms: float = 1.0
    syn_scale_e: float = 0.02
    syn_scale_i: float = 0.02


class LIFBrain:
    """Dual-τ conductance LIF. Weights are magnitude tensors, not signed.

    Legacy single-tensor bundles (from the pre-dual-τ graph builder) are
    detected and loaded as "all excitatory, no inhibition" so they still run
    (they just produce the old pathological dynamics; use the new builder).
    """

    def __init__(
        self,
        n_neurons: int,
        w_exc_csr: torch.Tensor,
        w_inh_csr: torch.Tensor,
        params: LIFParams,
        device: str = "cpu",
    ):
        assert w_exc_csr.is_sparse_csr and w_inh_csr.is_sparse_csr
        assert w_exc_csr.shape == (n_neurons, n_neurons)
        assert w_inh_csr.shape == (n_neurons, n_neurons)
        self.N = n_neurons
        self.W_exc = w_exc_csr.to(device)
        self.W_inh = w_inh_csr.to(device)
        self.p = params
        self.device = device

        self.v = torch.full((n_neurons,), params.v_rest, device=device)
        self.g_e = torch.zeros(n_neurons, device=device)
        self.g_i = torch.zeros(n_neurons, device=device)
        self.refrac = torch.zeros(n_neurons, device=device)

        self.decay_m = float(torch.exp(torch.tensor(-params.dt_ms / params.tau_m_ms)))
        self.decay_e = float(torch.exp(torch.tensor(-params.dt_ms / params.tau_e_ms)))
        self.decay_i = float(torch.exp(torch.tensor(-params.dt_ms / params.tau_i_ms)))

        self._v_reset_t = torch.tensor(params.v_reset, device=device)
        self._refrac_t = torch.tensor(params.refractory_ms, device=device)

        self.bundle: dict[str, Any] | None = None
        # Compatibility shim for callers inspecting .W (deprecated, prefer W_exc/W_inh)
        self.W = w_exc_csr

    def reset(self) -> None:
        self.v.fill_(self.p.v_rest)
        self.g_e.zero_()
        self.g_i.zero_()
        self.refrac.zero_()

    def step(self, external_input: torch.Tensor) -> torch.Tensor:
        """Advance 1 ms. Returns float32 binary spike vector (N,).

        Caller supplies external drive (descending/sensory). The internal E/I
        conductances carry recurrent synaptic effects from the previous spike,
        so you do NOT separately call propagate().
        """
        # Decay the conductances toward 0 (fast E, slow I).
        self.g_e = self.g_e * self.decay_e
        self.g_i = self.g_i * self.decay_i

        # Membrane leak + conductance effect + external drive.
        self.v = self.p.v_rest + (self.v - self.p.v_rest) * self.decay_m
        self.v = self.v + self.g_e - self.g_i + external_input

        active = self.refrac <= 0
        self.refrac = torch.clamp(self.refrac - self.p.dt_ms, min=0.0)

        spikes = (self.v >= self.p.v_thresh) & active
        self.v = torch.where(spikes, self._v_reset_t, self.v)
        self.refrac = torch.where(spikes, self._refrac_t, self.refrac)

        # Spike propagation: add synaptic current pulses to conductances for
        # the NEXT step. Exc goes to g_e, inh goes to g_i (as magnitude).
        spikes_f = spikes.to(torch.float32)
        exc_input = torch.sparse.mm(self.W_exc, spikes_f.unsqueeze(1)).squeeze(1) * self.p.syn_scale_e
        inh_input = torch.sparse.mm(self.W_inh, spikes_f.unsqueeze(1)).squeeze(1) * self.p.syn_scale_i
        self.g_e = self.g_e + exc_input
        self.g_i = self.g_i + inh_input

        return spikes_f

    def propagate(self, spikes: torch.Tensor) -> torch.Tensor:
        """Removed in the dual-τ model: step() handles recurrence internally.

        Callers must pass ONLY external inputs (descending drive, sensory) to
        step(). Recurrent synaptic current is computed and applied via g_e/g_i
        inside step(). Calling propagate() and adding to external_input would
        double-count the recurrence.
        """
        raise RuntimeError(
            "LIFBrain.propagate() is removed in the dual-τ model. "
            "step() now handles recurrence internally. "
            "Pass only external inputs (descending drive, sensory) to step()."
        )

    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        params: LIFParams | None = None,
        device: str = "cpu",
    ) -> "LIFBrain":
        """Load a graph bundle. Accepts both the new (W_exc, W_inh) format and
        the legacy (weights_csr signed) format.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = torch.load(path, weights_only=False)

        n = bundle["meta"]["n_neurons"]
        if "w_exc_csr" in bundle and "w_inh_csr" in bundle:
            W_exc = bundle["w_exc_csr"]
            W_inh = bundle["w_inh_csr"]
        else:
            # Legacy signed tensor → split by sign.
            W = bundle["weights_csr"].to_sparse_coo().coalesce()
            vals = W.values()
            idx = W.indices()
            exc_mask = vals > 0
            inh_mask = vals < 0
            W_exc = torch.sparse_coo_tensor(
                idx[:, exc_mask], vals[exc_mask], (n, n)
            ).coalesce().to_sparse_csr()
            W_inh = torch.sparse_coo_tensor(
                idx[:, inh_mask], -vals[inh_mask], (n, n)  # negate so stored as magnitude
            ).coalesce().to_sparse_csr()

        brain = cls(n_neurons=n, w_exc_csr=W_exc, w_inh_csr=W_inh,
                    params=params or LIFParams(), device=device)
        brain.bundle = bundle
        return brain
