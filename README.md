# Fly VNC Walking Simulation

**Closed-loop simulation of the *Drosophila* male ventral nerve cord driving a biomechanical fly in MuJoCo.** The brain is a dual-timeconstant leaky integrate-and-fire reconstruction of the Sept 2026 male-cns:v1.0 connectome release (24,115 neurons, 1.7M weighted edges). Only a thin command interface is learned; the connectome itself is never modified.

Includes a live FastAPI + Three.js viewer that streams spikes from the running sim at 50 Hz and renders every neuron at its anatomical position with a stylized 3D fly in the corner.

---

## Scientific claim

Walking is the native function of the VNC. If a complete connectome is sufficient to reproduce fly behavior, a closed-loop simulation of the VNC driving a biomechanically accurate fly model should (a) reproduce published descending-command experiments, and (b) walk under novel sensory conditions.

Validation experiments documented in `src/validation/` compare the sim against published results from Bidaye et al. 2020, Bidaye et al. 2014, Cande et al. 2018, and Rayshubskiy et al. 2020. Current status in [Scientific status](#scientific-status) below.

---

## Repository layout

```
src/
  data/            neuPrint fetch + VNC subset extraction
  sim/             dual-τ LIF simulator, closed-loop runner
  encoding/        sensor → spike encoders, motor spike → torque decoders
  biomech/         NeuroMechFly bridge
  validation/      the mandatory reproduction experiments + figures
  server/          FastAPI + WebSocket for the live viewer
viewer/            Three.js live viewer (vanilla JS, no build step)
scripts/           numbered pipeline steps (fetch → build → run → validate)
data/              cached connectome subset (gitignored; reproducible from scripts)
```

---

## Install

### Prerequisites

- Python 3.12 or 3.13 (3.14 works but torch wheels can lag)
- macOS, Linux, or WSL on Windows
- About 300 MB disk for the cached connectome subset
- A free [neuPrint](https://neuprint.janelia.org) account for the auth token

### Steps

```bash
# clone
git clone https://github.com/sadiqkhzn/fly-vnc-walking-sim.git
cd fly-vnc-walking-sim

# venv
python3 -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# or for exact reproducibility of the author's working environment:
pip install -r requirements-lock.txt
```

### Credentials

Create `.env` (gitignored) at the repo root:

```
NEUPRINT_SERVER=https://neuprint.janelia.org
NEUPRINT_DATASET=male-cns:v1.0
NEUPRINT_TOKEN=<paste-your-token-here>
```

Get your token at <https://neuprint.janelia.org> → Account page.

---

## Reproduce

Each script is numbered and idempotent. First run of `01_fetch_connectome.py` takes ~2 minutes over the network and caches to `data/`. Subsequent runs are near-instant.

```bash
# 1. Fetch the VNC subset from neuPrint (24,115 neurons, 1.77M edges)
python scripts/01_fetch_connectome.py

# 2. Sanity-check cached data (bilateral symmetry, NT distribution, orphan edges)
python scripts/02_sanity.py

# 3. Build the signed sparse weight tensor (one-shot, writes data/vnc_graph.pt)
python scripts/03_build_graph.py

# 4. Brain-only test: MDN drive → motor pool activation (no biomechanics yet)
python scripts/06_brain_only.py

# 5. Fetch per-motor-neuron VNC-ROI assignment (needed by validation experiments)
python scripts/10a_motor_rois.py

# 6. Validation experiments (produce figures + JSON reports)
python src/validation/test_tripod_gait.py
python src/validation/test_stop_command.py

# 7. Closed-loop smoke test: brain spikes → fly actuators → fly moves
python scripts/09_closed_loop.py
```

### Live viewer

```bash
python -m src.server.main
```

Open <http://127.0.0.1:8000>. Buttons on the right drive the sim; the brain
spikes in real time via WebSocket. Drag anywhere to orbit (TrackballControls,
free in any direction).

---

## Simulator design

### Dual-τ conductance LIF (`src/sim/lif.py`)

Standard conductance-based model (Dayan & Abbott 2001, Ch 5):

```
g_E[t+1] = g_E[t] * exp(-dt/τ_E) + (W_exc @ spikes[t]) * syn_scale_e
g_I[t+1] = g_I[t] * exp(-dt/τ_I) + (W_inh @ spikes[t]) * syn_scale_i
V  [t+1] = V_rest + (V[t] - V_rest) * exp(-dt/τ_m) + g_E - g_I + external_input
```

Defaults (from `LIFParams`):
- `τ_m = 20 ms` — membrane
- `τ_E =  5 ms` — fast nAChR
- `τ_I = 150 ms` — slow mixed GABA / GluCl in insect VNC

The asymmetry gives rise to winner-take-all and gating dynamics that collapse when `τ_E = τ_I`. Weight tensors are magnitudes (both non-negative), sign applied via Dale's law in the graph builder.

### Graph build (`scripts/03_build_graph.py`)

Edges are split by presynaptic neurotransmitter class into two sparse CSR tensors:

- **Excitatory** (acetylcholine): 1,026,234 edges
- **Inhibitory** (GABA, glutamate): 693,369 edges
- Modulators (serotonin, dopamine, octopamine, histamine, unclear) are excluded from the LIF layer

### NeuroMechFly bridge (`src/biomech/fly_bridge.py`)

- 42 position-controlled actuators (6 legs × 7 DOF)
- Full 3D foot contact force vectors + binary contact + contact position
- Standing posture produces bilaterally symmetric foot forces matching real fly weight distribution (hind > middle > front)

---

## Scientific status

**Working end-to-end:**
- Fetched 24,115 VNC + descending neurons from `male-cns:v1.0` (Sept 2026)
- 1.77M weight ≥ 3 edges, 22.4M total synapses
- Bilateral symmetry: L=8472, R=8413, M=314
- Signed sparse tensor: 57.9% excitatory / 39.1% inhibitory / 2.9% modulatory
- Brain-only test: MDN drive raises motor pool from 0 → ~30 Hz with 37.5% active cells
- Closed loop: fly produces 1.17 mm of movement in 400 ms of MDN drive

**Validation results (honest):**

| Experiment | Verdict | Detail |
|---|---|---|
| V2 — MDN raises motor pool | ✅ PASS | 349× over baseline |
| V2 — DNp09 suppression | ⚠️ PARTIAL | ratio 0.84 (16% suppression) with correct biological latency (25 ms). Full suppression requires shunting inhibition not representable in current-based point-neuron LIF. |
| V2 — latency criterion | ✅ PASS | 25 ms, matches fly literature |
| V1 — oscillation present | ✅ | rhythmic motor response, not noise |
| V1 — tripod anti-phase | ❌ | brain-only sim cannot break hemisegment symmetry without proprioceptive feedback (Mantziaris 2020) |

The project documents these findings transparently rather than tuning parameters to force passes. The two open gaps map onto two concrete next milestones:

1. **Shunting inhibition** — upgrade LIF to a conductance-based model with explicit reversal potentials (Dayan & Abbott Ch 5.4).
2. **Closed sensory loop** — route the already-exposed `foot_force_vec`, `foot_contact`, and `joint_angles` into chordotonal / campaniform sensory neurons in the VNC. The sensors are wired through FlyBridge; what's left is the encoding function and the appropriate sensory neuron body IDs.

The connectome itself is never modified. The simulator is the thing we iterate on.

---

## References

Primary data:
- **male-cns:v1.0** — [neuprint.janelia.org](https://neuprint.janelia.org), Sept 2026 release
- **NeuroMechFly v2** — <https://github.com/NeLy-EPFL/flygym>

Validation targets:
- Bidaye et al. 2014 — DNp09 stops walking (*Science*)
- Bidaye et al. 2020 — MDN moonwalker command neurons (*Nature*)
- Cande et al. 2018 — Optogenetic activation reveals CPG tripod coordination (*eLife*)
- Rayshubskiy et al. 2020 — Neural circuits for turning (*bioRxiv*)
- Phelps et al. 2021 — Motor neuron atlas (FANC, *Cell*)
- Mantziaris et al. 2020 — Sensory feedback and insect CPGs (*Current Opinion in Insect Science*)

Methods:
- Dayan P. & Abbott L.F. 2001 — *Theoretical Neuroscience*. MIT Press. (Chapter 5 — conductance-based neuron models)

---

## Citation

If you use this work, please cite:

```bibtex
@software{khan_fly_vnc_walking_sim_2026,
  author = {Khan, Sadiq},
  title  = {Fly VNC Walking Simulation},
  year   = 2026,
  url    = {https://github.com/sadiqkhzn/fly-vnc-walking-sim},
  version = {0.1.0}
}
```

Or use the "Cite this repository" button on GitHub (powered by [CITATION.cff](CITATION.cff)).

---

## License

MIT — see [LICENSE](LICENSE).
