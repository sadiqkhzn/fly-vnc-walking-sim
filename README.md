# Fly VNC → Robot

A biologically-grounded simulation of the *Drosophila* male ventral nerve cord
(VNC) driving a hexapod robot in physics simulation. The brain is a fixed
leaky-integrate-and-fire reconstruction of the Sept 2026 male CNS connectome
(Cell, 166,700 neurons, 125M connections). Only a thin command interface is
learned; the connectome itself is never modified.

## Claim

Walking is the native function of the VNC. If a complete connectome is
sufficient to reproduce fly behavior, a closed-loop simulation of the VNC
driving a biomechanically accurate fly model should (a) reproduce published
descending-command experiments, and (b) walk under novel sensory conditions.

## Scientific validation (must pass before anything else is interesting)

1. **Tripod gait from descending drive** — tonic MDN / bolt-protocerebrum walk
   DN stimulation produces alternating tripod stepping without sensory input.
   Target: Bidaye et al. 2020, Cande et al. 2018.
2. **Descending stop command** — DNp09 activation halts walking within one
   step cycle. Target: Bidaye et al. 2014.
3. **Asymmetric descending drive → turning** — unilateral DNa01 / DNa02
   activation produces smooth ipsilateral turn. Target: Rayshubskiy et al. 2020.

Each validation experiment outputs a figure alongside the published reference
in `validation/figures/`. Project is not shippable until all three pass.

## Stack

| Layer | Tool |
|---|---|
| Connectome | neuPrint + CAVE (Janelia / HHMI, Sept 2026 male CNS release) |
| Brain sim | PyTorch sparse LIF, 1 ms timestep |
| Biomechanics | NeuroMechFly 2 (EPFL Ramdya lab) on MuJoCo |
| Streaming | FastAPI + WebSockets |
| Viewer | Three.js, neuron positions from neuPrint |

Everything free and open source. See `requirements.txt` for Python deps.

## Scope restriction

Load only the VNC + descending neurons from brain (~15-20k neurons), not the
full 166,700. This is the motor system. Olfaction, memory, and central complex
circuits are out of scope for v1.

## Honest expected results

- Walks reliably on flat ground.
- Handles mild slopes and small obstacles via reflex arcs.
- Fails on rough terrain — the VNC alone cannot plan footholds.
- A trained RL policy would beat it. That is not the point. The point is that
  the connectome, unchanged, reproduces published fly motor behavior.

## Current status (as of Pass 11)

**Working end-to-end:**
- Fetched 24,115 VNC + descending neurons from `male-cns:v1.0` (Sept 2026 release),
  1.77M weight ≥ 3 edges, 22.4M total synapses. Full bilateral symmetry (L=8472, R=8413).
- Dual-timeconstant conductance LIF (fast E, slow I). Split sparse tensors:
  1,026,234 excitatory + 693,369 inhibitory edges. Dale's law via `predictedNt`.
- Brain-only test: zero baseline, MDN drive → motor pool activates widely with
  realistic 5–30 Hz rates and healthy sparseness (27% active).
- Closed loop: MDN drive → spikes → actuators → fly walks (1.17 mm in 400 ms).
- NeuroMechFly bridge: 42 position-controlled actuators, full 3D foot force
  sensors, bilaterally symmetric standing posture matching fly biomechanics.
- Live viewer: FastAPI + Three.js, 24k neurons as InstancedMesh, spike-driven
  glow at 50 Hz, interactive drive controls.

**Validation results (honest):**

| Experiment | Verdict | Detail |
|---|---|---|
| V2 — MDN raises motor pool | ✅ PASS | 349× over baseline |
| V2 — DNp09 suppression | ⚠️ PARTIAL | ratio 0.84 (16% suppression), with correct biological latency (25 ms). Full suppression requires shunting inhibition not representable in current-based point-neuron LIF. |
| V2 — latency criterion | ✅ PASS | 25 ms, matches fly literature |
| V1 — oscillation present | ✅ | rhythmic motor response, not noise |
| V1 — tripod anti-phase | ❌ | brain-only sim cannot break hemisegment symmetry without proprioceptive feedback (Mantziaris 2020) |

**Scientific stance:** we **document findings transparently** instead of tuning
to force passes. The two open gaps map onto two concrete, well-defined next
milestones:

1. **Shunting inhibition** — upgrade LIF to a conductance-based model with
   explicit reversal potentials. This is the standard next step beyond
   current-based LIF (Dayan & Abbott Ch 5.4).
2. **Closed sensory loop** — route the already-exposed `foot_force_vec`,
   `foot_contact`, and `joint_angles` into chordotonal / campaniform sensory
   neurons in the VNC. The sensors are wired through FlyBridge; what's left is
   the encoding function and the appropriate sensory neuron body IDs.

The connectome itself is never modified. The simulator is the thing we iterate on.

## Directory layout

```
src/
  data/        neuPrint fetch, VNC subset extraction
  sim/         LIF simulator, closed-loop runner
  encoding/    sensor → spike encoders, motor spike → torque decoders
  biomech/     NeuroMechFly bridge
  validation/  the three mandatory reproduction experiments
  server/      FastAPI + WebSocket streaming
viewer/        Three.js live visualization
scripts/       one-shot pipeline steps (fetch → build → validate)
```

## Timeline

| Week | Deliverable |
|---|---|
| 1 | neuPrint access, VNC subset extracted, LIF sim produces spikes |
| 2 | NeuroMechFly closed loop with hand-tuned descending commands |
| 3 | All three validation experiments pass |
| 4 | Readout trained over varied terrain |
| 5 | Three.js live viewer streaming over WebSocket |
| 6 | Writeup, demo video, arXiv preprint, LinkedIn post |

## Citations to add on first commit of writeup

- Sept 2026 Cell paper (male CNS connectome)
- June 2026 Nature "Distributed control circuits across a brain-and-cord connectome"
- Lobato-Rios et al. 2022 (NeuroMechFly) + 2024/25 updates
- Bidaye, Cande, Rayshubskiy validation references above
- Phelps et al. 2021 (FANC motor neuron → muscle mapping)
