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

## Current status (as of Pass 10)

**Working end-to-end:**
- Fetched 24,115 VNC + descending neurons from `male-cns:v1.0` (Sept 2026 release),
  1.77M weight ≥ 3 edges, 22.4M total synapses. Full bilateral symmetry (L=8472, R=8413).
- Signed LIF sparse weight tensor (34 MB). Dale's law via `predictedNt`.
  57.9% excitatory / 39.1% inhibitory, matching standard fly ratios.
- Brain-only test (Pass 6): zero baseline, MDN drive → 334 Hz on MDN cells,
  motor pool jumps 0 → 29.4 Hz, 37.5% of motor cells active. No pathology.
- Closed loop (Pass 9): MDN drive → spikes → actuators → fly moves 1.17 mm in
  400 ms. 15× slower than real-time on CPU.
- NeuroMechFly bridge: 42 position-controlled actuators across 6 legs; standing
  posture produces bilaterally symmetric foot forces matching known fly weight
  distribution (hind > middle > front).

**Open scientific question flagged (Pass 10):**
Validation 1 (tripod gait from tonic MDN) shows motor-pool oscillation (confirmed
rhythm, not noise) but at 24–48 Hz rather than the expected 5–15 Hz, and both
tripod groups fire co-actively (`r(A,B) = +0.20`) rather than anti-phase.
Parameter sweeps over `tau_m ∈ {20, 50, 100} ms` and inhibition scale
`∈ {1, 2}` did not find a point passing both frequency and phasing criteria.

This matches published findings for insect-CPG simulations: reciprocal
half-center oscillation typically requires **proprioceptive load feedback**
to break hemisegment symmetry (Mantziaris et al. 2020). The brain-only
simulation lacks this closed loop; both hemisegments receive identical
descending drive and respond symmetrically.

The project chooses to **document this transparently** rather than tune
parameters to force a passing result. The next milestone is a closed sensory
loop (foot-load → VNC sensory neurons) which is where the biology predicts
anti-phase coordination should emerge.

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
