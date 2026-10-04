# Validation experiments

**These three experiments gate the project.** Nothing else matters until all
three reproduce their published references.

Each script:
1. Loads the connectome subset and builds the LIF brain.
2. Opens FlyBridge (or runs headless if the experiment is pure neural stimulation).
3. Injects a well-defined descending protocol via `command_fn`.
4. Logs motor neuron spike rasters + fly kinematics.
5. Writes a side-by-side figure into `src/validation/figures/` comparing to the
   published figure (paste the published figure in manually as a PNG).

| Script | Protocol | Expected result | Reference |
|---|---|---|---|
| `test_tripod_gait.py` | tonic MDN/bolt walk DNs for 2 s | alternating tripod stepping, ~5-15 Hz | Bidaye 2020, Cande 2018 |
| `test_stop_command.py` | walk for 1 s, then DNp09 for 1 s | halt within one step cycle | Bidaye 2014 |
| `test_turning.py` | unilateral DNa01 or DNa02 drive | smooth ipsilateral turn | Rayshubskiy 2020 |

A failure here does not mean give up — it means the sensory/motor mapping or
the synaptic gain is wrong. Debug there, not in the connectome.
