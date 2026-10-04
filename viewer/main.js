// Three.js live spike viewer.
//
// Loads neuron positions once from /neurons, then opens WS /spikes.
// InstancedMesh keeps draw calls at 1 even for 24k points. Spikes are drawn
// by writing per-instance color intensities to a GPU buffer, so updating
// activity each frame costs O(spikes), not O(neurons).

import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { createFly } from "./fly_model.js";

// -------------- role → base color (RGB tuple) --------------
const COLORS = {
  command:    [1.00, 0.38, 0.53],  // #ff6188
  descending: [1.00, 0.85, 0.40],  // #ffd866
  motor:      [0.66, 0.86, 0.46],  // #a9dc76
  local:      [0.47, 0.66, 1.00],  // #78a9ff
};
const BASE_INTENSITY = 0.08;     // dim at rest so spikes pop
const SPIKE_INTENSITY = 1.0;
const GLOW_DECAY = 0.94;         // slower decay → spikes linger ~400 ms, readable at 60 fps
const POINT_SIZE = 1.1;

// -------------- Three.js scene setup --------------
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x050607, 1);
renderer.autoClear = false;  // we composite main scene + corner fly in the same frame

const scene = new THREE.Scene();
// Fog adds depth: distant neurons fade toward the background. Range tuned
// to brain radius (loadNeurons scales so maxR → 100 world units) with
// enough slack that at side-profile viewing distances most of the volume
// is still legible.
scene.fog = new THREE.Fog(0x050607, 260, 650);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 5000);
// Camera direction matches the corner fly view EXACTLY:
//   fly camera at (7.0, 4.4, 9.5) → direction (0.515, 0.345, 0.802)
// Scaled to brain radius (~100, so distance ~280) → (144, 97, 225).
// Both views now point at the same unit vector; brain and fly share pose.
camera.position.set(144, 97, 225);

// TrackballControls (not OrbitControls) because OrbitControls locks the
// camera's up vector to world +Y, which creates a gimbal-lock feel near
// the poles. TrackballControls allows free tumbling in any direction.
const controls = new TrackballControls(camera, canvas);
controls.rotateSpeed = 3.5;
controls.zoomSpeed = 1.2;
controls.panSpeed = 0.8;
controls.noZoom = false;
controls.noPan = false;
controls.staticMoving = false;
controls.dynamicDampingFactor = 0.15;

// -------------- state --------------
let instanced = null;           // THREE.InstancedMesh
let colorAttr = null;           // Float32Array of instance colors (N * 3)
let glow = null;                // Float32Array of per-neuron glow level
let baseColor = null;           // Float32Array of role base colors (N * 3)
let indexCount = 0;
let scaleFactor = 1;
let brainPivot = null;          // module scope so the tick loop can auto-rotate
let autoRotate = true;          // horizontal auto-rotation; pauses while the user drags

// -------------- corner fly --------------
// Procedural Drosophila-ish fly, lit by a key + rim light so it reads 3D.
//
// Anatomy (fly convention: +Z = forward, +Y = up):
//   head  (small sphere + 2 compound eyes + 2 antennae)
//   thorax (slightly wider than head, iridescent dark blue-green)
//   abdomen (3 tapered segments, dark with lighter rings between)
//   wings  (teardrop shape, translucent, hinged above thorax; buzz at ~22 Hz)
//   6 legs (coxa + femur + tibia, jointed for proper bend)
//
// All sized in the fly's local units; the camera framing fits the whole fly.
const flyScene = new THREE.Scene();
const flyCamera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
// Pulled back from (5.5, 3.5, 7.5) so the abdomen tip isn't clipped by the
// scissor viewport edges. lookAt still centred on the thorax.
flyCamera.position.set(7.0, 4.4, 9.5);
flyCamera.lookAt(0, -0.3, -1.4);

// Lighting — hemisphere for ambient skydome + key for form + rim for silhouette
flyScene.add(new THREE.HemisphereLight(0xffffff, 0x1a1218, 0.9));
const key = new THREE.DirectionalLight(0xfff4e2, 1.3);
key.position.set(4, 5, 4);
flyScene.add(key);
const rim = new THREE.DirectionalLight(0x8ac4ff, 0.7);
rim.position.set(-4, 2, -3);
flyScene.add(rim);

let flyWings = [];
let flyBrainGlow = null;
let flyBrainGlowBase = 0.2;

// --- Fly avatar ---
// The procedural fly is imported from fly_model.js so the viewer file stays
// focused on brain spike streaming. The imported createFly() returns:
//   group       — the whole fly as a THREE.Group (origin at thorax)
//   wingPivots  — [THREE.Group, THREE.Group] for the wing-buzz animation
//   headGroup   — empty group fixed at the head position; we attach the
//                 mini-brain point cloud as a child so it stays glued to the head
//   eyes        — [Mesh, Mesh] in case we want to pulse eye glow with firing
const fly = createFly();
flyWings = fly.wingPivots.map((pivot, i) => ({ pivot, sx: i === 0 ? 1 : -1 }));
const flyGroup = fly.group;
flyScene.add(flyGroup);


// Mini brain attached to the head group so it tracks any head motion.
(function attachMiniBrain() {
  const n = 110;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let x, y, z, r2;
    do {
      x = Math.random() * 2 - 1; y = Math.random() * 2 - 1; z = Math.random() * 2 - 1;
      r2 = x*x + y*y + z*z;
    } while (r2 > 1);
    const s = 0.22;
    pos[i*3] = x * s; pos[i*3 + 1] = y * s; pos[i*3 + 2] = z * s;
  }
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x5fd3ff, size: 0.06, transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  flyBrainGlow = new THREE.Points(geo, mat);
  fly.headGroup.add(flyBrainGlow);
})();

let totalSpikesInWindow = 0;
let lastRateT = performance.now();
const rateWindowMs = 1000;
let currentFiringHz = 0;

// -------------- API helpers --------------
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return await r.json();
}

async function loadMeta() {
  const m = await fetchJSON("/meta");
  document.getElementById("m-dataset").textContent = m.dataset;
  document.getElementById("m-neurons").textContent = m.n_neurons.toLocaleString();
  document.getElementById("m-edges").textContent = m.n_edges.toLocaleString();
  document.getElementById("m-rate").textContent = `${m.wall_tick_hz} tick/s`;
  document.getElementById("m-drive").textContent = m.drive;
  highlightActiveDrive(m.drive);
  return m;
}

async function loadNeurons() {
  const data = await fetchJSON("/neurons");
  const n = data.length;

  // Pre-pass: compute mean position + extent so we can center and scale.
  let cx = 0, cy = 0, cz = 0, c = 0;
  let maxR = 0;
  for (const d of data) {
    if (d.x == null) continue;
    cx += d.x; cy += d.y; cz += d.z; c++;
  }
  if (c) { cx /= c; cy /= c; cz /= c; }
  for (const d of data) {
    if (d.x == null) continue;
    const r = Math.hypot(d.x - cx, d.y - cy, d.z - cz);
    if (r > maxR) maxR = r;
  }
  // Scale so the whole brain fits in a sphere of radius ~BRAIN_RADIUS world units.
  const BRAIN_RADIUS = 115;              // size tuned for the default framing
  scaleFactor = maxR > 0 ? BRAIN_RADIUS / maxR : 1;
  // male-cns coordinate frame has y ~ dorso-ventral and z ~ anterior-posterior
  // in nanometers. We lay flat with y-up, so swap y <-> z and invert.

  // Point size in world units.
  const POINT_WORLD_SIZE = 0.6;
  const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 });
  instanced = new THREE.InstancedMesh(new THREE.SphereGeometry(POINT_WORLD_SIZE, 6, 6), mat, n);
  instanced.frustumCulled = false;
  // Use Three.js's built-in per-instance color (instanceColor); works cleanly
  // with MeshBasicMaterial without any custom shader surgery.
  instanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  instanced.instanceColor.setUsage(THREE.DynamicDrawUsage);
  colorAttr = instanced.instanceColor;

  glow = new Float32Array(n);
  baseColor = new Float32Array(n * 3);

  const m = new THREE.Matrix4();
  const cc = new THREE.Color();
  let withPos = 0;
  for (let i = 0; i < n; i++) {
    const d = data[i];
    const role = d.role in COLORS ? d.role : "local";
    const col = COLORS[role];
    baseColor[i * 3 + 0] = col[0];
    baseColor[i * 3 + 1] = col[1];
    baseColor[i * 3 + 2] = col[2];
    cc.setRGB(col[0] * BASE_INTENSITY, col[1] * BASE_INTENSITY, col[2] * BASE_INTENSITY);
    instanced.setColorAt(i, cc);

    if (d.x != null) {
      withPos++;
      const px = (d.x - cx) * scaleFactor;
      const py = (d.y - cy) * scaleFactor;
      const pz = (d.z - cz) * scaleFactor;
      // Lay the brain so its long axis (anterior-posterior) is HORIZONTAL in
      // the viewer, matching how the corner fly is drawn: head on the left,
      // body extending right. Mapping:
      //   viewer_x =  pz   anterior (low raw_z) → -X (left), posterior → +X (right)
      //   viewer_y = -py   dorsal (low raw_y)  → +Y (up)
      //   viewer_z =  px   bilateral left-right of the fly, into/out of screen
      m.makeTranslation(pz, -py, px);
    } else {
      m.makeTranslation(1e6, 1e6, 1e6);
    }
    instanced.setMatrixAt(i, m);
  }
  instanced.instanceMatrix.needsUpdate = true;
  instanced.instanceColor.needsUpdate = true;

  // Wrap the instanced mesh in a pivot so we can auto-rotate the whole brain
  // around the vertical axis. Initial X rotation flips the brain upside-down
  // (ventral up), Y rotation gives the 3/4 pose that matches the corner fly
  // on load; the tick loop increments Y after.
  brainPivot = new THREE.Group();
  brainPivot.rotation.x = Math.PI;
  brainPivot.rotation.y = -0.3;
  brainPivot.position.y = 35;     // lift the brain slightly up in the viewport
  brainPivot.add(instanced);
  scene.add(brainPivot);
  indexCount = n;
  // Camera position was set at module load (see top of file). We only tune
  // the orbit limits here; moving the camera again would clobber the
  // deliberately chosen fly-matching angle.
  controls.target.set(0, 0, 0);
  controls.minDistance = 20;
  controls.maxDistance = 800;
  controls.update();

  console.log(`[viewer] mounted ${n} neurons, ${withPos} with coords, scale ${scaleFactor.toExponential(2)}`);
}

// -------------- spike stream --------------
function openSpikeStream() {
  const url = `ws://${location.host}/spikes`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => console.log("[viewer] spike WS connected");
  ws.onclose = () => {
    console.warn("[viewer] spike WS closed; reconnecting in 1s");
    setTimeout(openSpikeStream, 1000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") return;
    const dv = new DataView(ev.data);
    const t = dv.getUint32(0, true);
    const n = dv.getUint16(4, true);
    document.getElementById("m-time").textContent = `${t} ms`;
    totalSpikesInWindow += n;
    if (!glow) return;
    const base = 6;
    for (let k = 0; k < n; k++) {
      const idx = dv.getUint32(base + k * 4, true);
      if (idx < glow.length) glow[idx] = SPIKE_INTENSITY;
    }
  };
}

// -------------- drive buttons --------------
function highlightActiveDrive(name) {
  for (const btn of document.querySelectorAll("#controls button[data-drive]")) {
    btn.classList.toggle("active", btn.dataset.drive === name);
  }
  document.getElementById("m-drive").textContent = name;
}

async function setDrive(name) {
  const r = await fetch("/drive", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command: name }),
  });
  if (r.ok) highlightActiveDrive(name);
}

async function reset() {
  await fetch("/reset", { method: "POST" });
}

for (const btn of document.querySelectorAll("#controls button[data-drive]")) {
  btn.addEventListener("click", () => setDrive(btn.dataset.drive));
}
for (const btn of document.querySelectorAll("#controls button[data-action='reset']")) {
  btn.addEventListener("click", reset);
}

// -------------- animation loop --------------
function animateFly(elapsedS) {
  // Wing buzz: real flies ~200 Hz; visual cap at ~22 Hz. Flap axis is Z
  // (longitudinal) so wings go up/down relative to the thorax.
  const buzzHz = 22;
  const amp = 0.75;
  const theta = Math.sin(elapsedS * buzzHz * 2 * Math.PI) * amp;
  for (const w of flyWings) {
    w.pivot.rotation.z = w.sx * (0.05 + theta * 0.6);
  }

  // Mini-brain intensity tracks the main-sim firing rate smoothly.
  // Target brightness proportional to firing / 5000 Hz, clamped.
  const targetBrightness = Math.min(1.0, currentFiringHz / 5000);
  flyBrainGlowBase += (targetBrightness - flyBrainGlowBase) * 0.08;
  if (flyBrainGlow) {
    const b = 0.25 + flyBrainGlowBase * 0.9;
    flyBrainGlow.material.color.setRGB(0.37 * b * 2, 0.83 * b * 2, 1.0 * b * 2);
    flyBrainGlow.material.opacity = 0.4 + flyBrainGlowBase * 0.55;
    flyBrainGlow.material.size = 0.035 + flyBrainGlowBase * 0.03;
  }

  // Gentle idle sway for the whole fly. The imported model sets its own baseline
  // rotation.y = -0.3; we add a small sinusoidal perturbation on top so the fly
  // breathes without jumping to a different stable orientation.
  flyGroup.rotation.y = -0.3 + Math.sin(elapsedS * 0.5) * 0.06;
  flyGroup.position.y = Math.sin(elapsedS * 1.2) * 0.05;
}

// Pause auto-rotation while the user is interacting with the trackball, resume
// after they let go. Keeps it from fighting their drag.
canvas.addEventListener("mousedown", () => { autoRotate = false; });
canvas.addEventListener("mouseup",   () => { autoRotate = true;  });
canvas.addEventListener("mouseleave", () => { autoRotate = true; });

function tick() {
  requestAnimationFrame(tick);
  controls.update();

  if (autoRotate && brainPivot) {
    brainPivot.rotation.y -= 0.004;   // anti-clockwise (viewed from above)
  }

  if (glow && colorAttr) {
    const arr = colorAttr.array;
    const base = baseColor;
    for (let i = 0; i < glow.length; i++) {
      glow[i] *= GLOW_DECAY;
      const g = glow[i];
      const o = i * 3;
      const intensity = BASE_INTENSITY + g * (1 - BASE_INTENSITY);
      arr[o]     = base[o]     * intensity;
      arr[o + 1] = base[o + 1] * intensity;
      arr[o + 2] = base[o + 2] * intensity;
    }
    colorAttr.needsUpdate = true;
  }

  // Firing-rate HUD, computed over a rolling 1 s window.
  const now = performance.now();
  if (now - lastRateT >= rateWindowMs) {
    currentFiringHz = totalSpikesInWindow / ((now - lastRateT) / 1000);
    document.getElementById("m-firing").textContent =
      `${currentFiringHz.toFixed(0)} spikes/s`;
    totalSpikesInWindow = 0;
    lastRateT = now;
  }

  animateFly(now / 1000);

  // Composite: main scene full-viewport, then overlay fly in bottom-right.
  renderer.clear();
  renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
  renderer.setScissor(0, 0, window.innerWidth, window.innerHeight);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);

  // Bottom-right corner. flyY in WebGL coords = pixels up from bottom.
  // Give the fly extra right-side padding so its abdomen isn't clipped by
  // the viewport edge, and keep the footer line (36 px) clear.
  const flySize = Math.min(320, window.innerWidth * 0.24);
  const flyX = window.innerWidth - flySize - 36;
  const flyY = 44;
  renderer.setViewport(flyX, flyY, flySize, flySize);
  renderer.setScissor(flyX, flyY, flySize, flySize);
  renderer.setScissorTest(true);
  renderer.clearDepth();  // keep main-scene colour but fresh depth for the overlay
  renderer.render(flyScene, flyCamera);
  renderer.setScissorTest(false);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Poll /meta periodically for drive + tick-rate changes.
setInterval(async () => {
  try {
    const m = await fetchJSON("/meta");
    document.getElementById("m-rate").textContent = `${m.wall_tick_hz} tick/s`;
    highlightActiveDrive(m.drive);
  } catch {}
}, 2000);

// Boot.
loadMeta()
  .then(loadNeurons)
  .then(openSpikeStream)
  .then(() => tick())
  .catch((e) => {
    document.getElementById("m-neurons").textContent = "ERROR";
    console.error(e);
  });
