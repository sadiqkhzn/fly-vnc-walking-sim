// Three.js live spike viewer.
//
// Loads neuron positions once from /neurons, then opens WS /spikes.
// InstancedMesh keeps draw calls at 1 even for 24k points. Spikes are drawn
// by writing per-instance color intensities to a GPU buffer, so updating
// activity each frame costs O(spikes), not O(neurons).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

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

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 5000);
camera.position.set(0, 0, 260);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.9;

// -------------- state --------------
let instanced = null;           // THREE.InstancedMesh
let colorAttr = null;           // Float32Array of instance colors (N * 3)
let glow = null;                // Float32Array of per-neuron glow level
let baseColor = null;           // Float32Array of role base colors (N * 3)
let indexCount = 0;
let scaleFactor = 1;

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
  const BRAIN_RADIUS = 100;              // world units; camera sits at ~2.5x this
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
      m.makeTranslation(px, -pz, -py);
    } else {
      m.makeTranslation(1e6, 1e6, 1e6);
    }
    instanced.setMatrixAt(i, m);
  }
  instanced.instanceMatrix.needsUpdate = true;
  instanced.instanceColor.needsUpdate = true;

  scene.add(instanced);
  indexCount = n;
  // Position camera so the whole brain fits.
  camera.position.set(0, 0, 260);
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
function tick() {
  requestAnimationFrame(tick);
  controls.update();

  if (glow && colorAttr) {
    const arr = colorAttr.array;
    const base = baseColor;
    // Decay all glow values and write combined color.
    // Hot loop — kept pure arithmetic for speed.
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

  renderer.render(scene, camera);
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
