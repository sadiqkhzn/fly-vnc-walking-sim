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
renderer.autoClear = false;  // we composite main scene + corner fly in the same frame

const scene = new THREE.Scene();
// Fog adds depth: distant neurons fade toward the background. Range tuned
// to brain radius (loadNeurons scales so maxR → 100 world units).
scene.fog = new THREE.Fog(0x050607, 150, 420);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 5000);
// Start at an angled 3D view (anterior-dorsal perspective) so the volume
// reads as 3D from first paint, instead of looking like a flat drawing.
camera.position.set(140, 90, 220);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.9;
controls.autoRotate = false;       // user can enable with a button later
controls.autoRotateSpeed = 0.3;

// -------------- state --------------
let instanced = null;           // THREE.InstancedMesh
let colorAttr = null;           // Float32Array of instance colors (N * 3)
let glow = null;                // Float32Array of per-neuron glow level
let baseColor = null;           // Float32Array of role base colors (N * 3)
let indexCount = 0;
let scaleFactor = 1;

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
flyCamera.position.set(5.5, 3.5, 7.5);
flyCamera.lookAt(0, -0.2, -1.6);

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

// Coordinate convention used below (matches many fly viewers):
//   +Z  → anterior (head direction)
//   -Z  → posterior (abdomen)
//   +Y  → dorsal (up)
// Fly is laid out ~5 units long so lights/distances read nicely.
function buildFly() {
  const g = new THREE.Group();

  // --- Materials (shared) ---
  const matChitin = new THREE.MeshStandardMaterial({
    color: 0x4a3524, roughness: 0.5, metalness: 0.3,  // mid-brown exoskeleton
  });
  const matChitinDark = new THREE.MeshStandardMaterial({
    color: 0x2a1d14, roughness: 0.55, metalness: 0.25,
  });
  const matAbd = new THREE.MeshStandardMaterial({
    color: 0x3d2a1b, roughness: 0.5, metalness: 0.3,
  });
  const matRing = new THREE.MeshStandardMaterial({
    color: 0xb08a4a, roughness: 0.55, metalness: 0.4,  // golden ring between abd segments
  });
  const matLeg = new THREE.MeshStandardMaterial({
    color: 0x1f1510, roughness: 0.75, metalness: 0.1,
  });

  // Translucent head with clearcoat so the mini-brain inside is visible
  const matHead = new THREE.MeshPhysicalMaterial({
    color: 0x8f6c4d, roughness: 0.3, metalness: 0.15,
    transparent: true, opacity: 0.32, clearcoat: 0.75, clearcoatRoughness: 0.3,
    depthWrite: false,
  });

  // Compound eyes — reddish, slightly emissive
  const matEye = new THREE.MeshStandardMaterial({
    color: 0xa02020, roughness: 0.3, metalness: 0.25,
    emissive: 0x401010, emissiveIntensity: 0.4,
  });

  // Iridescent wings — the single trick that makes a procedural fly look alive
  const matWing = new THREE.MeshPhysicalMaterial({
    color: 0xaad4f0, roughness: 0.15, metalness: 0.0,
    transparent: true, opacity: 0.36, side: THREE.DoubleSide,
    iridescence: 0.95, iridescenceIOR: 1.3,
    depthWrite: false,
  });

  // --- THORAX (origin) ---
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(1.0, 40, 32), matChitin);
  thorax.scale.set(1.15, 1.0, 1.3);
  thorax.position.set(0, 0, -1.0);
  g.add(thorax);

  // --- HEAD ---
  const headZ = 0.65;
  const headPos = new THREE.Vector3(0, 0.08, headZ);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.95, 40, 32), matHead);
  head.position.copy(headPos);
  g.add(head);

  // Compound eyes (hemispheres on the sides, tilted outward)
  for (const sx of [-1, 1]) {
    const eye = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.6),
      matEye
    );
    eye.position.set(sx * 0.72, 0.08, headZ + 0.1);
    eye.lookAt(sx * 4, 0.1, headZ + 1.2);
    eye.rotateX(-Math.PI / 2);
    g.add(eye);
  }

  // Antennae — a short pedicel bump + a thin arista bristle, one per side
  for (const sx of [-1, 1]) {
    const ped = new THREE.Mesh(new THREE.SphereGeometry(0.08, 10, 8), matChitinDark);
    ped.position.set(sx * 0.18, 0.42, headZ + 0.4);
    g.add(ped);
    const arista = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.03, 0.4, 8), matChitinDark
    );
    arista.position.set(sx * 0.22, 0.6, headZ + 0.55);
    arista.rotation.x = Math.PI / 3;
    arista.rotation.z = -sx * 0.5;
    g.add(arista);
  }

  // Proboscis (mouthparts): short tapered cylinder jutting down and forward
  const proboscis = new THREE.Mesh(
    new THREE.CylinderGeometry(0.08, 0.13, 0.55, 12), matChitinDark
  );
  proboscis.position.set(0, -0.6, headZ + 0.3);
  proboscis.rotation.x = 0.35;
  g.add(proboscis);

  // --- MINI BRAIN inside the head (same role as the main scene point cloud) ---
  const brainPts = 90;
  const brainGeo = new THREE.BufferGeometry();
  const pos = new Float32Array(brainPts * 3);
  for (let i = 0; i < brainPts; i++) {
    let x, y, z, r2;
    do {
      x = Math.random() * 2 - 1; y = Math.random() * 2 - 1; z = Math.random() * 2 - 1;
      r2 = x*x + y*y + z*z;
    } while (r2 > 1);
    const s = 0.5;
    pos[i*3] = x * s; pos[i*3 + 1] = y * s; pos[i*3 + 2] = z * s;
  }
  brainGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const brainMat = new THREE.PointsMaterial({
    color: 0x5fd3ff, size: 0.08, transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  flyBrainGlow = new THREE.Points(brainGeo, brainMat);
  flyBrainGlow.position.copy(headPos);
  g.add(flyBrainGlow);

  // --- ABDOMEN: ellipsoid with 4 raised torus rings for tergite segmentation ---
  const abdomen = new THREE.Mesh(new THREE.SphereGeometry(1.0, 40, 32), matAbd);
  abdomen.scale.set(0.95, 0.88, 1.9);
  abdomen.position.set(0, -0.2, -3.1);
  g.add(abdomen);

  for (let i = 0; i < 4; i++) {
    const ringR = 0.78 - i * 0.11;
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(ringR, 0.035, 8, 48), matRing
    );
    ring.position.set(0, -0.2, -2.3 - i * 0.55);
    ring.scale.set(1.1, 1.0, 1);  // squish to match abdomen cross-section
    g.add(ring);
  }

  // --- WINGS (two) — iridescent teardrop, hinged above thorax ---
  const wingShape = new THREE.Shape();
  wingShape.moveTo(0, 0);
  wingShape.bezierCurveTo(1.2, 0.6, 3.2, 0.9, 4.2, 0.2);
  wingShape.bezierCurveTo(3.6, -0.5, 1.8, -0.7, 0, -0.25);
  const wingGeo = new THREE.ShapeGeometry(wingShape, 24);

  for (const sx of [-1, 1]) {
    const pivot = new THREE.Group();
    pivot.position.set(sx * 0.55, 0.75, -1.1);
    const wing = new THREE.Mesh(wingGeo, matWing);
    wing.rotation.x = -Math.PI / 2;
    wing.scale.x = sx;
    wing.rotation.z = sx * 0.25;
    pivot.add(wing);
    g.add(pivot);
    flyWings.push({ pivot, sx });
  }

  // --- LEGS: 6, each with hip → femur → knee → tibia (two cylinders, two groups) ---
  for (const sx of [-1, 1]) {
    [-0.3, -1.1, -2.0].forEach((rowZ, i) => {
      const hip = new THREE.Group();
      hip.position.set(sx * 0.9, -0.6, rowZ);
      g.add(hip);

      const femur = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055, 0.04, 1.3, 8), matLeg
      );
      femur.position.set(sx * 0.55, 0.1, 0);
      femur.rotation.z = sx * 1.1;
      femur.rotation.y = (i - 1) * 0.45 * sx;
      hip.add(femur);

      const knee = new THREE.Group();
      knee.position.set(sx * 1.15, 0.42, (i - 1) * 0.5);
      hip.add(knee);

      const tibia = new THREE.Mesh(
        new THREE.CylinderGeometry(0.04, 0.023, 1.75, 8), matLeg
      );
      tibia.position.set(sx * 0.22, -0.85, 0);
      tibia.rotation.z = sx * 0.25;
      knee.add(tibia);
    });
  }

  g.rotation.y = -0.35;
  g.position.y = 0.3;
  flyScene.add(g);
  return g;
}

const flyGroup = buildFly();

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

  // Gentle idle sway for the whole fly
  flyGroup.rotation.y = -0.2 + Math.sin(elapsedS * 0.5) * 0.07;
  flyGroup.position.y = Math.sin(elapsedS * 1.2) * 0.05;
}

function tick() {
  requestAnimationFrame(tick);
  controls.update();

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
  // We leave 36px for the footer, then anchor the fly directly above it.
  const flySize = Math.min(300, window.innerWidth * 0.22);
  const flyX = window.innerWidth - flySize - 20;
  const flyY = 36;
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
