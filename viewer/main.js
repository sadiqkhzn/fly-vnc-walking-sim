// Three.js live spike viewer.
//
// Loads neuron positions once from /neurons, then opens WS /spikes and
// pulses matching points when they fire. One InstancedMesh keeps draw
// calls at 1 for ~20k points.

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const API = "http://127.0.0.1:8000";

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050607);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 10000);
camera.position.set(0, 0, 400);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

let instanced = null;
let glow = null; // Float32Array per-neuron glow level in [0, 1]
let indexById = new Map();

async function loadNeurons() {
  const res = await fetch(`${API}/neurons`);
  const data = await res.json();
  if (!Array.isArray(data)) {
    document.getElementById("neuron-count").textContent = "server not ready";
    return;
  }
  const n = data.length;
  document.getElementById("neuron-count").textContent = n.toLocaleString();

  const geo = new THREE.SphereGeometry(0.6, 6, 6);
  const mat = new THREE.MeshBasicMaterial({ color: 0x5fd3ff });
  instanced = new THREE.InstancedMesh(geo, mat, n);
  glow = new Float32Array(n);

  const m = new THREE.Matrix4();
  for (let i = 0; i < n; i++) {
    const { body_id, xyz } = data[i];
    indexById.set(body_id, i);
    m.makeTranslation(xyz[0], xyz[1], xyz[2]);
    instanced.setMatrixAt(i, m);
  }
  instanced.instanceMatrix.needsUpdate = true;
  scene.add(instanced);
}

function openSpikeStream() {
  const ws = new WebSocket(`ws://127.0.0.1:8000/spikes`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") return; // heartbeat JSON
    if (!glow) return;
    const view = new DataView(ev.data);
    const t = view.getUint32(0, true);
    document.getElementById("sim-t").textContent = `${t} ms`;
    const nActive = view.getUint16(4, true);
    for (let k = 0; k < nActive; k++) {
      const idx = view.getUint32(6 + k * 4, true);
      glow[idx] = 1.0;
    }
  };
}

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  if (instanced && glow) {
    // Simple decay; a shader-based color attribute would be prettier.
    for (let i = 0; i < glow.length; i++) glow[i] *= 0.92;
  }
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

loadNeurons().then(openSpikeStream).then(tick);
