import * as THREE from "three";

const UP = new THREE.Vector3(0, 1, 0);
const DOWN = new THREE.Vector3(0, -1, 0);
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const N = (x, y, z) => new THREE.Vector3(x, y, z).normalize();
const C = (hex) => new THREE.Color(hex);
const clamp01 = (x) => Math.min(Math.max(x, 0), 1);
const smooth = (a, b, x) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const mix = (a, b, t) => a.clone().lerp(b, clamp01(t));

function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- surfaces: a deformed unit SphereGeometry, u (unit vector) -> point ----------

function surfaceNormal(fn, u) {
  const e = 1e-3;
  const t1 = (Math.abs(u.y) < 0.9 ? V(0, 1, 0) : V(1, 0, 0)).cross(u).normalize();
  const t2 = u.clone().cross(t1).normalize();
  const a = fn(u.clone().addScaledVector(t1, e).normalize()).sub(fn(u.clone().addScaledVector(t1, -e).normalize()));
  const b = fn(u.clone().addScaledVector(t2, e).normalize()).sub(fn(u.clone().addScaledVector(t2, -e).normalize()));
  const n = a.cross(b).normalize();
  return n.dot(u) < 0 ? n.negate() : n;
}

function surfaceGeometry(fn, colorFn, ws, hs, polesZ = false) {
  const g = new THREE.SphereGeometry(1, ws, hs);
  if (polesZ) g.rotateX(Math.PI / 2);
  const p = g.attributes.position;
  const n = g.attributes.normal;
  const colors = new Float32Array(p.count * 3);
  const u = V();
  for (let i = 0; i < p.count; i++) {
    u.fromBufferAttribute(p, i).normalize();
    const q = fn(u);
    const nn = surfaceNormal(fn, u);
    p.setXYZ(i, q.x, q.y, q.z);
    n.setXYZ(i, nn.x, nn.y, nn.z);
    const c = colorFn(u);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return g;
}

function surfaceFrame(fn, u, inset = 0) {
  const n = surfaceNormal(fn, u);
  return { pos: fn(u).addScaledVector(n, -inset), n };
}

// ---------- primitives ----------

function capsuleGeometry(len, r0, r1, bulge = 0, radial = 10, rings = 16) {
  const pts = [];
  for (let k = 0; k <= rings; k++) {
    const h = k / rings;
    const cap = Math.sqrt(Math.min(1, h / 0.1, (1 - h) / 0.1));
    const r = (r1 + (r0 - r1) * h + bulge * Math.sin(Math.PI * h)) * cap;
    pts.push(new THREE.Vector2(Math.max(r, 1e-4), -len * (1 - h)));
  }
  return new THREE.LatheGeometry(pts, radial);
}

function taperedTube(curve, r0, r1, tubular = 32, radial = 6) {
  const g = new THREE.TubeGeometry(curve, tubular, 1, radial, false);
  const p = g.attributes.position;
  const c = V();
  const v = V();
  for (let i = 0; i <= tubular; i++) {
    curve.getPointAt(i / tubular, c);
    const r = r0 + (r1 - r0) * (i / tubular);
    for (let j = 0; j <= radial; j++) {
      const k = i * (radial + 1) + j;
      v.fromBufferAttribute(p, k).sub(c).multiplyScalar(r).add(c);
      p.setXYZ(k, v.x, v.y, v.z);
    }
  }
  return g;
}

function bristleGeometry(baseRatio, bend, radial = 5, segs = 6) {
  const g = new THREE.CylinderGeometry(baseRatio * 0.12, baseRatio, 1, radial, segs, true);
  g.translate(0, 0.5, 0);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const h = p.getY(i);
    p.setZ(i, p.getZ(i) + bend * h * h);
  }
  g.computeVertexNormals();
  return g;
}

function basisQuaternion(yAxis, zHint) {
  const y = yAxis.clone().normalize();
  const z = zHint.clone().addScaledVector(y, -zHint.dot(y));
  if (z.lengthSq() < 1e-8) z.crossVectors(y, Math.abs(y.x) < 0.9 ? V(1, 0, 0) : V(0, 0, 1));
  z.normalize();
  const x = V().crossVectors(y, z);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}

// hairs: [{ pos, dir, curl, len }]; local +Y of the geometry follows dir, the bend curls toward curl
function hairs(list, geo, mat) {
  const mesh = new THREE.InstancedMesh(geo, mat, list.length);
  const m = new THREE.Matrix4();
  const s = V();
  list.forEach((h, i) => {
    m.compose(h.pos, basisQuaternion(h.dir, h.curl ?? V(0, 0, -1)), s.set(h.len, h.len, h.len));
    mesh.setMatrixAt(i, m);
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
  return mesh;
}

function ellipsoid(r, mat, ws = 24, hs = 16) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, ws, hs), mat);
  m.scale.set(r[0], r[1], r[2]);
  return m;
}

// ---------- body part shapes ----------

const thoraxFn = (u) => {
  const w = 1 + 0.05 * u.z - 0.1 * u.z * u.z;
  const h = 1 + 0.1 * smooth(-0.3, 1, u.y) * (1 - 0.6 * u.z * u.z) - 0.1 * smooth(-0.3, -1, u.y);
  return V(u.x * 0.84 * w, u.y * 0.68 * h, u.z);
};

const thoraxColor = (() => {
  const pleura = C(0x93714a), scutum = C(0x6b4728), vitta = C(0x4a2f1a), venter = C(0x5a3e26);
  return (u) => {
    const dorsal = smooth(0.05, 0.65, u.y);
    let c = mix(pleura, scutum, dorsal);
    const stripes = Math.exp(-((u.x / 0.07) ** 2)) * 0.6 + Math.exp(-(((Math.abs(u.x) - 0.3) / 0.08) ** 2));
    c = mix(c, vitta, 0.45 * stripes * dorsal * smooth(-0.9, -0.2, u.z) * smooth(1, 0.5, u.z));
    return mix(c, venter, smooth(-0.35, -0.8, u.y));
  };
})();

const HEAD = { x: 0.78, y: 0.66 };
const headFn = (u) => V(u.x * HEAD.x, u.y * HEAD.y * (1 - 0.04 * u.z), u.z * (0.59 + 0.03 * Math.tanh(3 * u.z)));

const headColor = (() => {
  const base = C(0x9a6a3c), frons = C(0xc8692a), face = C(0xdcb676), occiput = C(0x5e4630), ocellar = C(0x2e1c10);
  return (u) => {
    let c = mix(base, occiput, smooth(0.1, -0.5, u.z));
    c = mix(c, frons, smooth(0.25, 0.5, u.z) * smooth(-0.05, 0.2, u.y) * smooth(0.5, 0.3, Math.abs(u.x)));
    c = mix(c, face, smooth(0.35, 0.6, u.z) * smooth(0.05, -0.15, u.y) * smooth(0.45, 0.25, Math.abs(u.x)));
    const tri = Math.hypot(u.x / 0.13, (u.z + 0.08) / 0.16);
    return mix(c, ocellar, smooth(1.1, 0.8, tri) * smooth(0.85, 0.95, u.y));
  };
})();

const AB = { x: 0.72, y: 0.6, z: 0.98 };
const AB_BOUNDS = [1, 0.58, 0.22, -0.12, -0.42, -0.68, -1];
function abSegment(t) {
  for (let i = 0; i < AB_BOUNDS.length - 1; i++) {
    if (t >= AB_BOUNDS[i + 1]) return { i, f: (AB_BOUNDS[i] - t) / (AB_BOUNDS[i] - AB_BOUNDS[i + 1]) };
  }
  return { i: AB_BOUNDS.length - 2, f: 1 };
}
const abBase = (t) => Math.pow(Math.max(1 - t * t, 0), t < 0 ? 0.8 : 0.5) * (1 + 0.12 * t);
const abLip = (t) => {
  const { i, f } = abSegment(t);
  return i === 0 || i === 5 ? 1 : 0.975 + 0.05 * smooth(0, 0.95, f);
};
const abdomenFn = (u) => {
  const rho = Math.hypot(u.x, u.y);
  const r = rho > 1e-9 ? (abBase(u.z) * abLip(u.z)) / rho : 0;
  return V(u.x * AB.x * r, u.y * AB.y * r, u.z * AB.z);
};

const abdomenColor = (() => {
  const tergite = C(0xc99a52), band = C(0x3a2312), venter = C(0xe2c999), sternite = C(0xc4a067);
  return (u) => {
    const { i, f } = abSegment(u.z);
    const dorsal = smooth(-0.45, 0.35, u.y);
    const th = 0.52 + 0.35 * (1 - dorsal);
    let b = smooth(th, th + 0.12, f) * smooth(-0.3, 0.1, u.y);
    if (i === 0) b *= 0.25;
    if (i >= 4) b = Math.max(b, (i === 5 ? 0.85 : 0.35) * smooth(-0.2, 0.3, u.y));
    let low = mix(venter, sternite, smooth(0.32, 0.22, Math.abs(u.x)) * smooth(0.2, 0.75, f) * smooth(0.05, 0.3, 1 - f));
    return mix(mix(low, tergite, dorsal), band, b);
  };
})();

// ---------- wing ----------

const WING_L = 3.5;
const wingCamber = (x, y) => {
  const u = x / WING_L, v = y / WING_L;
  return 0.03 * WING_L * Math.sin(Math.PI * clamp01(u)) * Math.max(0, 1 - ((v - 0.05) / 0.34) ** 2) * smooth(0, 0.25, u);
};
const wingPt = (u, v, lift = 0.006) => V(u * WING_L, v * WING_L, wingCamber(u * WING_L, v * WING_L) + lift);

function wingShape() {
  const L = WING_L;
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(0.05 * L, -0.06 * L, 0.1 * L, -0.08 * L, 0.18 * L, -0.09 * L);
  s.bezierCurveTo(0.4 * L, -0.12 * L, 0.65 * L, -0.19 * L, 0.82 * L, -0.19 * L);
  s.bezierCurveTo(0.97 * L, -0.19 * L, 1.02 * L, -0.05 * L, 0.99 * L, 0.05 * L);
  s.bezierCurveTo(0.96 * L, 0.18 * L, 0.82 * L, 0.3 * L, 0.62 * L, 0.31 * L);
  s.bezierCurveTo(0.42 * L, 0.32 * L, 0.25 * L, 0.24 * L, 0.16 * L, 0.16 * L);
  s.bezierCurveTo(0.12 * L, 0.17 * L, 0.08 * L, 0.13 * L, 0.07 * L, 0.09 * L);
  s.bezierCurveTo(0.05 * L, 0.07 * L, 0.02 * L, 0.03 * L, 0, 0);
  return s;
}

function wingMembraneGeometry(shape) {
  let src = new THREE.ShapeGeometry(shape, 24).toNonIndexed().attributes.position.array;
  for (let k = 0; k < 3; k++) {
    const out = new Float32Array(src.length * 4);
    let o = 0;
    for (let t = 0; t < src.length; t += 9) {
      const a = [src[t], src[t + 1]], b = [src[t + 3], src[t + 4]], c = [src[t + 6], src[t + 7]];
      const ab = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const bc = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2];
      const ca = [(c[0] + a[0]) / 2, (c[1] + a[1]) / 2];
      for (const tri of [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]) {
        for (const q of tri) {
          out[o++] = q[0];
          out[o++] = q[1];
          out[o++] = 0;
        }
      }
    }
    src = out;
  }
  const normals = new Float32Array(src.length);
  const e = 1e-3;
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i], y = src[i + 1];
    src[i + 2] = wingCamber(x, y);
    const n = V(-(wingCamber(x + e, y) - wingCamber(x - e, y)) / (2 * e), -(wingCamber(x, y + e) - wingCamber(x, y - e)) / (2 * e), 1).normalize();
    normals.set([n.x, n.y, n.z], i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(src, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  return g;
}

const VEINS = {
  L1: [[0.03, -0.03], [0.15, -0.075], [0.3, -0.11], [0.43, -0.135]],
  L2: [[0.2, -0.04], [0.4, -0.08], [0.6, -0.13], [0.8, -0.175], [0.86, -0.183]],
  L3: [[0.04, -0.01], [0.25, -0.03], [0.5, -0.045], [0.75, -0.05], [0.985, -0.035]],
  L4: [[0.06, 0.015], [0.25, 0.045], [0.5, 0.085], [0.75, 0.12], [0.94, 0.14]],
  L5: [[0.07, 0.04], [0.25, 0.1], [0.45, 0.17], [0.62, 0.23], [0.75, 0.275]],
  L6: [[0.08, 0.06], [0.18, 0.12], [0.3, 0.18]],
};

function veinAt(pts, u) {
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    if (u >= a[0] && u <= b[0]) return [u, a[1] + ((b[1] - a[1]) * (u - a[0])) / (b[0] - a[0])];
  }
  return pts[pts.length - 1];
}

function buildWing(M, G, R) {
  const plane = new THREE.Group();
  plane.rotation.x = -Math.PI / 2;
  const shape = wingShape();

  const membrane = new THREE.Mesh(wingMembraneGeometry(shape), M.wing);
  membrane.renderOrder = 3;
  plane.add(membrane);

  const veinMesh = (pts, r0, r1) => {
    const curve = new THREE.CatmullRomCurve3(pts.map(([u, v]) => wingPt(u, v)));
    const m = new THREE.Mesh(taperedTube(curve, r0, r1, Math.max(8, pts.length * 12), 6), M.vein);
    m.renderOrder = 2;
    plane.add(m);
    return curve;
  };

  const outline = (from, to, step = 0.04) => {
    const pts = [];
    for (let i = from; i < to; i++) {
      for (let t = 0; t < 1; t += step) pts.push(shape.curves[i].getPoint(t));
    }
    return pts;
  };

  const costaPts = outline(0, 2).concat([0, 0.25, 0.5, 0.7].map((t) => shape.curves[2].getPoint(t)));
  const costa = veinMesh(costaPts.map((p) => [p.x / WING_L, p.y / WING_L + 0.004]), 0.017, 0.009);
  veinMesh(VEINS.L1, 0.012, 0.008);
  veinMesh(VEINS.L2, 0.009, 0.006);
  veinMesh(VEINS.L3, 0.011, 0.006);
  veinMesh(VEINS.L4, 0.009, 0.005);
  veinMesh(VEINS.L5, 0.009, 0.005);
  veinMesh(VEINS.L6, 0.008, 0.003);
  veinMesh([veinAt(VEINS.L3, 0.4), veinAt(VEINS.L4, 0.4)], 0.008, 0.008);
  veinMesh([veinAt(VEINS.L4, 0.6), veinAt(VEINS.L5, 0.66)], 0.008, 0.008);
  veinMesh([[0.12, -0.085], [0.12, -0.04]], 0.007, 0.007);
  const marginPts = outline(2, 7, 0.02).slice(36).map((p) => [p.x / WING_L, p.y / WING_L]);
  veinMesh(marginPts, 0.005, 0.004);

  const fringe = [];
  for (let t = 0.02; t < 0.98; t += 0.016) {
    const p = costa.getPointAt(t);
    const tan = costa.getTangentAt(t);
    fringe.push({ pos: p, dir: tan.clone().multiplyScalar(1.4).add(V(0, -1, 0.35)), curl: V(0, 0, 1), len: 0.05 + 0.03 * R() });
  }
  for (let i = 0; i < marginPts.length; i += 2) {
    const [u, v] = marginPts[i];
    const p = wingPt(u, v, 0.002);
    const centre = V(0.55 * WING_L, 0.05 * WING_L, p.z);
    fringe.push({ pos: p, dir: p.clone().sub(centre).setZ(0).normalize().add(V(0.25, 0, 0.05)), curl: V(0, 0, 1), len: 0.05 + 0.02 * R() });
  }
  plane.add(hairs(fringe, G.micro, M.hair));

  const base = ellipsoid([0.09, 0.05, 0.035], M.sclerite, 12, 8);
  base.position.set(0.02, 0, 0.01);
  plane.add(base);
  return plane;
}

// ---------- legs ----------

const LEGS = [
  { name: "prothoracic", hipU: [0.36, -0.78, 0.55], coxa: [0.06, -0.28, 0.12], knee: [0.95, -0.42, 1.02], tibia: [1.2, -1.04, 1.5], heading: [0.35, 0, 1] },
  { name: "mesothoracic", hipU: [0.5, -0.82, 0.05], coxa: [0.1, -0.26, 0], knee: [1.3, -0.3, 0.02], tibia: [1.82, -1.04, -0.25], heading: [1, 0, -0.25] },
  { name: "metathoracic", hipU: [0.42, -0.78, -0.45], coxa: [0.08, -0.26, -0.12], knee: [1.12, -0.32, -1.0], tibia: [1.5, -1.04, -1.75], heading: [0.45, 0, -1] },
];
const TARSI = [
  [0.2, 0.026, 0.022, 0.55],
  [0.09, 0.022, 0.02, 0.3],
  [0.075, 0.02, 0.019, 0.15],
  [0.065, 0.019, 0.018, 0.05],
  [0.08, 0.018, 0.016, 0.0],
];

function link(parent, parentQ, offset, dir) {
  const g = new THREE.Group();
  g.position.set(0, -offset, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(DOWN, dir.clone().normalize());
  g.quaternion.copy(parentQ.clone().invert().multiply(q));
  parent.add(g);
  return [g, q];
}

function segmentHairs(len, r, count, lenRange, R, start = 0.12) {
  const list = [];
  for (let i = 0; i < count; i++) {
    const h = start + (1 - start - 0.06) * ((i + R() * 0.6) / count);
    const a = i * 2.39996 + R() * 0.4;
    const n = V(Math.cos(a), 0, Math.sin(a));
    list.push({ pos: n.clone().multiplyScalar(r * 0.9).setY(-h * len), dir: n.clone().multiplyScalar(0.5).add(DOWN), curl: n.clone().negate(), len: lenRange[0] + R() * (lenRange[1] - lenRange[0]) });
  }
  return list;
}

function buildLeg(spec, s, M, G, R) {
  const mir = (a) => V(a[0] * s, a[1], a[2]);
  const hip = thoraxFn(N(spec.hipU[0] * s, spec.hipU[1], spec.hipU[2])).multiplyScalar(0.95);
  const coxaEnd = hip.clone().add(mir(spec.coxa));
  const knee = mir(spec.knee);
  const tibiaEnd = mir(spec.tibia);

  const root = new THREE.Group();
  root.name = `${spec.name}_${s > 0 ? "R" : "L"}`;
  root.position.copy(hip);

  const segs = [
    ["coxa", coxaEnd.clone().sub(hip), 0.075, 0.06, 0.012, 6, [0.04, 0.07]],
    ["femur", knee.clone().sub(coxaEnd), 0.056, 0.042, 0.02, 28, [0.05, 0.09]],
    ["tibia", tibiaEnd.clone().sub(knee), 0.042, 0.033, 0.004, 34, [0.04, 0.08]],
  ];
  let parent = root, parentQ = new THREE.Quaternion(), offset = 0;
  for (const [key, d, r0, r1, bulge, count, hl] of segs) {
    const len = d.length();
    const [g, q] = link(parent, parentQ, offset, d);
    g.name = key;
    g.add(new THREE.Mesh(capsuleGeometry(len, r0, r1, bulge), M.leg));
    g.add(hairs(segmentHairs(len, (r0 + r1) / 2 + bulge * 0.7, count, hl, R), G.micro, M.hair));
    if (key === "femur") {
      const tro = new THREE.Mesh(capsuleGeometry(0.1, 0.05, 0.05, 0.01, 8, 8), M.joint);
      tro.position.y = 0.03;
      g.add(tro);
    }
    if (key === "tibia") {
      const spurs = [0, 2.1].map((a) => ({ pos: V(Math.cos(a) * 0.03, -len * 0.97, Math.sin(a) * 0.03), dir: V(Math.cos(a) * 0.3, -1, Math.sin(a) * 0.3), curl: V(-Math.cos(a), 0, -Math.sin(a)), len: 0.07 }));
      spurs.push({ pos: V(0.035, -len * 0.82, 0), dir: V(0.6, -1, 0), curl: V(-1, 0, 0), len: 0.1 });
      g.add(hairs(spurs, G.macro, M.hair));
    }
    root.userData[key] = g;
    parent = g;
    parentQ = q;
    offset = len;
  }

  const heading = mir(spec.heading).setY(0).normalize();
  const tarsus = [];
  for (const [len, r0, r1, pitch] of TARSI) {
    const dir = heading.clone().multiplyScalar(Math.cos(pitch)).addScaledVector(DOWN, Math.sin(pitch) + 0.02);
    const [g, q] = link(parent, parentQ, offset, dir);
    g.add(new THREE.Mesh(capsuleGeometry(len, r0, r1, 0.003, 8, 10), M.leg));
    g.add(hairs(segmentHairs(len, (r0 + r1) / 2, 5, [0.03, 0.05], R, 0.2), G.micro, M.hair));
    tarsus.push(g);
    parent = g;
    parentQ = q;
    offset = len;
  }
  root.userData.tarsus = tarsus;

  const inv = parentQ.clone().invert();
  const toLocal = (v) => v.clone().applyQuaternion(inv);
  const side = V().crossVectors(heading, UP).normalize();
  const claws = [-1, 1].map((k) => ({
    pos: V(0, -offset, 0).add(toLocal(side).multiplyScalar(0.012 * k)),
    dir: toLocal(heading.clone().addScaledVector(DOWN, 0.6).addScaledVector(side, 0.35 * k)),
    curl: toLocal(DOWN),
    len: 0.06,
  }));
  parent.add(hairs(claws, G.claw, M.hair));
  for (const k of [-1, 1]) {
    const pad = ellipsoid([0.018, 0.01, 0.026], M.pulvillus, 10, 8);
    pad.position.copy(V(0, -offset + 0.005, 0).add(toLocal(side.clone().multiplyScalar(0.014 * k).addScaledVector(DOWN, 0.014))));
    pad.quaternion.copy(basisQuaternion(toLocal(DOWN).negate(), toLocal(heading)));
    parent.add(pad);
  }
  return root;
}

// ---------- fly ----------

export function createFly() {
  const R = rng(7);
  const group = new THREE.Group();
  group.name = "Drosophila";

  const M = {
    thorax: new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      metalness: 0.15,
      roughness: 0.42,
      clearcoat: 0.55,
      clearcoatRoughness: 0.35,
      iridescence: 0.45,
      iridescenceIOR: 1.5,
      iridescenceThicknessRange: [250, 650],
      sheen: 0.25,
      sheenColor: new THREE.Color(0x9a8a78),
      sheenRoughness: 0.7,
    }),
    head: new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      roughness: 0.3,
      clearcoat: 0.7,
      clearcoatRoughness: 0.15,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
    }),
    abdomen: new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      roughness: 0.48,
      clearcoat: 0.45,
      clearcoatRoughness: 0.3,
      iridescence: 0.15,
      iridescenceIOR: 1.4,
      sheen: 0.4,
      sheenColor: new THREE.Color(0xbfa98a),
      sheenRoughness: 0.6,
    }),
    tergite: new THREE.MeshPhysicalMaterial({ color: 0x3a2312, roughness: 0.4, clearcoat: 0.5, clearcoatRoughness: 0.3 }),
    leg: new THREE.MeshPhysicalMaterial({ color: 0x5a3820, roughness: 0.45, clearcoat: 0.35, clearcoatRoughness: 0.4 }),
    joint: new THREE.MeshStandardMaterial({ color: 0x3a2414, roughness: 0.5 }),
    hair: new THREE.MeshStandardMaterial({ color: 0x140d08, roughness: 0.45, metalness: 0.25 }),
    antenna: new THREE.MeshStandardMaterial({ color: 0x9c5a2a, roughness: 0.55 }),
    funiculus: new THREE.MeshPhysicalMaterial({ color: 0xb36d34, roughness: 0.8, sheen: 1, sheenColor: new THREE.Color(0xe8c9a0), sheenRoughness: 0.5 }),
    rostrum: new THREE.MeshStandardMaterial({ color: 0xc9a46c, roughness: 0.6 }),
    haustellum: new THREE.MeshPhysicalMaterial({ color: 0x8a5a30, roughness: 0.45, clearcoat: 0.4 }),
    labellum: new THREE.MeshPhysicalMaterial({ color: 0xd8b483, roughness: 0.5, sheen: 0.6, sheenColor: new THREE.Color(0xffe6c0) }),
    ocellus: new THREE.MeshPhysicalMaterial({ color: 0x5a1a0a, roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.05 }),
    haltere: new THREE.MeshPhysicalMaterial({ color: 0xe3c890, roughness: 0.45, clearcoat: 0.3 }),
    neck: new THREE.MeshStandardMaterial({ color: 0xa88454, roughness: 0.6 }),
    spiracle: new THREE.MeshStandardMaterial({ color: 0x1e140c, roughness: 0.6 }),
    pulvillus: new THREE.MeshPhysicalMaterial({ color: 0xe0cc9e, roughness: 0.3, transmission: 0.2, clearcoat: 0.6 }),
    sclerite: new THREE.MeshStandardMaterial({ color: 0x3e2816, roughness: 0.5 }),
    wing: new THREE.MeshPhysicalMaterial({
      color: 0xdde6ee,
      metalness: 0,
      roughness: 0.12,
      iridescence: 0.95,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [200, 650],
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
    vein: new THREE.MeshStandardMaterial({ color: 0x4a3420, roughness: 0.5, transparent: true, opacity: 0.75, depthWrite: false }),
  };
  M.scutellum = M.thorax;

  const G = {
    macro: bristleGeometry(0.03, 0.12, 5, 6),
    micro: bristleGeometry(0.05, 0.18, 4, 3),
    claw: bristleGeometry(0.12, 0.45, 5, 5),
  };

  // Thorax
  const thorax = new THREE.Mesh(surfaceGeometry(thoraxFn, thoraxColor, 64, 48), M.thorax);
  thorax.name = "thorax";
  group.add(thorax);

  const scutTop = thoraxFn(N(0, Math.sqrt(1 - 0.8 * 0.8), -0.8));
  const scutFn = (u) => V(u.x * 0.3, u.y * 0.15, u.z * 0.24);
  const scutellum = new THREE.Mesh(surfaceGeometry(scutFn, (u) => mix(C(0x6b4728), C(0x8a6036), smooth(0, -1, u.z)), 32, 20), M.scutellum);
  scutellum.position.set(0, scutTop.y - 0.04, -0.8);
  group.add(scutellum);

  for (const s of [1, -1]) {
    for (const [u, sz] of [[[0.86, 0.12, 0.48], 0.035], [[0.82, 0.05, -0.42], 0.03]]) {
      const f = surfaceFrame(thoraxFn, N(u[0] * s, u[1], u[2]));
      const sp = ellipsoid([sz, sz * 0.4, sz * 0.7], M.spiracle, 10, 6);
      sp.position.copy(f.pos);
      sp.quaternion.copy(basisQuaternion(f.n, V(0, 1, 0)));
      group.add(sp);
    }
  }

  const macro = [];
  const thoraxBristle = (u, len, lateral, s) => {
    const f = surfaceFrame(thoraxFn, N(u[0] * s, u[1], u[2]), 0.01);
    macro.push({ pos: f.pos, dir: f.n.clone().multiplyScalar(0.5).add(V(lateral * s, 0, -1)), curl: f.n.clone().negate(), len });
  };
  for (const s of [1, -1]) {
    thoraxBristle([0.26, 0.92, 0.1], 0.42, 0.05, s);
    thoraxBristle([0.27, 0.86, -0.38], 0.56, 0.05, s);
    thoraxBristle([0.68, 0.42, 0.6], 0.34, 0.7, s);
    thoraxBristle([0.72, 0.3, 0.52], 0.28, 0.8, s);
    thoraxBristle([0.8, 0.42, 0.28], 0.4, 0.6, s);
    thoraxBristle([0.82, 0.4, 0.05], 0.4, 0.6, s);
    thoraxBristle([0.62, 0.6, 0.32], 0.34, 0.4, s);
    thoraxBristle([0.66, 0.62, -0.12], 0.42, 0.45, s);
    thoraxBristle([0.6, 0.6, -0.32], 0.36, 0.4, s);
    thoraxBristle([0.5, 0.6, -0.6], 0.4, 0.3, s);
    thoraxBristle([0.42, 0.62, -0.72], 0.34, 0.25, s);
    thoraxBristle([0.88, -0.2, 0.32], 0.3, 0.4, s);
    thoraxBristle([0.86, -0.32, 0.12], 0.26, 0.4, s);
    const sp = scutellum.position;
    macro.push({ pos: V(0.24 * s, 0.04, -0.12).add(sp), dir: V(0.25 * s, 0.45, -1), curl: V(0, -1, 0), len: 0.45 });
    macro.push({ pos: V(0.09 * s, 0.05, -0.22).add(sp), dir: V(-0.3 * s, 0.4, -1), curl: V(0, -1, 0), len: 0.66 });
  }
  group.add(hairs(macro, G.macro, M.hair));

  const micro = [];
  for (let a = -0.62; a <= 0.62; a += 0.062) {
    for (let b = -0.66; b <= 0.82; b += 0.068) {
      const x = a + (R() - 0.5) * 0.03, z = b + (R() - 0.5) * 0.04;
      const yy = 1 - x * x - z * z;
      if (yy < 0.18) continue;
      const f = surfaceFrame(thoraxFn, N(x, Math.sqrt(yy), z), 0.004);
      micro.push({ pos: f.pos, dir: f.n.clone().multiplyScalar(0.35).add(V(x * 0.3, 0, -1)), curl: f.n.clone().negate(), len: 0.07 + 0.03 * R() });
    }
  }
  group.add(hairs(micro, G.micro, M.hair));

  // Neck
  const neck = new THREE.Mesh(capsuleGeometry(0.3, 0.17, 0.2, 0.02, 16, 10), M.neck);
  neck.rotation.x = -Math.PI / 2;
  neck.position.set(0, 0.04, 0.86);
  group.add(neck);

  // Head
  const head = new THREE.Group();
  head.name = "head";
  head.position.set(0, 0.05, 1.6);
  group.add(head);

  const headMesh = new THREE.Mesh(surfaceGeometry(headFn, headColor, 64, 48), M.head);
  headMesh.renderOrder = 4;
  head.add(headMesh);

  const headGroup = new THREE.Group();
  headGroup.name = "headAnchor";
  head.add(headGroup);

  const eyes = [];
  const eyeR = 0.4 * HEAD.x;
  for (const s of [1, -1]) {
    const eyeMat = new THREE.MeshPhysicalMaterial({
      color: 0xa3120f,
      emissive: 0x5a0505,
      emissiveIntensity: 0.35,
      roughness: 0.38,
      clearcoat: 0.6,
      clearcoatRoughness: 0.25,
    });
    const f = surfaceFrame(headFn, N(s, 0.12, 0.22), 0.11);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(eyeR, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2), eyeMat);
    eye.name = s > 0 ? "eye_R" : "eye_L";
    eye.position.copy(f.pos);
    eye.quaternion.copy(basisQuaternion(f.n, V(0, 1, 0)));
    eye.scale.set(1, 0.62, 1.32);

    const count = 760;
    const facet = new THREE.InstancedMesh(new THREE.SphereGeometry(0.0165, 6, 2, 0, Math.PI * 2, 0, Math.PI / 2), eyeMat, count);
    const m = new THREE.Matrix4();
    const sc = V(1, 0.7, 1);
    for (let i = 0; i < count; i++) {
      const y = 1 - ((i + 0.5) / count) * 0.94;
      const r = Math.sqrt(1 - y * y);
      const th = i * 2.399963;
      const n = V(r * Math.cos(th), y, r * Math.sin(th));
      m.compose(n.clone().multiplyScalar(eyeR * 0.99), new THREE.Quaternion().setFromUnitVectors(UP, n), sc);
      facet.setMatrixAt(i, m);
    }
    facet.instanceMatrix.needsUpdate = true;
    facet.frustumCulled = false;
    eye.add(facet);
    head.add(eye);
    eyes.push(eye);
  }

  for (const s of [1, -1]) {
    for (const [u, sz] of [[[0, 0.99, -0.02], 0.032], [[0.07 * s, 0.96, -0.16], 0.028]]) {
      if (u[0] === 0 && s < 0) continue;
      const f = surfaceFrame(headFn, N(...u));
      const o = ellipsoid([sz, sz * 0.6, sz], M.ocellus, 12, 8);
      o.position.copy(f.pos);
      o.quaternion.copy(basisQuaternion(f.n, V(0, 0, 1)));
      head.add(o);
    }
  }

  const headHairs = [];
  const headMicro = [];
  const headBristle = (u, dir, len, list = headHairs) => {
    const f = surfaceFrame(headFn, N(...u), 0.008);
    list.push({ pos: f.pos, dir: N(...dir).add(f.n.clone().multiplyScalar(0.3)), curl: f.n.clone().negate(), len });
  };
  for (const s of [1, -1]) {
    headBristle([0.05 * s, 0.98, -0.06], [0.4 * s, 0.6, 0.8], 0.3);
    headBristle([0.06 * s, 0.9, -0.42], [-0.5 * s, 0.4, -1], 0.24);
    headBristle([0.32 * s, 0.86, -0.3], [-0.3 * s, 0.5, -1], 0.36);
    headBristle([0.48 * s, 0.78, -0.3], [0.8 * s, 0.4, -0.7], 0.32);
    headBristle([0.4 * s, 0.76, 0.42], [-0.1 * s, 0.8, -0.6], 0.28);
    headBristle([0.4 * s, 0.66, 0.58], [0.1 * s, 0.7, 0.7], 0.22);
    headBristle([0.38 * s, 0.55, 0.68], [0, 0.8, -0.5], 0.2);
    headBristle([0.22 * s, -0.45, 0.85], [0.2 * s, -0.4, 1], 0.32);
    for (let i = 0; i < 9; i++) {
      const y = -0.45 + i * 0.11;
      headBristle([0.78 * s, y, -0.55], [0.3 * s, 0.1, -1], 0.07 + 0.02 * R(), headMicro);
    }
    for (let i = 0; i < 10; i++) {
      headBristle([(0.12 + 0.22 * R()) * s, 0.2 + 0.6 * R(), 0.75], [0, 1, 0.2], 0.05 + 0.02 * R(), headMicro);
      headBristle([(0.3 + 0.4 * R()) * s, -0.55 - 0.3 * R(), 0.1 + 0.4 * R()], [0.2 * s, -1, 0.2], 0.06 + 0.03 * R(), headMicro);
    }
  }
  head.add(hairs(headHairs, G.macro, M.hair));
  head.add(hairs(headMicro, G.micro, M.hair));

  // Antennae
  for (const s of [1, -1]) {
    const pa = surfaceFrame(headFn, N(0.13 * s, 0.24, 0.96), 0.01).pos;
    const antenna = new THREE.Group();
    antenna.name = s > 0 ? "antenna_R" : "antenna_L";
    antenna.position.copy(pa);
    head.add(antenna);

    const scape = ellipsoid([0.035, 0.035, 0.04], M.antenna, 12, 8);
    scape.position.set(0.01 * s, 0, 0.03);
    const pedicel = ellipsoid([0.06, 0.055, 0.06], M.antenna, 16, 12);
    pedicel.position.set(0.025 * s, 0, 0.085);
    const p2 = V(0.045 * s, -0.11, 0.13);
    const funiculus = ellipsoid([0.06, 0.12, 0.055], M.funiculus, 20, 14);
    funiculus.position.copy(p2);
    funiculus.rotation.x = -0.3;
    antenna.add(scape, pedicel, funiculus);

    const b = p2.clone().add(V(0.04 * s, 0.07, 0.03));
    const curve = new THREE.QuadraticBezierCurve3(b, b.clone().add(V(0.12 * s, 0.1, 0.22)), b.clone().add(V(0.22 * s, 0.34, 0.4)));
    antenna.add(new THREE.Mesh(taperedTube(curve, 0.012, 0.003, 24, 6), M.hair));
    const branches = [];
    for (let i = 0; i < 11; i++) {
      const dorsal = i < 7;
      const t = dorsal ? 0.3 + i * 0.1 : 0.4 + (i - 7) * 0.13;
      const p = curve.getPointAt(Math.min(t, 0.95));
      const tan = curve.getTangentAt(Math.min(t, 0.95));
      const side = UP.clone().addScaledVector(tan, -UP.dot(tan)).normalize().multiplyScalar(dorsal ? 1 : -1);
      branches.push({ pos: p, dir: tan.clone().multiplyScalar(0.7).add(side), curl: tan, len: (dorsal ? 0.11 : 0.08) * (1.1 - t * 0.6) });
    }
    branches.push({ pos: pedicel.position.clone().add(V(0.02 * s, 0.04, 0.02)), dir: V(0.3 * s, 1, 0.4), curl: V(0, 0, 1), len: 0.08 });
    antenna.add(hairs(branches, G.micro, M.hair));
  }

  // Proboscis
  const pb = surfaceFrame(headFn, N(0, -0.8, 0.6), 0.04).pos;
  const proboscis = new THREE.Group();
  proboscis.name = "proboscis";
  proboscis.position.copy(pb);
  proboscis.rotation.x = -0.5;
  head.add(proboscis);
  proboscis.add(new THREE.Mesh(capsuleGeometry(0.2, 0.12, 0.095, 0.01, 16, 12), M.rostrum));
  const haustellum = new THREE.Mesh(capsuleGeometry(0.2, 0.08, 0.062, 0.006, 14, 12), M.haustellum);
  haustellum.position.y = -0.16;
  proboscis.add(haustellum);
  const lobeHairs = [];
  for (const s of [1, -1]) {
    const lobe = ellipsoid([0.058, 0.05, 0.095], M.labellum, 20, 14);
    lobe.position.set(0.05 * s, -0.38, 0.02);
    lobe.rotation.z = -0.25 * s;
    proboscis.add(lobe);
    for (let i = 0; i < 6; i++) {
      const a = -0.6 + i * 0.25;
      lobeHairs.push({ pos: V(0.08 * s, -0.38, 0.02 + a * 0.12), dir: V(0.8 * s, -0.4, a), curl: V(-s, 0, 0), len: 0.05 });
    }
    const palp = ellipsoid([0.035, 0.07, 0.035], M.antenna, 12, 10);
    palp.position.set(0.09 * s, -0.06, 0.1);
    palp.rotation.x = 0.9;
    proboscis.add(palp);
    for (let i = 0; i < 3; i++) {
      lobeHairs.push({ pos: V(0.1 * s, -0.04 + i * 0.02, 0.14 + i * 0.02), dir: V(0.3 * s, -0.2, 1), curl: V(0, -1, 0), len: 0.08 });
    }
  }
  proboscis.add(hairs(lobeHairs, G.micro, M.hair));

  // Abdomen
  const abdomen = new THREE.Group();
  abdomen.name = "abdomen";
  abdomen.position.set(0, -0.08, -1.8);
  abdomen.rotation.x = -0.14;
  group.add(abdomen);
  abdomen.add(new THREE.Mesh(surfaceGeometry(abdomenFn, abdomenColor, 72, 120, true), M.abdomen));

  for (const t of AB_BOUNDS.slice(2, 6)) {
    const arc = Math.PI * 1.35;
    const geo = new THREE.TorusGeometry(abBase(t) * AB.x, 0.022, 10, 96, arc);
    geo.rotateZ(Math.PI / 2 - arc / 2);
    const ring = new THREE.Mesh(geo, M.tergite);
    ring.scale.y = AB.y / AB.x;
    ring.position.z = t * AB.z;
    abdomen.add(ring);
  }

  const tip = new THREE.Mesh(capsuleGeometry(0.22, 0.1, 0.02, 0.01, 12, 10), M.tergite);
  tip.rotation.x = Math.PI / 2 - 0.15;
  tip.position.set(0, -0.03, -AB.z * 0.9);
  abdomen.add(tip);

  const abHairs = [];
  for (let i = 1; i < AB_BOUNDS.length - 1; i++) {
    for (const f of [0.45, 0.78, 0.93]) {
      const t = AB_BOUNDS[i] - (AB_BOUNDS[i] - AB_BOUNDS[i + 1]) * f;
      const n = Math.round(10 + 14 * abBase(t));
      for (let k = 0; k < n; k++) {
        const a = -0.15 * Math.PI + (1.3 * Math.PI * (k + R() * 0.5)) / n;
        const rr = Math.sqrt(1 - t * t);
        const fr = surfaceFrame(abdomenFn, N(Math.cos(a) * rr, Math.sin(a) * rr, t), 0.004);
        abHairs.push({ pos: fr.pos, dir: fr.n.clone().multiplyScalar(0.35).add(V(0, 0, -1)), curl: fr.n.clone().negate(), len: (f > 0.9 ? 0.1 : 0.075) + 0.02 * R() });
      }
    }
  }
  abdomen.add(hairs(abHairs, G.micro, M.hair));

  // Wings
  const wingPivots = [];
  for (const s of [1, -1]) {
    const pivot = new THREE.Group();
    pivot.name = s > 0 ? "wingPivot_R" : "wingPivot_L";
    pivot.position.set(0.34 * s, 0.62, 0.3);
    const mirror = new THREE.Group();
    mirror.scale.x = s;
    const yaw = new THREE.Group();
    yaw.rotation.y = 1.02;
    const tilt = new THREE.Group();
    tilt.rotation.set(0.1, 0, 0.07);
    tilt.add(buildWing(M, G, R));
    yaw.add(tilt);
    mirror.add(yaw);
    pivot.add(mirror);
    group.add(pivot);
    wingPivots.push(pivot);
  }

  // Halteres
  for (const s of [1, -1]) {
    const f = surfaceFrame(thoraxFn, N(0.75 * s, 0.25, -0.55), 0.02);
    const haltere = new THREE.Group();
    haltere.name = s > 0 ? "haltere_R" : "haltere_L";
    haltere.position.copy(f.pos);
    haltere.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(UP, N(s, 0.3, -0.6)));
    const stalk = new THREE.Mesh(capsuleGeometry(0.24, 0.022, 0.016, 0, 8, 8), M.haltere);
    stalk.rotation.x = Math.PI;
    const knob = ellipsoid([0.055, 0.075, 0.05], M.haltere, 16, 12);
    knob.position.y = 0.27;
    haltere.add(stalk, knob);
    group.add(haltere);
  }

  // Legs
  for (const s of [1, -1]) {
    for (const spec of LEGS) group.add(buildLeg(spec, s, M, G, R));
  }

  group.rotation.y = -0.3;

  return { group, wingPivots, headGroup, eyes };
}
