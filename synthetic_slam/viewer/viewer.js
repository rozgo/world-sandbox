// SLAM trace viewer: replays one AMB3R-SLAM run from scene.json, points.bin and video.mp4.
// Coordinates are three.js's (y up); poses are camera-to-world matrices, row-major.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

const DATA = new URLSearchParams(location.search).get('data') || '.';
const $ = (id) => document.getElementById(id);
const COL = { rust: 0xe0703a, teal: 0x4fb3a9, steel: 0x8d97a1, yellow: 0xe8b931, red: 0xe0524f, white: 0xe9e6e1 };

async function fetchBuffer(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

const S = await (await fetch(`${DATA}/scene.json`)).json();
const buf = await fetchBuffer(`${DATA}/points.bin`, (p) => { $('loading').textContent = `Loading the map… ${Math.round(p * 100)}%`; });
$('loading').remove();

const N = S.frames, FPS = S.fps;
const [IW, IH] = S.image;
const [FX, FY, CX, CY] = S.K;

// ---- poses ------------------------------------------------------------------------------
const mat = (a) => new THREE.Matrix4().set(...a);
const poses = S.poses.map(mat);
const pos = poses.map((m) => new THREE.Vector3().setFromMatrixPosition(m));
const quat = poses.map((m) => new THREE.Quaternion().setFromRotationMatrix(m));
const fwd = poses.map((m) => new THREE.Vector3(-m.elements[8], -m.elements[9], -m.elements[10]));
const UP = new THREE.Vector3(0, 1, 0);
function smoothed(arr, half) {
  return arr.map((_, i) => {
    const v = new THREE.Vector3();
    let n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(N - 1, i + half); j++) { v.add(arr[j]); n++; }
    return v.divideScalar(n);
  });
}
const sPos = smoothed(pos, 10);
const sFwd = smoothed(fwd, 14).map((v) => v.sub(UP.clone().multiplyScalar(v.dot(UP))).normalize());
const pathLen = pos.reduce((s, p, i) => (i ? s + p.distanceTo(pos[i - 1]) : 0), 0);
const unit = pathLen / 10;  // overlay sizes scale with the run

// ---- renderer, scene ------------------------------------------------------------------------
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f11);
const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.01 * unit, 400 * unit);
const controls = new OrbitControls(camera, canvas);
controls.enabled = false;
controls.enableDamping = true;

const video = $('video');
video.src = `${DATA}/video.mp4`;
video.loop = true;
const videoTex = new THREE.VideoTexture(video);
videoTex.colorSpace = THREE.SRGBColorSpace;

// ---- points -----------------------------------------------------------------------------
const count = S.count;
const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, 0, count * 3), 3));
geom.setAttribute('rgb', new THREE.BufferAttribute(new Uint8Array(buf, count * 12, count * 3), 3, true));
geom.setAttribute('frame', new THREE.BufferAttribute(new Uint8Array(buf, count * 15, count), 1));
geom.setAttribute('submap', new THREE.BufferAttribute(new Uint8Array(buf, count * 16, count), 1));
geom.setAttribute('conf', new THREE.BufferAttribute(new Uint8Array(buf, count * 17, count), 1, true));

const committed = S.submaps.filter((s) => !s.replaced).sort((a, b) => a.index - b.index);
const subPasses = S.passes.filter((p) => p.stage === 'submap');
const mappedAt = committed.map((s) => subPasses[s.pass].at);
const subColour = committed.map((s) => new THREE.Color(s.colour));
while (mappedAt.length < 8) { mappedAt.push(1e9); subColour.push(new THREE.Color(0x888888)); }

const pointMat = new THREE.ShaderMaterial({
  uniforms: {
    uFrame: { value: 0 }, uMode: { value: 0 }, uReveal: { value: 0 }, uSize: { value: 1.2 },
    uScale: { value: 1 }, uN: { value: N }, uMapped: { value: mappedAt }, uSub: { value: subColour },
  },
  vertexShader: /* glsl */`
    attribute vec3 rgb; attribute float frame; attribute float submap; attribute float conf;
    uniform float uFrame, uSize, uScale, uN; uniform int uMode, uReveal;
    uniform float uMapped[8]; uniform vec3 uSub[8];
    varying vec3 vColor;
    vec3 turbo(float x) {
      const vec4 kr = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
      const vec4 kg = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
      const vec4 kb = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
      const vec2 kr2 = vec2(-152.94239396, 59.28637943), kg2 = vec2(4.27729857, 2.82956604),
                 kb2 = vec2(-89.90310912, 27.34824973);
      x = clamp(x, 0.0, 1.0);
      vec4 v4 = vec4(1.0, x, x * x, x * x * x); vec2 v2 = v4.zw * v4.z;
      return vec3(dot(v4, kr) + dot(v2, kr2), dot(v4, kg) + dot(v2, kg2), dot(v4, kb) + dot(v2, kb2));
    }
    vec3 viridis(float t) {
      const vec3 c0 = vec3(0.2777, 0.0054, 0.3341), c1 = vec3(0.1051, 1.4046, 1.3846), c2 = vec3(-0.3309, 0.2148, 0.0951),
                 c3 = vec3(-4.6342, -5.7991, -19.3324), c4 = vec3(6.2283, 14.1799, 56.6906),
                 c5 = vec3(4.7764, -13.7451, -65.3530), c6 = vec3(-5.4355, 4.6459, 26.3124);
      t = clamp(t, 0.0, 1.0);
      return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
    }
    void main() {
      int k = int(submap + 0.5);
      float from = uReveal == 0 ? frame : (uReveal == 1 ? uMapped[k] : -1.0);
      if (from > uFrame + 0.001) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vColor = vec3(0.0); return; }
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = clamp(uSize * uScale / max(-mv.z, 1e-3), 1.0, 9.0);
      vec3 c = rgb;
      if (uMode == 1) c = uSub[k] * (0.45 + 0.75 * dot(rgb, vec3(0.333)));
      else if (uMode == 2) c = viridis(conf);
      else if (uMode == 3) c = turbo(frame / max(uN - 1.0, 1.0));
      float age = uFrame - from;
      if (uReveal != 2 && age < 5.0) c = mix(c, vec3(1.0, 0.86, 0.6), 0.35 * (1.0 - age / 5.0));
      vColor = c;
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      if (dot(d, d) > 0.25) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`,
});
const points = new THREE.Points(geom, pointMat);
points.frustumCulled = false;
scene.add(points);

// ---- overlay helpers ----------------------------------------------------------------------
const lineMats = [];
function fatLine(pts, colour, width, opts = {}) {
  const g = new LineGeometry();
  g.setPositions(pts.flatMap((p) => [p.x, p.y, p.z]));
  const m = new LineMaterial({ color: colour, linewidth: width, transparent: true, opacity: opts.opacity ?? 1,
    dashed: !!opts.dashed, dashSize: 0.12 * unit, gapSize: 0.08 * unit, depthTest: opts.depthTest ?? true });
  lineMats.push(m);
  const l = new Line2(g, m);
  if (opts.dashed) l.computeLineDistances();
  return l;
}

function frustumLines(depth) {
  const c = [[0, 0], [IW, 0], [IW, IH], [0, IH]].map(([u, v]) =>
    new THREE.Vector3((u - CX) / FX * depth, -(v - CY) / FY * depth, -depth));
  const o = new THREE.Vector3();
  const segs = [o, c[0], o, c[1], o, c[2], o, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]];
  return new THREE.BufferGeometry().setFromPoints(segs);
}

function frustum(depth, colour, opacity = 1) {
  const l = new THREE.LineSegments(frustumLines(depth), new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity }));
  l.matrixAutoUpdate = false;
  return l;
}

// The SLAM camera: its frustum and the current frame on the image plane.
const camDepth = 0.55 * unit;
const camGroup = new THREE.Group();
camGroup.matrixAutoUpdate = false;
camGroup.add(new THREE.LineSegments(frustumLines(camDepth), new THREE.LineBasicMaterial({ color: COL.yellow })));
const planeW = IW / FX * camDepth, planeH = IH / FY * camDepth;
const plane = new THREE.Mesh(new THREE.PlaneGeometry(planeW, planeH),
  new THREE.MeshBasicMaterial({ map: videoTex, transparent: true, opacity: 0.92, side: THREE.DoubleSide }));
plane.position.set((IW / 2 - CX) / FX * camDepth, -(IH / 2 - CY) / FY * camDepth, -camDepth * 0.995);
camGroup.add(plane);
const axes = new THREE.AxesHelper(0.25 * unit);
camGroup.add(axes);
scene.add(camGroup);

// Path: the final trajectory, drawn up to the current frame, over a faint full path.
const pathFull = fatLine(pos, COL.steel, 1.5, { opacity: 0.35 });
const pathNow = fatLine(pos, COL.rust, 3.5);
const pathGroup = new THREE.Group();
pathGroup.add(pathFull, pathNow);
scene.add(pathGroup);

// Front-end live estimate, after the first re-anchoring (before it the front-end has its own origin).
const firstAnchor = Math.min(...S.passes.map((p) => p.at));
const liveIdx = S.frontend.map((p, i) => (p && i > firstAnchor ? i : -1)).filter((i) => i >= 0);
const livePts = liveIdx.map((i) => new THREE.Vector3().setFromMatrixPosition(mat(S.frontend[i])));
const liveLine = livePts.length > 1 ? fatLine(livePts, 0xb7c2cc, 2, { opacity: 0.85 }) : new THREE.Group();
scene.add(liveLine);

// Keyframes the front-end promoted.
const kfGroup = new THREE.Group();
const kfFrustums = S.keyframes.map((f) => {
  const fr = frustum(0.28 * unit, COL.yellow, 0.55);
  fr.matrix.copy(poses[f]);
  fr.userData.frame = f;
  kfGroup.add(fr);
  return fr;
});
scene.add(kfGroup);

// Front-end window: keyframe (teal), recent frames (steel), the new frame (rust).
const feGroup = new THREE.Group();
const fePool = [0, 1, 2, 3, 4].map(() => { const f = frustum(0.34 * unit, COL.steel); feGroup.add(f); return f; });
scene.add(feGroup);

// The latest backend submap pass: one small frustum per view, in the submap's colour.
const subGroup = new THREE.Group();
const subPool = Array.from({ length: 40 }, () => { const f = frustum(0.2 * unit, COL.teal, 0.9); subGroup.add(f); return f; });
scene.add(subGroup);

// Pose graph: a node per submap at its middle frame, and its edges.
const graphGroup = new THREE.Group();
const nodePos = committed.map((s) => pos[Math.floor((s.start + s.end - 1) / 2)].clone().add(UP.clone().multiplyScalar(0.6 * unit)));
const nodes = committed.map((s, k) => {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.12 * unit, 20, 14), new THREE.MeshBasicMaterial({ color: s.colour }));
  m.position.copy(nodePos[k]);
  graphGroup.add(m);
  return m;
});
const edgeStyle = { adjacent: [COL.white, 3, false], span: [COL.teal, 2.5, true], long_context: [COL.teal, 2.5, false],
  long_context_rejected: [COL.red, 2, true], loop: [0x7cc96b, 3.5, false] };
const edges = S.edges.map((e) => {
  const [c, w, dashed] = edgeStyle[e.type] || [COL.white, 2, false];
  const a = nodePos[e.a], b = nodePos[e.b];
  const lift = e.type === 'adjacent' ? 0 : (e.type === 'long_context_rejected' ? -0.5 : 0.45) * unit * (1 + Math.abs(e.b - e.a) * 0.3);
  const mid = a.clone().add(b).multiplyScalar(0.5).add(UP.clone().multiplyScalar(lift));
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const l = fatLine(curve.getPoints(24), c, w, { dashed });
  l.userData = { a: e.a, b: e.b };
  graphGroup.add(l);
  return l;
});
graphGroup.visible = false;
scene.add(graphGroup);

// Loop probes: a dot where retrieval queried the database (grey: no match).
const probeGroup = new THREE.Group();
const probeDots = S.probes.map((p) => {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.025 * unit, 10, 8),
    new THREE.MeshBasicMaterial({ color: p.match >= 0 ? 0x7cc96b : 0x9a9690 }));
  m.position.copy(pos[p.frame]).add(UP.clone().multiplyScalar(0.18 * unit));
  m.userData.frame = p.frame;
  probeGroup.add(m);
  return m;
});
probeGroup.visible = false;
scene.add(probeGroup);

// Ground grid, at the floor below the camera path.
const posAttr = geom.attributes.position.array;
const ys = [];
for (let i = 0; i < count; i += 37) {
  const x = posAttr[3 * i], y = posAttr[3 * i + 1], z = posAttr[3 * i + 2];
  for (let f = 0; f < N; f += 12) {
    const p = pos[f];
    if ((x - p.x) ** 2 + (z - p.z) ** 2 < (0.8 * unit) ** 2 && y < p.y) { ys.push(y); break; }
  }
}
ys.sort((a, b) => a - b);
const groundY = ys.length ? ys[Math.floor(ys.length * 0.3)] : pos[0].y - unit * 0.3;
const grid = new THREE.GridHelper(60 * unit, 60, 0x3a4047, 0x22272c);
grid.position.set(pos[0].x, groundY, pos[0].z);
scene.add(grid);

// Ego overlay: the video in front of the camera, blended over the map.
const egoPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ map: videoTex, transparent: true, opacity: 0, depthTest: false, depthWrite: false }));
egoPlane.renderOrder = 10;
camera.add(egoPlane);
scene.add(camera);

// ---- state and UI ---------------------------------------------------------------------------
const state = { view: 'follow', frame: 0, playing: false, scrubbing: false };
const layers = { points, camera: camGroup, path: pathGroup, front: feGroup, keyframes: kfGroup, submap: subGroup,
  graph: graphGroup, probes: probeGroup, live: liveLine, grid };
liveLine.visible = false;

document.querySelectorAll('.seg').forEach((seg) => seg.addEventListener('click', (ev) => {
  const b = ev.target.closest('button');
  if (!b) return;
  seg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
  const name = seg.dataset.name, v = b.dataset.v;
  if (name === 'view') setView(v);
  if (name === 'colour') { pointMat.uniforms.uMode.value = +v; legend(); }
  if (name === 'reveal') pointMat.uniforms.uReveal.value = +v;
}));
document.querySelectorAll('[data-layer]').forEach((cb) => cb.addEventListener('change', () => {
  layers[cb.dataset.layer].visible = cb.checked;
}));
$('size').addEventListener('input', (e) => { pointMat.uniforms.uSize.value = +e.target.value; });
$('blend').addEventListener('input', (e) => { egoPlane.material.opacity = +e.target.value; });
$('panelToggle').addEventListener('click', () => {
  const p = $('panel');
  p.classList.toggle('closed');
  $('panelToggle').setAttribute('aria-expanded', String(!p.classList.contains('closed')));
});
if (innerWidth < 700) $('panel').classList.add('closed');

const scrub = $('scrub');
scrub.max = N - 1;
const seek = (f) => {
  state.frame = Math.max(0, Math.min(N - 1, f));
  video.currentTime = (Math.round(state.frame) + 0.5) / FPS;
};
scrub.addEventListener('input', () => { pause(); seek(+scrub.value); });
$('back').addEventListener('click', () => { pause(); seek(Math.round(state.frame) - 1); });
$('fwd').addEventListener('click', () => { pause(); seek(Math.round(state.frame) + 1); });
$('speed').addEventListener('change', (e) => { video.playbackRate = +e.target.value; });
function play() { video.playbackRate = +$('speed').value; video.play().catch(() => {}); state.playing = true; $('play').textContent = '❚❚'; }
function pause() { video.pause(); state.playing = false; $('play').textContent = '▶'; }
$('play').addEventListener('click', () => (state.playing ? pause() : play()));
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.code === 'Space') { e.preventDefault(); state.playing ? pause() : play(); }
  if (e.code === 'ArrowRight') { pause(); seek(Math.round(state.frame) + 1); }
  if (e.code === 'ArrowLeft') { pause(); seek(Math.round(state.frame) - 1); }
});

function setView(v) {
  state.view = v;
  controls.enabled = v === 'orbit';
  $('blendGroup').style.display = v === 'ego' ? 'block' : 'none';
  egoPlane.visible = v === 'ego';
  if (v === 'orbit') controls.target.copy(pos[Math.round(state.frame)]);
  camera.fov = v === 'ego' ? THREE.MathUtils.radToDeg(2 * Math.atan(IH / 2 / FY)) : 55;
  camera.updateProjectionMatrix();
  pointMat.uniforms.uScale.value = baseScale * (v === 'ego' ? 1.8 : 1);
}

function legend() {
  const rows = [];
  const mode = pointMat.uniforms.uMode.value;
  if (mode === 1) committed.forEach((s) => rows.push([s.colour, `submap ${s.index}: frames ${s.start}–${s.end - 1}`]));
  if (mode === 2) rows.push(['linear-gradient(90deg,#440154,#21918c,#fde725)', 'confidence, low → high']);
  if (mode === 3) rows.push(['linear-gradient(90deg,#30123b,#28bceb,#a4fc3c,#fb8022,#7a0403)', 'source frame, first → last']);
  rows.push(['#e8b931', 'SLAM camera, keyframes'], ['#e0703a', 'camera path'], ['#4fb3a9', 'front-end keyframe'],
    ['#e0524f', 'rejected long-context link']);
  $('legend').innerHTML = rows.map(([c, t]) => `<div class="row"><span class="sw" style="background:${c}"></span>${t}</div>`).join('');
}
legend();

// ---- per-frame update -------------------------------------------------------------------------
const tmpM = new THREE.Matrix4();
function backendAt(f) {
  let last = null;
  for (const p of S.passes) if (p.at <= f) last = p;
  return last;
}

function update(ff) {
  const f = Math.max(0, Math.min(N - 1, Math.floor(ff)));
  const a = Math.min(1, ff - f), g = Math.min(N - 1, f + 1);
  pointMat.uniforms.uFrame.value = ff;

  // the SLAM camera, interpolated between frames
  const p = pos[f].clone().lerp(pos[g], a);
  const q = quat[f].clone().slerp(quat[g], a);
  camGroup.matrix.compose(p, q, new THREE.Vector3(1, 1, 1));
  camGroup.visible = state.view !== 'ego' && $$('camera');
  pathNow.geometry.instanceCount = Math.max(0, f);
  if (liveLine.geometry) liveLine.geometry.instanceCount = Math.max(0, liveIdx.filter((i) => i <= f).length - 1);

  kfFrustums.forEach((k) => { k.visible = k.userData.frame <= f; });

  const views = S.frontend_views[f] || [f];
  const kf = views.length > 1 ? views[0] : null;
  fePool.forEach((fr, i) => {
    fr.visible = i < views.length && views.length > 1;
    if (!fr.visible) return;
    const v = views[i];
    fr.matrix.copy(poses[v]);
    fr.material.color.setHex(v === f ? COL.rust : v === kf ? COL.teal : COL.steel);
  });

  const bp = backendAt(f);
  const sp = bp && bp.stage === 'submap' ? bp : [...S.passes].reverse().find((x) => x.stage === 'submap' && x.at <= f);
  subPool.forEach((fr, i) => {
    const show = sp && i < sp.frames.length;
    fr.visible = !!show;
    if (!show) return;
    fr.matrix.copy(poses[sp.frames[i]]);
    const s = S.submaps.find((x) => x.pass === subPasses.indexOf(sp));
    fr.material.color.set(s ? s.colour : '#8d97a1');
  });

  nodes.forEach((n, k) => { n.visible = mappedAt[k] <= f; });
  edges.forEach((e) => { e.visible = mappedAt[e.userData.a] <= f && mappedAt[e.userData.b] <= f; });
  probeDots.forEach((d) => { d.visible = d.userData.frame <= f; });

  // HUD
  $('hudFrame').textContent = `Frame ${f} of ${N - 1} · ${(f / FPS).toFixed(2)} s of video`;
  $('hudFront').textContent = views.length > 1
    ? `Front-end: tracked with keyframe ${kf}${views.length > 2 ? ` and frames ${views.slice(1, -1).join(', ')}` : ''}`
      + (S.keyframes.includes(f) ? ' · new keyframe' : '')
    : 'Front-end: the first frame anchors the map';
  if (!bp) $('hudBack').textContent = `Backend: waiting for the first ${subPasses[0]?.frames.length ?? ''} views`;
  else if (bp.stage === 'submap') {
    const s = S.submaps.find((x) => x.pass === subPasses.indexOf(bp));
    $('hudBack').textContent = `Backend: ${s && !s.replaced ? `submap ${s.index}` : 'warm-up submap, replaced later'} · `
      + `${bp.views} views in one pass, ${bp.seconds.toFixed(2)} s at frame ${bp.at}`;
  } else if (bp.stage === 'long_context') {
    $('hudBack').textContent = `Backend: long-context window over ${bp.views} views; link rejected, then the pose graph`;
  }

  scrub.value = f;
  $('time').textContent = `${f.toString().padStart(3, ' ')} / ${N - 1} · ${(ff / FPS).toFixed(2)} s`;
}
function $$(name) { return document.querySelector(`[data-layer="${name}"]`).checked; }

function placeCamera(ff) {
  const f = Math.max(0, Math.min(N - 1, Math.floor(ff)));
  const a = Math.min(1, ff - f), g = Math.min(N - 1, f + 1);
  if (state.view === 'ego') {
    camera.position.copy(pos[f]).lerp(pos[g], a);
    camera.quaternion.copy(quat[f]).slerp(quat[g], a);
    const d = 0.05 * unit, h = 2 * d * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    egoPlane.position.set(0, 0, -d);
    egoPlane.scale.set(h * camera.aspect, h, 1);
  } else if (state.view === 'follow') {
    const p = sPos[f].clone().lerp(sPos[g], a), d = sFwd[f].clone().lerp(sFwd[g], a).normalize();
    camera.position.copy(p).addScaledVector(d, -2.9 * unit).addScaledVector(UP, 1.35 * unit);
    camera.lookAt(p.clone().addScaledVector(d, 2.6 * unit).addScaledVector(UP, -0.1 * unit));
  } else {
    controls.update();
  }
}

let baseScale = 1;
function resize() {
  const st = $('stage');
  const w = st.clientWidth, h = st.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const focal = h * renderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(55) / 2));
  baseScale = focal * 0.006 * unit;
  pointMat.uniforms.uScale.value = baseScale * (state.view === 'ego' ? 1.8 : 1);
  lineMats.forEach((m) => m.resolution.set(w, h));
}
addEventListener('resize', resize);
resize();

function loop() {
  if (state.playing && !video.paused) state.frame = Math.min(N - 1, video.currentTime * FPS);
  update(state.frame);
  placeCamera(state.frame);
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
setView('follow');
video.addEventListener('loadeddata', () => { if (!state.playing) seek(0); }, { once: true });
play();
loop();
