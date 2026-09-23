import './style.css';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { loadAssets } from './data.js';
import { World, DT } from './sim/world.js';
import { makeActivityMapper } from './sim/activity.js';
import { Fly } from './render/fly.js';
import { BrainView } from './render/brain.js';
import { Swamp } from './render/swamp.js';
import { Props } from './render/props.js';
import { Plumes } from './render/plumes.js';
import { CameraRig } from './render/cameras.js';
import { Hud } from './ui/hud.js';

const BRAIN_LAYER = 1;
const SPEEDS = [1, 2, 4, 8];
const query = new URLSearchParams(location.search);
const LITE = query.has('lite'); // cheap settings for software renderers and screenshots

const hud = new Hud();
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(LITE ? 1 : Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = !LITE;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.4, 6000);
const pipCam = new THREE.PerspectiveCamera(32, 1, 0.02, 60);
pipCam.layers.set(BRAIN_LAYER);

const names = { 'circuit.json': 'проводку', 'anatomy.json': 'анатомию', 'fly_rig.json': 'суставы', 'neurons.bin': 'скелеты нейронов', 'somas.bin': 'сомы', 'cns.glb': 'нейропили', 'fly.glb': 'тело мухи' };
let loaded = 0;
const A = await loadAssets((name) => hud.loading(`загружено ${++loaded}/7 · ${names[name] ?? name}`));

const swamp = new Swamp(scene, renderer);
let world = new World(A.circuit, { seed: query.has('seed') ? +query.get('seed') : (Math.random() * 1e9) | 0 });
const props = new Props(scene);
const plumes = new Plumes(scene);
const fly = new Fly(A.fly, A.rig);
fly.meshes.forEach((m) => m.layers.enable(BRAIN_LAYER));
scene.add(fly.root);
const brain = new BrainView(A);
brain.group.traverse((o) => o.layers.set(BRAIN_LAYER));
fly.model.add(brain.group);
const collect = makeActivityMapper(A.circuit);

// Trail of recent positions, shown on the map
const TRAIL = 600;
const trailPos = new Float32Array(TRAIL * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xff6a4d, transparent: true, opacity: 0.9, fog: false }));
trail.frustumCulled = false;
scene.add(trail);
let trailN = 0, trailT = 0;

// Map markers: glowing rings for every entity, only shown in the overview
const markers = {
  group: new THREE.Group(),
  pool: new Map(),
  colors: { fruit: 0xffb347, sundew: 0xff4f8b, frog: 0x7bd88f },
  update(world, on) {
    this.group.visible = on;
    if (!on) return;
    const seen = new Set();
    for (const e of world.entities) {
      seen.add(e.id);
      let m = this.pool.get(e.id);
      if (!m) {
        m = new THREE.Mesh(new THREE.RingGeometry(e.radius + 6, e.radius + 10, 48).rotateX(-Math.PI / 2),
          new THREE.MeshBasicMaterial({ color: this.colors[e.kind], transparent: true, opacity: 0.8, depthWrite: false, fog: false }));
        this.group.add(m);
        this.pool.set(e.id, m);
      }
      m.position.set(e.x, 2, e.z);
    }
    for (const [id, m] of this.pool) if (!seen.has(id)) { this.group.remove(m); m.geometry.dispose(); this.pool.delete(id); }
  },
};
scene.add(markers.group);

const rig = new CameraRig(camera, canvas);
const focus = new THREE.Vector3();
const headLocal = new THREE.Vector3(0.745, 0, -0.035); // brain centre in the fly (MuJoCo frame, mm), see build_anatomy.py

// Bloom lights up the neurons; software renderers (SwiftShader) mangle it, so skip it there.
const glInfo = renderer.getContext().getExtension('WEBGL_debug_renderer_info');
const glName = glInfo ? renderer.getContext().getParameter(glInfo.UNMASKED_RENDERER_WEBGL) : '';
const software = /swiftshader|llvmpipe/i.test(glName);
const useBloom = (!software || query.has('bloom')) && !query.has('nobloom');
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.55, 0.45, 0.72);
composer.addPass(bloom);
composer.addPass(new OutputPass());

let mode = 'follow';
let speedIdx = 0;
let placing = null;
let acc = 0;

function flyFocus(target) {
  const f = world.fly;
  if (mode === 'xray' || mode === 'brain') return fly.model.localToWorld(target.copy(headLocal));
  if (mode === 'map') return target.set(0, 0, 0);
  return target.set(f.x, fly.root.position.y, f.z);
}

function setMode(m) {
  mode = m;
  const xray = m === 'xray' || m === 'brain';
  fly.setXray(xray, m === 'brain' ? 0.35 : 1);
  swamp.group.visible = props.group.visible = m !== 'brain';
  plumes.points.visible = plumes.enabled && m !== 'brain';
  camera.layers.set(0);
  if (xray) camera.layers.enable(BRAIN_LAYER);
  brain.setContext({ cloud: true, shells: true, neuropils: m === 'brain' });
  scene.background.set(m === 'brain' ? 0x020405 : 0x0b1714);
  // neurons are already luminous: softer bloom when they fill the screen
  Object.assign(bloom, xray ? { strength: 0.32, threshold: 0.88, radius: 0.35 } : { strength: 0.55, threshold: 0.72, radius: 0.45 });
  rig.setMode(m, flyFocus(focus));
  hud.setCamera(m);
}

hud.buildLegend(brain.groups, (name, on) => brain.setGroupVisible(name, on));
hud.bind({
  camera: setMode,
  place: (kind) => { placing = placing === kind ? null : kind; hud.setPlacing(placing); },
  sugar: () => world.giveSugar(),
  shock: () => world.giveShock(),
  speed: () => { speedIdx = (speedIdx + 1) % SPEEDS.length; return SPEEDS[speedIdx]; },
  arms: () => (world.armsRace = !world.armsRace),
  plumes: () => (plumes.enabled = !plumes.enabled),
  wind: () => { world.wind.angle += Math.PI / 4; world.log('info', 'Ветер повернул — шлейфы запаха потекли иначе'); },
  reset: () => { world.mb.resetMemory(); world.log('info', 'Память стёрта: все синапсы KC→MBON вернулись к исходным'); },
});

// Placing props / hovering neurons
const ray = new THREE.Raycaster();
ray.params.Points.threshold = 0.006;
const ndc = new THREE.Vector2();
const waterPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
canvas.addEventListener('pointerdown', (e) => { canvas._down = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!placing || !canvas._down) return;
  if (Math.hypot(e.clientX - canvas._down[0], e.clientY - canvas._down[1]) > 6) return; // was a drag
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const p = ray.ray.intersectPlane(waterPlane, new THREE.Vector3());
  if (p && Math.hypot(p.x, p.z) < 300) {
    world.add(placing, p.x, p.z);
    world.log('info', `Поставлено: ${{ fruit: 'гнилой фрукт', sundew: 'росянка', frog: 'лягушка' }[placing]}`);
  }
});
canvas.addEventListener('pointermove', (e) => {
  if (mode !== 'brain' && mode !== 'xray') return hud.tooltip(null);
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  ray.layers.set(BRAIN_LAYER);
  const hit = ray.intersectObject(brain.somaPoints)[0];
  if (!hit) return hud.tooltip(null);
  const i = hit.index, n = A.circuit.neurons;
  const g = brain.groups[brain.groupIndex[i]].name;
  hud.tooltip(`<b>${n.instance[i] || A.circuit.types[n.type[i]]}</b><br>${g} · ${A.circuit.nts[n.nt[i]]}<br>bodyId ${n.bodyId[i]} · активность ${(brain.act[4 * i] * 100).toFixed(0)}%`, e.clientX, e.clientY);
});

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloom.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const px = h * renderer.getPixelRatio() / (2 * Math.tan((camera.fov * Math.PI) / 360));
  swamp.setPixelScale(px / renderer.getPixelRatio());
  plumes.setPixelScale(px / renderer.getPixelRatio());
  brain.setViewport(h * renderer.getPixelRatio(), camera.fov);
}
addEventListener('resize', resize);
resize();

function readout() {
  const mb = world.mb;
  let kc = 0;
  for (let i = 0; i < mb.nK; i++) if (mb.kc[i] > 0.02) kc++;
  let app = 0, av = 0;
  for (let m = 0; m < mb.nM; m++) (mb.valenceW[m] > 0 ? (app += mb.valenceW[m] * mb.mbon[m]) : (av -= mb.valenceW[m] * mb.mbon[m]));
  let pam = 0, ppl = 0;
  for (let d = 0; d < mb.nD; d++) (d < mb.nPam ? (pam += mb.dan[d]) : (ppl += mb.dan[d]));
  return { kcFrac: kc / mb.nK, approach: app / 20, avoid: av / 20, pam: pam / mb.nPam, ppl: ppl / (mb.nD - mb.nPam) };
}

const euler = new THREE.Euler(0, 0, 0, 'YZX');
let lastNow = performance.now();
let hudT = 0, frames = 0;
for (let i = 0; i < (+query.get('warm') || 0) / DT; i++) world.step(DT); // dev: start later in the fly's life
setMode(query.get('mode') ?? 'follow');
if (query.has('warm')) rig.blend = 0.001;
hud.ready(A.circuit);

function frame(now) {
  frames++;
  const dt = Math.min(Math.max(0, (now - lastNow) / 1000), 0.05);
  lastNow = now;
  const speed = SPEEDS[speedIdx];
  acc += dt * speed;
  let steps = 0, simDt = 0;
  while (acc >= DT && steps < 40) {
    world.step(DT);
    acc -= DT;
    simDt += DT;
    steps++;
  }
  if (steps === 40) acc = 0;

  // fly pose from the world
  const f = world.fly;
  const perched = f.state === 'feed' || f.state === 'trapped';
  const y = perched ? props.perchHeight(world, f.perch) : f.y;
  fly.root.position.set(f.x, y, f.z);
  euler.set(perched ? 0 : f.roll, f.heading, perched ? 0 : f.pitch);
  fly.root.quaternion.setFromEuler(euler);
  fly.animate(dt, world.t, f.state, world.motor);

  if (simDt > 0) brain.update(simDt, collect(world.mb, world.motor), world.sense.u);
  props.sync(world, simDt || 0);
  plumes.update(world, simDt);
  swamp.update(dt, fly.root.position);

  if ((trailT += simDt) > 0.1) {
    trailT = 0;
    trailPos.copyWithin(3, 0, (TRAIL - 1) * 3);
    trailPos.set([f.x, y, f.z], 0);
    trailN = Math.min(TRAIL, trailN + 1);
    trailGeo.setDrawRange(0, trailN);
    trailGeo.attributes.position.needsUpdate = true;
  }
  trail.visible = mode === 'map';
  markers.update(world, mode === 'map');

  rig.update(dt, flyFocus(focus));

  if ((hudT += dt) > 0.1) {
    hudT = 0;
    hud.update(world, readout());
  }

  // main view
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  if (useBloom) composer.render(); else renderer.render(scene, camera);

  // picture-in-picture: the live brain while following the fly
  const pip = mode === 'follow' || mode === 'map';
  if (pip) {
    const s = Math.min(260, Math.round(Math.min(innerWidth, innerHeight) * 0.3));
    const x = innerWidth - s - 16, yb = innerWidth < 900 ? innerHeight - s - 190 : 16 + 64;
    const head = fly.model.localToWorld(focus.copy(headLocal));
    const a = world.t * 0.25;
    const side = new THREE.Vector3(Math.cos(a), 0.35, Math.sin(a)).normalize().multiplyScalar(2.4);
    pipCam.position.copy(head).add(side);
    pipCam.lookAt(head);
    pipCam.aspect = 1;
    pipCam.updateProjectionMatrix();
    fly.setXray(true, 0.35);
    brain.setViewport(s * renderer.getPixelRatio(), pipCam.fov);
    const bg = scene.background.clone();
    scene.background.set(0x020405);
    renderer.setScissorTest(true);
    renderer.setScissor(x, yb, s, s);
    renderer.setViewport(x, yb, s, s);
    renderer.render(scene, pipCam);
    renderer.setScissorTest(false);
    scene.background.copy(bg);
    fly.setXray(false);
    brain.setViewport(innerHeight * renderer.getPixelRatio(), camera.fov);
    hud.placePip({ x, y: innerHeight - yb - s });
  } else hud.placePip(null);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

window.__flyswamp = { world, get mode() { return mode; }, get frames() { return frames; }, useBloom, glName };
window.__shotInfo = window.__flyswamp;
