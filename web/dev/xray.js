// Dev harness: fly in x-ray with the CNS inside, driven by the real circuit smelling fruit.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { loadAssets } from '../src/data.js';
import { Fly } from '../src/render/fly.js';
import { BrainView } from '../src/render/brain.js';
import { MushroomBody } from '../src/sim/brain.js';
import { makeOdorLibrary } from '../src/sim/odors.js';
import { makeActivityMapper } from '../src/sim/activity.js';

const q = new URLSearchParams(location.search);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x03060a);
const cam = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.01, 100);
const views = { head: [1.35, 0.45, 1.1], brain: [0.74, 1.6, 0.001], brainfront: [2.4, 0.0, 0.0], brainside: [0.74, 0.0, 1.8], side: [0.2, 0.3, 4.2], top: [0.05, 4.2, 0.01], full: [2.5, 2.2, 4.5], front: [3.5, 0.2, 0] };
cam.position.set(...(views[q.get('view')] ?? views.head));
if (q.get('view')?.startsWith('brain')) cam.up.set(q.get('view') === 'brain' ? 1 : 0, q.get('view') === 'brain' ? 0 : 1, 0);
cam.lookAt(['full', 'side', 'top'].includes(q.get('view')) ? new THREE.Vector3(0.2, -0.1, 0) : new THREE.Vector3(0.72, 0.02, 0));

const A = await loadAssets();
const fly = new Fly(A.fly, A.rig);
scene.add(fly.root);
if (q.get('fly') === '0') fly.meshes.forEach((m) => (m.visible = false));
fly.setXray(q.get('x') !== '0');
const brain = new BrainView(A);
brain.setViewport(innerHeight, cam.fov);
fly.model.add(brain.group);
for (const g of (q.get('hide') ?? '').split(',')) if (g) brain.setGroupVisible(g, false);
if (q.get('part')) for (const [k, o] of Object.entries({ meshes: brain.cns, lines: brain.skeletons, somas: brain.somaPoints, cloud: brain.cloud, fly: fly.model })) if (!q.get('part').split(',').includes(k)) { if (k === 'fly') fly.meshes.forEach((m) => (m.visible = false)); else o.visible = false; }

const mb = new MushroomBody(A.circuit);
const lib = makeOdorLibrary(A.circuit.glomeruli);
mb.calibrate([lib.fruit, lib.sundew, lib.frog]);
const collect = makeActivityMapper(A.circuit);
const odor = lib[q.get('odor') ?? 'fruit'].map((x) => x * 0.6);
const reward = q.get('reward') ? 1 : 0, punish = q.get('punish') ? 1 : 0;
for (let t = 0; t < 1.0; t += 0.02) {
  mb.step(odor, 0.02, { reward: t > 0.6 ? reward : 0, punish: t > 0.6 ? punish : 0 });
  brain.update(0.02, collect(mb, { turn: 0.5, escape: q.get('escape') ? 1 : 0 }), mb.u);
}

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, cam));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.9, 0.5, 0.15));
composer.addPass(new OutputPass());
const composer2 = new EffectComposer(renderer);
composer2.addPass(new RenderPass(scene, cam));
composer2.addPass(new OutputPass());
const loop = () => { if (q.get('nobloom')) renderer.render(scene, cam); else if (q.get('comp') === 'plain') composer2.render(); else composer.render(); requestAnimationFrame(loop); };
loop();
window.__shotInfo = { floatRT: renderer.extensions.has('EXT_color_buffer_float'), halfRT: renderer.extensions.has('EXT_color_buffer_half_float'), kcActive: mb.kc.filter((x) => x > 0.01).length, valence: mb.valence, meshes: Object.keys(brain.meshes).length, comps: Object.keys(brain.compartmentDans).length, gloms: brain.glomMeshes.length };
