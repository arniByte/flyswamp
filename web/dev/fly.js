// Dev harness: renders the fly rig from a chosen view (?view=side|top|front|persp&x=1&pose=...).
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { loadFly } from '../src/render/fly.js';

const q = new URLSearchParams(location.search);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101418);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.DirectionalLight(0xffffff, 2).translateY(5).translateX(3));
const cam = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.01, 100);
const views = { side: [0, 0.3, 9], top: [0.01, 9, 0], front: [9, 0.4, 0], persp: [5, 4, 6], back: [-6, 3, 4] };
cam.position.set(...(views[q.get('view')] ?? views.persp));
cam.lookAt(0, 0, 0);
const fly = await loadFly('/data/fly.glb', '/data/fly_rig.json');
scene.add(fly.root);
if (q.get('x')) { fly.setXray(true); }
for (const [k, v] of q.entries()) if (k.startsWith('j.')) fly.set(k.slice(2), +v);
fly.applyPose();
scene.add(new THREE.AxesHelper(2));
renderer.render(scene, cam);
window.__shotInfo = { joints: Object.keys(fly.joints).length, bodies: Object.keys(fly.bodies).length, meshes: fly.meshes.length };
