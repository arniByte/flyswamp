import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BASE = `${import.meta.env?.BASE_URL ?? '/'}data/`;

async function get(name, kind, onProgress) {
  const r = await fetch(BASE + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  const v = kind === 'json' ? await r.json() : await r.arrayBuffer();
  onProgress?.(name);
  return v;
}

export async function loadAssets(onProgress) {
  const loader = new GLTFLoader();
  const glb = (name) => get(name, 'bin', onProgress).then((b) => loader.parseAsync(b, BASE));
  const [circuit, anatomy, rig, neurons, somas, cns, fly] = await Promise.all([
    get('circuit.json', 'json', onProgress),
    get('anatomy.json', 'json', onProgress),
    get('fly_rig.json', 'json', onProgress),
    get('neurons.bin', 'bin', onProgress),
    get('somas.bin', 'bin', onProgress),
    glb('cns.glb'),
    glb('fly.glb'),
  ]);
  return { circuit, anatomy, rig, neurons, somas, cns: cns.scene, fly: fly.scene };
}
