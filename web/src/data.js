import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BASE = `${import.meta.env?.BASE_URL ?? '/'}data/`;
// Hosts that serve no binary types (claude.ai artifacts) get binaries as base64 text; see scripts/artifact.mjs.
const B64 = globalThis.FLYSWAMP_B64 === true;

function fromBase64(text) {
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(text.trim()).buffer;
  const s = atob(text.trim());
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out.buffer;
}

async function get(name, kind, onProgress) {
  const b64 = kind === 'bin' && B64;
  const r = await fetch(BASE + name + (b64 ? '.b64.txt' : ''));
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  const v = kind === 'json' ? await r.json() : b64 ? fromBase64(await r.text()) : await r.arrayBuffer();
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
