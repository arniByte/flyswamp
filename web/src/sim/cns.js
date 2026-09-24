// The whole MaleCNS as a spiking network inside the game (P2-validated model: Shiu et al. 2024 LIF,
// w_syn from the P2 gate, graph from pipeline/build_web_graph.py). Runs in a Web Worker against the wall
// clock; the main thread keeps a decaying activity value per neuron for the x-ray and reads motor neurons.
// Input from the world is only what passed the P2 gate: taste during feeding drives the right labellar
// sugar GRNs at 150 Hz, exactly the gate's stimulus. Smell reaches the spiking model after the KC benchmark.
import { CNSRunner } from './cns-runner.js';
import { concat, decodeCNS, gunzip } from './cnsgraph.js';

const TAU_ACT = 0.15; // s, decay of the displayed activity after a spike
const SUGAR_HZ = 150;
const RATE_WINDOW = 0.5; // s, for the MN9 read-out

export class LiveCNS {
  constructor(header, neurons, chunks) {
    this.header = header;
    this.n = header.n;
    const L = header.neurons.layout, buf = neurons;
    const view = (k, T) => new T(buf, L[k].offset, L[k].length);
    this.bodyId = view('bodyId', Uint32Array);
    this.position = view('position', Int16Array);
    this.nameIndex = view('name', Uint16Array);
    this.region = view('region', Uint8Array);
    this.flags = view('flags', Uint8Array);
    this.names = header.neurons.names;
    this.act = new Float32Array(this.n);
    this.spikesPerSec = 0;
    this.factor = 1;
    this.simMs = 0;
    this.status = 'загрузка';
    this.mn9 = header.sets.mn9.MN9_L;
    this._mn9Times = [];
    this._taste = false;
    this._pending = [];
    this._start(chunks);
  }

  _start(chunks) {
    this._chunks = chunks; // kept until the worker is up, for the main-thread fallback
    try {
      this.worker = new Worker(new URL('./cns-worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e) => this._message(e.data);
      this.worker.onerror = () => { if (this._chunks) this._startLocal(this._chunks); };
      this.worker.postMessage({ type: 'init', header: this.header, chunks });
      this.mode = 'worker';
    } catch {
      this._startLocal(chunks);
    }
  }

  // no workers (some hosts): the same runner on the main thread, a few ms per frame
  _startLocal(chunks) {
    this.worker?.terminate();
    this.worker = null;
    this.mode = 'main thread';
    gunzip(concat(chunks)).then((raw) => {
      this.local = new CNSRunner(decodeCNS(this.header, raw), this.header.engine);
      this._ready();
    }).catch((err) => { this.status = `ошибка: ${err.message}`; });
  }

  _ready() {
    this.status = 'работает';
    this._chunks = null;
    for (const m of this._pending) this._send(m);
    this._pending = [];
  }

  _send(m) {
    if (this.worker) this.worker.postMessage(m);
    else if (this.local && m.type === 'drive') this.local.setDrive(m.entries);
    else if (m.type === 'pause') this._paused = m.value;
  }

  _post(m) {
    if (this.status === 'работает') this._send(m); else this._pending.push(m);
  }

  _message(m) {
    if (m.type === 'ready') this._ready();
    else if (m.type === 'tick') { this._spikes(m.spikes, m.simMs); this.factor += 0.2 * (m.factor - this.factor); }
    else if (m.type === 'error') this.status = `ошибка: ${m.message}`;
  }

  _spikes(spikes, simMs) {
    const act = this.act, t = simMs / 1000;
    for (let k = 0; k < spikes.length; k++) {
      const i = spikes[k];
      act[i] = Math.min(1, act[i] + 0.6);
      if (i === this.mn9) this._mn9Times.push(t);
    }
    const dt = (simMs - this.simMs) / 1000;
    if (dt > 0) this.spikesPerSec = 0.8 * this.spikesPerSec + 0.2 * (spikes.length / dt);
    this.simMs = simMs;
  }

  // world: the fly is feeding → taste; speed: world time per wall second
  setWorld({ feeding, speed }) {
    if (feeding !== this._taste) {
      this._taste = feeding;
      this._post({ type: 'drive', entries: feeding ? this.header.sets.sugar_R.map((i) => [i, SUGAR_HZ]) : [] });
    }
    if (speed !== this._speed) { this._speed = speed; this._post({ type: 'speed', value: speed }); }
  }

  // hidden tab: stop simulating instead of burning a core for nobody
  pause(on) { this._post({ type: 'pause', value: on }); }

  // called every frame with wall-clock dt
  update(dt) {
    if (this.local && this.status === 'работает' && !this._paused) {
      const target = this.local.simMs + dt * 1000 * (this._speed ?? 1);
      const t0 = this.local.simMs;
      this._spikes(this.local.advance(target, 6), this.local.simMs);
      this.factor = (this.local.simMs - t0) / Math.max(1e-3, dt * 1000);
    }
    const k = Math.exp(-dt / TAU_ACT), act = this.act;
    for (let i = 0; i < act.length; i++) if (act[i] > 1e-3) act[i] *= k; else act[i] = 0;
  }

  get mn9Hz() {
    const t = this._mn9Times, from = this.simMs / 1000 - RATE_WINDOW;
    let k = 0;
    while (k < t.length && t[k] <= from) k++;
    if (k) t.splice(0, k);
    return t.length / RATE_WINDOW;
  }
  get tasting() { return this._taste; }

  describe(i) {
    return { name: this.names[this.nameIndex[i]], bodyId: this.bodyId[i], region: this.header.neurons.regions[this.region[i]], activity: this.act[i] };
  }
}

// Loads the header, graph chunks and neuron table through the game's data loader; null when absent.
export async function loadCNS(get, onProgress) {
  let header;
  try { header = await get('cns/cns.json', 'json'); } catch { return null; }
  let done = 0;
  const parts = await Promise.all([...header.chunks, header.neurons.file].map((f) =>
    get(`cns/${f}`, 'bin').then((b) => { onProgress?.(++done, header.chunks.length + 1); return b; })));
  const neurons = parts.pop();
  return new LiveCNS(header, neurons, parts);
}
