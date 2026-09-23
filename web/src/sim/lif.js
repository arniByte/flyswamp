// Leaky integrate-and-fire network: a port of the Shiu et al. 2024 (Nature) Brian2 model,
// github.com/philshiu/Drosophila_brain_model/blob/main/model.py. Same equations and constants:
//   dv/dt = (v_0 - v + g) / t_mbr,  dg/dt = -g / tau   (both frozen while refractory)
//   spike when v > v_th; then v = v_rst, g = 0; presynaptic spike → g += w_syn · signed synapse count
//   after a 1.8 ms delay; Poisson inputs add w_syn · f_poi to v and remove the target's refractory period.
// Integration is exact for the linear system (Brian2 method='linear'). Each step follows Brian2's
// schedule: state update → threshold → synapses (delayed spikes, Poisson inputs) → reset.
// As in Brian2, `(unless refractory)` makes v and g conditional writes: synaptic and Poisson input
// reaching a neuron that is refractory, or that spiked this step, is dropped, not stored.
//
// Only neurons away from rest are updated. A neuron whose |v - v_0| and |g| both fall below
// `restEpsilon` is put back exactly at rest; this is the one deviation from the reference and is
// checked by test/lif.test.js (dense and sparse runs give identical spikes).
import { mulberry32 } from './rng.js';

export const SHIU_2024 = {
  dt: 0.1, // ms, Brian2 default clock
  vRest: -52, // mV, Kakaria & de Bivort 2017
  vReset: -52,
  vThresh: -45,
  tauMem: 20, // ms
  tauSyn: 5, // ms, Jürgensen et al. 2021
  tRefractory: 2.2, // ms, Lazar et al. 2021
  delay: 1.8, // ms, Paul et al. 2015
  wSyn: 0.275, // mV per synapse, the model's one free parameter
  poissonScale: 250, // Poisson event = wSyn · poissonScale
  restEpsilon: 1e-6, // mV, see header
};

export class LIFNetwork {
  constructor(graph, params = {}) {
    this.p = { ...SHIU_2024, ...params };
    const p = this.p;
    this.graph = graph;
    const n = (this.n = graph.n);
    this.A = Math.exp(-p.dt / p.tauMem);
    this.C = Math.exp(-p.dt / p.tauSyn);
    this.B = (p.tauSyn / (p.tauSyn - p.tauMem)) * (this.C - this.A);
    this.delaySteps = Math.round(p.delay / p.dt);
    this.refSteps = Math.round(p.tRefractory / p.dt);
    if (this.delaySteps < 1) throw new Error('delay must be at least one step');
    this.v = new Float64Array(n);
    this.g = new Float64Array(n);
    this.lastSpike = new Int32Array(n);
    this.refractory = new Int32Array(n);
    this.silenced = new Uint8Array(n);
    this.active = new Int32Array(n);
    this.inActive = new Uint8Array(n);
    this.pinned = new Uint8Array(n);
    this.counts = new Uint32Array(n);
    this.spikes = new Int32Array(n);
    this.nSpikes = 0;
    this.ring = Array.from({ length: this.delaySteps }, () => ({ ids: new Int32Array(64), n: 0 }));
    this.poisson = { ids: new Int32Array(0), p: new Float64Array(0) };
    this.dense = false; // true updates every neuron every step (reference path for tests)
    this.reset(1);
  }

  // Clears state, spike queue and counts; keeps Poisson inputs and silencing.
  reset(seed) {
    const p = this.p;
    this.rand = mulberry32(seed);
    this.v.fill(p.vRest);
    this.g.fill(0);
    this.lastSpike.fill(-(1 << 30));
    this.refractory.fill(this.refSteps);
    for (const i of this.poisson.ids) this.refractory[i] = 0;
    this.inActive.fill(0);
    this.nActive = 0;
    for (const i of this.poisson.ids) this._activate(i);
    for (const s of this.ring) s.n = 0;
    this.counts.fill(0);
    this.nSpikes = 0;
    this.stepIndex = 0;
  }

  // Poisson drive as in Shiu et al.: one input per neuron at `hz`, each event adds wSyn·poissonScale to v.
  setPoisson(indices, hz) {
    for (const i of this.poisson.ids) { this.pinned[i] = 0; this.refractory[i] = this.refSteps; }
    const rate = Array.isArray(hz) ? hz : indices.map(() => hz);
    this.poisson = { ids: Int32Array.from(indices), p: Float64Array.from(rate, (r) => r * this.p.dt * 1e-3) };
    for (const i of indices) { this.pinned[i] = 1; this.refractory[i] = 0; this._activate(i); }
  }

  // Zero all outgoing synapses of these neurons (Shiu et al. `neu_slnc`).
  silence(indices, on = true) {
    for (const i of indices) this.silenced[i] = on ? 1 : 0;
  }

  resetCounts() {
    this.counts.fill(0);
  }

  _activate(i) {
    if (!this.inActive[i]) { this.inActive[i] = 1; this.active[this.nActive++] = i; }
  }

  run(ms) {
    const steps = Math.round(ms / this.p.dt);
    for (let k = 0; k < steps; k++) this.step();
  }

  step() {
    const { v, g, lastSpike, refractory, active, inActive, pinned, spikes, A, B, C } = this;
    const { vRest, vReset, vThresh, restEpsilon, wSyn } = this.p;
    const s = this.stepIndex;
    if (this.dense && this.nActive < this.n) for (let i = 0; i < this.n; i++) this._activate(i);

    // groups + thresholds
    let ns = 0, keep = 0;
    for (let a = 0, na = this.nActive; a < na; a++) {
      const i = active[a];
      if (s - lastSpike[i] >= refractory[i]) {
        const u = v[i] - vRest, gi = g[i];
        v[i] = vRest + u * A + gi * B;
        g[i] = gi * C;
        if (v[i] > vThresh) { spikes[ns++] = i; lastSpike[i] = s; }
      }
      if (!this.dense && !pinned[i] && Math.abs(v[i] - vRest) < restEpsilon && Math.abs(g[i]) < restEpsilon) {
        v[i] = vRest; g[i] = 0; inActive[i] = 0;
      } else active[keep++] = i;
    }
    this.nActive = keep;

    // synapses: spikes emitted delaySteps ago arrive now; this step's spikes take their slot
    const slot = this.ring[s % this.delaySteps];
    const { indptr, indices, weight } = this.graph, silenced = this.silenced;
    for (let k = 0; k < slot.n; k++) {
      const pre = slot.ids[k];
      if (silenced[pre]) continue;
      for (let e = indptr[pre], end = indptr[pre + 1]; e < end; e++) {
        const j = indices[e], since = s - lastSpike[j];
        if (since === 0 || since < refractory[j]) continue;
        g[j] += weight[e] * wSyn;
        if (!inActive[j]) { inActive[j] = 1; active[this.nActive++] = j; }
      }
    }
    if (slot.ids.length < ns) slot.ids = new Int32Array(Math.max(ns, slot.ids.length * 2));
    slot.ids.set(spikes.subarray(0, ns));
    slot.n = ns;

    const { ids: pIds, p: pProb } = this.poisson, pw = wSyn * this.p.poissonScale, rand = this.rand;
    for (let k = 0; k < pIds.length; k++) {
      const i = pIds[k], since = s - lastSpike[i];
      if (rand() < pProb[k] && since !== 0 && since >= refractory[i]) v[i] += pw;
    }

    // resets
    const counts = this.counts;
    for (let k = 0; k < ns; k++) {
      const i = spikes[k];
      v[i] = vReset; g[i] = 0;
      counts[i]++;
    }
    this.nSpikes = ns;
    this.stepIndex = s + 1;
    return ns;
  }
}
