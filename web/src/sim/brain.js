// Rate model of the fly olfactory learning circuit, wired with MaleCNS v1.0 synapse counts.
//
//   glomerular input u ──(ORN→PN synapses)──▶ PN ──(PN→KC)──▶ KC ──(KC→MBON × plastic gain)──▶ MBON ──▶ valence
//                                                            ▲  │
//                                                   APL ◀────┘  └── eligibility trace
//                                   reward ▶ PAM ─┐
//                                   punish ▶ PPL1 ┴─(DAN→MBON wiring picks the compartment)─▶ depress KC→MBON gain
//
// No DOM here: the same code runs in the browser and in headless Node experiments.
import { mulberry32, gaussian } from './rng.js';

const DEFAULTS = {
  seed: 1,
  apl: true,            // control: false removes APL feedback inhibition
  dopamine: true,       // control: false blocks plasticity
  shufflePnKc: false,   // control: degree-preserving shuffle of PN→KC wiring
  targetSparsity: 0.07, // fraction of odor-driven KCs active at calibration
  aplStrength: 3.0,
  pnSigma: 0.15,        // Olsen et al. 2010 style divisive normalization
  pnM: 0.08,
  pnNoise: 0.03,
  kcGain: 4.0,
  mbonScale: 0.05,
  traceTau: 2.0,        // s, KC eligibility trace
  learnRate: 0.7,
  recoveryTau: 180,     // s, slow return of depressed synapses (forgetting)
  danBase: 0.05,
  valenceGain: 3.0,
};

export class MushroomBody {
  constructor(circuit, opts = {}) {
    this.opts = { ...DEFAULTS, ...opts };
    this.rand = mulberry32(this.opts.seed);
    const grp = Object.fromEntries(circuit.groups.map((g) => [g.name, g]));
    this.groups = grp;
    this.glomeruli = circuit.glomeruli;
    this.nG = circuit.glomeruli.length;
    this.nP = grp.PN.count;
    this.nK = grp.KC.count;
    this.nM = grp.MBON.count;
    this.nPam = circuit.dan.n_pam;
    this.nD = circuit.dan.n_pam + circuit.dan.n_ppl1;
    this.nA = grp.APL.count;
    this.valenceW = Float32Array.from(circuit.mbon.valence);

    const E = circuit.edges;
    this.glomPn = normalizedCsr(E.glom_pn, this.nP);
    let pnKc = E.pn_kc;
    if (this.opts.shufflePnKc) pnKc = shufflePre(pnKc, mulberry32(this.opts.seed + 99));
    this.pnKc = normalizedCsr(pnKc, this.nK);

    // KC → MBON edges keep raw weights; the plastic gain multiplies each synapse group.
    const km = E.kc_mbon;
    const nE = km.length / 3;
    this.kmPre = new Int32Array(nE);
    this.kmPost = new Int32Array(nE);
    this.kmW = new Float32Array(nE);
    this.kmGain = new Float32Array(nE).fill(1);
    this.mbonNorm = new Float32Array(this.nM);
    for (let e = 0; e < nE; e++) {
      this.kmPre[e] = km[3 * e];
      this.kmPost[e] = km[3 * e + 1];
      this.kmW[e] = km[3 * e + 2];
      this.mbonNorm[this.kmPost[e]] += km[3 * e + 2];
    }

    // APL: KC→APL averaging weights and APL→KC inhibition weights (mean 1 over innervated KCs).
    this.kcApl = new Float32Array(this.nK * this.nA);
    const aplSum = new Float32Array(this.nA);
    forEdges(E.kc_apl, (k, a, w) => { this.kcApl[k * this.nA + a] = w; aplSum[a] += w; });
    for (let k = 0; k < this.nK; k++) for (let a = 0; a < this.nA; a++) this.kcApl[k * this.nA + a] /= aplSum[a] || 1;
    this.aplKc = new Float32Array(this.nK * this.nA);
    const inhSum = new Float32Array(this.nA), inhN = new Float32Array(this.nA);
    forEdges(E.apl_kc, (a, k, w) => { this.aplKc[k * this.nA + a] = w; inhSum[a] += w; inhN[a]++; });
    for (let k = 0; k < this.nK; k++) for (let a = 0; a < this.nA; a++) {
      this.aplKc[k * this.nA + a] /= (inhSum[a] / Math.max(inhN[a], 1)) || 1;
    }

    // DAN → MBON wiring decides which MBON's KC synapses a dopamine neuron can modify.
    // Rows are scaled by min(1, total/50) so weakly innervated MBONs get little dopamine.
    this.danMbon = normalizedCsr(E.dan_mbon, this.nM, 50);

    // State (rates in 0..1)
    this.u = new Float32Array(this.nG);
    this.pn = new Float32Array(this.nP);
    this.kc = new Float32Array(this.nK);
    this.kcInput = new Float32Array(this.nK);
    this.trace = new Float32Array(this.nK);
    this.apl = new Float32Array(this.nA);
    this.mbon = new Float32Array(this.nM);
    this.mbonNaive = new Float32Array(this.nM); // same KC drive with all plastic gains = 1
    this.dan = new Float32Array(this.nD).fill(this.opts.danBase);
    this.da = new Float32Array(this.nM);
    this.valence = 0;
    this.refOutput = 1;
    this.theta = 0.3;
    this._scratch = this.makeBuffers();
    this._scratch2 = this.makeBuffers();
  }

  makeBuffers() {
    const f = (n) => new Float32Array(n);
    return { pn: f(this.nP), kc: f(this.nK), kcIn: f(this.nK), mbon: f(this.nM), mbonNaive: f(this.nM), apl: f(this.nA) };
  }

  // Pick the KC threshold so that reference odors activate ~targetSparsity of odor-driven KCs,
  // and record the typical naive MBON output used to scale readout confidence.
  calibrate(refOdors) {
    let lo = 0, hi = 1;
    for (let it = 0; it < 30; it++) {
      this.theta = 0.5 * (lo + hi);
      let frac = 0;
      for (const u of refOdors) frac += this.evaluate(u).sparsity;
      frac /= refOdors.length;
      if (frac > this.opts.targetSparsity) lo = this.theta; else hi = this.theta;
    }
    let tot = 0;
    const b = this._scratch;
    for (const u of refOdors) {
      this.evaluate(u, b);
      for (let m = 0; m < this.nM; m++) tot += Math.abs(this.valenceW[m]) * b.mbonNaive[m];
    }
    this.refOutput = tot / refOdors.length || 1;
    return { theta: this.theta, refOutput: this.refOutput };
  }

  // Stateless forward pass. Returns valence and code statistics; fills optional buffers.
  evaluate(u, bufs = this._scratch, noise = 0) {
    const { pn, kc, kcIn, mbon, mbonNaive, apl } = bufs;
    this._pn(u, pn, noise);
    this._kc(pn, kc, kcIn, apl);
    this._mbon(kc, mbon, mbonNaive);
    const raw = this._readout(mbon, mbonNaive);
    let driven = 0, active = 0;
    for (let k = 0; k < this.nK; k++) {
      if (kcIn[k] > 0.01) driven++;
      if (kc[k] > 0.01) active++;
    }
    return { raw, valence: Math.tanh(this.opts.valenceGain * raw), sparsity: active / Math.max(driven, 1), active };
  }

  // Advance the circuit by dt seconds. reward/punish are teaching signals in 0..1.
  step(u, dt, { reward = 0, punish = 0 } = {}) {
    const o = this.opts;
    this.u.set(u);
    const pn = this._scratch.pn;
    this._pn(u, pn, o.pnNoise);
    lowpass(this.pn, pn, dt, 0.03);
    this._kc(this.pn, this.kc, this.kcInput, this.apl);
    lowpass(this.trace, this.kc, dt, o.traceTau);
    const b = this._scratch;
    this._mbon(this.kc, b.mbon, b.mbonNaive);
    lowpass(this.mbon, b.mbon, dt, 0.05);
    lowpass(this.mbonNaive, b.mbonNaive, dt, 0.05);
    this.valence = Math.tanh(o.valenceGain * this._readout(this.mbon, this.mbonNaive));

    for (let d = 0; d < this.nD; d++) {
      const drive = o.danBase + (d < this.nPam ? reward : punish) + 0.02 * gaussian(this.rand);
      this.dan[d] += (Math.max(0, drive) - this.dan[d]) * Math.min(1, dt / 0.08);
    }
    this._plasticity(dt);
  }

  _pn(u, out, noise) {
    const { ptr, idx, w } = this.glomPn;
    const o = this.opts;
    let s = 0;
    for (let g = 0; g < this.nG; g++) s += u[g];
    const ms = Math.pow(o.pnM * s, 1.5), sg = Math.pow(o.pnSigma, 1.5);
    for (let p = 0; p < this.nP; p++) {
      let x = 0;
      for (let e = ptr[p]; e < ptr[p + 1]; e++) x += w[e] * u[idx[e]];
      const x15 = Math.pow(x, 1.5);
      let r = x15 / (x15 + sg + ms + 1e-9);
      if (noise) r += noise * gaussian(this.rand);
      out[p] = r < 0 ? 0 : r > 1 ? 1 : r;
    }
  }

  _kc(pn, out, kcIn, apl) {
    const { ptr, idx, w } = this.pnKc;
    for (let k = 0; k < this.nK; k++) {
      let x = 0;
      for (let e = ptr[k]; e < ptr[k + 1]; e++) x += w[e] * pn[idx[e]];
      kcIn[k] = x;
    }
    const nA = this.nA, o = this.opts, th = this.theta;
    apl.fill(0);
    const iters = o.apl ? 6 : 1;
    for (let it = 0; it < iters; it++) {
      for (let k = 0; k < this.nK; k++) {
        let inh = 0;
        if (o.apl) for (let a = 0; a < nA; a++) inh += this.aplKc[k * nA + a] * apl[a];
        const v = (kcIn[k] - th - o.aplStrength * inh) * o.kcGain;
        out[k] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
      if (!o.apl) break;
      for (let a = 0; a < nA; a++) {
        let s = 0;
        for (let k = 0; k < this.nK; k++) s += this.kcApl[k * nA + a] * out[k];
        apl[a] = 0.5 * apl[a] + 0.5 * s; // damped fixed-point iteration
      }
    }
  }

  _mbon(kc, out, naive) {
    out.fill(0);
    naive.fill(0);
    for (let e = 0; e < this.kmPre.length; e++) {
      const x = this.kmW[e] * kc[this.kmPre[e]];
      if (x === 0) continue;
      out[this.kmPost[e]] += x * this.kmGain[e];
      naive[this.kmPost[e]] += x;
    }
    for (let m = 0; m < this.nM; m++) {
      const s = 1 / (this.mbonNorm[m] * this.opts.mbonScale + 1e-9);
      out[m] = Math.min(2, out[m] * s);
      naive[m] = Math.min(2, naive[m] * s);
    }
  }

  // Learned valence: shift of the approach/avoid MBON balance relative to the naive circuit,
  // normalized by total output and faded out for weak odors. Exactly 0 before any learning.
  _readout(mbon, naive) {
    let a = 0, b = 0, an = 0, bn = 0;
    for (let m = 0; m < this.nM; m++) {
      const v = this.valenceW[m];
      if (v > 0) { a += v * mbon[m]; an += v * naive[m]; } else { b -= v * mbon[m]; bn -= v * naive[m]; }
    }
    const tot = an + bn;
    const conf = Math.min(1, tot / (0.3 * this.refOutput));
    return (((a - b) - (an - bn)) / (tot + 1e-6)) * conf;
  }

  _plasticity(dt) {
    const o = this.opts;
    const { ptr, idx, w } = this.danMbon;
    let any = false;
    for (let m = 0; m < this.nM; m++) {
      let x = 0;
      for (let e = ptr[m]; e < ptr[m + 1]; e++) x += w[e] * Math.max(0, this.dan[idx[e]] - o.danBase - 0.05);
      this.da[m] = o.dopamine ? x : 0;
      if (this.da[m] > 1e-3) any = true;
    }
    const rec = dt / o.recoveryTau;
    const g = this.kmGain;
    if (any) {
      const lr = o.learnRate * dt;
      for (let e = 0; e < g.length; e++) {
        const d = this.da[this.kmPost[e]];
        let v = g[e];
        if (d > 0) v -= lr * d * this.trace[this.kmPre[e]] * v;
        v += rec * (1 - v);
        g[e] = v < 0.02 ? 0.02 : v;
      }
    } else {
      for (let e = 0; e < g.length; e++) g[e] += rec * (1 - g[e]);
    }
  }

  resetMemory() {
    this.kmGain.fill(1);
    this.trace.fill(0);
  }

  // Mean plastic gain per MBON, for display.
  mbonGain(out = new Float32Array(this.nM)) {
    const n = new Float32Array(this.nM);
    out.fill(0);
    for (let e = 0; e < this.kmGain.length; e++) {
      out[this.kmPost[e]] += this.kmGain[e] * this.kmW[e];
      n[this.kmPost[e]] += this.kmW[e];
    }
    for (let m = 0; m < this.nM; m++) out[m] = n[m] ? out[m] / n[m] : 1;
    return out;
  }
}

function lowpass(state, target, dt, tau) {
  const k = Math.min(1, dt / tau);
  for (let i = 0; i < state.length; i++) state[i] += (target[i] - state[i]) * k;
}

function forEdges(flat, fn) {
  for (let e = 0; e < flat.length; e += 3) fn(flat[e], flat[e + 1], flat[e + 2]);
}

// Compressed rows keyed by the postsynaptic neuron; weights divided by max(row sum, floor).
function normalizedCsr(flat, nPost, floor = 0) {
  const n = flat.length / 3;
  const count = new Int32Array(nPost + 1);
  for (let e = 0; e < n; e++) count[flat[3 * e + 1] + 1]++;
  for (let i = 0; i < nPost; i++) count[i + 1] += count[i];
  const ptr = Int32Array.from(count);
  const fill = Int32Array.from(count);
  const idx = new Int32Array(n);
  const w = new Float32Array(n);
  for (let e = 0; e < n; e++) {
    const post = flat[3 * e + 1];
    const at = fill[post]++;
    idx[at] = flat[3 * e];
    w[at] = flat[3 * e + 2];
  }
  for (let i = 0; i < nPost; i++) {
    let s = 0;
    for (let e = ptr[i]; e < ptr[i + 1]; e++) s += w[e];
    const d = Math.max(s, floor);
    if (d > 0) for (let e = ptr[i]; e < ptr[i + 1]; e++) w[e] /= d;
  }
  return { ptr, idx, w };
}

// Degree-preserving control: permute presynaptic partners across edges.
function shufflePre(flat, rand) {
  const out = flat.slice();
  const n = flat.length / 3;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = out[3 * i];
    out[3 * i] = out[3 * j];
    out[3 * j] = t;
  }
  return out;
}
