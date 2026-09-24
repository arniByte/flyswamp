// Runs the whole-CNS LIF network (lif.js, lazy update) against the wall clock, for the game.
// The world tells it which sensory neurons fire at what Poisson rate; it returns every spike.
// Used inside cns-worker.js, or on the main thread when workers are unavailable.
import { LIFNetwork } from './lif.js';

export class CNSRunner {
  constructor(graph, params, seed = 1) {
    this.net = new LIFNetwork(graph, params);
    this.net.reset(seed);
    this.simMs = 0;
    this.drive = '';
    this.buf = new Int32Array(1 << 16);
  }

  // entries: [[neuron index, Hz], ...]; unchanged drive keeps the Poisson state as it is
  setDrive(entries) {
    const key = entries.map((e) => e.join(':')).join(',');
    if (key === this.drive) return;
    this.drive = key;
    this.net.setPoisson(entries.map((e) => e[0]), entries.map((e) => e[1]));
  }

  // Steps until simulated time reaches targetMs or budgetMs of wall time is spent; returns the spikes.
  advance(targetMs, budgetMs) {
    const net = this.net, dt = net.p.dt, t0 = performance.now();
    let n = 0;
    while (this.simMs < targetMs) {
      for (let k = 0; k < 10; k++) {
        const ns = net.step();
        if (n + ns > this.buf.length) { const b = new Int32Array(Math.max(this.buf.length * 2, n + ns)); b.set(this.buf.subarray(0, n)); this.buf = b; }
        this.buf.set(net.spikes.subarray(0, ns), n);
        n += ns;
      }
      this.simMs += 10 * dt;
      if (performance.now() - t0 > budgetMs) break;
    }
    return this.buf.slice(0, n);
  }
}
