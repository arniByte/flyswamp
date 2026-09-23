import test from 'node:test';
import assert from 'node:assert/strict';
import { LIFNetwork, SHIU_2024 } from '../src/sim/lif.js';
import { mulberry32 } from '../src/sim/rng.js';

// edges: [[pre, post, signedCount], ...]
function makeGraph(n, edges) {
  edges = [...edges].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const indptr = new Int32Array(n + 1);
  for (const [pre] of edges) indptr[pre + 1]++;
  for (let i = 0; i < n; i++) indptr[i + 1] += indptr[i];
  return {
    n, nnz: edges.length, ids: Array.from({ length: n }, (_, i) => String(i)), indptr,
    indices: Int32Array.from(edges, (e) => e[1]), weight: Int16Array.from(edges, (e) => e[2]),
  };
}

function randomGraph(n, perNeuron, seed) {
  const r = mulberry32(seed), edges = [];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < perNeuron; k++) {
      const j = Math.floor(r() * n);
      if (j !== i) edges.push([i, j, Math.round((r() < 0.75 ? 1 : -1) * (1 + 12 * r()))]);
    }
  }
  return makeGraph(n, edges);
}

test('membrane follows the exact solution of the linear system', () => {
  const net = new LIFNetwork(makeGraph(1, []));
  const { vRest, tauMem: tm, tauSyn: ts, dt } = SHIU_2024;
  const g0 = 4, u0 = 1.5;
  net.v[0] = vRest + u0; net.g[0] = g0; net._activate(0);
  for (let k = 1; k <= 300; k++) {
    net.step();
    const t = k * dt, K = (g0 * ts) / (ts - tm);
    const u = u0 * Math.exp(-t / tm) + K * (Math.exp(-t / ts) - Math.exp(-t / tm));
    assert.ok(Math.abs(net.v[0] - vRest - u) < 1e-12, `step ${k}`);
    assert.ok(Math.abs(net.g[0] - g0 * Math.exp(-t / ts)) < 1e-12);
  }
});

test('spikes arrive after the synaptic delay and respect the refractory period', () => {
  // neuron 0 is driven by a Poisson input that fires every step; 0 → 1 with 40 synapses
  const net = new LIFNetwork(makeGraph(3, [[0, 1, 40], [2, 1, 1]]));
  net.setPoisson([0], 1e4 / SHIU_2024.dt);
  net.reset(1);
  net.step(); // Poisson event at step 0 lifts v above threshold
  assert.equal(net.step(), 1); // spike detected at step 1
  const first = 1, delay = Math.round(SHIU_2024.delay / SHIU_2024.dt);
  for (let s = 2; s < first + delay; s++) { net.step(); assert.equal(net.g[1], 0, `no input before delay, step ${s}`); }
  net.step();
  assert.ok(Math.abs(net.g[1] - 40 * SHIU_2024.wSyn) < 1e-12, 'g jumps by count · w_syn at step spike+delay');
  // Poisson targets have no refractory period: neuron 0 spikes every other step (reset, then next event)
  net.resetCounts();
  net.run(100);
  assert.equal(net.counts[0], 500);
  // neuron 1, driven hard by neuron 0, cannot fire faster than once per 2.2 ms refractory + reset
  assert.ok(net.counts[1] > 0 && net.counts[1] <= Math.floor(100 / SHIU_2024.tRefractory) + 1, `neuron 1: ${net.counts[1]}`);
});

test('input reaching a refractory neuron is dropped, as Brian2 conditional writes do', () => {
  const net = new LIFNetwork(makeGraph(2, [[0, 1, 40]]));
  net.setPoisson([0], 1e4 / SHIU_2024.dt);
  net.reset(1);
  net.step(); net.step(); // neuron 0 spikes at step 1
  const delay = Math.round(SHIU_2024.delay / SHIU_2024.dt), arrival = 1 + delay;
  while (net.stepIndex < arrival) net.step();
  net.lastSpike[1] = arrival - 5; // neuron 1 is inside its 2.2 ms refractory period when the spike lands
  net.step();
  assert.equal(net.g[1], 0);
  // the Poisson event drawn in the step a neuron spikes is lost as well: 1 spike per 2 steps, not 1 per step
  net.resetCounts(); net.run(10);
  assert.equal(net.counts[0], 50);
});

test('sparse update gives the same spikes as updating every neuron', () => {
  const graph = randomGraph(400, 60, 7);
  const drive = Array.from({ length: 20 }, (_, i) => i * 7);
  const run = (dense) => {
    const net = new LIFNetwork(graph);
    net.dense = dense;
    net.setPoisson(drive, 150);
    net.reset(42);
    net.run(500);
    return Array.from(net.counts);
  };
  const sparse = run(false), dense = run(true);
  assert.ok(sparse.reduce((a, b) => a + b) > 2000, 'network is active');
  assert.ok(sparse.filter((c, i) => c > 0 && !drive.includes(i)).length > 50, 'activity spreads past the driven neurons');
  assert.deepEqual(sparse, dense);
});

test('silencing a neuron removes its synaptic output', () => {
  const graph = makeGraph(2, [[0, 1, 40]]);
  const net = new LIFNetwork(graph);
  net.setPoisson([0], 200);
  net.reset(3);
  net.run(1000);
  assert.ok(net.counts[1] > 0);
  net.silence([0]);
  net.reset(3);
  net.run(1000);
  assert.ok(net.counts[0] > 100);
  assert.equal(net.counts[1], 0);
});
