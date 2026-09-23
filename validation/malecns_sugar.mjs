// First P2 probe: the Shiu et al. 2024 model, parameters unchanged, on a MaleCNS graph (pipeline/build_graph.py).
// Drives the left labellar sugar GRNs found by map_flywire_to_malecns.py and reports both MN9s, how many
// neurons respond, by superclass, and speed.
// usage: node validation/malecns_sugar.mjs <graph_prefix> [poisson_hz] [trials] [out.json]
//        LIF_PARAMS='{"dt":0.2}' DENSE=1 to change engine settings
import fs from 'node:fs';
import { loadGraph } from './run_lif.mjs';
import { indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';

const [prefix, hzArg, trialsArg, out] = process.argv.slice(2);
const hz = Number(hzArg ?? 150), trials = Number(trialsArg ?? 5);
const here = new URL('.', import.meta.url).pathname;
const sets = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets;
const sugar = sets.neu_sugar.filter((r) => r.side === 'L' && /^LB/.test(r.type ?? '')).map((r) => String(r.bodyId));
const { header, graph } = loadGraph(prefix);
const meta = JSON.parse(fs.readFileSync(`${prefix}.meta.json`, 'utf8'));
const mn9 = meta.instance.flatMap((x, i) => (/^MN9_/.test(x ?? '') ? [[x, i]] : []));

const params = JSON.parse(process.env.LIF_PARAMS ?? '{}');
const net = new LIFNetwork(graph, params);
net.dense = process.env.DENSE === '1';
net.setPoisson(indexOf(graph, sugar), hz);
const mn = mn9.map(() => []), bySuper = {};
let responding = 0;
const t0 = performance.now();
for (let k = 0; k < trials; k++) {
  net.reset(k + 1);
  net.run(1000);
  mn9.forEach(([, i], j) => mn[j].push(net.counts[i]));
  for (let i = 0; i < graph.n; i++) {
    if (!net.counts[i]) continue;
    responding++;
    const s = meta.superclass[i] ?? 'unknown';
    bySuper[s] = (bySuper[s] ?? 0) + net.counts[i];
  }
}
const wall = (performance.now() - t0) / 1000 / trials;
const stat = (x) => { const m = x.reduce((a, b) => a + b, 0) / x.length; return [m, Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1))]; };
const res = {
  graph: header.source, unclear_nt: header.unclear_nt, nnz: header.nnz, stimulus: { neurons: sugar, hz }, trials,
  params: net.p, dense: net.dense, mn9: Object.fromEntries(mn9.map(([x], j) => [x, stat(mn[j])])),
  responding_neurons: responding / trials,
  spikes_per_s_by_superclass: Object.fromEntries(Object.entries(bySuper).sort((a, b) => b[1] - a[1]).map(([s, c]) => [s, c / trials])),
  wall_s_per_sim_s: wall,
};
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));
console.log(`${sugar.length} sugar GRNs (L) @${hz} Hz: ${Object.entries(res.mn9).map(([x, [m, s]]) => `${x} ${m.toFixed(1)} ± ${s.toFixed(1)} Hz`).join(', ')}; ` +
  `${res.responding_neurons.toFixed(0)} neurons respond; ${wall.toFixed(2)} s wall per simulated s`);
