// Engine speed: wall-clock seconds per simulated second on a whole graph, with the sugar GRNs driven at
// 150 Hz as in the P1/P2 runs, for the default (lazy) update and the dense reference path.
// usage: node validation/speed.mjs <graph_prefix> flywire|malecns [simulated_ms] [out.json]
//        LIF_PARAMS as in run_lif.mjs
import fs from 'node:fs';
import os from 'node:os';
import { loadGraph } from './run_lif.mjs';
import { indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';

const [prefix, which, msArg, out] = process.argv.slice(2);
const ms = Number(msArg ?? 1000);
const here = new URL('.', import.meta.url).pathname;
const stim = which === 'flywire'
  ? JSON.parse(fs.readFileSync(`${here}reference/sugar_150hz.json`, 'utf8')).meta.stimulated
  : JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets.neu_sugar
    .filter((r) => r.side === 'R' && /^LB/.test(r.type ?? '')).map((r) => String(r.bodyId));
const { header, graph } = loadGraph(prefix);
const params = JSON.parse(process.env.LIF_PARAMS ?? '{}');
const runs = {};
for (const mode of ['lazy', 'dense']) {
  const net = new LIFNetwork(graph, params);
  net.dense = mode === 'dense';
  net.setPoisson(indexOf(graph, stim), 150);
  net.reset(1);
  const t0 = performance.now();
  let active = 0;
  const steps = Math.round(ms / net.p.dt);
  for (let k = 0; k < steps; k++) { net.step(); active += net.nActive; }
  const wall = (performance.now() - t0) / 1000;
  runs[mode] = { wall_s_per_sim_s: wall / (ms / 1000), spikes: net.counts.reduce((a, b) => a + b, 0), counts: net.counts, active: active / steps };
}
const same = runs.lazy.counts.every((c, i) => c === runs.dense.counts[i]);
const res = {
  graph: prefix.split('/').pop(), neurons: header.n, connections: header.nnz, simulated_ms: ms, params: { ...params },
  cpu: os.cpus()[0]?.model ?? null, node: process.version,
  lazy_s_per_sim_s: runs.lazy.wall_s_per_sim_s, dense_s_per_sim_s: runs.dense.wall_s_per_sim_s,
  lazy_mean_updated_per_step: runs.lazy.active,
  spikes: runs.lazy.spikes, identical_spikes: same,
};
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));
console.log(`${res.graph}: lazy ${res.lazy_s_per_sim_s.toFixed(2)} s (${res.lazy_mean_updated_per_step.toFixed(0)} neurons updated per step), dense ${res.dense_s_per_sim_s.toFixed(2)} s per simulated s; ${res.spikes} spikes, identical: ${same}`);
