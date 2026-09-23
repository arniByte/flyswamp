// Run our LIF engine (web/src/sim/lif.js) with the stimulus of a reference file and write per-trial
// spike counts in the same schema, for compare.mjs.
// usage: node validation/run_lif.mjs <graph_prefix> <reference.json> <out.json> [trials] [seed] [poisson_hz]
// poisson_hz overrides the reference's rate, to test what rate a reference was actually run at.
// LIF_PARAMS='{"dt":0.2}' overrides engine parameters (see SHIU_2024 in lif.js).
import fs from 'node:fs';
import { parseCSR, indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';

export function loadGraph(prefix) {
  const header = JSON.parse(fs.readFileSync(`${prefix}.json`, 'utf8'));
  const buf = fs.readFileSync(`${prefix}.bin`);
  return { header, graph: parseCSR(header, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)) };
}

function main() {
  const [prefix, refPath, out, trialsArg, seedArg, hzArg] = process.argv.slice(2);
  const ref = JSON.parse(fs.readFileSync(refPath, 'utf8')).meta;
  if (hzArg) ref.poisson_rate_hz = Number(hzArg);
  const trials = Number(trialsArg ?? ref.n_trials), seed = Number(seedArg ?? 1);
  const { header, graph } = loadGraph(prefix);
  const net = new LIFNetwork(graph, JSON.parse(process.env.LIF_PARAMS ?? '{}'));
  net.setPoisson(indexOf(graph, ref.stimulated), ref.poisson_rate_hz);
  const counts = new Map();
  const t0 = performance.now();
  for (let k = 0; k < trials; k++) {
    net.reset(seed + k);
    net.run(ref.t_run_s * 1000);
    for (let i = 0; i < graph.n; i++) {
      if (!net.counts[i]) continue;
      if (!counts.has(i)) counts.set(i, new Array(trials).fill(0));
      counts.get(i)[k] = net.counts[i];
    }
  }
  const wall = (performance.now() - t0) / 1000;
  const meta = {
    ...ref, source: 'flyswamp web/src/sim/lif.js', dataset: header.source, n_trials: trials, seed, dt_s: net.p.dt / 1000,
    params: net.p, elapsed_s: wall, realtime_factor: (trials * ref.t_run_s) / wall,
  };
  const obj = { meta, counts: Object.fromEntries([...counts].map(([i, c]) => [graph.ids[i], c])) };
  fs.writeFileSync(out, JSON.stringify(obj));
  const mn9 = obj.counts[ref.mn9] ?? [0];
  console.log(`${counts.size} active neurons, MN9 ${(mn9.reduce((a, b) => a + b) / trials / ref.t_run_s).toFixed(1)} Hz, ` +
    `${wall.toFixed(1)} s wall for ${trials * ref.t_run_s} s simulated (${meta.realtime_factor.toFixed(2)}× real time)`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
