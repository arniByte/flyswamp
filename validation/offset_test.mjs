// Sanity benchmark: after the stimulus stops, activity must die out (no input, no spontaneous firing in
// the model). Drives sugar GRNs for 1 s, then records 1 s without input.
// Repeated over seeds, since reverberation can start by chance; the benchmark passes only if every seed goes quiet.
// usage: node validation/offset_test.mjs <graph_prefix> flywire|malecns [out.json] [seeds]   (LIF_PARAMS as in run_lif.mjs)
import fs from 'node:fs';
import { loadGraph } from './run_lif.mjs';
import { indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';

const [prefix, which, out, seedsArg] = process.argv.slice(2);
const seeds = Number(seedsArg ?? 1);
const here = new URL('.', import.meta.url).pathname;
const stim = which === 'flywire'
  ? JSON.parse(fs.readFileSync(`${here}reference/sugar_150hz.json`, 'utf8')).meta.stimulated
  : JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets.neu_sugar
    .filter((r) => r.side === 'L' && /^LB/.test(r.type ?? '')).map((r) => String(r.bodyId));
const { graph } = loadGraph(prefix);
const meta = fs.existsSync(`${prefix}.meta.json`) ? JSON.parse(fs.readFileSync(`${prefix}.meta.json`, 'utf8')) : null;
const net = new LIFNetwork(graph, JSON.parse(process.env.LIF_PARAMS ?? '{}'));
net.dense = true;
const stimIdx = indexOf(graph, stim), perBin = Math.round(100 / net.p.dt), runs = [];
let persisting = [], persistCounts = null;
for (let seed = 1; seed <= seeds; seed++) {
  net.setPoisson(stimIdx, 150);
  net.reset(seed);
  const bins = [];
  for (let b = 0; b < 20; b++) {
    if (b === 10) { net.setPoisson([], 0); net.resetCounts(); }
    let s = 0;
    for (let k = 0; k < perBin; k++) s += net.step();
    bins.push(s);
  }
  const p = [...net.counts.keys()].filter((i) => net.counts[i] > 0).sort((a, b) => net.counts[b] - net.counts[a]);
  if (p.length > persisting.length) { persisting = p; persistCounts = Uint32Array.from(net.counts); }
  runs.push({ seed, spikes_per_100ms: bins, quiet_at_end: bins[19] === 0, neurons_active_after_offset: p.length });
}
const res = {
  graph: prefix.split('/').pop(), params: net.p, stimulus_off_after_bin: 10, runs,
  passes: runs.every((r) => r.quiet_at_end),
  top_after_offset: persisting.slice(0, 30).map((i) => ({ id: graph.ids[i], instance: meta?.instance[i] ?? null, nt: meta?.nt[i] ?? null, spikes: persistCounts[i] })),
};
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));
for (const r of runs) console.log(`${res.graph} seed ${r.seed}: spikes/100 ms ${r.spikes_per_100ms.join(' ')}`);
console.log(`${res.graph}: ${runs.filter((r) => r.quiet_at_end).length}/${seeds} seeds quiet → ${res.passes ? 'PASS' : 'FAIL'}`);
