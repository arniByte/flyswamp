// Which recurrent circuit keeps the network firing after the stimulus is gone. Replays one seed of
// offset_test.mjs (sugar GRNs 1 s, then 1 s without input), keeps the neurons still spiking in the last
// 500 ms and finds the strongly connected components of the excitatory graph among them: a component
// of two or more neurons is a loop that can sustain itself. For every persistent neuron it also lists its
// strongest input from other persistent neurons, to show who drives whom. Each loop edge also gets the
// peak depolarisation one presynaptic spike causes in the model (synapses · wSyn · κ, κ the peak of the
// normalised PSP kernel for tauMem, tauSyn), to compare with the 7 mV from rest to threshold.
// usage: node validation/loop_anatomy.mjs <graph_prefix> <seed> [out.json]   (LIF_PARAMS as in run_lif.mjs)
import fs from 'node:fs';
import { loadGraph } from './run_lif.mjs';
import { indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';

const [prefix, seedArg, out] = process.argv.slice(2);
const seed = Number(seedArg);
const here = new URL('.', import.meta.url).pathname;
const stim = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets.neu_sugar
  .filter((r) => r.side === 'L' && /^LB/.test(r.type ?? '')).map((r) => String(r.bodyId));
const { graph } = loadGraph(prefix);
const meta = JSON.parse(fs.readFileSync(`${prefix}.meta.json`, 'utf8'));
const net = new LIFNetwork(graph, JSON.parse(process.env.LIF_PARAMS ?? '{}'));
net.dense = true;
net.setPoisson(indexOf(graph, stim), 150);
net.reset(seed);
net.run(1000);
net.setPoisson([], 0);
net.run(500);
net.resetCounts();
net.run(500);

const live = [...net.counts.keys()].filter((i) => net.counts[i] > 0);
const inLive = new Map(live.map((i, k) => [i, k]));
const { indptr, indices, weight } = graph;
// excitatory edges among persistent neurons, and the reverse lists for input summaries
const out_ = live.map(() => []), in_ = live.map(() => []);
for (const i of live) {
  for (let e = indptr[i]; e < indptr[i + 1]; e++) {
    const k = inLive.get(indices[e]);
    if (k === undefined) continue;
    in_[k].push({ from: i, w: weight[e] });
    if (weight[e] > 0) out_[inLive.get(i)].push(k);
  }
}

// Tarjan's strongly connected components, iterative
const idx = new Int32Array(live.length).fill(-1), low = new Int32Array(live.length), onStack = new Uint8Array(live.length);
const stack = [], sccs = [];
let counter = 0;
for (let r = 0; r < live.length; r++) {
  if (idx[r] >= 0) continue;
  const work = [[r, 0]];
  idx[r] = low[r] = counter++; stack.push(r); onStack[r] = 1;
  while (work.length) {
    const top = work[work.length - 1], [v, pos] = top;
    if (pos < out_[v].length) {
      top[1]++;
      const w = out_[v][pos];
      if (idx[w] < 0) { idx[w] = low[w] = counter++; stack.push(w); onStack[w] = 1; work.push([w, 0]); }
      else if (onStack[w]) low[v] = Math.min(low[v], idx[w]);
    } else {
      work.pop();
      if (work.length) { const u = work[work.length - 1][0]; low[u] = Math.min(low[u], low[v]); }
      if (low[v] === idx[v]) {
        const comp = [];
        let w;
        do { w = stack.pop(); onStack[w] = 0; comp.push(w); } while (w !== v);
        if (comp.length > 1) sccs.push(comp);
      }
    }
  }
}

const { tauMem: tm, tauSyn: ts, wSyn } = net.p, tPeak = Math.log(tm / ts) * tm * ts / (tm - ts);
const kappa = (ts / (tm - ts)) * (Math.exp(-tPeak / tm) - Math.exp(-tPeak / ts));
const psp = (n) => Math.round(n * wSyn * kappa * 10) / 10;
const name = (i) => meta.instance[i] ?? graph.ids[i];
const describe = (i) => ({ id: graph.ids[i], instance: name(i), nt: meta.nt[i], hz: net.counts[i] * 2 });
const loops = sccs.sort((a, b) => b.length - a.length).map((comp) => {
  const set = new Set(comp.map((k) => live[k]));
  const edges = [];
  for (const i of set) for (let e = indptr[i]; e < indptr[i + 1]; e++) {
    if (weight[e] > 0 && set.has(indices[e])) edges.push({ from: name(i), to: name(indices[e]), synapses: weight[e], peak_psp_mv: psp(weight[e]) });
  }
  return { neurons: [...set].map(describe), edges: edges.sort((a, b) => b.synapses - a.synapses) };
});
const drivers = live.sort((a, b) => net.counts[b] - net.counts[a]).slice(0, 40).map((i) => {
  const k = inLive.get(i);
  const top = in_[k].sort((a, b) => b.w - a.w).slice(0, 3).map((x) => ({ from: name(x.from), synapses: x.w }));
  return { ...describe(i), strongest_live_inputs: top };
});
const res = { graph: prefix.split('/').pop(), params: net.p, seed, psp_kernel_peak: kappa, window_ms: [1500, 2000], n_persistent: live.length, loops, drivers };
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));
console.log(`${res.graph} seed ${seed}: ${live.length} neurons still firing 0.5–1 s after offset, ${loops.length} excitatory loop(s)`);
for (const l of loops) {
  console.log(`  loop of ${l.neurons.length}: ${l.neurons.map((n) => `${n.instance}(${n.nt ?? '—'}, ${n.hz} Hz)`).join(', ')}`);
  for (const e of l.edges.slice(0, 12)) console.log(`    ${e.from} → ${e.to}: ${e.synapses} synapses, ${e.peak_psp_mv} mV peak PSP`);
}
for (const d of drivers.slice(0, 15)) console.log(`  ${d.instance} ${d.hz} Hz ← ${d.strongest_live_inputs.map((x) => `${x.from} ${x.synapses}`).join(', ')}`);
