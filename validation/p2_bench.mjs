// P2 benchmark suite for the LIF engine on a whole connectome. Every benchmark drives labellar GRN sets
// from the Shiu et al. 2024 experiments (figures.ipynb) with Poisson input and reads out MN9:
//   sugar         sugar GRNs at 150 Hz, 1 s                                   Shiu Fig. 1
//   sugar_bitter  sugar at 150 Hz plus bitter GRNs at 100 Hz, the second Poisson class (r_poi2), Shiu Fig. 3a
//   bitter        bitter alone                                                MN9 should stay quiet
//   water         water GRNs at 200 Hz (at 150 Hz FlyWire MN9 barely answers)  Shiu Fig. 4
//   offset        sugar for 1 s, then 1 s without input: activity must die out
// MN9 is read on both sides and the stronger one counts. In FlyWire the Shiu GRN sets are left-side and
// the contralateral MN9_R answers most. MaleCNS MN9_R has a tenth of the input synapses of MN9_L
// (water_grn_targets.py), so the MaleCNS suite mirrors the arrangement: right GRNs, contralateral MN9_L.
// The FlyWire 630 run of the same suite with our engine, which matches the authors' Brian2 model (P1), is
// the reference the MaleCNS criteria are relative to.
//
// --shuffle SEED rewires the graph before running: every connection keeps its presynaptic neuron and
// signed weight, and the postsynaptic ends are permuted over all connections. In- and out-degree of every
// neuron stay exact (configuration model); self-connections are re-drawn, repeated pairs are allowed.
//
// usage: node validation/p2_bench.mjs <graph_prefix> flywire|malecns <seeds> [out.json] [--shuffle SEED]
//        seeds as "1-5" or "3,7,9"; LIF_PARAMS='{"wSyn":0.2}' and DENSE=1 as in the other scripts;
//        BITTER_HZ (default 100) and WATER_HZ (default 200) set those rates; BENCH=sugar,offset runs a subset;
//        STIM_SIDE=L|R|both picks the MaleCNS GRNs (default R).
import fs from 'node:fs';
import { loadGraph } from './run_lif.mjs';
import { indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';
import { mulberry32 } from '../web/src/sim/rng.js';

const args = process.argv.slice(2);
const shuffleAt = args.indexOf('--shuffle');
const shuffleSeed = shuffleAt >= 0 ? Number(args.splice(shuffleAt, 2)[1]) : null;
const [prefix, which, seedsArg, out] = args;
const seeds = seedsArg.includes('-')
  ? (([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k))(seedsArg.split('-').map(Number))
  : seedsArg.split(',').map(Number);
const HZ = 150, BITTER_HZ = Number(process.env.BITTER_HZ ?? 100), WATER_HZ = Number(process.env.WATER_HZ ?? 200);
const here = new URL('.', import.meta.url).pathname;

// stimulus sets: FlyWire 630 ids from the authors' notebook; MaleCNS sets carried over by connectivity
// (map_flywire_to_malecns.py), right labellar GRNs (see above)
let sets, mn9Ids;
if (which === 'flywire') {
  const s = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets.json`, 'utf8')).sets;
  sets = { sugar: s.neu_sugar, bitter: s.neu_bitter, water: s.neu_water };
  // sides from the FlyWire 783 annotations; the notebook's '# left' on the first id disagrees with them
  const typed = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_783.json`, 'utf8')).neurons;
  mn9Ids = Object.fromEntries(s.ids_mn9.map((id) => [`MN9_${typed[id].side[0].toUpperCase()}`, id]));
} else {
  const s = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets;
  const side = process.env.STIM_SIDE ?? 'R';
  const pick = (rows) => rows.filter((r) => (side === 'both' || r.side === side) && /^LB/.test(r.type ?? '')).map((r) => r.bodyId);
  sets = { sugar: pick(s.neu_sugar), bitter: pick(s.neu_bitter), water: pick(s.neu_water) };
  mn9Ids = null;
}

const { header, graph } = loadGraph(prefix);
if (shuffleSeed !== null) shuffle(graph, shuffleSeed);
const idx = Object.fromEntries(Object.entries(sets).map(([k, v]) => [k, indexOf(graph, v.map(String))]));
let mn9;
if (mn9Ids) mn9 = Object.fromEntries(Object.entries(mn9Ids).map(([k, id]) => [k, indexOf(graph, [id])[0]]));
else {
  const meta = JSON.parse(fs.readFileSync(`${prefix}.meta.json`, 'utf8'));
  mn9 = Object.fromEntries(meta.instance.flatMap((x, i) => (/^MN9_/.test(x ?? '') ? [[x, i]] : [])));
}

const net = new LIFNetwork(graph, JSON.parse(process.env.LIF_PARAMS ?? '{}'));
net.dense = process.env.DENSE === '1';
const only = process.env.BENCH ? new Set(process.env.BENCH.split(',')) : null;
const t0 = performance.now();
let simMs = 0;

// one 1 s trial per seed; returns MN9 rates per side and responding-neuron counts
function drive(stim) {
  net.setPoisson(stim.flatMap(([ids]) => ids), stim.flatMap(([ids, hz]) => ids.map(() => hz)));
  const trials = seeds.map((seed) => {
    net.reset(seed);
    net.run(1000);
    simMs += 1000;
    let responding = 0, spikes = 0;
    for (let i = 0; i < graph.n; i++) if (net.counts[i]) { responding++; spikes += net.counts[i]; }
    return { seed, mn9: Object.fromEntries(Object.entries(mn9).map(([k, i]) => [k, net.counts[i]])), responding, spikes };
  });
  const side = Object.keys(mn9).map((k) => [k, stat(trials.map((t) => t.mn9[k]))]);
  const best = side.reduce((a, b) => (b[1][0] > a[1][0] ? b : a));
  return {
    mn9: Object.fromEntries(side), mn9_best: { side: best[0], hz: best[1] },
    responding: stat(trials.map((t) => t.responding)), total_spikes: stat(trials.map((t) => t.spikes)), trials,
  };
}

function offset() {
  const perBin = Math.round(100 / net.p.dt);
  const runs = seeds.map((seed) => {
    net.setPoisson(idx.sugar, HZ);
    net.reset(seed);
    const bins = [];
    for (let b = 0; b < 20; b++) {
      if (b === 10) net.setPoisson([], 0);
      let s = 0;
      for (let k = 0; k < perBin; k++) s += net.step();
      bins.push(s);
    }
    simMs += 2000;
    return { seed, spikes_per_100ms: bins, quiet_at_end: bins[19] === 0 };
  });
  return { runs, quiet_seeds: runs.filter((r) => r.quiet_at_end).length, n_seeds: runs.length };
}

const res = {
  graph: header.source, prefix: prefix.split('/').pop(), which, seeds, shuffle_seed: shuffleSeed,
  params: net.p, dense: net.dense, poisson_hz: HZ, bitter_hz: BITTER_HZ, water_hz: WATER_HZ, stim_side: which === 'malecns' ? process.env.STIM_SIDE ?? 'R' : 'as in the notebook',
  stimulus_sizes: Object.fromEntries(Object.entries(idx).map(([k, v]) => [k, v.length])), benchmarks: {},
};
const run = (name, fn) => { if (!only || only.has(name)) res.benchmarks[name] = fn(); };
run('sugar', () => drive([[idx.sugar, HZ]]));
run('sugar_bitter', () => drive([[idx.sugar, HZ], [idx.bitter, BITTER_HZ]]));
run('bitter', () => drive([[idx.bitter, BITTER_HZ]]));
run('water', () => drive([[idx.water, WATER_HZ]]));
run('offset', offset);
const b = res.benchmarks;
if (b.sugar && b.sugar_bitter) {
  const s = b.sugar.mn9_best.hz[0];
  res.bitter_suppression = s > 0 ? 1 - b.sugar_bitter.mn9[b.sugar.mn9_best.side][0] / s : null;
}
res.wall_s = (performance.now() - t0) / 1000;
res.wall_s_per_sim_s = res.wall_s / (simMs / 1000);
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));

const f = (x) => `${x[0].toFixed(1)} ± ${x[1].toFixed(1)}`;
console.log(`${res.prefix}${shuffleSeed !== null ? ` shuffled(${shuffleSeed})` : ''} wSyn ${net.p.wSyn}, seeds ${seedsArg}`);
for (const k of ['sugar', 'sugar_bitter', 'bitter', 'water']) {
  if (!b[k]) continue;
  console.log(`  ${k.padEnd(12)} MN9 ${Object.entries(b[k].mn9).map(([s, x]) => `${s} ${f(x)}`).join(', ')} Hz; ${b[k].responding[0].toFixed(0)} neurons respond`);
}
if (res.bitter_suppression != null) console.log(`  bitter suppresses sugar → MN9 by ${(100 * res.bitter_suppression).toFixed(0)} %`);
if (b.offset) console.log(`  offset       ${b.offset.quiet_seeds}/${b.offset.n_seeds} seeds quiet 1 s after the stimulus`);
console.log(`  ${res.wall_s.toFixed(0)} s wall, ${res.wall_s_per_sim_s.toFixed(1)} s per simulated s`);

function stat(x) {
  const m = x.reduce((a, c) => a + c, 0) / x.length;
  return [m, x.length > 1 ? Math.sqrt(x.reduce((a, c) => a + (c - m) ** 2, 0) / (x.length - 1)) : 0];
}

function shuffle(g, seed) {
  const rand = mulberry32(seed), post = g.indices.slice(), pre = new Int32Array(g.nnz);
  for (let i = 0; i < g.n; i++) pre.fill(i, g.indptr[i], g.indptr[i + 1]);
  for (let e = post.length - 1; e > 0; e--) {
    const k = Math.floor(rand() * (e + 1));
    [post[e], post[k]] = [post[k], post[e]];
  }
  // a self-connection swaps its target with a random connection where the swap creates none
  for (let e = 0; e < post.length; e++) {
    while (post[e] === pre[e]) {
      const k = Math.floor(rand() * post.length);
      if (post[k] !== pre[e] && post[e] !== pre[k]) [post[e], post[k]] = [post[k], post[e]];
    }
  }
  g.indices = post;
}
