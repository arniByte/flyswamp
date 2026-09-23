// Where a taste pathway to MN9 breaks between FlyWire and MaleCNS. Drives the same Shiu et al. 2024 GRN set
// in both graphs (FlyWire 783, whose neurons carry FlyWire cell types, and MaleCNS, whose neurons carry
// flywireType), records every neuron's rate over the seeds and compares, by FlyWire cell type:
//   layer 1   the types the stimulated GRNs synapse onto, with synapses and rate in each dataset
//   drivers   the active presynaptic partners of the answering MN9, ranked by rate × synapses
// A type that fires in FlyWire and stays silent in MaleCNS, or that exists in both but lacks the synapses
// onto MN9 in one, is where the pathway breaks.
// usage: node validation/pathway_compare.mjs water|sugar|bitter <hz> <seeds> [out.json]
//        FlyWire runs with the Shiu parameters; LIF_PARAMS applies to the MaleCNS run only.
import fs from 'node:fs';
import { loadGraph } from './run_lif.mjs';
import { indexOf } from '../web/src/sim/graph.js';
import { LIFNetwork } from '../web/src/sim/lif.js';

const [setName, hzArg, seedsArg, out] = process.argv.slice(2);
const hz = Number(hzArg);
const [s0, s1] = seedsArg.split('-').map(Number);
const seeds = Array.from({ length: (s1 ?? s0) - s0 + 1 }, (_, k) => s0 + k);
const here = new URL('.', import.meta.url).pathname, cache = `${here}../.cache/graphs`;

const fwSets = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets.json`, 'utf8')).sets;
const m783 = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_783.json`, 'utf8')).neurons;
const mcSets = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets;
const key = `neu_${setName}`;

const datasets = [
  {
    name: 'FlyWire 783', prefix: `${cache}/flywire783`, params: {},
    stim: fwSets[key].map((id) => m783[id].root_783),
    mn9: Object.fromEntries(fwSets.ids_mn9.map((id) => [`MN9_${m783[id].side[0].toUpperCase()}`, m783[id].root_783])),
    typeOf: (meta, i) => meta.cell_type[i],
  },
  {
    name: 'MaleCNS v1.0', prefix: `${cache}/malecns_min1`, params: JSON.parse(process.env.LIF_PARAMS ?? '{}'),
    stim: mcSets[key].filter((r) => r.side === 'L' && /^LB/.test(r.type ?? '')).map((r) => String(r.bodyId)),
    mn9: null,
    typeOf: (meta, i) => meta.flywireType[i] ?? meta.type[i],
  },
];

const results = datasets.map((d) => {
  const { graph } = loadGraph(d.prefix);
  const meta = JSON.parse(fs.readFileSync(`${d.prefix}.meta.json`, 'utf8'));
  const mn9 = d.mn9
    ? Object.fromEntries(Object.entries(d.mn9).map(([k, id]) => [k, indexOf(graph, [id])[0]]))
    : Object.fromEntries(meta.instance.flatMap((x, i) => (/^MN9_/.test(x ?? '') ? [[x, i]] : [])));
  const stim = indexOf(graph, d.stim);
  const net = new LIFNetwork(graph, d.params);
  net.setPoisson(stim, hz);
  const rate = new Float64Array(graph.n);
  for (const seed of seeds) {
    net.reset(seed);
    net.run(1000);
    for (let i = 0; i < graph.n; i++) rate[i] += net.counts[i] / seeds.length;
  }
  const type = (i) => d.typeOf(meta, i) ?? `#${graph.ids[i]}`;
  const { indptr, indices, weight } = graph;

  // layer 1: synapses from the stimulated set onto each type, and that type's mean rate
  const stimSet = new Set(stim), l1 = new Map();
  for (const s of stim) for (let e = indptr[s]; e < indptr[s + 1]; e++) {
    const j = indices[e];
    if (stimSet.has(j)) continue;
    const t = type(j), row = l1.get(t) ?? { synapses: 0, neurons: new Set() };
    row.synapses += weight[e];
    row.neurons.add(j);
    l1.set(t, row);
  }
  const layer1 = [...l1].map(([t, r]) => ({ type: t, synapses: r.synapses, neurons: r.neurons.size, hz: mean([...r.neurons].map((j) => rate[j])) }))
    .sort((a, b) => Math.abs(b.synapses) - Math.abs(a.synapses));

  // drivers of the answering MN9: presynaptic partners weighted by their rate
  const [side, target] = Object.entries(mn9).reduce((a, b) => (rate[b[1]] > rate[a[1]] ? b : a));
  const drv = new Map();
  for (let i = 0; i < graph.n; i++) for (let e = indptr[i]; e < indptr[i + 1]; e++) {
    if (indices[e] !== target) continue;
    const t = type(i), row = drv.get(t) ?? { synapses: 0, drive: 0, active: 0, neurons: 0 };
    row.synapses += weight[e];
    row.drive += weight[e] * rate[i];
    row.neurons++;
    if (rate[i] > 0) row.active++;
    drv.set(t, row);
  }
  const drivers = [...drv].map(([t, r]) => ({ type: t, ...r })).sort((a, b) => Math.abs(b.drive) - Math.abs(a.drive));
  // rate of every type, to look up the other dataset's drivers
  const byType = new Map();
  for (let i = 0; i < graph.n; i++) {
    const t = type(i), r = byType.get(t) ?? [];
    r.push(rate[i]);
    byType.set(t, r);
  }
  return {
    name: d.name, params: net.p, stimulated: stim.length, mn9: Object.fromEntries(Object.entries(mn9).map(([k, i]) => [k, rate[i]])),
    answering: side, layer1, drivers, byType, responding: rate.filter((x) => x > 0).length,
  };
});

const [fw, mc] = results;
const typeHz = (r, t) => (r.byType.has(t) ? mean(r.byType.get(t)) : null);
const drvOf = (r, t) => r.drivers.find((x) => x.type === t) ?? null;
const compare = fw.drivers.slice(0, 20).map((x) => ({
  type: x.type, fw_hz: typeHz(fw, x.type), fw_synapses_to_mn9: x.synapses, fw_drive: x.drive,
  mc_hz: typeHz(mc, x.type), mc_synapses_to_mn9: drvOf(mc, x.type)?.synapses ?? 0, mc_drive: drvOf(mc, x.type)?.drive ?? 0,
}));
const l1compare = fw.layer1.slice(0, 20).map((x) => ({
  type: x.type, fw_synapses: x.synapses, fw_hz: x.hz,
  mc_synapses: mc.layer1.find((y) => y.type === x.type)?.synapses ?? 0, mc_hz: typeHz(mc, x.type),
}));
const res = {
  stimulus: setName, hz, seeds, datasets: results.map(({ byType, ...r }) => ({ ...r, layer1: r.layer1.slice(0, 40), drivers: r.drivers.slice(0, 40) })),
  mn9_drivers_by_flywire_type: compare, layer1_by_flywire_type: l1compare,
};
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));

const f1 = (x) => (x == null ? '—' : x.toFixed(1));
for (const r of results) console.log(`${r.name}: ${r.stimulated} ${setName} GRNs at ${hz} Hz, wSyn ${r.params.wSyn}; MN9 ${Object.entries(r.mn9).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(', ')} Hz; ${r.responding} neurons respond`);
console.log(`\nlayer 1 (FlyWire types)          FW syn  FW Hz   MC syn  MC Hz`);
for (const x of l1compare) console.log(`  ${x.type.padEnd(28)} ${String(x.fw_synapses).padStart(6)} ${f1(x.fw_hz).padStart(6)}  ${String(x.mc_synapses).padStart(6)} ${f1(x.mc_hz).padStart(6)}`);
console.log(`\ndrivers of FlyWire ${fw.answering}          FW syn  FW Hz   MC syn  MC Hz  (MC ${mc.answering})`);
for (const x of compare) console.log(`  ${x.type.padEnd(28)} ${String(x.fw_synapses_to_mn9).padStart(6)} ${f1(x.fw_hz).padStart(6)}  ${String(x.mc_synapses_to_mn9).padStart(6)} ${f1(x.mc_hz).padStart(6)}`);

function mean(x) { return x.length ? x.reduce((a, b) => a + b, 0) / x.length : 0; }
