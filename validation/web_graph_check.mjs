// The game's whole-CNS graph (web/public/data/cns, pipeline/build_web_graph.py) against the packed graph it
// came from: the browser decoder (web/src/sim/cnsgraph.js) must give back the same CSR arrays bit for bit,
// the neuron sets must be the ones p2_bench.mjs uses, and the game's runner must reproduce sugar -> MN9.
// usage: node validation/web_graph_check.mjs [out.json]
import fs from 'node:fs';
import { loadGraph } from './run_lif.mjs';
import { concat, decodeCNS, gunzip } from '../web/src/sim/cnsgraph.js';
import { CNSRunner } from '../web/src/sim/cns-runner.js';

const here = new URL('.', import.meta.url).pathname, dir = `${here}../web/public/data/cns`;
const header = JSON.parse(fs.readFileSync(`${dir}/cns.json`, 'utf8'));
const t0 = performance.now();
const raw = await gunzip(concat(header.chunks.map((c) => fs.readFileSync(`${dir}/${c}`))));
const web = decodeCNS(header, raw);
const decodeMs = performance.now() - t0;
const { graph } = loadGraph(`${here}../.cache/graphs/${header.graph}`);
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const identical = same(web.indptr, graph.indptr) && same(web.indices, graph.indices) && same(web.weight, graph.weight);
const nb = fs.readFileSync(`${dir}/${header.neurons.file}`), L = header.neurons.layout.bodyId;
const bodyIds = new Uint32Array(nb.buffer, nb.byteOffset + L.offset, L.length);
const idsMatch = graph.ids.every((id, i) => String(bodyIds[i]) === id);

const sets = JSON.parse(fs.readFileSync(`${here}reference/shiu_neuron_sets_malecns.json`, 'utf8')).sets;
const index = new Map(graph.ids.map((id, i) => [id, i]));
const pick = (rows, side) => rows.filter((r) => r.side === side && /^LB/.test(r.type ?? '')).map((r) => index.get(String(r.bodyId))).sort((a, b) => a - b);
const setsMatch = ['sugar', 'bitter', 'water'].every((k) => ['L', 'R'].every((s) => same(header.sets[`${k}_${s}`], pick(sets[`neu_${k}`], s))));

// sugar on the right GRNs, read MN9_L, as in the P2 gate, through the game's runner
const mn9 = header.sets.mn9.MN9_L, rates = [];
for (let seed = 1; seed <= 5; seed++) {
  const r = new CNSRunner(web, header.engine, seed);
  r.setDrive(header.sets.sugar_R.map((i) => [i, 150]));
  const spikes = r.advance(1000, Infinity);
  rates.push(spikes.filter((i) => i === mn9).length);
}
const mean = rates.reduce((a, b) => a + b) / rates.length;
const res = { graph: header.graph, gzip_mb: header.gzip_bytes / 1e6, decode_ms: decodeMs, identical_csr: identical, body_ids_match: idsMatch, sets_match_p2_bench: setsMatch, mn9_l_hz_seeds_1_5: rates, mn9_l_mean_hz: mean };
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(res, null, 1));
console.log(`${header.graph}: decoded ${web.nnz} connections in ${decodeMs.toFixed(0)} ms, identical CSR: ${identical}, bodyIds match: ${idsMatch}, sets match: ${setsMatch}; runner sugar_R → MN9_L ${mean.toFixed(1)} Hz (${rates.join(', ')})`);
process.exit(identical && idsMatch && setsMatch ? 0 : 1);
