import test from 'node:test';
import assert from 'node:assert/strict';
import { concat, decodeCNS, gunzip } from '../src/sim/cnsgraph.js';
import { CNSRunner } from '../src/sim/cns-runner.js';
import { LIFNetwork } from '../src/sim/lif.js';
import { mulberry32 } from '../src/sim/rng.js';

// The encoder of pipeline/build_web_graph.py, in JS: degrees, row-wise index deltas, zigzag weights, LEB128.
function varints(xs) {
  const out = [];
  for (let x of xs) {
    do { let b = x & 0x7f; x >>>= 7; if (x) b |= 0x80; out.push(b); } while (x);
  }
  return Uint8Array.from(out);
}

async function gzip(bytes) {
  const s = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

function randomCSR(n, perNeuron, seed) {
  const r = mulberry32(seed), rows = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let k = 0; k < perNeuron; k++) row.push([Math.floor(r() * n), Math.round((r() < 0.75 ? 1 : -1) * (1 + 300 * r()))]);
    rows.push(row.sort((a, b) => a[0] - b[0]));
  }
  const indptr = new Int32Array(n + 1);
  rows.forEach((row, i) => { indptr[i + 1] = indptr[i] + row.length; });
  const flat = rows.flat();
  return { n, nnz: flat.length, ids: null, indptr, indices: Int32Array.from(flat, (e) => e[0]), weight: Int16Array.from(flat, (e) => e[1]) };
}

async function encode(g, chunk) {
  const deg = [], delta = [], zig = [];
  for (let i = 0; i < g.n; i++) {
    deg.push(g.indptr[i + 1] - g.indptr[i]);
    for (let e = g.indptr[i]; e < g.indptr[i + 1]; e++) delta.push(e === g.indptr[i] ? g.indices[e] : g.indices[e] - g.indices[e - 1]);
  }
  for (const w of g.weight) zig.push(((w << 1) ^ (w >> 31)) >>> 0);
  const streams = [varints(deg), varints(delta), varints(zig)];
  const blob = await gzip(concat(streams.map((s) => s.buffer)));
  const chunks = [];
  for (let k = 0; k < blob.length; k += chunk) chunks.push(blob.slice(k, k + chunk).buffer);
  return { header: { format: 'flyswamp-cns-1', n: g.n, nnz: g.nnz, streams: streams.map((s) => s.length) }, chunks };
}

test('the browser graph decodes back to the same CSR, across chunk borders', async () => {
  const g = randomCSR(3000, 20, 7);
  const { header, chunks } = await encode(g, 5000);
  assert.ok(chunks.length > 2);
  const d = decodeCNS(header, await gunzip(concat(chunks)));
  assert.deepEqual(d.indptr, g.indptr);
  assert.deepEqual(d.indices, g.indices);
  assert.deepEqual(d.weight, g.weight);
  const raw = await gunzip(concat(chunks));
  assert.throws(() => decodeCNS({ ...header, streams: [header.streams[0] + 1, ...header.streams.slice(1)] }, raw), /corrupt/);
});

test('the game runner gives the same spikes as the engine stepped directly', () => {
  const g = randomCSR(800, 30, 11);
  const params = { wSyn: 0.2 }, drive = [[1, 150], [2, 150], [3, 150], [4, 150]];
  const runner = new CNSRunner(g, params, 5);
  runner.setDrive(drive);
  const got = [...runner.advance(300, Infinity)];
  runner.setDrive(drive); // unchanged drive is a no-op
  got.push(...runner.advance(400, Infinity));
  const net = new LIFNetwork(g, params);
  net.reset(5);
  net.setPoisson(drive.map((e) => e[0]), drive.map((e) => e[1]));
  const want = [];
  for (let s = 0; s < 4000; s++) { const ns = net.step(); want.push(...net.spikes.subarray(0, ns)); }
  assert.ok(want.length > 100);
  assert.deepEqual(got, want);
});
