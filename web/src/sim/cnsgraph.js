// Decoder for the whole-CNS graph written by pipeline/build_web_graph.py (format flyswamp-cns-1):
// gzip over three LEB128 varint streams — out-degree per neuron, postsynaptic indices as deltas within
// each row, zigzag-coded signed synapse counts. Returns the CSR graph the LIF engine takes.

export async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function concat(buffers) {
  const out = new Uint8Array(buffers.reduce((s, b) => s + b.byteLength, 0));
  let o = 0;
  for (const b of buffers) { out.set(new Uint8Array(b), o); o += b.byteLength; }
  return out;
}

export function decodeCNS(header, raw) {
  if (header.format !== 'flyswamp-cns-1') throw new Error(`unknown graph format ${header.format}`);
  const { n, nnz, streams } = header;
  const indptr = new Int32Array(n + 1), indices = new Int32Array(nnz), weight = new Int16Array(nnz);
  let p = 0;
  const next = () => {
    let x = 0, shift = 0, b;
    do { b = raw[p++]; x |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    return x;
  };
  for (let i = 0; i < n; i++) indptr[i + 1] = indptr[i] + next();
  if (p !== streams[0] || indptr[n] !== nnz) throw new Error('corrupt degree stream');
  for (let i = 0; i < n; i++) {
    let q = 0;
    for (let e = indptr[i], end = indptr[i + 1]; e < end; e++) { q = e === indptr[i] ? next() : q + next(); indices[e] = q; }
  }
  if (p !== streams[0] + streams[1]) throw new Error('corrupt index stream');
  for (let e = 0; e < nnz; e++) { const z = next(); weight[e] = (z >>> 1) ^ -(z & 1); }
  if (p !== raw.length) throw new Error('corrupt weight stream');
  return { n, nnz, ids: null, indptr, indices, weight };
}
