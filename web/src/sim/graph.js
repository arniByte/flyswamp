// Reader for the CSR connectome graphs written by pipeline/csr.py.
const TYPES = { int32: Int32Array, int16: Int16Array };

export function parseCSR(header, buffer) {
  if (header.format !== 'flyswamp-csr-1') throw new Error(`unknown graph format ${header.format}`);
  const arr = (name) => {
    const { dtype, offset, length } = header.arrays[name];
    const T = TYPES[dtype];
    if (offset % T.BYTES_PER_ELEMENT) throw new Error(`${name}: misaligned offset`);
    return new T(buffer, offset, length);
  };
  const g = { n: header.n, nnz: header.nnz, ids: header.ids, indptr: arr('indptr'), indices: arr('indices'), weight: arr('weight') };
  if (g.indptr.length !== g.n + 1 || g.indptr[g.n] !== g.nnz) throw new Error('corrupt indptr');
  return g;
}

// Indices of the given neuron ids; throws on ids missing from the graph.
export function indexOf(graph, ids) {
  const map = graph._index ?? (graph._index = new Map(graph.ids.map((id, i) => [id, i])));
  return ids.map((id) => {
    const i = map.get(String(id));
    if (i === undefined) throw new Error(`neuron ${id} not in graph`);
    return i;
  });
}
