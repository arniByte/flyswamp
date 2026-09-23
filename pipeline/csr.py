"""Signed synapse-count graph in CSR form, read by the LIF engine (web/src/sim/graph.js).

<prefix>.json is the header, <prefix>.bin holds little-endian arrays at the offsets it lists:
  indptr   int32[n+1]  row pointers, rows are presynaptic neurons
  indices  int32[nnz]  postsynaptic neuron index
  weight   int16[nnz]  signed synapse count (+ excitatory, - inhibitory)
Neuron ids are strings: FlyWire root ids do not fit in a float64.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "flyswamp-csr-1"


def write_csr(prefix: Path, ids, pre, post, weight, meta: dict) -> dict:
    n = len(ids)
    pre, post, weight = np.asarray(pre), np.asarray(post), np.asarray(weight)
    if len(pre) and (pre.min() < 0 or pre.max() >= n or post.min() < 0 or post.max() >= n):
        raise ValueError("edge index out of range")
    if np.abs(weight).max(initial=0) > np.iinfo(np.int16).max:
        raise ValueError("weight does not fit int16")
    order = np.lexsort((post, pre))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    arrays = [("indptr", indptr.astype("<i4")), ("indices", post[order].astype("<i4")), ("weight", weight[order].astype("<i2"))]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    layout, offset = {}, 0
    with open(prefix.with_suffix(".bin"), "wb") as f:
        for name, a in arrays:
            layout[name] = {"dtype": {"<i4": "int32", "<i2": "int16"}[a.dtype.str], "offset": offset, "length": len(a)}
            f.write(a.tobytes())
            offset += a.nbytes
    header = {"format": FORMAT, "n": n, "nnz": int(len(post)), **meta, "arrays": layout, "ids": [str(i) for i in ids]}
    prefix.with_suffix(".json").write_text(json.dumps(header))
    return header
