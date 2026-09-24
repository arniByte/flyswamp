"""The whole-CNS graph for the game: a packed graph from build_graph.py, compressed for the browser, with
each neuron's soma position in the x-ray frame and the neuron sets the game drives or reads.

Outputs (web/public/data/cns/, not in git: rebuild with scripts/setup.sh webgraph):
  cns.json          header: sizes, stream layout, chunk names, engine parameters, neuron sets, attribution
  cns.bin.N         the edges, gzip-compressed and split into 8 MB chunks (artifact hosts limit file sizes)
  cns_neurons.bin   per neuron, in graph order: uint32 bodyId, int16 xyz soma (fly frame, same scale as somas.bin),
                    uint16 index into the type names, uint8 region, uint8 flags (1 = has a position); layout in cns.json

Edge encoding, rows in neuron order: out-degree per neuron, then postsynaptic indices as deltas within a
row, then signed weights zigzag-coded, each as LEB128 varints; web/src/sim/cnsgraph.js decodes it.

usage: python build_web_graph.py [--graph NAME] [--w-syn MV]   (default malecns_min2, 0.2)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ANNOTATIONS, ATTRIBUTION, CACHE, WEB_DATA, flat
from build_anatomy import NM_PER_VOXEL, Q, REGION, quantize, to_fly

CHUNK = 8_000_000  # base64 on artifact hosts: 10.7 MB per text file, under the 16 MB limit
VALIDATION = Path(__file__).resolve().parent.parent / "validation"


def varint(x: np.ndarray) -> bytes:
    """LEB128 of non-negative integers below 2^28, vectorised."""
    x = np.asarray(x, np.int64)
    assert x.min(initial=0) >= 0 and x.max(initial=0) < 1 << 28
    nbytes = 1 + (x >= 1 << 7) + (x >= 1 << 14) + (x >= 1 << 21)
    out = np.zeros(int(nbytes.sum()), np.uint8)
    pos = np.concatenate([[0], np.cumsum(nbytes)[:-1]])
    for k in range(4):
        m = nbytes > k
        more = (nbytes > k + 1)[m]
        out[pos[m] + k] = ((x[m] >> (7 * k)) & 0x7F) | (more.astype(np.int64) << 7)
    return out.tobytes()


def read_csr(prefix: Path):
    h = json.loads(prefix.with_suffix(".json").read_text())
    buf = prefix.with_suffix(".bin").read_bytes()
    a = h["arrays"]
    arr = lambda k, dt: np.frombuffer(buf, dt, a[k]["length"], a[k]["offset"])  # noqa: E731
    return h, arr("indptr", "<i4"), arr("indices", "<i4"), arr("weight", "<i2")


def region_code(superclass) -> int:
    s = superclass if isinstance(superclass, str) else ""
    for k, v in REGION.items():
        if s.startswith(k) or k in s:
            return v
    return 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_min2")
    ap.add_argument("--w-syn", type=float, default=0.2)
    a = ap.parse_args()
    prefix = CACHE / "graphs" / a.graph
    h, indptr, indices, weight = read_csr(prefix)
    meta = json.loads(prefix.with_suffix(".meta.json").read_text())
    n, ids = h["n"], [int(x) for x in h["ids"]]
    index = {b: i for i, b in enumerate(ids)}

    deg = np.diff(indptr)
    pre = np.repeat(np.arange(n), deg)
    delta = np.diff(indices, prepend=0).astype(np.int64)
    delta[indptr[:-1][deg > 0]] = indices[indptr[:-1][deg > 0]]  # first entry of each row is absolute
    zig = (weight.astype(np.int64) << 1) ^ (weight.astype(np.int64) >> 63)
    streams = [varint(deg), varint(delta), varint(zig)]
    assert (np.diff(pre) >= 0).all()
    blob = gzip.compress(b"".join(streams), 6)
    out = WEB_DATA / "cns"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("cns.bin.*"):
        old.unlink()
    chunks = []
    for k in range(0, len(blob), CHUNK):
        name = f"cns.bin.{len(chunks)}"
        (out / name).write_bytes(blob[k:k + CHUNK])
        chunks.append(name)

    # soma positions in graph order; neurons whose soma lies outside the CNS volume get tosomaLocation, or none
    ann = pd.read_feather(flat(ANNOTATIONS), columns=["bodyId", "somaLocation", "tosomaLocation", "superclass", "type", "instance", "somaSide", "rootSide"])
    ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(ids)
    loc = [s if isinstance(s, (list, np.ndarray)) else t for s, t in zip(ann.somaLocation, ann.tosomaLocation)]
    has = np.array([isinstance(p, (list, np.ndarray)) for p in loc])
    pos = np.zeros((n, 3), np.int16)
    pos[has] = quantize(to_fly(np.stack([np.asarray(p, float) for p, h_ in zip(loc, has) if h_]) * NM_PER_VOXEL))
    region = np.array([region_code(s) for s in ann.superclass], np.uint8)
    flags = has.astype(np.uint8)
    names = [x if isinstance(x, str) and x else None for x in ann.instance]
    names = [x or (t if isinstance(t, str) and t else None) or "?" for x, t in zip(names, ann["type"])]
    types = sorted(set(names))
    tix = {t: i for i, t in enumerate(types)}
    type_idx = np.array([tix[x] for x in names], np.uint16)
    assert len(types) < 65536
    parts = [("bodyId", np.array(ids, np.uint32)), ("position", pos), ("name", type_idx), ("region", region), ("flags", flags)]
    layout, off, blobs = {}, 0, []
    for name, arr in parts:
        layout[name] = {"offset": off, "dtype": arr.dtype.name, "length": int(arr.size)}
        blobs.append(arr.tobytes())
        off += arr.nbytes
    (out / "cns_neurons.bin").write_bytes(b"".join(blobs))

    # neuron sets the game drives or reads, as graph indices
    grn = json.loads((VALIDATION / "reference" / "shiu_neuron_sets_malecns.json").read_text())["sets"]
    sets = {}
    for key, name in (("neu_sugar", "sugar"), ("neu_bitter", "bitter"), ("neu_water", "water")):
        for side in ("L", "R"):
            sets[f"{name}_{side}"] = sorted(index[r["bodyId"]] for r in grn[key]
                                            if r["side"] == side and str(r.get("type") or "").startswith("LB") and r["bodyId"] in index)
    inst = meta["instance"]
    sets["mn9"] = {x: i for i, x in enumerate(inst) if isinstance(x, str) and x.startswith("MN9_")}
    typ = ann["type"].fillna("")
    orn = {}
    for i, t in enumerate(typ):
        if t.startswith("ORN_"):
            orn.setdefault(t[4:], []).append(i)
    sets["orn"] = orn

    header = {
        "format": "flyswamp-cns-1", "n": n, "nnz": int(len(indices)), "graph": a.graph,
        "source": h["source"], "weight_meaning": h["weight_meaning"], "attribution": ATTRIBUTION,
        "streams": [len(s) for s in streams], "chunks": chunks, "gzip_bytes": len(blob),
        "engine": {"wSyn": a.w_syn}, "validation": "docs/SCIENCE.md, P2: taste gate on held-out seeds for this graph",
        "neurons": {"file": "cns_neurons.bin", "layout": layout, "names": types, "scale": Q, "with_position": int(has.sum()),
                    "regions": ["central brain", "optic lobe", "VNC", "descending/ascending", "sensory", "other"]},
        "sets": sets,
    }
    (out / "cns.json").write_text(json.dumps(header))
    print(f"{out}: {n} neurons ({int(has.sum())} with a soma position), {len(indices)} connections, "
          f"{len(blob) / 1e6:.1f} MB in {len(chunks)} chunks", file=sys.stderr)


if __name__ == "__main__":
    main()
