"""Whole-CNS MaleCNS v1.0 graph for the LIF engine (web/src/sim/lif.js): every Traced neuron, synapse
counts signed by neurotransmitter, in the CSR format of csr.py plus a per-neuron metadata file.

Signs follow Shiu et al. 2024, read off their Connectivity_783.parquet against the FlyWire NT annotations:
GABA and glutamate -1; acetylcholine, dopamine, octopamine, serotonin +1. MaleCNS also calls histamine,
which the FlyWire classifier lacked; the fly's histamine receptors (Ort, HisCl1) are chloride channels,
so histamine is -1. Neurons without a consensus call ('unclear' or missing) follow --unclear:
  drop       their outgoing synapses are removed; they still receive input (default)
  predicted  use their own predicted_nt when that is not 'unclear', else drop

--drop-superclass REGEX removes whole classes of neurons, e.g. "^vnc_" for a brain-only graph like FlyWire.

usage: python build_graph.py [--min-syn N] [--unclear drop|predicted] [--drop-superclass REGEX] [--out PREFIX]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

from common import ANNOTATIONS, ATTRIBUTION, CACHE, NEUROTRANSMITTERS, WEIGHTS, flat
from csr import write_csr

SIGN = {"acetylcholine": 1, "dopamine": 1, "octopamine": 1, "serotonin": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
META = ["type", "flywireType", "superclass", "class", "somaSide", "instance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-syn", type=int, default=1)
    ap.add_argument("--unclear", choices=["drop", "predicted"], default="drop")
    ap.add_argument("--drop-superclass", default=None)
    ap.add_argument("--out", default=str(CACHE / "graphs" / "malecns"))
    a = ap.parse_args()

    ann = pd.read_feather(flat(ANNOTATIONS), columns=["bodyId", "status", *META])
    ann = ann[ann.status == "Traced"]
    if a.drop_superclass:
        ann = ann[~ann.superclass.fillna("").str.contains(a.drop_superclass)]
    ann = ann.sort_values(["superclass", "bodyId"], na_position="last").reset_index(drop=True)
    nt = pd.read_feather(flat(NEUROTRANSMITTERS), columns=["body", "consensus_nt", "predicted_nt"]).set_index("body")
    nt = nt.reindex(ann.bodyId)
    call = nt.consensus_nt.where(nt.consensus_nt.isin(SIGN.keys()))
    if a.unclear == "predicted":
        call = call.fillna(nt.predicted_nt.where(nt.predicted_nt.isin(SIGN.keys())))
    sign = call.map(SIGN).fillna(0).astype(np.int8).to_numpy()

    index = pd.Series(np.arange(len(ann)), index=ann.bodyId)
    t = pf.read_table(flat(WEIGHTS), memory_map=True)
    body_ids = pa.array(ann.bodyId.to_numpy(), type=pa.int64())
    keep = pc.and_(pc.is_in(t["body_pre"], value_set=body_ids), pc.is_in(t["body_post"], value_set=body_ids))
    keep = pc.and_(keep, pc.greater_equal(t["weight"], a.min_syn))
    t = t.filter(keep)
    pre = index.reindex(t["body_pre"].to_numpy()).to_numpy()
    post = index.reindex(t["body_post"].to_numpy()).to_numpy()
    count = t["weight"].to_numpy()
    s = sign[pre]
    nz = s != 0
    w = (count * s)[nz]
    pre, post = pre[nz], post[nz]
    if np.abs(w).max() > np.iinfo(np.int16).max:
        raise SystemExit("weight exceeds int16")

    prefix = Path(a.out)
    header = write_csr(prefix, ann.bodyId.to_numpy(), pre, post, w, {
        "source": f"{ATTRIBUTION}; Traced neurons, synapse confidence >= 0.5, >= {a.min_syn} synapses per connection",
        "weight_meaning": "synapse count signed by consensus neurotransmitter (Shiu et al. 2024 convention, histamine -1)",
        "unclear_nt": a.unclear,
        "dropped_superclass": a.drop_superclass,
    })
    meta = {k: [None if pd.isna(v) else v for v in ann[k]] for k in META}
    meta["nt"] = [None if pd.isna(v) else v for v in call]
    meta["sign"] = sign.tolist()
    prefix.with_suffix(".meta.json").write_text(json.dumps(meta))
    n_drop = int((sign == 0).sum())
    print(f"{prefix}: {header['n']} neurons, {header['nnz']} connections "
          f"({int(count.sum())} synapses kept before sign filter); {n_drop} neurons without a sign emit nothing")


if __name__ == "__main__":
    main()
