"""Neurotransmitter calls and mutual synapse counts in MaleCNS v1.0 for the neurons that keep the whole-CNS
model firing after the stimulus (validation/loop_anatomy.mjs). The sign the graph gives a neuron comes from
consensus_nt (pipeline/build_graph.py); this shows where that call is weak or missing, and the synapse
counts between the loop types, to set against other connectomes (MANC, BANC, FlyWire via Virtual Fly Brain).

usage: python loop_nt.py [out.json] [TYPE ...]
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from common import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS, flat  # noqa: E402

TYPES = ["IN10B003", "IN08B004", "DNg33", "AN27X013", "AN09A005", "DLMn a, b", "DLMn c-f"]
NT_COLS = ["consensus_nt", "predicted_nt", "predicted_nt_confidence", "celltype_predicted_nt",
           "celltype_predicted_nt_confidence", "ground_truth"]


def plain(v):
    return None if pd.isna(v) else (round(float(v), 3) if isinstance(v, float) else v)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    types = sys.argv[2:] or TYPES
    ann = pd.read_feather(flat(ANNOTATIONS), columns=["bodyId", "status", "type", "instance"])
    ann = ann[ann.type.isin(types) & (ann.status == "Traced")]
    nt = pd.read_feather(flat(NEUROTRANSMITTERS), columns=["body", *NT_COLS]).set_index("body")
    nt = nt.reindex(ann.bodyId)
    neurons = [{"bodyId": str(b), "type": t, "instance": i, **{c: plain(nt.at[b, c]) for c in NT_COLS}}
               for b, t, i in zip(ann.bodyId, ann.type, ann.instance)]

    ids = pa.array(ann.bodyId.to_numpy(), type=pa.int64())
    w = pf.read_table(flat(WEIGHTS), memory_map=True)
    w = w.filter(pc.and_(pc.is_in(w["body_pre"], value_set=ids), pc.is_in(w["body_post"], value_set=ids))).to_pandas()
    tp = dict(zip(ann.bodyId, ann.type))
    w["pre_type"], w["post_type"] = w.body_pre.map(tp), w.body_post.map(tp)
    by_type = w.groupby(["pre_type", "post_type"]).weight.agg(["sum", "count"]).reset_index()
    type_pairs = [{"pre": a, "post": b, "synapses": int(s), "connections": int(n)}
                  for a, b, s, n in by_type.itertuples(index=False)]
    inst = dict(zip(ann.bodyId, ann.instance))
    pairs = [{"pre": f"{inst[a]} {a}", "post": f"{inst[b]} {b}", "synapses": int(s)}
             for a, b, s in w.sort_values("weight", ascending=False)[["body_pre", "body_post", "weight"]].itertuples(index=False)]

    res = {"source": "MaleCNS v1.0 flat connectome, synapse confidence >= 0.5, Traced neurons",
           "types": types, "neurons": neurons, "type_pairs": type_pairs, "neuron_pairs": pairs}
    if out:
        Path(out).write_text(json.dumps(res, indent=1, allow_nan=False, ensure_ascii=False))
    for n in neurons:
        print(f"{n['instance']:>14} {n['bodyId']:>10}  consensus {n['consensus_nt']}, predicted {n['predicted_nt']} "
              f"({n['predicted_nt_confidence']}), type {n['celltype_predicted_nt']} ({n['celltype_predicted_nt_confidence']}), "
              f"ground truth {n['ground_truth']}")
    for p in type_pairs:
        print(f"{p['pre']:>10} → {p['post']:<10} {p['synapses']:>6} synapses in {p['connections']} connections")


if __name__ == "__main__":
    main()
