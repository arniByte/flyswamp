"""Carry the Shiu et al. 2024 experiment sets from FlyWire 783 to MaleCNS v1.0 by connectivity.

Cell type names do not settle it: sugar and water GRNs are both LB3 in FlyWire, while MaleCNS splits LB3
into LB3a-d. So each neuron is described by whom it talks to: synapse counts to and from partners,
grouped by partner cell type (FlyWire cell_type; MaleCNS flywireType, which names the same types).
Every MaleCNS candidate neuron is compared (cosine) with the mean profile of each FlyWire set and joins
the best set if it is clearly ahead of the runner-up.

usage: python map_flywire_to_malecns.py [out.json]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from common import ANNOTATIONS, WEIGHTS, flat  # noqa: E402

from refs import FLYWIRE_ANNOTATIONS, SHIU_REPO  # noqa: E402

REPO = SHIU_REPO
ANN = FLYWIRE_ANNOTATIONS / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
SETS = ["neu_sugar", "neu_water", "neu_bitter", "neu_ir94e", "ids_mn9"]
MIN_COSINE, MIN_MARGIN = 0.5, 0.05


def profiles(pre, post, w, types, targets):
    """{target: {('out'|'in', partner type): synapses}}; composite types 'A,B' split the weight."""
    prof = {t: defaultdict(float) for t in targets}
    tset = set(targets)
    for a, b, n in zip(pre, post, w):
        for me, other, d in ((a, b, "out"), (b, a, "in")):
            if me in tset:
                ty = types.get(other)
                if isinstance(ty, str) and ty:
                    parts = ty.split(",")
                    for p in parts:
                        prof[me][(d, p.strip())] += n / len(parts)
    return prof


def text(x):
    return x if isinstance(x, str) else None


def unit(p, keys):
    """Profile on the FlyWire keys, normalised by its full norm: synapses onto partner types that FlyWire
    lacks (nerve cord, for instance) count against the match instead of being ignored."""
    v = np.array([p.get(k, 0.0) for k in keys])
    n = np.sqrt(sum(x * x for x in p.values())) if isinstance(p, dict) else np.linalg.norm(v)
    return v / n if n else v


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reference" / "shiu_neuron_sets_malecns.json"
    m783 = json.loads((HERE / "reference" / "shiu_neuron_sets_783.json").read_text())["neurons"]
    sets = json.loads((HERE / "reference" / "shiu_neuron_sets.json").read_text())["sets"]
    fw = {s: [int(m783[i]["root_783"]) for i in sets[s]] for s in SETS}

    ann = pd.read_csv(ANN, sep="\t", usecols=["root_id", "cell_type"], low_memory=False).dropna()
    fw_type = dict(zip(ann.root_id, ann.cell_type))
    con = pd.read_parquet(REPO / "Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    all_fw = {i for v in fw.values() for i in v}
    c = con[con.Presynaptic_ID.isin(all_fw) | con.Postsynaptic_ID.isin(all_fw)]
    fw_prof = profiles(c.Presynaptic_ID.to_numpy(), c.Postsynaptic_ID.to_numpy(), c.Connectivity.to_numpy(float), fw_type, all_fw)

    mc = pd.read_feather(flat(ANNOTATIONS), columns=["bodyId", "status", "type", "flywireType", "class", "rootSide", "instance"])
    mc = mc[mc.status == "Traced"]
    cand = mc[(mc["class"] == "gustatory") | (mc.type == "MN9")]
    mc_type = dict(zip(mc.bodyId, mc.flywireType))
    t = pf.read_table(flat(WEIGHTS), memory_map=True)
    ids = pa.array(cand.bodyId.to_numpy(), type=pa.int64())
    t = t.filter(pc.or_(pc.is_in(t["body_pre"], value_set=ids), pc.is_in(t["body_post"], value_set=ids)))
    mc_prof = profiles(t["body_pre"].to_numpy(), t["body_post"].to_numpy(), t["weight"].to_numpy().astype(float),
                       mc_type, set(cand.bodyId))

    keys = sorted({k for p in fw_prof.values() for k in p})
    centroid = {s: (lambda c: c / np.linalg.norm(c))(sum(unit(fw_prof[i], keys) for i in fw[s])) for s in SETS}
    rows = []
    for _, r in cand.iterrows():
        v = unit(mc_prof[r.bodyId], keys)
        sims = sorted(((float(v @ centroid[s]), s) for s in SETS), reverse=True)
        (c1, s1), (c2, _) = sims[0], sims[1]
        rows.append({"bodyId": int(r.bodyId), "type": text(r.type), "side": text(r.rootSide), "best": s1, "cosine": round(c1, 3),
                     "margin": round(c1 - c2, 3), "assigned": s1 if c1 >= MIN_COSINE and c1 - c2 >= MIN_MARGIN else None})
    df = pd.DataFrame(rows)
    print(pd.crosstab(df.type.fillna("?"), df.assigned.fillna("-")).loc[lambda x: x.drop(columns="-", errors="ignore").sum(axis=1) > 0])
    result = {s: sorted((r for r in rows if r["assigned"] == s), key=lambda r: r["bodyId"]) for s in SETS}
    for s in SETS:
        side = pd.Series([r["side"] for r in result[s]]).value_counts().to_dict()
        print(f"{s}: FlyWire {len(fw[s])} → MaleCNS {len(result[s])} {side}")
    out.write_text(json.dumps({"method": __doc__.strip().splitlines()[0], "min_cosine": MIN_COSINE, "min_margin": MIN_MARGIN,
                               "profile_keys": len(keys), "sets": result}, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
