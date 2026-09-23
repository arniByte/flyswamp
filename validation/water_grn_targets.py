"""Per-GRN synapses onto the interneurons where the water -> MN9 pathway differs between FlyWire and
MaleCNS (validation/pathway_compare.mjs): is the difference in a few GRNs of the Shiu water set, or in every
water GRN? Also the total input synapses of each MN9. FlyWire 783 connectivity from the Shiu et al.
repository, MaleCNS v1.0 flat connectome; targets are FlyWire cell types, found in MaleCNS through flywireType.

usage: python water_grn_targets.py [out.json] [TYPE ...]
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
from common import ANNOTATIONS, WEIGHTS, flat  # noqa: E402

from refs import FLYWIRE_ANNOTATIONS, SHIU_REPO  # noqa: E402

TARGETS = ["CB0192", "CB0407"]
GRN_TYPES = ["LB2d", "LB3a", "LB3b", "LB3c", "LB3d", "LB4b"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    targets = sys.argv[2:] or TARGETS
    m783 = json.loads((HERE / "reference" / "shiu_neuron_sets_783.json").read_text())["neurons"]
    sets = json.loads((HERE / "reference" / "shiu_neuron_sets.json").read_text())["sets"]
    ann = pd.read_csv(FLYWIRE_ANNOTATIONS / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv",
                      sep="\t", usecols=["root_id", "cell_type", "side"], dtype={"root_id": str}, low_memory=False)
    fw_type = dict(zip(ann.root_id, ann.cell_type))
    con = pd.read_parquet(SHIU_REPO / "Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    con["pre"], con["post"] = con.Presynaptic_ID.astype(str), con.Postsynaptic_ID.astype(str)

    flywire = {}
    for s in ("neu_water", "neu_sugar"):
        grns = [m783[i]["root_783"] for i in sets[s]]
        c = con[con.pre.isin(grns)]
        c = c.assign(t=c.post.map(fw_type))
        rows = []
        for g in grns:
            cg = c[c.pre == g]
            rows.append({"root_id": g, "type": fw_type.get(g), "total_out": int(cg.Connectivity.sum()),
                         **{t: int(cg[cg.t == t].Connectivity.sum()) for t in targets}})
        flywire[s] = rows

    mc = pd.read_feather(flat(ANNOTATIONS), columns=["bodyId", "status", "type", "flywireType", "rootSide", "somaSide"])
    mc = mc[mc.status == "Traced"]
    grn = mc[mc.type.isin(GRN_TYPES)]
    tgt = mc[mc.flywireType.isin(targets)]
    w = pf.read_table(flat(WEIGHTS), memory_map=True)
    w = w.filter(pc.is_in(w["body_pre"], value_set=pa.array(grn.bodyId.to_numpy(), type=pa.int64()))).to_pandas()
    tt = dict(zip(tgt.bodyId, tgt.flywireType))
    w["t"] = w.body_post.map(tt)
    malecns = []
    for b, ty, side in zip(grn.bodyId, grn.type, grn.rootSide):
        wb = w[w.body_pre == b]
        malecns.append({"bodyId": str(b), "type": ty, "side": side if isinstance(side, str) else None,
                        "total_out": int(wb.weight.sum()), **{t: int(wb[wb.t == t].weight.sum()) for t in targets}})

    # total input to each MN9: a motor neuron with far fewer input synapses than its partner cannot answer alike
    mn9_fw = {m783[i]["side"]: m783[i]["root_783"] for i in sets["ids_mn9"]}
    mn9_in = {f"FlyWire {s}": int(con[con.post == r].Connectivity.sum()) for s, r in mn9_fw.items()}
    mn9_mc = mc[mc.type == "MN9"]
    wi = pf.read_table(flat(WEIGHTS), memory_map=True)
    wi = wi.filter(pc.is_in(wi["body_post"], value_set=pa.array(mn9_mc.bodyId.to_numpy(), type=pa.int64()))).to_pandas()
    for b, side in zip(mn9_mc.bodyId, mn9_mc.rootSide.fillna(mn9_mc.somaSide)):
        mn9_in[f"MaleCNS {side}"] = int(wi[wi.body_post == b].weight.sum())

    res = {"targets": targets, "flywire783": flywire, "malecns": malecns,
           "malecns_target_neurons": {t: int((tgt.flywireType == t).sum()) for t in targets}, "mn9_input_synapses": mn9_in}
    if out:
        Path(out).write_text(json.dumps(res, indent=1, allow_nan=False))
    for s, rows in flywire.items():
        df = pd.DataFrame(rows)
        print(f"FlyWire {s}: {len(df)} GRNs; per GRN " + ", ".join(
            f"{t} median {df[t].median():.0f}, >=5 synapses in {(df[t] >= 5).sum()}" for t in targets))
    print("MN9 input synapses:", mn9_in)
    df = pd.DataFrame(malecns)
    for ty, g in df.groupby("type"):
        print(f"MaleCNS {ty}: {len(g)} GRNs; per GRN " + ", ".join(
            f"{t} median {g[t].median():.0f}, >=5 synapses in {(g[t] >= 5).sum()}" for t in targets))


if __name__ == "__main__":
    main()
