"""Extract the olfactory learning circuit from MaleCNS v1.0.

Output: web/public/data/circuit.json

Neuron groups (global index = group offset + local index):
  ORN   sampled olfactory receptor neurons, for display only; activity = input of their glomerulus
  PN    all antennal lobe projection neurons (class ALPN)
  KC    all Kenyon cells
  MBON  all mushroom body output neurons
  PAM   dopaminergic neurons, PAM cluster  (reward teaching signal)
  PPL1  dopaminergic neurons, PPL1 cluster (punishment teaching signal)
  APL   the two anterior paired lateral neurons (feedback inhibition onto KCs)
  DN    a few descending neurons, for display only (motor readout)

Every weight below is a synapse count taken from the connectome.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from common import ANNOTATIONS, ATTRIBUTION, NEUROTRANSMITTERS, WEB_DATA, WEIGHTS, flat

# Minimum synapse counts per connection. Low counts are mostly noise; see README.
MIN_W = {
    "glom_pn": 5,   # summed over all ORNs of a glomerulus
    "pn_kc": 3,
    "kc_mbon": 2,
    "kc_apl": 3,
    "apl_kc": 3,
    "dan_mbon": 3,
    "mbon_dan": 3,
}
ORN_PER_GLOM_SIDE = 4           # ORNs kept for display per glomerulus and side
DAN_GATE_MIN = 20               # MBONs with less DAN input get valence 0
DN_TYPES = ["DNa01", "DNa02", "DNp01", "MDN", "DNp09"]


def main() -> None:
    ann = pd.read_feather(flat(ANNOTATIONS))
    nt = pd.read_feather(flat(NEUROTRANSMITTERS)).set_index("body")
    typ = ann["type"].fillna("")
    cls = ann["class"].fillna("")

    orn_all = ann[(cls == "olfactory") & typ.str.startswith("ORN_")].copy()
    orn_all["glom"] = orn_all["type"].str[4:]
    # ORN somata sit in the antenna, outside the imaged CNS; the side is in the instance name.
    orn_all["somaSide"] = orn_all["instance"].str.extract(r"_([LR])$", expand=False)
    glomeruli = sorted(orn_all["glom"].unique())
    gidx = {g: i for i, g in enumerate(glomeruli)}

    sel = {
        "PN": ann[cls == "ALPN"],
        "KC": ann[cls == "Kenyon_Cell"],
        "MBON": ann[cls == "MBON"],
        "PAM": ann[typ.str.match(r"^PAM\d")],
        "PPL1": ann[typ.str.match(r"^PPL1\d")],
        "APL": ann[typ == "APL"],
        "DN": ann[typ.isin(DN_TYPES)],
    }
    for k in sel:
        sel[k] = sel[k].sort_values(["type", "bodyId"]).reset_index(drop=True)

    ids_needed = np.concatenate([orn_all.bodyId.values] + [d.bodyId.values for d in sel.values()])
    edges = load_edges(ids_needed)

    # ---- ORN glomerulus -> PN (aggregate ORN axons per glomerulus) ----
    orn_glom = dict(zip(orn_all.bodyId, orn_all.glom))
    pn_local = local_index(sel["PN"])
    e = edges[edges.body_pre.isin(orn_glom.keys()) & edges.body_post.isin(pn_local.keys())].copy()
    e["g"] = e.body_pre.map(orn_glom).map(gidx)
    e["p"] = e.body_post.map(pn_local)
    glom_pn = e.groupby(["g", "p"]).weight.sum().reset_index()
    glom_pn = glom_pn[glom_pn.weight >= MIN_W["glom_pn"]]

    # ---- sample ORNs for display: strongest PN drivers per glomerulus & side ----
    out_w = e.groupby("body_pre").weight.sum()
    orn_all["w_out"] = orn_all.bodyId.map(out_w).fillna(0)
    orn_disp = (
        orn_all[(orn_all.w_out > 0) & orn_all.somaSide.notna()]
        .sort_values("w_out", ascending=False)
        .groupby(["glom", "somaSide"])
        .head(ORN_PER_GLOM_SIDE)
        .sort_values(["glom", "bodyId"])
        .reset_index(drop=True)
    )

    groups = ["ORN", "PN", "KC", "MBON", "PAM", "PPL1", "APL", "DN"]
    frames = {"ORN": orn_disp, **sel}
    offsets, off = {}, 0
    for gname in groups:
        offsets[gname] = off
        off += len(frames[gname])

    loc = {g: local_index(frames[g]) for g in groups}
    dan = pd.concat([sel["PAM"], sel["PPL1"]]).reset_index(drop=True)
    dan_local = local_index(dan)  # PAM first, then PPL1

    def pair(pre: dict, post: dict, key: str) -> pd.DataFrame:
        x = edges[edges.body_pre.isin(pre.keys()) & edges.body_post.isin(post.keys())]
        x = x[x.weight >= MIN_W[key]]
        return pd.DataFrame({"i": x.body_pre.map(pre).values, "j": x.body_post.map(post).values, "w": x.weight.values})

    E = {
        "glom_pn": pd.DataFrame({"i": glom_pn.g.values, "j": glom_pn.p.values, "w": glom_pn.weight.values}),
        "pn_kc": pair(loc["PN"], loc["KC"], "pn_kc"),
        "kc_mbon": pair(loc["KC"], loc["MBON"], "kc_mbon"),
        "kc_apl": pair(loc["KC"], loc["APL"], "kc_apl"),
        "apl_kc": pair(loc["APL"], loc["KC"], "apl_kc"),
        "dan_mbon": pair(dan_local, loc["MBON"], "dan_mbon"),
        "mbon_dan": pair(loc["MBON"], dan_local, "mbon_dan"),
    }

    # ---- MBON valence from which DAN cluster innervates its compartment ----
    n_pam = len(sel["PAM"])
    dm = E["dan_mbon"]
    pam_in = np.bincount(dm.j[dm.i < n_pam], weights=dm.w[dm.i < n_pam], minlength=len(sel["MBON"]))
    ppl_in = np.bincount(dm.j[dm.i >= n_pam], weights=dm.w[dm.i >= n_pam], minlength=len(sel["MBON"]))
    tot = pam_in + ppl_in
    # Aso et al. 2014 (eLife 3:e04580): MBONs in PAM compartments promote avoidance,
    # MBONs in PPL1 compartments promote approach. Sign and magnitude come from the wiring.
    valence = np.where(tot >= DAN_GATE_MIN, (ppl_in - pam_in) / np.maximum(tot, 1), 0.0)

    # ---- assemble ----
    all_df = pd.concat([frames[g].assign(_g=gi) for gi, g in enumerate(groups)]).reset_index(drop=True)
    types = sorted(all_df["type"].fillna("?").unique())
    tidx = {t: i for i, t in enumerate(types)}
    side_code = {"L": 0, "R": 1}
    nts = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "histamine", "unclear"]
    nt_col = all_df.bodyId.map(nt["consensus_nt"]).fillna("unclear")
    nt_col = nt_col.where(nt_col.isin(nts), "unclear")

    out = {
        "meta": {
            "source": "MaleCNS v1.0 flat connectome (minconf 0.5)",
            "attribution": ATTRIBUTION,
            "license": "CC-BY-4.0",
            "generated": date.today().isoformat(),
            "min_weights": MIN_W,
            "notes": "Weights are synapse counts. Edges are [pre_local, post_local, weight] triples, flattened.",
        },
        "glomeruli": glomeruli,
        "groups": [{"name": g, "offset": offsets[g], "count": len(frames[g])} for g in groups],
        "types": types,
        "nts": nts,
        "neurons": {
            "bodyId": [int(b) for b in all_df.bodyId],
            "type": [tidx[t] for t in all_df["type"].fillna("?")],
            "instance": [str(s) for s in all_df["instance"].fillna("")],
            "side": [side_code.get(s, 2) for s in all_df.somaSide.fillna("")],
            "nt": [nts.index(n) for n in nt_col],
        },
        "orn_glom": [gidx[g] for g in orn_disp.glom],
        "mbon": {
            "valence": [round(float(v), 4) for v in valence],
            "pam_in": [int(v) for v in pam_in],
            "ppl1_in": [int(v) for v in ppl_in],
        },
        "dan": {"n_pam": n_pam, "n_ppl1": len(sel["PPL1"])},
        "edges": {k: flatten(v) for k, v in E.items()},
    }

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA / "circuit.json"
    path.write_text(json.dumps(out, separators=(",", ":")))

    # ---- report ----
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    for g in groups:
        print(f"  {g:5s} {len(frames[g]):5d} neurons")
    for k, v in E.items():
        print(f"  {k:9s} {len(v):6d} edges {int(v.w.sum()):8d} synapses")
    mb = sel["MBON"].assign(valence=valence, pam=pam_in, ppl1=ppl_in)
    print(mb.groupby("type")[["valence", "pam", "ppl1"]].mean().round(2).to_string())


def load_edges(ids: np.ndarray) -> pd.DataFrame:
    t = feather.read_table(flat(WEIGHTS))
    vs = pa.array(np.unique(ids))
    m = pc.and_(pc.is_in(t["body_pre"], value_set=vs), pc.is_in(t["body_post"], value_set=vs))
    return t.filter(m).to_pandas()


def local_index(df: pd.DataFrame) -> dict:
    return {int(b): i for i, b in enumerate(df.bodyId)}


def flatten(df: pd.DataFrame) -> list:
    df = df.sort_values(["i", "j"])
    return np.stack([df.i.values, df.j.values, df.w.values], axis=1).astype(int).ravel().tolist()


if __name__ == "__main__":
    main()
