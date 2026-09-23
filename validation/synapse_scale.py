"""How many more input synapses a neuron has in MaleCNS v1.0 than the same cell type in FlyWire 783.

The Shiu et al. 2024 weight per synapse (w_syn = 0.275 mV) was fitted on FlyWire counts. If MaleCNS
reports systematically more synapses per connection (different detection, more complete proofreading),
the same w_syn drives every neuron harder. The ratio of median input synapses per neuron, over cell
types matched by name (FlyWire cell_type = MaleCNS flywireType), gives the rescaling without fitting.

usage: python synapse_scale.py [out.json]
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

REPO = SHIU_REPO
ANN = FLYWIRE_ANNOTATIONS / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
MIN_INPUT = 50  # ignore types whose neurons have almost no inputs in either dataset


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "p2_synapse_scale.json"
    ann = pd.read_csv(ANN, sep="\t", usecols=["root_id", "cell_type"], low_memory=False).dropna()
    con = pd.read_parquet(REPO / "Connectivity_783.parquet", columns=["Postsynaptic_ID", "Connectivity"])
    fw_in = con.groupby("Postsynaptic_ID").Connectivity.sum()
    fw = ann.assign(inp=ann.root_id.map(fw_in)).dropna().groupby("cell_type").inp.median()

    mc = pd.read_feather(flat(ANNOTATIONS), columns=["bodyId", "status", "flywireType"])
    traced = mc[mc.status == "Traced"]
    t = pf.read_table(flat(WEIGHTS), memory_map=True)
    t = t.filter(pc.is_in(t["body_pre"], value_set=pa.array(traced.bodyId.to_numpy(), type=pa.int64())))
    agg = t.group_by("body_post").aggregate([("weight", "sum")]).to_pandas()
    mc_in = agg.set_index("body_post")["weight_sum"]
    typed = traced[traced.flywireType.notna() & ~traced.flywireType.str.contains(",")]
    mcm = typed.assign(inp=typed.bodyId.map(mc_in)).dropna().groupby("flywireType").inp.median()

    j = pd.concat([fw.rename("flywire"), mcm.rename("malecns")], axis=1).dropna()
    j = j[(j.flywire >= MIN_INPUT) & (j.malecns >= MIN_INPUT)]
    r = j.malecns / j.flywire
    res = {"types": int(len(j)), "median_ratio": float(r.median()), "iqr": [float(r.quantile(0.25)), float(r.quantile(0.75))],
           "w_syn_flywire": 0.275, "w_syn_malecns": 0.275 / float(r.median())}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
