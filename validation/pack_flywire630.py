"""Pack the FlyWire 630 graph shipped with Shiu et al. 2024 into our CSR format, unchanged:
same neuron order as their completeness csv, weight = their "Excitatory x Connectivity" column.
The 783 graph from the same repository packs the same way and also gets a .meta.json with cell type,
side, super class and transmitter from the FlyWire annotations (Schlegel et al. 2024), so analysis
scripts can name its neurons.

usage: python pack_flywire630.py [out_prefix] [630|783]   (default .cache/graphs/flywire630, 630)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from common import CACHE  # noqa: E402
from csr import write_csr  # noqa: E402

from refs import FLYWIRE_ANNOTATIONS, SHIU_REPO  # noqa: E402

REPO = SHIU_REPO
FILES = {"630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
         "783": ("Completeness_783.csv", "Connectivity_783.parquet")}
META = ["cell_type", "side", "super_class", "top_nt"]


def main():
    version = sys.argv[2] if len(sys.argv) > 2 else "630"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE / "graphs" / f"flywire{version}"
    comp, conn = FILES[version]
    ids = pd.read_csv(REPO / comp, index_col=0).index.to_list()
    con = pd.read_parquet(REPO / conn, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    h = write_csr(out, ids, con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values,
                  con["Excitatory x Connectivity"].values,
                  {"source": f"FlyWire {version} as shipped with Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model)",
                   "weight_meaning": "Excitatory x Connectivity: synapse count signed by predicted neurotransmitter"})
    if version == "783":
        ann = pd.read_csv(FLYWIRE_ANNOTATIONS / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv",
                          sep="\t", usecols=["root_id", *META], dtype={"root_id": str}, low_memory=False)
        ann = ann.drop_duplicates("root_id").set_index("root_id").reindex([str(i) for i in ids])
        meta = {k: [v if isinstance(v, str) else None for v in ann[k]] for k in META}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, allow_nan=False))
    print(f"{out}: {h['n']} neurons, {h['nnz']} edges")


if __name__ == "__main__":
    main()
