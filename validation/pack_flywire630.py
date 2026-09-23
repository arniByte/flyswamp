"""Pack the FlyWire 630 graph shipped with Shiu et al. 2024 into our CSR format, unchanged:
same neuron order as their completeness csv, weight = their "Excitatory x Connectivity" column.

usage: python pack_flywire630.py [out_prefix]   (default .cache/graphs/flywire630)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from common import CACHE  # noqa: E402
from csr import write_csr  # noqa: E402

from refs import SHIU_REPO  # noqa: E402

REPO = SHIU_REPO


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE / "graphs" / "flywire630"
    ids = pd.read_csv(REPO / "2023_03_23_completeness_630_final.csv", index_col=0).index.to_list()
    con = pd.read_parquet(REPO / "2023_03_23_connectivity_630_final.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    h = write_csr(out, ids, con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values,
                  con["Excitatory x Connectivity"].values,
                  {"source": "FlyWire 630 as shipped with Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model)",
                   "weight_meaning": "Excitatory x Connectivity: synapse count signed by predicted neurotransmitter"})
    print(f"{out}: {h['n']} neurons, {h['nnz']} edges")


if __name__ == "__main__":
    main()
