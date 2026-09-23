"""Find the FlyWire 783 neuron for each neuron in the Shiu et al. 2024 experiment sets (FlyWire 630 ids),
so the sets can be typed with the FlyWire annotations (Schlegel et al. 2024) and carried to MaleCNS.

Most of them keep their root id in 783 and map to themselves. The few that were re-proofread (new root id)
are matched by connectivity: about 106k neurons keep their id across versions and serve as anchors, a
neuron is described by its synapse counts to and from anchors, and its 783 match is the changed-id
candidate with the most similar description (cosine). A match is kept only if it is clearly better than
the runner-up and no other target claims the same neuron.

usage: SHIU_REPO=... FLYWIRE_ANNOTATIONS=... python map_630_to_783.py [out.json]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(os.environ["SHIU_REPO"])
ANN = Path(os.environ["FLYWIRE_ANNOTATIONS"]) / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
HERE = Path(__file__).resolve().parent
MIN_COSINE, MIN_MARGIN = 0.5, 0.1


def load(ids_csv, con_parquet):
    ids = pd.read_csv(REPO / ids_csv, index_col=0).index.to_numpy()
    con = pd.read_parquet(REPO / con_parquet, columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
    return ids, con


def features(ids, con, anchor_col, rows=None):
    """Row per neuron: synapse counts to anchors (outputs) then from anchors (inputs)."""
    n_anchor = int(anchor_col.max()) + 1
    col = np.full(len(ids), -1)
    col[anchor_col >= 0] = anchor_col[anchor_col >= 0]
    pre, post, w = con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy(), con["Connectivity"].to_numpy(np.float32)
    out_ok, in_ok = col[post] >= 0, col[pre] >= 0
    r = np.concatenate([pre[out_ok], post[in_ok]])
    c = np.concatenate([col[post[out_ok]], n_anchor + col[pre[in_ok]]])
    m = sp.csr_matrix((np.concatenate([w[out_ok], w[in_ok]]), (r, c)), shape=(len(ids), 2 * n_anchor))
    if rows is not None:
        m = m[rows]
    norm = np.sqrt(m.multiply(m).sum(axis=1)).A1
    norm[norm == 0] = 1
    return sp.diags(1 / norm) @ m


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reference" / "shiu_neuron_sets_783.json"
    sets = json.loads((HERE / "reference" / "shiu_neuron_sets.json").read_text())["sets"]
    ids630, con630 = load("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet")
    ids783, con783 = load("Completeness_783.csv", "Connectivity_783.parquet")
    anchors = np.intersect1d(ids630, ids783)
    a_index = {a: k for k, a in enumerate(anchors)}
    col630 = np.array([a_index.get(i, -1) for i in ids630])
    col783 = np.array([a_index.get(i, -1) for i in ids783])
    kept_id = set(ids783.tolist())
    targets = sorted({int(i) for v in sets.values() for i in v} - kept_id)
    same = sorted({int(i) for v in sets.values() for i in v} & kept_id)
    pos630 = {f: k for k, f in enumerate(ids630)}
    rows = [pos630[t] for t in targets]
    f630 = features(ids630, con630, col630, rows)
    candidates = np.flatnonzero(col783 < 0)  # neurons whose root id changed, as the targets' did
    f783 = features(ids783, con783, col783)[candidates]
    sim = (f630 @ f783.T).toarray()

    order = np.argsort(-sim, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows_idx = np.arange(len(targets))
    cos, margin = sim[rows_idx, best], sim[rows_idx, best] - sim[rows_idx, second]
    match = ids783[candidates[best]]
    dup = pd.Series(match).duplicated(keep=False).to_numpy()

    ann = pd.read_csv(ANN, sep="\t", usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"], low_memory=False).set_index("root_id")
    mapping = {str(t): {"root_783": str(t), "cosine": None, "margin": None, "how": "same root id",
                        **{k: (None if pd.isna(v) else v) for k, v in (ann.loc[t].to_dict() if t in ann.index else {}).items()}}
               for t in same}
    for t, m, c, g, d in zip(targets, match, cos, margin, dup):
        ok = c >= MIN_COSINE and g >= MIN_MARGIN and not d
        a = ann.loc[m].to_dict() if ok and m in ann.index else {}
        mapping[str(t)] = {"root_783": str(m) if ok else None, "cosine": round(float(c), 3), "margin": round(float(g), 3),
                           "how": "connectivity" if ok else "unmatched",
                           **{k: (None if pd.isna(v) else v) for k, v in a.items()}}
    summary = {}
    for name, v in sets.items():
        kept = [mapping[i] for i in v if mapping[i]["root_783"]]
        summary[name] = {"n": len(v), "matched": len(kept),
                         "types": pd.Series([f"{k.get('cell_class')}/{k.get('cell_type')}" for k in kept]).value_counts().to_dict()}
        print(name, summary[name]["matched"], "/", len(v), summary[name]["types"])
    out.write_text(json.dumps({"method": __doc__.strip().splitlines()[0], "min_cosine": MIN_COSINE, "min_margin": MIN_MARGIN,
                               "anchors": int(len(anchors)), "summary": summary, "neurons": mapping}, indent=1))


if __name__ == "__main__":
    main()
