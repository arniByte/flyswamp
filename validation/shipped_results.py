"""Convert the spike tables the authors ship in the Shiu et al. 2024 repo (results/example/*.parquet)
into our per-trial count schema, so they can be compared with compare.mjs like any other run.

usage: SHIU_REPO=... python shipped_results.py <name> <poisson_hz> <out.json>

sugarR.parquet was run at 200 Hz, not the 150 Hz default in model.py: its sugar GRNs fire at ~197 Hz, which
is what 200 Hz Poisson input gives after the events lost in spike steps, and our engine at 200 Hz matches
it neuron by neuron (validation/results/lif_vs_shipped_sugarR.json). sugarR_100Hz.parquet is 100 Hz.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from shiu_reference import MN9, SUGAR

REPO = Path(os.environ["SHIU_REPO"])


def main():
    name, hz, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    df = pd.read_parquet(REPO / "results" / "example" / f"{name}.parquet")
    n_trials = int(df.trial.max()) + 1
    per = df.groupby(["flywire_id", "trial"]).size().unstack(fill_value=0).reindex(columns=range(n_trials), fill_value=0)
    counts = {str(i): row.astype(int).tolist() for i, row in per.iterrows()}
    meta = {"source": f"authors' shipped results, results/example/{name}.parquet", "dataset": "FlyWire 630 (repo files)",
            "experiment": name, "stimulated": [str(f) for f in SUGAR], "poisson_rate_hz": hz, "t_run_s": 1.0,
            "n_trials": n_trials, "mn9": str(MN9)}
    Path(out).write_text(json.dumps({"meta": meta, "counts": counts}))
    print(f"{name}: {len(counts)} active neurons, {n_trials} trials")


if __name__ == "__main__":
    main()
