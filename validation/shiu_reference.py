"""Ground truth for our spiking engine: the published Shiu et al. 2024 Brian2 model, run unmodified.

Uses model.py from https://github.com/philshiu/Drosophila_brain_model (cloned, path via SHIU_REPO)
on its own FlyWire 630 connectivity, and writes per-trial spike counts for every neuron that fired, so the JS engine can be compared
neuron by neuron and against the reference's own trial-to-trial noise.

usage: python shiu_reference.py <experiment> <n_trials> <n_proc> <out.json> [poisson_hz]
experiments: sugar (the 21 sugar GRNs from the repo's example notebook; Poisson rate defaults to the model's 150 Hz)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from refs import SHIU_REPO

REPO = SHIU_REPO
sys.path.insert(0, str(REPO))
from model import default_params, run_trial  # noqa: E402
from brian2 import Hz  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

# From the repo's example.ipynb ("neu_sugar"), FlyWire 630 root ids
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
         720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
         720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265
EXPERIMENTS = {"sugar": SUGAR}

def main():
    exp, n_trials, n_proc, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    params = dict(default_params)
    if len(sys.argv) > 5:
        params["r_poi"] = float(sys.argv[5]) * Hz
    comp = REPO / "2023_03_23_completeness_630_final.csv"
    con = REPO / "2023_03_23_connectivity_630_final.parquet"
    ids = pd.read_csv(comp, index_col=0).index.to_list()
    flyid2i = {f: i for i, f in enumerate(ids)}
    exc = [flyid2i[f] for f in EXPERIMENTS[exp]]
    t0 = time.time()
    res = Parallel(n_jobs=n_proc)(delayed(run_trial)(exc, [], [], comp, con, params) for _ in range(n_trials))
    elapsed = time.time() - t0
    t_run = float(params["t_run"])
    counts = {}
    for trial, spk in enumerate(res):
        for i, times in spk.items():
            counts.setdefault(int(i), np.zeros(n_trials))[trial] = len(times)
    counts = {str(ids[i]): c.astype(int).tolist() for i, c in counts.items()}
    meta = {"source": "Shiu et al. 2024 model.py, unmodified", "dataset": "FlyWire 630 (repo files)", "experiment": exp,
            "stimulated": [str(f) for f in EXPERIMENTS[exp]], "poisson_rate_hz": float(params["r_poi"]), "t_run_s": t_run,
            "n_trials": n_trials, "dt_s": 1e-4, "elapsed_s": elapsed, "mn9": str(MN9)}
    Path(out).write_text(json.dumps({"meta": meta, "counts": counts}))
    mn9 = np.array(counts.get(str(MN9), [0] * n_trials)) / t_run
    print(f"{exp}: {len(counts)} active neurons, MN9 {mn9.mean():.1f} ± {mn9.std():.1f} Hz, {elapsed:.0f}s")

if __name__ == "__main__":
    main()
