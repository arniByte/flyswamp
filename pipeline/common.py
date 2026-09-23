"""Shared paths and download helpers for the flyswamp data pipeline.

All raw data comes from the public MaleCNS v1.0 release (CC-BY 4.0):
https://male-cns.janelia.org/download/
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("FLYSWAMP_CACHE", ROOT / ".cache"))
WEB_DATA = ROOT / "web" / "public" / "data"

GCS = "https://storage.googleapis.com/flyem-male-cns"
FLAT = f"{GCS}/v1.0/connectome-data/flat-connectome"

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

ATTRIBUTION = (
    "MaleCNS v1.0 connectome (FlyEM/HHMI Janelia, University of Cambridge, "
    "MRC LMB, Google Research), CC-BY 4.0, https://male-cns.janelia.org/"
)


def fetch(url: str, dest: Path) -> Path:
    """Download url to dest once; later calls reuse the cached file."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def flat(name: str) -> Path:
    return fetch(f"{FLAT}/{name}", CACHE / "malecns" / name)
