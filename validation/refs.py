"""Where the reference repositories live. scripts/setup.sh clones them into .cache/ref; the SHIU_REPO and
FLYWIRE_ANNOTATIONS environment variables override the defaults."""
import os
from pathlib import Path

REF = Path(os.environ.get("FLYSWAMP_CACHE", Path(__file__).resolve().parent.parent / ".cache")) / "ref"
SHIU_REPO = Path(os.environ.get("SHIU_REPO", REF / "Drosophila_brain_model"))
FLYWIRE_ANNOTATIONS = Path(os.environ.get("FLYWIRE_ANNOTATIONS", REF / "flywire_annotations"))
