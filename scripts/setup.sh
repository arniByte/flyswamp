#!/bin/bash
# Restore what git does not hold: reference repositories, the Brian2 reference environment, MaleCNS raw
# data (~1.1 GB) and the packed connectome graphs. Idempotent: every step skips work already done.
# usage: scripts/setup.sh [refs] [venv] [data] [graphs]      (no arguments = all four)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${FLYSWAMP_CACHE:-$ROOT/.cache}"
REF="$CACHE/ref"
STEPS="${*:-refs venv data graphs}"
has() { [[ " $STEPS " == *" $1 "* ]]; }

# name  url  commit this project was validated against
REPOS=(
  "Drosophila_brain_model https://github.com/philshiu/Drosophila_brain_model 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
  "flywire_annotations https://github.com/flyconnectome/flywire_annotations 8587524c1748ce5ef2080822a2fc890fc03bf597"
  "fly-brain https://github.com/Lulzx/fly-brain 36397dccbf2818e37c027c4b577a56a3cbe18d70"
  "DoOR.data https://github.com/ropensci/DoOR.data db323a496577c4b4a72b5c2fcd1859e07521ffb5"
  "DoOR.functions https://github.com/ropensci/DoOR.functions 15e415e4d84dfbbba6febefdfd0f2c1ddcf31cd6"
)

if has refs; then
  mkdir -p "$REF"
  for entry in "${REPOS[@]}"; do
    read -r name url commit <<<"$entry"
    dir="$REF/$name"
    if [ ! -d "$dir/.git" ]; then
      echo "clone $name"
      GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "$url" "$dir"
    fi
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
      git -C "$dir" fetch -q --depth 1 origin "$commit" && git -C "$dir" checkout -q "$commit" \
        || echo "warning: $name is not at the validated commit $commit"
    fi
  done
fi

if has venv; then
  if [ ! -x "$ROOT/.venv-ref/bin/python" ]; then
    echo "create .venv-ref (Brian2 reference)"
    python3 -m venv "$ROOT/.venv-ref"
  fi
  "$ROOT/.venv-ref/bin/pip" install -q -r "$ROOT/validation/requirements-ref.txt"
fi

if has data; then
  (cd "$ROOT/pipeline" && python3 -c "
from common import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS, flat
for name in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS):
    print(flat(name))
")
fi

if has graphs; then
  [ -f "$CACHE/graphs/flywire630.bin" ] || "$ROOT/.venv-ref/bin/python" "$ROOT/validation/pack_flywire630.py"
  [ -f "$CACHE/graphs/malecns_min1.bin" ] || (cd "$ROOT/pipeline" && python3 build_graph.py --min-syn 1 --out "$CACHE/graphs/malecns_min1")
fi

echo "setup done: $STEPS"
