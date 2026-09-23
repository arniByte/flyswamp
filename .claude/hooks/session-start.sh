#!/bin/bash
# Claude Code on the web: make tests and the data pipeline runnable at session start.
# Heavy, on-demand setup (reference repos, Brian2 env, 1.1 GB MaleCNS data) stays in scripts/setup.sh.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
(cd web && npm install --no-audit --no-fund)
python3 -m pip install -q -r pipeline/requirements.txt
