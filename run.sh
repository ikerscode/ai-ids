#!/usr/bin/env bash
# Start the ai-ids HUD/API on 127.0.0.1:8105 (creates the venv on first run).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m uvicorn aiids.api:app --host 127.0.0.1 --port "${PORT:-8105}" --log-level warning
