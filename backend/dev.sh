#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! .venv/bin/python -c "import PIL" >/dev/null 2>&1; then
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m uvicorn main:app --reload --port 8143
