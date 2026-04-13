#!/usr/bin/env bash
set -euo pipefail

uv run python -m apps.inference_api.main &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}

trap cleanup EXIT

exec uv run streamlit run apps/dashboard/app.py \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.address "${HOST:-0.0.0.0}"
