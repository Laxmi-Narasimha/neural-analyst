#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/ai-data-validator/backend"
FRONTEND_DIR="${ROOT_DIR}/ai-data-validator/frontend"

if [[ ! -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  echo "ERROR: Validator backend venv missing. Run: make setup-validator"
  exit 1
fi

cleanup() {
  echo ""
  echo "Shutting down validator services..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/2] Starting FastAPI backend (ai-data-validator)..."
cd "${BACKEND_DIR}"
./.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

echo "[2/2] Starting Streamlit frontend..."
cd "${FRONTEND_DIR}"
../backend/.venv/bin/python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost

