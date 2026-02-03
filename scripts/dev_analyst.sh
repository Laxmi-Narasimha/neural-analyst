#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/ai-data-analyst/backend"
FRONTEND_DIR="${ROOT_DIR}/ai-data-analyst/frontend"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node not found. See docs/SETUP.md (Next.js requires Node >= 20.9.0)."
  exit 1
fi

NODE_VER_RAW="$(node -v | sed 's/^v//')"
NODE_MAJOR="$(echo "${NODE_VER_RAW}" | cut -d. -f1)"
NODE_MINOR="$(echo "${NODE_VER_RAW}" | cut -d. -f2)"

if [[ "${NODE_MAJOR}" -lt 20 ]] || ([[ "${NODE_MAJOR}" -eq 20 ]] && [[ "${NODE_MINOR}" -lt 9 ]]); then
  echo "ERROR: Node ${NODE_VER_RAW} is too old. Next.js in this repo requires >= 20.9.0."
  exit 1
fi

if [[ ! -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  echo "ERROR: Analyst backend venv missing. Run: make setup-analyst"
  exit 1
fi

cleanup() {
  echo ""
  echo "Shutting down analyst services..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/2] Starting FastAPI backend (ai-data-analyst)..."
cd "${BACKEND_DIR}"
./.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

echo "[2/2] Starting Next.js frontend..."
cd "${FRONTEND_DIR}"
npm install
npm run dev

