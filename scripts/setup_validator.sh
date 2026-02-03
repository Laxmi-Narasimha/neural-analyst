#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.11}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/ai-data-validator/backend"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. See docs/SETUP.md for installing Python 3.11+."
  exit 1
fi

echo "[validator] Creating venv..."
cd "${BACKEND_DIR}"
if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

echo "[validator] Installing backend dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install streamlit

echo "[validator] Done."

