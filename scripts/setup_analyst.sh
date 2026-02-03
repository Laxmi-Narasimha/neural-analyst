#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.11}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/ai-data-analyst/backend"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. See docs/SETUP.md for installing Python 3.11+."
  exit 1
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PY_VER}" != "3.11" && "${PY_VER}" != "3.12" ]]; then
  echo "ERROR: ai-data-analyst requires Python 3.11+ (found ${PY_VER}). See docs/SETUP.md."
  exit 1
fi

echo "[analyst] Creating venv..."
cd "${BACKEND_DIR}"
if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

echo "[analyst] Installing backend dependencies (this is a large install)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "[analyst] Done."

