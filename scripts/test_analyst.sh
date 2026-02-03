#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.11}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/ai-data-analyst/backend"

if [[ ! -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  echo "ERROR: Analyst backend venv missing. Run: make setup-analyst"
  exit 1
fi

cd "${BACKEND_DIR}"
./.venv/bin/python -m pytest

