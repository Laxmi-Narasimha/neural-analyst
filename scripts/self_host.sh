#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_ENV="$ROOT/ai-data-analyst/backend/.env"
BACKEND_EXAMPLE="$ROOT/ai-data-analyst/backend/.env.example"

CELERY=0
MINIO=0
DETACH=0

for arg in "$@"; do
  case "$arg" in
    --celery) CELERY=1 ;;
    --minio) MINIO=1 ;;
    -d|--detach) DETACH=1 ;;
  esac
done

if [[ ! -f "$BACKEND_ENV" && -f "$BACKEND_EXAMPLE" ]]; then
  cp "$BACKEND_EXAMPLE" "$BACKEND_ENV"
  echo "Created $BACKEND_ENV from .env.example — set LLM_MODEL and your API key."
fi

cd "$ROOT"

FILES=(-f docker-compose.yml)
if [[ "$CELERY" -eq 1 ]]; then
  FILES+=(-f docker-compose.celery.yml)
fi
if [[ "$MINIO" -eq 1 ]]; then
  FILES+=(-f docker-compose.minio.yml)
fi

echo ""
echo "Neural Analyst self-host"
echo "  UI:  http://localhost:3000"
echo "  API: http://localhost:8000/docs"
echo "  BYOK setup: http://localhost:3000/setup-keys"
echo "  Keys/compute: edit ai-data-analyst/backend/.env (runs on YOUR RAM/CPU)"
echo ""

CMD=(docker compose "${FILES[@]}" up --build)
if [[ "$DETACH" -eq 1 ]]; then
  CMD+=(-d)
fi

exec "${CMD[@]}"