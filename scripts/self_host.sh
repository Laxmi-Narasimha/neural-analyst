#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

cd "$ROOT"

FILES=(-f docker-compose.yml)
if [[ "$CELERY" -eq 1 ]]; then
  FILES+=(-f docker-compose.celery.yml)
fi
if [[ "$MINIO" -eq 1 ]]; then
  FILES+=(-f docker-compose.minio.yml)
fi

CMD=(docker compose "${FILES[@]}" up --build)
if [[ "$DETACH" -eq 1 ]]; then
  CMD+=(-d)
fi

exec "${CMD[@]}"
