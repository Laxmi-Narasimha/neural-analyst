#!/usr/bin/env sh
set -eu

_is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if _is_true "${DB_MIGRATE_ON_STARTUP:-}"; then
  echo "[entrypoint] Running DB migrations (alembic upgrade head)..."
  tries="${DB_MIGRATE_MAX_TRIES:-30}"
  sleep_s="${DB_MIGRATE_SLEEP_SECONDS:-2}"

  while :; do
    if alembic -c alembic.ini upgrade head; then
      break
    fi
    tries="$((tries - 1))"
    if [ "${tries}" -le 0 ]; then
      echo "[entrypoint] DB migrations failed after retries."
      exit 1
    fi
    echo "[entrypoint] Waiting for DB... retrying migrations in ${sleep_s}s (${tries} left)"
    sleep "${sleep_s}"
  done
fi

exec "$@"

