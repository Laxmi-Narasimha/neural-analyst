# Setup

This repo currently requires:

- Python 3.11+
- Node 20.9+ (the Next.js app uses `.nvmrc`)

Optional (recommended for production-like runs):

- Postgres (primary DB target)
- Redis (for distributed rate limiting + background jobs when enabled)

## Windows (PowerShell)

### 1) Verify toolchain

```powershell
python --version
node --version
npm --version
```

### 2) Run the Analyst app (FastAPI + Next.js)

Backend:

```powershell
cd neural-analyst\ai-data-analyst\backend
if (!(Test-Path .\.venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Use SQLite for local dev (no Postgres required)
$env:DATABASE_URL="sqlite+aiosqlite:///./dev.db"

# Optional knobs
# - Allow up to ~1GB uploads (default is 1024MB)
$env:MAX_UPLOAD_SIZE_MB="1024"
# - Premium narrative rewrite (guarded; requires OPENAI_API_KEY)
# $env:NARRATOR_MODE="llm"
# $env:OPENAI_API_KEY="sk-..."

.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Frontend (new terminal):

```powershell
cd neural-analyst\ai-data-analyst\frontend
npm ci
$env:NEXT_PUBLIC_API_URL="http://localhost:8000/api/v1"
npm run dev
```

Open: `http://localhost:3000`

## macOS / Linux

Backend:

```bash
cd neural-analyst/ai-data-analyst/backend
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
export DATABASE_URL="sqlite+aiosqlite:///./dev.db"
export MAX_UPLOAD_SIZE_MB="1024"
# Optional: premium narrative rewrite (guarded; requires OPENAI_API_KEY)
# export NARRATOR_MODE="llm"
# export OPENAI_API_KEY="sk-..."
./.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd neural-analyst/ai-data-analyst/frontend
npm ci
export NEXT_PUBLIC_API_URL="http://localhost:8000/api/v1"
npm run dev
```

## Production Notes (high level)

- Prefer Postgres: `DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db"`
- Do not use `create_all`/auto-create in production; use migrations (Alembic) once wired.
- Run background jobs out-of-process (Celery/Redis) for long dataset processing and heavy compute.
- For multi-instance deployments, use an object store for uploads/artifacts:
  - set `OBJECT_STORE_BACKEND=s3`
  - set `OBJECT_STORE_S3_BUCKET=...` (and optionally `OBJECT_STORE_S3_ENDPOINT_URL=...` for MinIO)
  - keep `OBJECT_STORE_CACHE_DIR` on fast local disk for compute downloads

## Migrations (Alembic)

The backend includes Alembic scaffolding under `ai-data-analyst/backend/migrations`.

Run migrations:

```powershell
cd neural-analyst\ai-data-analyst\backend
$env:DATABASE_URL="sqlite+aiosqlite:///./dev.db"
.\.venv\Scripts\alembic.exe -c alembic.ini upgrade head
```

Create a new migration (after changing models):

```powershell
.\.venv\Scripts\alembic.exe -c alembic.ini revision --autogenerate -m "describe change"
```

## Distributed jobs (Celery)

By default, the API runs background work in-process (`JOB_EXECUTOR=local`).

To run distributed jobs (recommended for production-like scale):

1) Start Redis (or `docker compose up -d` from repo root).

2) Run a Celery worker (Windows note: use `--pool=solo`):

```powershell
cd neural-analyst\ai-data-analyst\backend
$env:JOB_EXECUTOR="celery"
.\.venv\Scripts\celery.exe -A app.workers.celery_app.celery_app worker -l info --pool=solo
```

## Optional: Full stack via Docker (recommended easiest self-host)

If you have Docker running locally:

```bash
cd neural-analyst
docker compose up --build
```

Notes:
- The backend container runs Alembic migrations on startup (`DB_MIGRATE_ON_STARTUP=true` in `docker-compose.yml`).
- `DB_AUTO_CREATE_TABLES=true` is kept for dev convenience; for production, prefer migrations and disable auto-create.

Then open:
- UI: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`

Convenience scripts:
- Windows: `scripts/self_host.ps1` (add `-Celery` for a worker, `-Minio` for S3-compatible storage via MinIO, `-Detach` to run in background)
- macOS/Linux: `scripts/self_host.sh` (add `--celery`, `--minio`, `--detach`)

### Optional: run out-of-process jobs (Celery worker)

```bash
cd neural-analyst
docker compose -f docker-compose.yml -f docker-compose.celery.yml up --build
```

### Optional: S3-compatible storage via MinIO (self-host)

This repo includes a ready-to-run MinIO override file for testing the S3 object-store backend locally.

Run (local jobs):

```bash
cd neural-analyst
docker compose -f docker-compose.yml -f docker-compose.minio.yml up --build
```

Run (Celery jobs + MinIO):

```bash
cd neural-analyst
docker compose -f docker-compose.yml -f docker-compose.celery.yml -f docker-compose.minio.yml up --build
```

MinIO endpoints:
- S3 endpoint: `http://localhost:9000`
- Console UI: `http://localhost:9001` (login: `minioadmin` / `minioadmin`)
