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

## Self-host architecture (BYOK + your compute)

Neural Analyst is designed for **self-host first**:

| Layer | Where it runs | What you provide |
|-------|---------------|------------------|
| Web UI | Your browser → your Docker/local frontend | Nothing extra |
| API + compute | Your container or `uvicorn` process | RAM/CPU on your machine or cloud VM |
| LLM chat | LiteLLM → your provider | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or free local **Ollama** |
| ML / AutoML | sklearn in-process or Celery worker | Same machine RAM (scales via reservoir sampling) |
| Large files | DuckDB pushdown on CSV/Parquet | Disk only — no full pandas load |

**Web UI vs CLI:** Use the web UI for upload, chat, and artifacts. Docker (`scripts/self_host.ps1`) is the easiest one-command setup. CLI is only needed for power users running `uvicorn`/`celery` directly.

Open **http://localhost:3000/setup-keys** after starting for BYOK checklist and live deployment status (`GET /api/v1/system/info`).

## Test datasets (download and try)

| Dataset | Size | Good for | Download |
|---------|------|----------|----------|
| Titanic | ~900 rows | Classification, missingness, ML baseline | [Kaggle Titanic](https://www.kaggle.com/c/titanic/data) |
| House Prices | ~1.5k rows | Regression, feature importance | [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) |
| Iris | 150 rows | Quick smoke test | [UCI Iris](https://archive.ics.uci.edu/ml/datasets/iris) |
| Wine Quality | ~6.5k rows | Regression, correlations | [UCI Wine](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| NYC TLC trips | Millions | DuckDB full-count, large-file sampling | [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| US government open data | Varies | Real-world EDA | [data.gov](https://data.gov/) |
| Dataset search | Any | Finding niche domains | [Google Dataset Search](https://datasetsearch.research.google.com/) |

**Suggested first test flow:**
1. Upload Titanic CSV → ask "show missing values" (Talk-to-Data, no LLM key needed)
2. Ask "train a baseline model for survival" → model comparison table appears in chat
3. Ask "analyze sentiment" on a reviews CSV, or "forecast next 8 months" on a time-series CSV
4. For million-row Parquet: ask "exact row count without loading" → DuckDB pushdown
