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

## Optional: Postgres + Redis via Docker

If you have Docker running locally:

```bash
cd neural-analyst
docker compose up -d
```

