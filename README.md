# neural-analyst (monorepo)

This repo is a **single product** (`ai-data-analyst/`) with optional legacy reference code.

- `ai-data-analyst/`: FastAPI + Next.js — Talk to Your Data, Data Speaks, Data Adequacy (merged validator), subscriptions (SaaS), self-host Docker.
- `ai-data-validator/`: **Deprecated** — merged into `ai-data-analyst` at `/app/quality`. See `ai-data-validator/DEPRECATED.md`.

## Local Dev (Windows)

Prereqs:

- Python 3.11+
- Node 20.9+ (see `.nvmrc`)

### 1) Backend (FastAPI)

```powershell
cd ai-data-analyst\backend
Copy-Item .env.example .env  # optional (local defaults work without it)
$env:DATABASE_URL="sqlite+aiosqlite:///./dev.db"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### 2) Frontend (Next.js)

```powershell
cd ai-data-analyst\frontend
npm ci
$env:NEXT_PUBLIC_API_URL="http://localhost:8000/api/v1"
npm run dev
```

App UI: `http://localhost:3000`

### Optional: One-command scripts

- `scripts/setup_analyst.ps1` (installs deps)
- `scripts/dev_analyst.ps1` (starts backend + frontend)

### Optional: Docker (one command self-host)

If you have Docker Desktop running:

```powershell
cd neural-analyst
docker compose up --build
```

- The backend runs Alembic migrations on startup (idempotent).
- UI: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`

To run background jobs out-of-process (Celery worker + Redis), use:

```powershell
cd neural-analyst
docker compose -f docker-compose.yml -f docker-compose.celery.yml up --build
```

Convenience:
- `scripts/self_host.ps1` (PowerShell) supports `-Celery`, `-Minio`, and `-Detach`.

Object storage (optional):
- Set `OBJECT_STORE_BACKEND=s3` + `OBJECT_STORE_S3_BUCKET=...` to store uploads/artifacts in S3-compatible storage (recommended for multi-instance deployments).
- For local S3-compatible testing, use MinIO:
  - `docker compose -f docker-compose.yml -f docker-compose.minio.yml up --build`
  - (Celery + MinIO) `docker compose -f docker-compose.yml -f docker-compose.celery.yml -f docker-compose.minio.yml up --build`

## Docs

- `docs/PROJECT_MAP.md`: folder map.
- `docs/ARCHITECTURE.md`: request flow for both apps.
- `docs/INDUSTRY_GRADE_AUDIT.md`: severity-ranked gaps + fixes/backlog.
- `docs/GITHUB_WORKFLOW.md`: how to commit/push/PR safely (with explanations).
- `docs/ROADMAP.md`: build sequence aligned to `spec_review/*`.
- `docs/SETUP.md`: setup runbook (Windows + macOS).
- `docs/DEPLOY_SAAS.md`: Cloudflare Pages + Render free-tier SaaS deployment.
- `docs/LEVELS_AND_SEQUENCE.md`: "low/medium/expert/core" capability ladder + sequential build rule.
- `spec_review/TALK_TO_YOUR_DATA_MASTER_SPEC.md`: spec pack entry point (product + engineering + security).
