# Project Map (Monorepo)

This repo is a **monorepo** containing two apps:

- `ai-data-analyst/` — dataset upload + analysis + chat (FastAPI + Next.js)
- `ai-data-validator/` — “data adequacy” validation for AI assistants (FastAPI + Streamlit)

## What is source vs generated vs runtime

### Source (keep in version control)
- Application code under:
  - `ai-data-analyst/backend/app/`
  - `ai-data-analyst/backend/tests/`
  - `ai-data-analyst/frontend/src/`
  - `ai-data-validator/backend/app/`
  - `ai-data-validator/frontend/streamlit_app.py`
- Config/templates:
  - `ai-data-analyst/backend/.env.example`
  - `ai-data-validator/backend/.env.template`
- Dependency manifests:
  - `ai-data-analyst/backend/pyproject.toml`, `ai-data-analyst/backend/requirements.txt`
  - `ai-data-analyst/frontend/package.json`, `ai-data-analyst/frontend/package-lock.json`
  - `ai-data-validator/backend/requirements.txt`

### Generated / vendor (should NOT be committed)
These exist in your current folder and inflate repo size; they should be treated as **local build artifacts**:

- Node:
  - `ai-data-analyst/frontend/node_modules/`
  - `ai-data-analyst/frontend/.next/`
- Python:
  - `ai-data-validator/.venv/` (and/or `*/backend/.venv/`)
  - `**/__pycache__/`
  - `**/.pytest_cache/`
  - `ai-data-analyst/backend/coverage_html/`
  - `ai-data-analyst/backend/.coverage`
- Nested git metadata (should not be inside another repo):
  - `ai-data-analyst/frontend/.git/`

### Runtime data (should NOT be committed)
- `ai-data-analyst/backend/uploads/`
- `ai-data-validator/backend/uploads/`
- Any DB files (SQLite):
  - `*.db`, `*.sqlite`

This repo’s root `.gitignore` is set up to ignore these categories.

---

## `ai-data-analyst/` overview

### Backend (`ai-data-analyst/backend/`)

- Entry point:
  - `ai-data-analyst/backend/app/main.py` — FastAPI app factory, middleware, exception handling, health endpoints.
- API layer:
  - `ai-data-analyst/backend/app/api/routes/` — route modules (datasets, chat, analyses, analytics, connections, auth, ml, data_quality).
  - `ai-data-analyst/backend/app/api/schemas/` — Pydantic request/response schemas.
- Services layer:
  - `ai-data-analyst/backend/app/services/data_ingestion.py` — file parsing + profiling (CSV/Excel/JSON/Parquet).
  - `ai-data-analyst/backend/app/services/llm_service.py` — OpenAI async client wrapper + retries + usage tracking.
  - `ai-data-analyst/backend/app/services/auth_service.py` — JWT + RBAC + API key service (dev-grade in-memory storage).
  - `ai-data-analyst/backend/app/services/database.py` — async SQLAlchemy engine/session management.
  - `ai-data-analyst/backend/app/services/*` — additional services (cache, reporting, connectors).
- Agent layer:
  - `ai-data-analyst/backend/app/agents/` — “ReAct-style” orchestrator + specialist agents.
- Data / ORM:
  - `ai-data-analyst/backend/app/models/database.py` — SQLAlchemy models and enums.
- “ML engines” (large surface area):
  - `ai-data-analyst/backend/app/ml/` — many analysis modules; tests primarily target these engines.
- Tests:
  - `ai-data-analyst/backend/tests/` — large unittest-style suite + synthetic data generator.

### Frontend (`ai-data-analyst/frontend/`)

- Next.js app router:
  - `ai-data-analyst/frontend/src/app/` — routes/pages.
  - `ai-data-analyst/frontend/src/app/app/*` — authenticated-ish “app” shell pages (datasets/analysis/quality/etc).
- API client:
  - `ai-data-analyst/frontend/src/lib/api.js` — fetch wrapper + typed-ish methods for backend endpoints.
- UI components:
  - `ai-data-analyst/frontend/src/components/` — landing + quality UI components.

---

## `ai-data-validator/` overview

### Backend (`ai-data-validator/backend/`)

- Entry point:
  - `ai-data-validator/backend/app/main.py` — FastAPI endpoints for validation sessions and report download.
- Orchestration:
  - `ai-data-validator/backend/app/orchestrator.py` — session state + ReAct-style sequencing:
    - QGen → Ingestion → Quality → Validation/Report
- Agents:
  - `ai-data-validator/backend/app/agents/*` — core agents + scaffolds.
- Utilities:
  - `ai-data-validator/backend/app/utils/*` — doc parsing, chunking, embeddings.
- Config:
  - `ai-data-validator/backend/app/config.py` — dotenv-based settings (uses SQLite by default for session db).

### Frontend (`ai-data-validator/frontend/`)

- Streamlit UI:
  - `ai-data-validator/frontend/streamlit_app.py`

