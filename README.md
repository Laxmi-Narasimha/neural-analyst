# neural-analyst (monorepo)

This repo contains two related apps:

- `ai-data-analyst/`: FastAPI backend + Next.js frontend for dataset upload, "Data Speaks" (safe operators), and chat.
- `ai-data-validator/`: FastAPI backend + Streamlit UI for "data adequacy" validation (uploads + Pinecone/OpenAI checks).

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

## Docs

- `docs/PROJECT_MAP.md`: folder map.
- `docs/ARCHITECTURE.md`: request flow for both apps.
- `docs/INDUSTRY_GRADE_AUDIT.md`: severity-ranked gaps + fixes/backlog.
- `docs/GITHUB_WORKFLOW.md`: how to commit/push/PR safely (with explanations).
- `docs/ROADMAP.md`: build sequence aligned to `spec_review/*`.
- `docs/SETUP.md`: setup runbook (Windows + macOS).
- `spec_review/TALK_TO_YOUR_DATA_MASTER_SPEC.md`: spec pack entry point (product + engineering + security).
