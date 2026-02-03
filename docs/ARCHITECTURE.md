# Architecture

This repo contains two apps. They can be run independently, but they are conceptually related:

- `ai-data-analyst/` focuses on analysis and dashboards.
- `ai-data-validator/` focuses on “is your data adequate/safe for an AI assistant goal?” validation.

---

## 1) AI Data Analyst (`ai-data-analyst/`)

### High-level flow

```mermaid
flowchart LR
  U["User (Browser)"] --> NX["Next.js UI<br/>ai-data-analyst/frontend/src"]
  NX --> API["API Client<br/>src/lib/api.js"]
  API --> FA["FastAPI<br/>ai-data-analyst/backend/app/main.py"]

  FA --> RT["Routes<br/>app/api/routes/*"]
  RT --> SV["Services<br/>app/services/*"]
  SV --> DB["PostgreSQL (optional but intended)<br/>SQLAlchemy async"]
  SV --> LLM["OpenAI API<br/>app/services/llm_service.py"]
  SV --> PC["Pinecone (optional)<br/>vectors"]
  SV --> RD["Redis/Celery (planned)<br/>TODOs in lifespan"]
  SV --> AG["Agents<br/>app/agents/*"]
  AG --> ML["ML Engines<br/>app/ml/*"]
```

### Request/response lifecycle

1. **UI action** (e.g. upload a dataset) triggers a call in `ai-data-analyst/frontend/src/lib/api.js`.
2. **FastAPI** receives it via route modules in `ai-data-analyst/backend/app/api/routes/*`.
3. Route handlers call:
   - **DB repositories** (SQLAlchemy models in `app/models/database.py`) for persistence, and/or
   - **services** for ingestion/LLM/analysis, and/or
   - **agents** that orchestrate multi-step workflows.
4. Responses follow a consistent Pydantic schema style in `ai-data-analyst/backend/app/api/schemas/*`.
5. Middleware in `ai-data-analyst/backend/app/main.py` injects request IDs and structured logging context.

---

## 2) AI Data Adequacy Validator (`ai-data-validator/`)

### High-level flow

```mermaid
flowchart LR
  U["User (Browser)"] --> ST["Streamlit UI<br/>ai-data-validator/frontend/streamlit_app.py"]
  ST --> FA["FastAPI<br/>ai-data-validator/backend/app/main.py"]

  FA --> ORCH["Orchestrator<br/>backend/app/orchestrator.py"]
  ORCH --> QG["Question Generation Agent<br/>backend/app/agents/qgen.py"]
  ORCH --> ING["Data Ingestion Agent<br/>backend/app/agents/ingestion.py"]
  ORCH --> QA["Quality Analysis Agent<br/>backend/app/agents/quality.py"]
  ORCH --> VR["Validation/Report Agent<br/>backend/app/agents/validation.py"]

  ING --> PARSE["Doc parsing + chunking<br/>backend/app/utils/*"]
  ING --> PC["Pinecone namespaces"]
  ORCH --> LLM["OpenAI API"]
```

### Session lifecycle

1. `POST /api/validate` starts a session.
2. If the user goal is unclear, orchestrator returns **clarifying questions** and a `session_id`.
3. `POST /api/validate/continue` submits answers; orchestrator runs ingestion → quality → report generation.
4. Optionally `POST /api/query` queries a Pinecone namespace.

