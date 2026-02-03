# “Industry-Grade” Audit (current state)

This audit is based on what exists on disk in `/Users/laxmi/Downloads/data_analyst` on **2026-02-03**.

## Scorecard (0–5)

- **Repo hygiene:** 1/5
  - Large generated/vendor artifacts are present in-tree (notably `ai-data-analyst/frontend/node_modules/`, `ai-data-analyst/frontend/.next/`, `ai-data-validator/.venv/`, coverage artifacts).
  - Nested git repo exists at `ai-data-analyst/frontend/.git/` (not suitable for a monorepo).
- **Runtime reproducibility:** 2/5
  - Some pinning exists (Python requirements, `package-lock.json`), but toolchain versions were not enforced at the repo root before.
  - Frontend requires **Node >= 20.9.0** (from `ai-data-analyst/frontend/node_modules/next/package.json` engines) while the plan initially assumed 18+.
- **Security:** 2/5
  - `ai-data-analyst` has RBAC concepts and JWT scaffolding, but the implementation had correctness issues (refresh flow, secret handling, per-request service instantiation).
  - CORS and upload limits exist, but there are still TODOs and demo-grade defaults.
- **Correctness:** 2/5
  - API routing and prefixes were inconsistent (double-prefix bugs were possible; frontend default API base didn’t match backend).
  - DB health check used a SQLAlchemy 2.0-incompatible execute pattern.
  - Startup previously failed hard if Postgres wasn’t running/configured.
- **Testing:** 3/5
  - `ai-data-analyst/backend/tests/` is extensive (mostly engine-level tests), but runtime dependency weight and environment requirements make it hard to run on a clean machine.
  - No CI config present in this workspace.
- **Operability:** 3/5
  - Structured logging + request IDs exist; `/health` + `/ready` endpoints exist.
  - Redis/Celery are referenced but not wired (TODOs), and there is no metrics/tracing integration.

---

## P0 blockers (must fix to run reliably)

1. **Toolchain mismatch**
   - Analyst backend requires Python 3.11+ but your machine currently has `python3` 3.9.6.
   - Analyst frontend requires Node >= 20.9.0; your machine currently has no Node installed.
2. **Frontend ↔ backend base URL mismatch**
   - Frontend client defaulted to `http://localhost:8000/api` but backend routes are under `/api/v1/*`.
3. **API routing inconsistencies**
   - Some route modules had their own `prefix=...` while the route aggregator also applied prefixes (risking double-prefix endpoints).
4. **DB health check / startup robustness**
   - Health check used a non-`text()` query string; and app startup previously failed hard when DB was unavailable, preventing `/health` from coming up.
5. **Auth correctness**
   - Token refresh path was logically broken; and the dependency provider returned a new auth service per request (losing in-memory users/sessions).

---

## P1 hardening (should fix before real users)

- **Persistence & migrations**
  - Auth and API keys should be backed by the database (not in-memory).
  - Add Alembic migrations (README references migrations, but no `alembic.ini`/migration tree exists).
- **Security defaults**
  - Require a strong `SECRET_KEY` in non-dev environments; tighten CORS; add CSRF strategy if cookies are used.
- **Upload and parsing hardening**
  - Add explicit content-type sniffing, antivirus hooks (if required), and sandboxing for unsafe document parsing.
- **Dependency footprint**
  - Consider splitting `ai-data-analyst/backend/requirements.txt` into a minimal “core” set + optional heavy ML extras.
- **CI**
  - Add a simple CI workflow (lint + unit tests) and a release checklist.

---

## P2 improvements (scale/perf/maintainability)

- Replace in-memory rate limiting with Redis-backed rate limiting.
- Add background job execution (Celery/Redis) for long analyses.
- Add tracing (OpenTelemetry) and metrics (Prometheus) hooks.
- Formalize API contracts (OpenAPI linting, typed client generation).
- Add end-to-end tests (Playwright) for the Next.js UI.

