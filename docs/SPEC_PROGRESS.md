# Spec Progress Tracker (Talk to Your Data)

Last updated: 2026-02-10

Source of truth specs:
- `spec_review/TALK_TO_YOUR_DATA_BACKLOG.md`
- `spec_review/TALK_TO_YOUR_DATA_FEATURE_SPEC.md`
- `spec_review/TALK_TO_YOUR_DATA_ENGINEERING_SPEC.md`
- `spec_review/TALK_TO_YOUR_DATA_SECURITY_EVAL_SPEC.md`

Legend:
- DONE: implemented + exercised via tests or direct flow
- PARTIAL: implemented, but missing some acceptance criteria / UX polish
- TODO: not implemented

## P0 (truthful end-to-end demo)

P0.1 Centralize dataset loading by dataset_id
- Status: DONE
- Key files:
  - `ai-data-analyst/backend/app/services/dataset_loader.py`
  - Used in: chat/analyses/analytics/datasets routes

P0.2 Replace placeholder orchestrator tools with compute-backed implementations
- Status: DONE
- Notes:
  - `app/agents/orchestrator.py` tools now call the safe compute layer and return artifact references (no fabricated values).
  - Runtime flows (chat + analyses) remain compute-backed without requiring orchestrator, but the orchestrator is now safe to wire in later.

P0.3 Safe compute layer (operator allow-list)
- Status: DONE
- Key files:
  - `ai-data-analyst/backend/app/compute/executor.py`
  - `ai-data-analyst/backend/app/compute/operators/eda.py`
  - `ai-data-analyst/backend/app/compute/plans.py`

P0.4 Artifact persistence + provenance
- Status: DONE
- Key files:
  - `ai-data-analyst/backend/app/compute/artifacts.py`
  - `ai-data-analyst/backend/app/services/artifact_index.py`
  - `ai-data-analyst/backend/app/api/routes/artifacts.py`

P0.5 Grounded chat (plan -> compute -> answer) + clarification loop
- Status: DONE (P0 scope)
- Notes:
  - Chat answers are compute-grounded when dataset_id is present (no numeric claims without evidence artifacts).
  - Minimal clarification loop exists for common ambiguity (group_by and time_column); broader ambiguity handling is tracked as P1 UX polish.
- Key files:
  - `ai-data-analyst/backend/app/api/routes/chat.py`
  - `ai-data-analyst/backend/app/api/schemas/base.py` (clarification field)

P0.6 Data Speaks workflow (API + UI)
- Status: DONE (MVP)
- Key files:
  - Backend: `ai-data-analyst/backend/app/api/routes/data_speaks.py`, `ai-data-analyst/backend/app/api/routes/analyses.py`
  - Worker/service: `ai-data-analyst/backend/app/services/analysis_execution.py`
  - Frontend: `ai-data-analyst/frontend/src/app/app/data-speaks/page.js`

P0.6a Evidence panels (UI organization of computed artifacts)
- Status: DONE (MVP)
- Notes:
  - Data Speaks includes a tabbed evidence view (Overview/Schema/Quality/Outliers/Relationships/Segments/Time/Privacy & Risk/All) built from computed steps.
  - PII-safe defaults: preview rows mask PII columns; schema snapshot strips example values for PII columns.
- Key files:
  - `ai-data-analyst/frontend/src/app/app/data-speaks/page.js`
  - `ai-data-analyst/frontend/src/app/app/data-speaks/page.module.css`
  - `ai-data-analyst/backend/app/compute/operators/eda.py`

P0.6b Suggested next actions + action feed (session-level)
- Status: DONE (MVP)
- Notes:
  - Backend generates deterministic `suggested_actions` from computed evidence.
  - UI renders "Suggested next actions" and a persisted action feed with re-run support.
  - Actions run as child analyses and are linked back to the parent session (feed updated on completion).
- Key files:
  - Backend: `ai-data-analyst/backend/app/api/routes/analyses.py`
  - Backend: `ai-data-analyst/backend/app/services/analysis_execution.py`
  - Backend: `ai-data-analyst/backend/app/services/insight_extraction.py`
  - Frontend: `ai-data-analyst/frontend/src/app/app/data-speaks/page.js`

P0.6c Drilldown operators (initial pack)
- Status: DONE (MVP)
- Notes:
  - Added bounded drilldown operators: missingness patterns, outlier explain, segment deep dive, privacy/risk scan, relationship explain, time anomaly/change-point scan.
  - Next step: add more per-panel "next actions" and deeper drilldowns where needed (e.g., driver analysis).
- Key files:
  - `ai-data-analyst/backend/app/compute/operators/eda.py`

P0.7 Stable identity end-to-end (owner_id everywhere)
- Status: DONE
- Key files:
  - `ai-data-analyst/backend/app/api/routes/auth.py`
  - `ai-data-analyst/backend/app/services/auth_service.py`

P0.8 Quality/Adequacy runs on dataset_id and persists
- Status: DONE (MVP)
- Notes:
  - Session state persisted in DB.
  - On completion, dataset + dataset version quality fields updated and artifacts generated.
- Key files:
  - `ai-data-analyst/backend/app/api/routes/data_quality.py`
  - `ai-data-analyst/frontend/src/components/quality/QualityDashboard.tsx`

P0.9 Analytics endpoints use real datasets + return artifacts
- Status: DONE (MVP)
- Notes:
  - Dataset-backed loads + dataset_version included.
  - Report/table/metric artifacts generated where applicable.
- Key files:
  - `ai-data-analyst/backend/app/api/routes/analytics.py`

P0.10 Minimal regression tests (grounding + core flows)
- Status: DONE
- Key files:
  - `ai-data-analyst/backend/tests/test_p0_integration_grounding.py`

## P1+ (next, recommended)

- Job queue + workers for compute at scale (Celery): DONE (Celery tasks + `JOB_EXECUTOR=celery`, `docker-compose.celery.yml`, requirements include celery/redis)
- Operator/artifact cache (reuse evidence by dataset_version + operator + params): DONE (artifact index stores `operator_params_hash`; compute executor reuses prior artifacts when safe; includes runtime meta in cache key to avoid incorrect reuse)
- Report sharing (read-only links for report artifacts): DONE (token-based share links + public report viewer route)
- Storage abstraction for uploads/artifacts (local -> S3): DONE (ObjectStore supports local + S3 URIs, local caching for compute, pre-signed artifact downloads, cache pruning, dataset purge/delete semantics, and orphan cleanup for both local + S3 via `/maintenance/storage-gc`)
- Dataset purge (delete stored blobs + hard-delete metadata): DONE (`POST /datasets/{dataset_id}/purge` + background job + worker support)
- Alembic migrations as prod source-of-truth: PARTIAL (migrations exist; Docker/Render runs `alembic upgrade head` on startup via `DB_MIGRATE_ON_STARTUP=true`; still need to tighten prod defaults and document upgrade workflow clearly)
- CI pipeline (tests + frontend build): DONE (`.github/workflows/ci.yml`)
- Observability (metrics + tracing): DONE (metrics now include operator + HTTP route + job runtime distributions; readiness includes object-store health checks)
- Transformation builder UX (guided steps, validation, versioning): DONE (UI; backend already supported)
- Dataset SQL + table artifact paging resiliency: DONE (DuckDB-first execution with automatic SQLite fallback for OSS/dev environments where DuckDB is unavailable)
- Transformation operator coverage (basic analyst workflows): DONE for current P1 scope (added row filters, sorting, row limits, time-feature derivation, and numeric binning with provenance metrics)
- Transformation plan assistant (deterministic): DONE (`POST /datasets/{dataset_id}/transform/suggest` generates explainable no-code cleaning steps and is wired into dataset transform UI via "Suggest plan")
- Transform workflow robustness (dataset UI): DONE for basic layer (SQL artifact table pagination wired via `/artifacts/{id}/rows`, and transform apply now tracks background job state to completion/failure directly in dataset view)
- Additional transformation operators (basic cleaning coverage): DONE for current scope (`clip_outliers`, `encode_categorical` added in backend + builder UI)

## P2 (started)

- Task inference + target selection: PARTIAL (deterministic target candidate scoring + leakage warnings + split strategy are available via `POST /ml/task-inference`; UI confirmation flow and training orchestration still pending)
