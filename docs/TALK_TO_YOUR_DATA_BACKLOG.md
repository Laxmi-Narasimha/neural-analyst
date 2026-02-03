# Neural Analyst — “Talk to Your Data” Implementation Backlog (P0/P1/P2)

**Date:** 2026‑02‑03  
**Purpose:** Decision-complete backlog of features and engineering work required to make the product behave like a real AI Data Analyst / AI Data Scientist assistant.  
**Rule:** No code snippets. This document is descriptive and references modules/files for implementation mapping.

This backlog is intentionally written to be “buildable”: each item states why it matters, what to build, where it likely lives in the current repo, and how to know it’s done.

Related docs:
- `docs/TALK_TO_YOUR_DATA_MASTER_SPEC.md` (vision + success criteria)
- `docs/TALK_TO_YOUR_DATA_FEATURE_SPEC.md` (UX flows)
- `docs/TALK_TO_YOUR_DATA_ENGINEERING_SPEC.md` (architecture)
- `docs/TALK_TO_YOUR_DATA_SECURITY_EVAL_SPEC.md` (safety + evaluation)

---

## 0) Definitions (how to read this backlog)

- **P0**: must fix to deliver a truthful “Talk to your data / Make the data speak” demo that works on real uploaded datasets.
- **P1**: should fix to make the product delightful and robust for real usage.
- **P2**: advanced features that move the product from analyst-assistant to data-scientist-assistant.
- **Acceptance criteria**: observable behaviors (API + UI) that prove the task is complete.

---

## 1) P0 — Make “Talk to your data” truthful end-to-end

### P0.1 Centralize dataset loading by `dataset_id` (single source of truth)

**Why it matters**
- Today, multiple routes either don’t load data at all (chat) or use mock/demo loaders (analytics).
- Without a single dataset loader and a consistent dataset registry, grounding is impossible.

**What to build**
- A dataset registry/loader module that:
  - resolves `dataset_id` to a dataset asset (file/blob reference) and version
  - provides safe preview and compute access (through the compute engine, not direct raw reads)
  - is used by chat, analyses, analytics endpoints, quality, and reporting

**Where it maps today**
- Dataset upload/process: `ai-data-analyst/backend/app/api/routes/datasets.py`
- Data ingestion/profile: `ai-data-analyst/backend/app/services/data_ingestion.py`
- Places currently missing loader integration:
  - chat: `ai-data-analyst/backend/app/api/routes/chat.py`
  - analytics demo loader: `ai-data-analyst/backend/app/api/routes/analytics.py`
  - orchestrator placeholders: `ai-data-analyst/backend/app/agents/orchestrator.py`

**Acceptance criteria**
- Given a dataset uploaded via the API, every endpoint that takes `dataset_id` actually uses that dataset’s contents.
- No route generates mock data when a dataset_id is provided.
- A single “dataset not found” behavior exists (consistent error shape).

---

### P0.2 Replace placeholder orchestrator tools with real compute-backed implementations

**Why it matters**
- The orchestrator is currently a “shape-only” stub returning fake results, which undermines trust.

**What to build**
- Tool implementations that call the safe compute engine and return artifact references, not fabricated values.
- A strict tool layer:
  - tools create artifacts
  - tools return artifact summaries and IDs
  - the narrator uses these artifacts only

**Where it maps today**
- Orchestrator tool stubs: `ai-data-analyst/backend/app/agents/orchestrator.py`
- Existing engines to wrap into tools/operators:
  - profiling: `ai-data-analyst/backend/app/ml/data_profiling.py`
  - quality: `ai-data-analyst/backend/app/ml/data_quality.py`
  - anomaly: `ai-data-analyst/backend/app/ml/anomaly_detection.py`
  - correlation: `ai-data-analyst/backend/app/ml/correlation_analysis.py`

**Acceptance criteria**
- Orchestrator summary/statistics/quality calls run against real dataset content.
- Orchestrator outputs include provenance: dataset_version + operator/tool name + parameters.
- No “demo” numbers appear in responses when dataset is missing; instead, return a clear error or ask for dataset selection.

---

### P0.3 Implement the safe compute layer (operator allow-list)

**Why it matters**
- “Dynamic code generation” must not mean executing arbitrary Python.
- A safe compute layer is the foundation for trust, performance, and scalability.

**What to build**
- An operator registry with:
  - validated parameters
  - row scan limits and timeouts
  - deterministic sampling behavior
- A first set of operators needed for Data Speaks:
  - schema/profile
  - missingness/uniqueness
  - top categories
  - numeric summaries and distributions
  - correlation/association
  - outlier scan
  - time trend summary when time exists

**Where it maps today**
- Many candidate implementations exist under `ai-data-analyst/backend/app/ml/*`, but need a unified interface and artifact outputs.
- Query-related scaffolding: `ai-data-analyst/backend/app/services/query_optimizer.py`, `ai-data-analyst/backend/app/ml/nl_query.py`

**Acceptance criteria**
- Every numeric output shown to the user comes from a compute operator artifact.
- Operators reject unsafe or unbounded operations (no full-table dumps to UI).
- Compute works on file-backed datasets (large data support) and clearly labels sampling.

---

### P0.4 Add artifact persistence and provenance (tables/charts/reports as first-class objects)

**Why it matters**
- Without artifacts, the system cannot be reproducible or provably grounded.
- The UI needs stable “things” to render and share.

**What to build**
- An Artifact model (and related storage) that supports:
  - metric/table/chart/report artifacts
  - preview payloads
  - provenance (dataset_version, operator, parameters)
  - parent-child artifact lineage

**Where it maps today**
- DB and models: `ai-data-analyst/backend/app/models/*`
- Analysis scaffolding: `ai-data-analyst/backend/app/api/routes/analyses.py`
- Report generator: `ai-data-analyst/backend/app/services/report_generator.py`

**Acceptance criteria**
- Every Data Speaks run produces a stable set of artifacts that can be reloaded.
- Chat answers reference artifact IDs; artifacts can be fetched independently.
- Reports can be generated from a session by selecting artifacts.

---

### P0.5 Wire grounded chat: chat must plan → compute → answer (not free-text only)

**Why it matters**
- The current chat route is “LLM-only”; it can hallucinate and won’t reflect dataset reality.

**What to build**
- A chat orchestrator pipeline:
  - intent classification (data question vs general advice)
  - plan generation (structured steps)
  - compute execution (operators)
  - narration from artifacts
- A clarification loop:
  - when target/time/metric is ambiguous, ask minimal questions with selectable options

**Where it maps today**
- Chat route: `ai-data-analyst/backend/app/api/routes/chat.py`
- Agent context scaffolding exists in analyses route: `ai-data-analyst/backend/app/api/routes/analyses.py`
- Orchestrator agent exists but must be made compute-backed: `ai-data-analyst/backend/app/agents/orchestrator.py`

**Acceptance criteria**
- Asking “How many rows?” returns a computed metric artifact and a short explanation.
- Asking “Top 10 categories by count” returns a computed table artifact.
- The assistant refuses to provide numeric claims without compute evidence.

---

### P0.6 Build the “Data Speaks” workflow (API + UI)

**Why it matters**
- This is the flagship “mesmerizing” button that differentiates the product.

**What to build**
- Backend:
  - start Data Speaks session
  - run the pipeline (profile → roles → quality → insights → narrative)
  - persist artifacts and session summary
- Frontend:
  - a Data Speaks screen with evidence panels and suggested actions
  - an execution log feed showing what ran

**Where it maps today**
- Frontend app pages live under: `ai-data-analyst/frontend/src/app/app/*`
- Backend should add a new route group for sessions (or analyses):
  - session lifecycle should be separate from “analysis jobs” unless they are unified intentionally

**Acceptance criteria**
- A user can select a dataset and click “Make the data speak”.
- The UI shows a narrative and multiple evidence panels populated from computed artifacts.
- Suggested prompts/actions are dataset-specific (reference inferred roles/columns).

---

### P0.7 Fix user identity end-to-end (stable user_id, datasets and conversations persist)

**Why it matters**
- Today, random user IDs are generated per request in multiple places; users will “lose” their datasets and chats.

**What to build**
- A stable identity model:
  - real user table
  - auth tokens/session
  - consistent owner_id usage for dataset, conversation, artifacts
- Ensure every route uses the authenticated user identity.

**Where it maps today**
- Demo user_id generation happens in:
  - `ai-data-analyst/backend/app/api/routes/datasets.py`
  - `ai-data-analyst/backend/app/api/routes/chat.py`
  - `ai-data-analyst/backend/app/api/routes/analyses.py`
- Auth and permission scaffolding exists:
  - `ai-data-analyst/backend/app/api/routes/auth.py`
  - `ai-data-analyst/backend/app/services/auth_service.py`

**Acceptance criteria**
- Upload a dataset, then list datasets: the uploaded dataset appears for the same user.
- Create a conversation, reload page, conversation history persists for the same user.
- Unauthorized users cannot access other users’ datasets.

---

### P0.8 Integrate “Quality / Adequacy” with real dataset objects (not file paths)

**Why it matters**
- The current quality flow passes `uploads/<filename>` strings without actually uploading/attaching those files in the quality request.
- The backend stores sessions in-memory, which breaks on restart and doesn’t scale.

**What to build**
- Quality should run on `dataset_id` and dataset_version.
- Persist sessions/results in DB (or Redis) instead of an in-memory dict.
- Unify validator logic with the analyst app:
  - either embed validator capabilities in the analyst backend
  - or keep validator as a separate service but share storage and dataset IDs

**Where it maps today**
- Quality UI: `ai-data-analyst/frontend/src/components/quality/QualityDashboard.tsx`
- Quality endpoints: `ai-data-analyst/backend/app/api/routes/data_quality.py`
- Validator app exists separately:
  - `ai-data-validator/backend/app/*`
  - `ai-data-validator/frontend/streamlit_app.py`

**Acceptance criteria**
- Running a quality check uses a dataset_id and returns artifacts tied to that dataset version.
- A quality session can be resumed after server restart.
- The readiness badge on dataset detail reflects the latest quality run.

---

### P0.9 Replace analytics endpoints’ mock dataset generator

**Why it matters**
- Advanced analytics endpoints currently ignore real datasets; this is a trust-breaker.

**What to build**
- Remove the demo `get_dataset()` generator and replace with dataset loader usage.
- Ensure every analytics endpoint:
  - validates required columns exist
  - returns artifacts and provenance

**Where it maps today**
- Demo generator is in: `ai-data-analyst/backend/app/api/routes/analytics.py`
- Engines are under: `ai-data-analyst/backend/app/ml/*`

**Acceptance criteria**
- Analytics endpoints return results computed from the real dataset.
- If required columns are missing, the endpoint returns a clear, user-safe error.

---

### P0.10 Minimal evaluation and regression tests (grounding + core flows)

**Why it matters**
- Without regression tests, the product will drift back into “demo mode”.

**What to build**
- A small test dataset suite (public, non-sensitive).
- Tests for:
  - upload → profile
  - Data Speaks session creation
  - one or two grounded chat queries
  - an injection-string dataset case (safety)

**Where it maps today**
- Backend tests likely under `ai-data-analyst/backend` (use existing pytest setup if present).

**Acceptance criteria**
- A CI-style run can validate core grounding behaviors quickly.

---

## 2) P1 — Make it delightful, reproducible, and usable for real analysts

### P1.1 Dataset transformations with preview and versioning
Build a transformation pipeline that:
- creates new dataset versions
- previews diffs and distribution changes
- persists transformation provenance

### P1.2 Insight library and ranking
Implement:
- a candidate insight generator set
- ranking and deduplication
- an “insight library” UI on the Data Speaks screen

### P1.3 Reporting and exports
Implement:
- report builder from artifacts
- export formats (markdown/HTML at minimum)
- shareable read-only report links (optional for OSS)

### P1.4 Streaming UX (progress and partial outputs)
Implement:
- progress events for Data Speaks runs
- partial artifact rendering as they become available

### P1.5 Connections hardening
Implement:
- secret storage and rotation
- read-only query enforcement
- query row limits and timeouts

### P1.6 Operability and observability
Add:
- structured logs with consistent correlation ids
- health/readiness endpoints that reflect DB/queue/storage status
- basic metrics (latency, error rates, job timings)

---

## 3) P2 — Become a real AI Data Scientist assistant (modeling workflows)

### P2.1 Task inference and target selection
Implement:
- deterministic target candidate scoring
- clarification UI to confirm target
- leakage warnings

### P2.2 Baseline training runs with correct evaluation
Implement:
- split strategies (random vs time split)
- metrics by task type
- per-segment evaluation

### P2.3 Explainability and diagnostics
Implement:
- global feature importance
- partial dependence or similar (optional)
- residual diagnostics for regression

### P2.4 Model registry and prediction workflows
Implement:
- persisted model runs
- prediction endpoints with schema checks
- prediction UI and exports

### P2.5 AutoML (opt-in, guarded)
Implement:
- guarded AutoML pipelines
- compute budgeting and user confirmation
- caching and reuse for repeated runs

---

## 4) P3 — Production-grade reliability (for SaaS-scale, optional for OSS)

### P3.1 Job queue + workers
Add:
- a durable queue (Redis-based or similar)
- worker processes
- retry and dead-letter behaviors

### P3.2 Storage abstraction
Add:
- object storage interface for uploads/artifacts
- migration from local disk to object storage

### P3.3 Multi-tenant controls and quotas
Add:
- workspace scoping
- per-workspace quotas
- audit logs

### P3.4 SLOs and monitoring
Define:
- time-to-first-evidence SLO
- job latency SLO
- error budget policies

