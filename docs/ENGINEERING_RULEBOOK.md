# Engineering Rulebook (Neural Analyst)

This project aims to behave like a *trustworthy* AI data analyst/data scientist. The way we achieve that is not by "better prompts" but by building a system where:
- all claims are grounded in computed artifacts derived from the user’s dataset/version
- all computation happens through a safe operator layer (allow-list + resource bounds)
- every result is reproducible (dataset_id + dataset_version + operator_name + params)
- the UI is an evidence browser first, and a chat UI second

This document is the operating system for how we build and ship changes without regressions.

---

## 1) Product Non-Negotiables

1. Evidence-first
   - No fabricated values, no “best guess” numbers.
   - The system can be uncertain, but it cannot be wrong confidently.
2. Reproducibility
   - Every analysis must be replayable via: dataset_id, dataset_version, plan, operators, params.
3. Safety by default
   - Only allow-listed operations; never execute arbitrary code on user data.
   - SQL is read-only and bounded; transformations are validated and versioned.
4. Scales to "real datasets"
   - Target: up to ~1GB uploads.
   - Use sampling/out-of-core patterns when needed and clearly disclose sampling.
5. Enterprise-grade reliability
   - Strict input validation, structured errors, timeouts, cancellation, idempotency.
   - Observability: logs, metrics, tracing hooks (even for OSS).
6. Cost & token discipline (LLM is optional)
   - Core compute never depends on an LLM. LLMs may improve wording, not truth.
   - Pass *evidence summaries* to LLMs (never raw rows by default; never secrets).
   - Keep prompts small: only the needed columns, metrics, and citations for the current question.
   - Prefer caching and reuse (dataset_version + operator/params hash) over repeated calls.
   - Always provide a deterministic fallback when LLM is off/unavailable.
7. Schema and upgrade discipline
   - Production schema changes must be shipped via Alembic migrations.
   - `create_all`/auto-create is for local/dev convenience only; it must not be the upgrade mechanism.

---

## 2) System Architecture (How to Add Features Correctly)

### 2.1 Datasets and Versions

- Dataset upload creates a dataset + an immutable dataset version.
- Transformations create additional immutable versions (never mutate in-place).
- Every compute execution pins a dataset_version (either “current” or explicitly chosen).

Implementation anchors:
- Backend routes: `ai-data-analyst/backend/app/api/routes/datasets.py`
- Loader: `ai-data-analyst/backend/app/services/dataset_loader.py`
- Transform pipeline: `ai-data-analyst/backend/app/services/dataset_transformations.py`

### 2.2 Operators and Artifacts (the "truth engine")

Operators:
- small, named functions that accept (df, schema/profile metadata, params)
- return: tables/charts + metrics + summary (no prose; no opinions)
- must be deterministic and bounded

Artifacts:
- persisted outputs: tables, charts, files
- always tied to dataset_id + dataset_version + operator_name + params
- referenced by UI and by narrator/insight extractor
 - eligible for reuse when the cache key is identical (dataset_version + operator_name + params + runtime meta)

Implementation anchors:
- Safe compute executor: `ai-data-analyst/backend/app/compute/executor.py`
- Operator catalog: `ai-data-analyst/backend/app/compute/operators/*`
- Artifact store: `ai-data-analyst/backend/app/compute/artifacts.py`

### 2.3 Narration (prose is a layer, not the source of truth)

- Deterministic narrative must always exist (based on computed evidence).
- Optional LLM rewrite can improve readability, but must not introduce new facts.
- If LLM output violates constraints, fall back to deterministic narration.

Implementation anchors:
- Narrator: `ai-data-analyst/backend/app/services/narrator.py`
- Execution integration: `ai-data-analyst/backend/app/services/analysis_execution.py`

### 2.4 Plans (how we orchestrate workflows)

A plan is a sequence of operator invocations (EDA, quality, segments, time, etc.).
Plans should evolve by:
- adding operators (new evidence types)
- adding plan templates (task-specific recipes)
- adding UI panels that surface evidence with guided next actions

Implementation anchors:
- Plans: `ai-data-analyst/backend/app/compute/plans.py`

---

## 3) Capability Ladder (Sequencing the "entire data science")

We build breadth *and* depth by working in tiers. Each tier is a complete product slice.

L0 (Basic Analyst)
- Upload, process, preview, schema, missingness, uniqueness, summaries
- Run safe SQL queries
- Generate an evidence-backed narrative and a compact insight list

L1 (Working Analyst)
- Segment analysis, relationships (corr/assoc), time resampling/aggregations
- Transformation builder (validated steps + versioning)
- Reports that cite evidence artifacts; shareable sessions (“Data Speaks”)

L2 (Expert Analyst / DS Assistant)
- Hypothesis testing, A/B testing, cohort/funnel/RFM packs
- Anomaly/change-point detection (time series)
- Feature engineering suggestions (evidence-backed, reversible transforms)

L3 (Core DS Workflows)
- Target inference, baseline modeling, cross-validation, explainability
- Model registry, batch predictions, drift monitoring
- Governance: audit logs, RBAC, quotas

Reference:
- `docs/LEVELS_AND_SEQUENCE.md`
- `docs/CAPABILITY_LADDER.md`

---

## 4) Definition of Done (DoD) for Any New Feature

For any new operator / plan / UI panel:

1) Operator correctness
- Validates required columns and parameter ranges
- Explicitly handles empty datasets / all-null columns / inf values
- Uses bounded operations (limit rows, top-k, sampling)

2) Artifacts and provenance
- Writes artifacts via ArtifactStore (never “return raw” without a reference)
- Attaches operator_name and operator_params
- Pins dataset_version

3) Security
- No arbitrary code execution
- SQL is read-only, single-statement, bounded; enforce allow-list on keywords

4) Reliability
- Structured error messages (predictable JSON shape)
- Timeouts/cancellation where relevant
- Idempotency for background jobs

5) Tests
- Unit tests for step/operator
- Integration test for route behavior (dataset-backed)
- Regression test for tricky edge cases

6) UX
- UI surfaces evidence, not just JSON dumps
- Links to the artifact viewer for “show me the data”
- Provides next actions (“run segment scan”, “transform missing values”, etc.)

7) No loose ends checklist
- No placeholder outputs / mock generators in any dataset-backed flow.
- Any new backend capability has a UI entry point (button/action/panel) and vice versa.
- Every new API endpoint has at least one test and is wired into CI.
- All user-visible failures are handled (timeouts, cancellations, validation errors) with a clear message.
- Any new "action" must be logged into the session feed with re-run support.

---

## 5) Repo Workflow (How We Ship OSS Safely)

Branches
- `main`: protected, green CI only
- feature branches: small slices with tests

Local gates
- Backend: `pytest -q`
- Frontend: `npm run build`
- Docker: `docker compose up --build` (for self-host checks)

CI gates
- GitHub Actions runs backend tests + frontend build.

Docs
- Every new feature updates:
  - `docs/SPEC_PROGRESS.md` (what moved forward)
  - `docs/LEVELS_AND_SEQUENCE.md` (where it lands in the ladder)
