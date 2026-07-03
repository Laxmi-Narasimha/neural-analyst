# Levels And Sequential Build Plan (Low -> Medium -> Expert -> Core)

This project is intentionally infinite in scope ("everything a great analyst can do").
The only industry-grade way to get there without collapsing into bugs is:

1) lock a **truthful, evidence-first core** (operators -> artifacts -> narrative), then
2) expand capabilities by adding operators + workflows, while
3) keeping contracts stable and adding regression tests per capability.

This doc maps your "low/medium/expert/core" framing to the existing spec pack in `spec_review/`
and the tracked progress in `docs/SPEC_PROGRESS.md`.

## Non-Negotiables (Always True)

- No dataset-specific numeric claim without evidence artifacts.
- No arbitrary code execution; only allow-listed operators and guarded SQL.
- Dataset content is untrusted input (prompt injection is expected).
- Every run stores provenance (dataset_version + operator + params).

## Level 1 (Low) - Trustworthy Analyst MVP (P0)

Goal: upload a dataset and immediately get a truthful "Make the data speak" session + grounded Q&A.

Capabilities:
- Ingestion: CSV/Excel/JSON/Parquet upload + processing (sampling-first for big files)
- Dataset registry/loader by dataset_id + dataset_version
- Safe compute operators (EDA allow-list)
- Artifact store + index + download
- Data Speaks sessions (Analyses) with SSE progress/events
- Grounded chat: plan -> compute -> answer, with minimal clarification loop
- Safety regression harness (injection-string dataset case)

Status: implemented (see `docs/SPEC_PROGRESS.md`).

## Level 2 (Medium) - Practical Daily Analyst Work (P1)

Goal: replace most notebook-style EDA and day-to-day transformations for real analysts.

Capabilities:
- Dataset SQL (read-only, guarded) for custom exploration
- Artifact viewing UX (paged tables; not just download)
- Transformations with preview + versioning + rollback
- Insight library UX: ranked insights, "why it matters", one-click follow-ups
- Reporting: export markdown/html reports from evidence
- Streaming UX polish: partial results, cancellation, resumable sessions
- Connections hardening: secrets encryption, read-only enforcement, row limits, auditability

Acceptance tests:
- Any "Run query" / "Run transform" produces artifacts tied to dataset_version.
- Switching dataset versions changes all downstream compute results.

## Level 3 (Expert) - Power Analyst + Domain Packs (P1/P2)

Goal: expert playbooks available as one-click guided workflows (still evidence-first).

Capabilities:
- Cohorts/retention, funnels, RFM/CLV heuristics (guarded, explainable)
- Time series anomaly/change-point summaries (guarded)
- Segmentation/clustering with stability metrics
- Statistical testing toolkit (with assumptions + effect sizes)
- "Driver analysis" workflows (feature importance-like insights without full ML)

Acceptance tests:
- Required-column validation for every domain operator.
- Clear assumptions + limitations in narrative output.

## Level 4 (Core) - Data Scientist + Enterprise Reliability (P2/P3)

Goal: correct-by-default modeling workflows + production reliability at scale.

Capabilities:
- Task inference -> target selection -> baseline modeling runs (leakage-aware)
- Proper evaluation (time splits when needed), per-segment evaluation
- Explainability + diagnostics (residuals, importance, drift)
- Model registry + predictions API + schema checks
- Scheduled runs + monitoring + alerts (quality decay, drift)
- Multi-tenant governance (workspaces, RBAC, quotas, audit logs)
- Storage abstraction (local -> S3) + durable worker queues + retries + DLQ

Acceptance tests:
- No silent leakage; target confirmation required.
- Reproducible model runs (dataset_version, seed, params pinned).

## The "Sequential" Rule (How We Implement)

We always build in this order for any new capability:
1) Operator(s) with validated params + bounded outputs
2) Artifact(s) stored + indexed + downloadable
3) Narrative/suggestions derived from artifacts (deterministic first; optional LLM polish)
4) Frontend UX wiring + loading/error states
5) Regression tests (unit + at least one integration test)

