# Neural Analyst - Workplan + Capability Roadmap

This doc is a living "north star" + sequential build plan.
It complements `spec_review/*` (which is the contractual spec) with:
- a capability ladder (basic -> pro -> advanced -> "god level"),
- non-functional requirements (reliability, scale, safety),
- and a practical execution order that keeps the product truthful at every step.

Last updated: 2026-02-04

## A) Non-negotiables (product trust)

1) No computed claim without evidence
- Any dataset-specific number shown to users must come from a stored artifact.
- If we cannot compute safely, we must say so and propose the next safe action.

2) No arbitrary code execution
- The system can feel "dynamic" via structured plans + allow-listed operators.
- Never execute LLM-generated Python/SQL except through strict allow-lists + sandboxes.

3) Dataset is untrusted input (prompt injection is expected)
- Do not treat dataset text as instructions.
- Do not concatenate raw data into prompts.
- Prefer compute-first pipelines; LLM sees summaries/artifacts only.

## B) Sequential build order (P0 first)

Reference checklist: `spec_review/TALK_TO_YOUR_DATA_BACKLOG.md`

### P0 - Truthful Talk-to-Data demo (must be rock solid)

P0.10 (Do early) Minimal integration + regression harness
- Upload -> process -> dataset READY
- Grounded chat query -> returns evidence artifacts
- Data Speaks run (EDA) -> creates artifacts + insights
- Injection-string dataset case -> no "instruction following"

P0.8 Quality/Adequacy must run on dataset_id + dataset_version
- Remove any "send file paths" contracts from UI/backend.
- Persist sessions in DB (already started) and bind to dataset_version.
- Add privacy controls (PII columns redaction policy) before LLM-heavy quality checks.

P0.5 Grounded Chat V1 (clarification loop)
- Ask the minimum question when the request is ambiguous.
- Return structured options (UI renders radio buttons).
- Only run compute after disambiguation is resolved.

P0.9 Analytics endpoints must stop returning "raw dicts"
- Every analytics endpoint should produce artifacts (tables/metrics/charts).
- Responses should reference artifact IDs and dataset_version.

P0.2 Orchestrator/tool layer (only if used in runtime)
- If an orchestrator exists, it must call the safe compute engine and return artifacts.
- If not used, we either deprecate it or rewire it later in P1/P2.

### P1 - Analyst-grade (delightful + reproducible)

Transformations (versioning + preview diffs + rollback)
- dataset version DAG, impact previews (rows changed, distribution shift)
- reproducible transformations with provenance

Insight library
- dedupe + ranking + save/share
- “what changed since last run” comparisons across dataset versions

Reports
- grounded executive report + read-only share links

Streaming UX
- consistent progress/events across analyses, quality, transforms, modeling

Operability
- structured logs, metrics backend, tracing, SLOs

### P2 - Data Scientist-grade

Modeling workflows (guarded + leakage-aware)
- task inference -> target selection -> baseline training -> evaluation
- explainability + diagnostics (residuals, PDP/ICE, SHAP-lite)
- model registry + prediction endpoints + drift monitoring

Domain packs (optional)
- marketing/product analytics, finance, experiments (A/B), customer analytics, etc.

### P3 - Enterprise-scale SaaS

- multi-tenant workspaces, quotas, audit logs
- storage abstraction (S3/GCS), compute workers, retries, DLQ
- governance: RBAC, data retention, PII policy enforcement

## C) Capability ladder (what the app can do, by maturity)

### Level 1 - Basic Data Analyst (P0)
- Ingest: upload/import, schema, preview, basic profiling
- EDA: missingness, uniqueness, distributions, correlations, associations, outliers, simple time summaries
- Evidence: artifacts + download + report export
- Grounded chat: compute-backed answers (no hallucinated dataset numbers)

### Level 2 - Pro Analyst (P1)
- Data cleaning pipelines (versioned transforms)
- Quality/readiness scoring per goal
- Interactive exploration: filters/segments/time windows
- "Insight library" with saved runs and comparisons

### Level 3 - Applied Data Scientist (P2)
- Target/task inference, baseline models, evaluation, explainability
- Feature engineering suggestions (guarded, reversible)
- Forecasting/anomaly pipelines for time series

### Level 4 - "God level" (directional, never fully done)
- Cross-dataset reasoning (joins, entity resolution)
- Causal inference suggestions (with explicit assumptions)
- AutoML with compute budgeting and governance
- Continuous monitoring: drift, data quality decay, alerts, scheduled reports

## D) Reliability and scaling principles (so this doesn't break later)

- Keep compute deterministic by default; sampling must be labeled with scan_ratio/confidence.
- Design every heavy path to run async via Jobs (Celery/worker) with cancellation checks.
- Enforce hard limits: rows scanned, runtime, memory, artifact sizes.
- Store provenance everywhere: dataset_version, operator, params, timestamps.

