# Neural Analyst Capability Ladder (What We Build, In What Order)

Last updated: 2026-02-05

This document is an implementation-oriented "map" of what the product needs to do to feel like:
- a reliable AI Data Analyst (EDA + QA + reporting), and then
- a real AI Data Scientist (modeling + experiments + explainability), and then
- an enterprise-grade system (scale, governance, observability).

It complements the specs under `spec_review/` by translating them into an ordered capability ladder.

---

## 0) Guiding Principle (Non-Negotiable)

If a user asks a dataset-specific question, the system must:
- plan -> compute (safe operators) -> produce artifacts -> narrate from artifacts
- refuse to provide numeric claims without evidence

Anything else is "demo mode" and will not scale to trust.

---

## 1) Level 0 (P0) - Truthful Talk-to-Data MVP

Goal: a user can upload a dataset, click "Make the data speak", and ask follow-ups without hallucinated numbers.

Core capabilities:
- Dataset registry/loader by `dataset_id` (single source of truth)
- Safe compute operator allow-list (no arbitrary Python execution)
- Artifact store + provenance (metric/table/report/chart artifacts)
- Grounded chat (plan -> compute -> answer) + minimal clarification loop
- Data Speaks (one-click EDA run) producing multiple evidence artifacts
- Drilldowns (bounded): missingness patterns, outlier explain, segment deep dive, relationship explain, time anomaly/change-point scan
- PII-safe defaults: mask PII values in previews; strip PII example values from schema outputs
- Stable identity + access control for datasets, conversations, artifacts
- Minimal regression tests (upload -> process -> grounded chat -> analysis run + injection case)

Status: see `docs/SPEC_PROGRESS.md` (P0 is largely DONE; chat clarification remains the main polish item).

---

## 2) Level 1 - Analyst Core (P1)

Goal: replace manual "notebook EDA" for most everyday analyst work, while staying safe and reproducible.

Capabilities to add:
- Transformations with preview + dataset versioning
  - filters, type casts, column rename, missing-value handling, dedupe, joins, time parsing
  - diff previews: row count delta, distribution shift, schema diff
- Insight library (ranked, deduped evidence-backed insights)
  - "top insights" with confidence + why it matters + provenance
- Reporting & exports
  - report builder from selected artifacts
  - markdown/html export, downloadable bundles, shareable read-only links (optional)
- Streaming UX + progress
  - progress events for long runs (Data Speaks, quality, heavy analytics)
  - partial artifact rendering as soon as an operator finishes
- Chart operators (safe chart specs)
  - deterministic chart selection + vega-lite specs derived from computed tables
- Connector hardening (read-only + limits)
  - query allow-listing, row limits, timeouts, secret encryption, audit logs

Engineering foundations to support Level 1:
- Durable job system (worker out-of-process)
- Storage abstraction (local -> S3/GCS) for uploads + artifacts
- CI: backend tests + frontend build + lint gates
- Observability: structured logs + traces + metrics + SLOs

---

## 3) Level 2 - Business Analytics Pack (P1/P2)

Goal: common "business analyst" playbooks available as one-click actions.

Modules/operators to support:
- Cohorts & retention (by signup date / first purchase / activation event)
- Funnels (step conversion, drop-off, segment comparisons)
- RFM, CLV, churn heuristics (guarded + explainable)
- Experimentation (A/B testing with assumptions + power checks)
- Growth decomposition (by segment, by channel, by product)

Non-negotiables:
- strict required-column validation
- artifact outputs for every computed result
- clear assumptions + limitations in the narrative layer

---

## 4) Level 3 - Data Scientist Workflows (P2)

Goal: guided modeling workflows that are correct by default (no silent leakage, no broken metrics).

Capabilities:
- Task inference + target selection
  - candidate target scoring
  - clarification UI to confirm target + time split strategy
  - leakage and post-outcome column warnings
- Baseline modeling runs
  - regression/classification + time-series forecasting (as separate tracks)
  - correct metrics + calibration + segment evaluation
- Explainability + diagnostics
  - feature importance, partial dependence (optional), residual checks
  - data drift checks between train/test/time windows
- Model registry + prediction workflows
  - persisted runs, versioning, schema checks
  - prediction endpoints and exports
- Guarded AutoML
  - compute budgets, user confirmation, caching/reuse

---

## 5) Level 4 - Enterprise & Scale (P3)

Goal: multiple users, multiple instances, and production operations without losing data or trust.

Capabilities:
- Multi-tenant controls
  - workspaces/projects, RBAC, dataset sharing rules
  - quotas (rows scanned, jobs/min, storage)
  - audit logs (who ran what on which dataset version)
- Production compute
  - job queue + workers + retries + cancellation + dead-letter
  - cache and reuse artifacts by dataset_version
  - out-of-core compute path (DuckDB/Polars lazy)
- Security
  - connector secret encryption (at rest) + rotation
  - prompt-injection resistance evaluation suite + gating
  - PII tagging, redaction rules, safe previewing
- Operability
  - tracing + metrics + dashboards (latency, cost, errors, job timings)
  - release pipelines + migrations + rollbacks
  - explicit SLOs: time-to-first-evidence, grounded-claim rate

---

## 6) What "No Missing Feature" Means (Realistic Interpretation)

"No missing feature compared to a human analyst" is not achievable as a one-time build.
The industry-grade way to satisfy the spirit of that goal is:
- build a strong operator layer + artifact model (extensible, safe, reproducible)
- continuously expand operator coverage and workflows
- keep correctness and provenance as the product's core identity
