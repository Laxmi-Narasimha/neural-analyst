# Neural Analyst — “Talk to Your Data” Documentation Pack (Master Spec + Gap Analysis)

**Date:** 2026‑02‑03  
**Repository:** `Laxmi-Narasimha/neural-analyst`  
**Audience:** Product owner, engineering, data science, security, QA  
**Constraint:** This document contains **no code snippets**; it specifies behavior, features, and implementation requirements.

This master spec is the entry point to a **30–40 page documentation pack** for building a world-class “AI Data Analyst / AI Data Scientist” experience that works primarily via prompts and guided UI (not by users writing code).

**Documentation pack (read in order):**
1. `docs/TALK_TO_YOUR_DATA_MASTER_SPEC.md` (this file) — vision, ground truth audit, feature coverage matrix, top-level roadmap.
2. `docs/TALK_TO_YOUR_DATA_FEATURE_SPEC.md` — detailed product requirements and UX flows for every major feature.
3. `docs/TALK_TO_YOUR_DATA_ENGINEERING_SPEC.md` — detailed architecture: dataset lifecycle, compute layer, agent tools, job system, persistence.
4. `docs/TALK_TO_YOUR_DATA_SECURITY_EVAL_SPEC.md` — threat model, safety constraints, evaluation harness, quality gates.
5. `docs/TALK_TO_YOUR_DATA_OPERATOR_CATALOG.md` — allow-listed compute primitives (operators) that make the assistant “dynamic” without unsafe code execution.
6. `docs/TALK_TO_YOUR_DATA_BACKLOG.md` — prioritized implementation backlog mapped to this repo’s modules (P0/P1/P2), with acceptance criteria.

---

## 0) Executive Summary

### 0.1 Your intended product (what “good” looks like)
The product should make analysts feel like they can:
- upload or connect data,
- press a single “Talk to your data / Make the data speak” button,
- immediately receive a truthful, evidence-backed story of what the data contains,
- then ask follow-up questions in natural language and receive grounded computations, charts, and reproducible steps.

The goal is not to replace analysts; it is to make them drastically faster and more effective.

### 0.2 What exists today (truth based on the repo)
The repo currently provides:
- A polished Next.js UI with a chat-like analysis page and dataset pages.
- A FastAPI backend with structured logging, configuration scaffolding, and SQLAlchemy models.
- Many analytics/ML “engine” modules under `ai-data-analyst/backend/app/ml/*`.
- A separate “data adequacy” validator (FastAPI + Streamlit), plus an embedded copy of its “data adequacy” manager under `ai-data-analyst/backend/app/agents/data_adequacy/*`.

### 0.3 The critical gap
The current system does **not** yet reliably execute analysis on the user’s actual uploaded datasets end-to-end. As a result:
- “chat” can produce plausible text without computed evidence,
- analytics endpoints often run on **mock data**,
- the “orchestrator” agent contains multiple **placeholder tool implementations**,
- several UI pages fall back to hard-coded example datasets.

This is not a “small missing feature”; it is the central product promise. The fastest way to unlock the wow factor is to build a real **dataset → compute → evidence → narrative** loop, then route all conversational answers through that loop.

### 0.4 The non-negotiable design choice
To meet your “dynamic, no hard-coded fields, no ambiguities” goal without compromising safety:
- Do **not** rely on arbitrary LLM-generated Python being executed.
- Instead, have the LLM generate a **constrained analysis plan** and execute it with a safe compute layer (SQL/DuckDB + approved operators).

This architecture can still feel “unlimited” to the user while remaining reliable and secure.

---

## 1) Product Principles (What makes the app “mesmerizing”)

### 1.1 Evidence-first, not vibes-first
Every insight must come with:
- a computed value (table/metric),
- the slice/filters used,
- and a short explanation of “why this matters”.

If the system cannot compute the answer, it must say so and propose the next best step (sampling, data selection, or a clarification question).

### 1.2 Dynamic, but deterministic
The system must infer:
- likely target columns,
- time columns,
- IDs and entity keys,
- categorical vs text fields,
- and risk flags (PII, leakage, drift indicators),
but do so in a reproducible way (heuristics + scoring + explainable confidence).

### 1.3 The assistant is a workflow, not a chatbot
The “Talk to your data” feature is not just LLM text. It is:
- a guided workflow,
- with tool calls and computation,
- with visual outputs,
- with saved artifacts and reproducibility.

### 1.4 Safe by default
The assistant must not be tricked by:
- prompt injection embedded inside dataset text,
- attempts to exfiltrate secrets,
- attempts to execute system commands,
- or adversarially-crafted data.

### 1.5 Performance-aware
For large datasets:
- compute should start with sampling and summaries,
- then progressively refine results,
- and offload heavy tasks to background jobs.

---

## 2) Ground Truth Audit (Specific shortcomings today)

This section highlights the most important gaps discovered in the current repo. It is not exhaustive; the detailed backlog lives in the feature/engineering docs.

### 2.1 “Talk to your data” is not grounded end-to-end
Key symptoms:
- Chat responses can be generated without querying dataset contents.
- Analytics endpoints can operate on generated mock data rather than real uploaded data.

Where it shows up:
- The analytics routes include a placeholder dataset loader that generates sample data instead of loading by `dataset_id`.
- The orchestrator’s internal tool methods return placeholder structures for summary/statistics/modeling.

### 2.2 API contract mismatches between frontend and backend
Examples:
- Dataset upload response fields expected by UI don’t match backend envelopes.
- Analysis button endpoints differ between UI and backend.
- Multiple pages rely on mock/fallback data due to contract errors.

### 2.3 User identity and persistence are incomplete
Symptoms:
- Routes use random user IDs in multiple places, so datasets and conversations don’t reliably show up for the “same user”.
- In-memory stores are used where persistence is required (sessions, connections).

### 2.4 Background execution and results persistence are incomplete
Symptoms:
- Analyses are queued, but results are not persisted in the DB in a consistent “analysis result” model.
- Heavy tasks run inline or are stubbed; no durable queue/worker model exists.

### 2.5 File lifecycle inconsistencies
Symptoms:
- Some flows refer to file paths that are never actually uploaded.
- Runtime uploads are stored locally without a unified abstraction, making scaling or multi-instance deployments hard.

---

## 3) Feature Coverage Matrix (What you have vs what you need)

Legend:
- ✅ Implemented (usable end-to-end)
- 🟡 Partial (exists but not integrated or lacks key pieces)
- 🧩 Stub/placeholder (returns fake results or doesn’t use real data)
- ❌ Missing

| Capability | Status | Notes |
|---|---:|---|
| Upload datasets (CSV/Excel/JSON/Parquet) | 🟡 | Upload exists, but end-to-end “use in chat/analytics” is not complete. |
| Dataset profiling (types/missingness/outliers) | 🟡 | Engines exist; needs automatic run + persistence + UI wiring. |
| “Talk to your data” autopilot story | ❌ | Needs narrative + evidence panels + suggested actions. |
| Grounded chat (tool-driven) | 🟡 | Chat exists, but grounding tools are not consistently enforced. |
| SQL/NL2SQL for dataset | 🟡 | NL2SQL engine exists; needs schema + execution + safety. |
| Analytics endpoints (forecasting, segmentation, etc.) | 🧩 | Some endpoints operate on mock data; need real dataset loading and results persistence. |
| Data quality and adequacy validation | 🟡 | Validator exists; needs unified file storage + dataset linkage + auth. |
| Visualization generation | 🟡 | Modules exist; needs deterministic spec + UI rendering + evidence links. |
| Report generation/export | 🟡 | Report generator exists; needs integration with analyses and UI download. |
| Connectors to databases | 🟡 | Connector scaffolding exists; needs secret handling, RBAC, UI workflows. |
| Multi-user auth + RBAC | 🟡 | Auth scaffolding exists; needs durable user store, sessions, security hardening. |
| Job queue for heavy compute | ❌ | Required for reliability with many concurrent users. |
| Reproducibility (“analysis as recipe”) | ❌ | Needs a first-class analysis plan and artifact model. |

---

## 4) What “Complete AI Data Analyst / Scientist” means (scope)

To “cover everything” for a modern AI data analyst/scientist assistant, the product should at minimum support:

### 4.1 Analyst core (required)
- Schema and profile understanding
- Fast summaries and slice/dice
- Trends, correlations, and drivers
- Outlier/anomaly discovery
- Data quality diagnostics and cleaning plans
- Visualization and storytelling
- Exportable reports

### 4.2 Data scientist core (required)
- Task inference (classification/regression/forecasting/clustering)
- Target suggestion with confidence + required user confirmation when ambiguous
- Baseline modeling with correct splits and metrics
- Explainability (global + local where feasible)
- Leakage and bias checks
- Reproducible artifacts and model registry

### 4.3 Advanced (optional but strongly differentiating)
- Cohorts, funnels, retention, RFM, CLV
- Experiment analysis and lift measurement
- Causal inference modules as an opt-in workflow with strong disclaimers
- Text analytics and embeddings-based clustering
- Geospatial workflows

This repo already contains many engine modules in these categories; the main work is integration, safety, and UX.

---

## 5) Top-Level Roadmap (Milestones)

### Milestone A (P0): Make “Talk to your data” truthful and end-to-end
- Real dataset loading by dataset ID everywhere
- Compute layer with safe operators (SQL + statistics + plotting)
- Tool-driven chat with strict grounding
- “Data Speaks” autopilot page + evidence panels
- Contract alignment across UI and API

### Milestone B (P1): Make it delightful
- “Insight library” (ranked insights with evidence)
- Transformations with preview and reproducibility
- Robust reporting and exports
- Integrated adequacy validation (with real uploads)

### Milestone C (P2): Make it a real “AI data scientist”
- Target/feature inference engine
- AutoML pipelines with leakage checks
- Model registry and prediction workflows
- Time series workflows

### Milestone D (P3): Production-grade reliability
- Job queue + workers
- Object storage abstraction for uploads and artifacts
- Multi-tenancy and quotas (optional for OSS, required for SaaS)
- Strong observability + SLOs

---

## 6) How to use the rest of this documentation pack

If your goal is to “build the best button that mesmerizes users”, start here:
- Read `docs/TALK_TO_YOUR_DATA_FEATURE_SPEC.md` and implement the “Data Speaks” screen and workflow first.
- Then use `docs/TALK_TO_YOUR_DATA_ENGINEERING_SPEC.md` to implement the compute layer and tool contracts.
- Use `docs/TALK_TO_YOUR_DATA_SECURITY_EVAL_SPEC.md` to ensure the system is safe and provably grounded.

---

## 7) Does the current app meet the “AI Data Analyst / AI Data Scientist” bar?

### 7.1 The honest answer
Structurally, it is close; functionally, not yet.

You already have:
- a UI that looks like an analyst assistant,
- a backend scaffold with dataset upload, storage, and many “analysis engines”,
- and a validator app that hints at a deeper “readiness” framework.

What you do not yet have (and what determines whether the product “works”):
- a single end-to-end “truth loop” that turns real datasets into computed evidence artifacts,
- strict enforcement that all answers come from that evidence,
- and durable session + artifact persistence so the experience feels like a workflow, not a chat demo.

### 7.2 What “complete” means (non-negotiable capability model)
AI Data Analyst Assistant (core) must reliably do:
- Load and understand any reasonable dataset (CSV/Excel/JSON/Parquet; mixed types; messy headers).
- Produce a trustworthy profile: shape, types, missingness, uniqueness, distributions, anomalies.
- Discover insights: segments, drivers, trends, outliers, relationships.
- Explain “why it matters” with evidence artifacts (tables, metrics, charts) that match the dataset.
- Offer next actions and prompts that are specific to the dataset (not generic).
- Export reports with reproducibility metadata.

AI Data Scientist Assistant (core) must reliably do:
- Infer task type when the user says “predict/forecast/classify/cluster” and ask for minimal clarifications.
- Suggest target variable candidates with confidence, show rationale, and require confirmation when ambiguous.
- Run baseline models with correct split strategies and metrics.
- Detect leakage risks, label leakage candidates, and propose safe feature sets.
- Provide explainability that’s honest and aligned with the chosen model and data type.
- Save model runs as artifacts (inputs, params, metrics, seed, dataset version).

The “mesmerizing” button requires both, but in a staged UX:
- Stage 1: Analyst wow (Data Speaks autopilot story + evidence).
- Stage 2: Analyst chat (interactive slice/dice and explanations).
- Stage 3: Scientist workflows (modeling, explainability, prediction).

### 7.3 Where you are today vs. the capability model
The repo contains many engines that could power these features, but orchestration currently does not enforce grounding. In practice, today:
- The chat route responds without executing compute tools.
- The orchestrator tools return placeholder outputs.
- Some analytics endpoints run against generated demo data.
- Dataset upload exists, but stable user identity is not wired, causing “missing datasets” from the user’s POV.

This is why the product can look “complete” while still failing the real promise when a user uploads their own data and expects truthful results.

---

## 8) The flagship experience (“Make the data talk”) — what it must do end-to-end

### 8.1 The Data Speaks pipeline (minimum)
When the user clicks the button for a dataset, the system should run a deterministic pipeline that produces a stable set of artifacts, and then generates the narrative from those artifacts:

1. Ingest and validate
   - confirm format + encoding + size limits
   - sanitize column names
   - infer stable column identifiers (so renames don’t break artifacts)

2. Profile
   - compute schema and type inference
   - compute missingness and uniqueness
   - compute distributions and quantiles for numeric fields
   - compute value counts and top categories for categoricals
   - compute basic text stats (length, language hints) for text fields

3. Role inference
   - identify candidate keys/IDs
   - identify candidate time columns
   - identify candidate targets (for later modeling)
   - detect PII and sensitive columns

4. Quality + risk scan
   - duplicates and potential keys
   - invalid values and format inconsistencies
   - outliers and heavy tails
   - leakage indicators (post-outcome columns)
   - representation risks (rare segments)

5. Insight generation
   - generate candidate insights (many)
   - rank them by usefulness and confidence
   - keep only top K for the “wow” story

6. Narrate with grounding
   - write a short story summarizing the top insights
   - attach the evidence artifacts (tables/charts)
   - include “what to do next” actions

7. Persist and index
   - store the session, plan, steps, and artifacts
   - allow re-run and export

### 8.2 What the user sees (minimum)
- A narrative header (top 3 takeaways + confidence + dataset scope).
- Evidence panels with overview, schema, quality, outliers, relationships, segments, time (if applicable), risk.
- Suggested next actions and prompts that reference their actual columns/entities.

---

## 9) The “dynamic code” requirement (how to satisfy it without unsafe execution)

### 9.1 Reframe the requirement
You want the system to behave like an expert analyst who reads the dataset, decides what’s important, chooses the right variables/targets, and produces the right computations and visualizations.

The key is: “dynamic” should mean dynamic planning, not arbitrary code execution.

### 9.2 The recommended approach
Use a constrained “Plan → Execute → Explain” loop:
- Plan: LLM generates a structured analysis plan from user intent + dataset profile.
- Execute: a deterministic compute engine executes the plan using an allow-list of operators.
- Explain: LLM narrates and proposes next steps, but only from artifacts.

This approach can handle unknown schemas and “any dataset” while remaining safe, reproducible, and scalable.

---

## 10) Success criteria (what “done” means for the flagship button)

### 10.1 User-visible acceptance criteria
The product meets the “Talk to your data” promise when a user can:
- Upload a dataset and immediately see the correct row/column counts and schema.
- Click “Make the data speak” and get a narrative that is consistent with computed evidence.
- Ask follow-ups (counts, group-bys, trends) and always receive computed outputs.
- Export a report and re-run the session later with the same dataset version.

### 10.2 Engineering acceptance criteria (non-negotiables)
- A single dataset loader is used across routes, agents, and analytics endpoints.
- Every numeric claim is traceable to an artifact created by the compute engine.
- Heavy operations run as background jobs with progress + cancellation.
- Artifact storage is abstracted so deployments can switch from local disk to object storage.

### 10.3 Quality acceptance criteria (initial targets)
These can be adjusted, but you need explicit targets to prevent “demo drift”:
- Grounding rate for numeric claims: 95%+ (production mode).
- Time-to-first-evidence for Data Speaks on a small dataset: under 10 seconds on a laptop.
- Time-to-first-evidence for a medium dataset: under 30 seconds with sampling-first.
- Injection success rate on adversarial suite: 0% for high-risk actions.

---

## 11) What to do next (implementation order)

This is the order that maximizes visible progress and prevents wasted work:

1. Implement the compute + artifact loop for core EDA operators (not all analytics).
2. Wire Data Speaks to use only computed artifacts.
3. Enforce grounding in chat (block numbers without evidence).
4. Replace mock dataset usage in analytics routes with real dataset loading.
5. Add durable user identity and dataset access control.
6. Integrate quality/adequacy as a first-class “readiness” feature tied to datasets.

The detailed task breakdown and mapping to current files is in:
- `docs/TALK_TO_YOUR_DATA_BACKLOG.md`

---

## 12) Glossary (shared language)

- **Artifact**: a stored output of computation (table, chart, report, metric) with provenance.
- **Dataset version**: an immutable fingerprint of dataset content (hash + metadata) used to ensure reproducibility.
- **Grounding**: the guarantee that answers come from computed artifacts rather than unverified model output.
- **Operator**: a safe, allow-listed compute primitive (profile, group-by, correlation, outlier scan, plot).
- **Plan**: a structured description of operators to run, with inputs/outputs and safety constraints.
- **Session (Data Speaks)**: the stored workflow instance created when the user clicks “Make the data speak”.
