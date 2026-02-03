# Neural Analyst — Feature Specification (Talk to Your Data / Data Speaks)

**Date:** 2026‑02‑03  
**Purpose:** Detailed product/UX specification for the end-user experience.  
**Rule:** No code snippets; this is behavioral and implementation guidance.

This document answers: “What features must exist so the product truly behaves like an AI Data Analyst / AI Data Scientist assistant that works primarily via prompts and guided UI?”

---

## 1) Product Promise (User-Facing)

### 1.1 The one-sentence promise
“Upload or connect your data, then talk to it: get evidence-backed insights, charts, and models without writing code.”

### 1.2 The “mesmerizing” moment you want
Within one click, a user should see:
- a crisp story of what the dataset is about,
- what is suspicious or risky,
- what is most important,
- and what to do next (as one-click actions and suggested prompts).

The experience must feel:
- interactive (not a static report),
- truthful (numbers match the dataset),
- and actionable (next steps are clear).

---

## 2) Information Architecture (IA)

### 2.1 Primary navigation
Minimum top-level areas:
1. **Datasets** (upload, connect, manage, preview)
2. **Talk to Your Data** (main chat + actions)
3. **Data Speaks** (autopilot story + evidence)
4. **Quality** (data quality + adequacy + readiness)
5. **Connections** (databases, warehouses, APIs)
6. **Models** (trained models, evaluations, predictions)
7. **Reports** (generated artifacts, sharing/export)
8. **Settings** (API keys, privacy, compute limits)

### 2.2 First-time user path
1. Land on a minimal “Welcome” page.
2. Upload a dataset (or connect to a source).
3. Immediately see the dataset preview and a single primary CTA:
   - “Talk to your data” (primary)
   - “Make the data speak” (secondary)

---

## 3) The Main Button: “Talk to your data / Make the data speak”

### 3.1 Placement
The CTA should appear:
- on the dashboard,
- on the dataset detail page,
- and inside the chat page as a “Start” action.

### 3.2 What it triggers (UX behavior)
When pressed, the system creates a “Data Speaks session” and shows progress:
1. “Reading your data…”  
2. “Understanding the schema…”  
3. “Checking quality and risks…”  
4. “Finding key insights…”  
5. “Preparing your data story…”

Users should be able to cancel at any time and still keep partial results.

### 3.3 Output layout (Data Speaks screen)
The Data Speaks screen must include:

**A) Narrative header**
- Dataset name + version + last updated
- Confidence indicator (how complete the scan was)
- “Top 3 takeaways” in one paragraph

**B) Evidence panels (tabs or sections)**
1. Overview (rows, columns, memory estimate, time span, entities)
2. Schema & Column Dictionary (types, roles, example values)
3. Data Quality (missingness, duplicates, validity, consistency)
4. Distribution & Outliers (key columns; extreme values; heavy tails)
5. Relationships (correlation / association, likely drivers)
6. Segments (top segments, cohorts, clusters)
7. Time (trends, seasonality, change points) if a time column exists
8. Privacy & Risk (PII flags; leakage risk; sensitive columns)

**C) “Suggested next actions”**
- 5–10 one-click actions, ordered by value and confidence.
Examples:
- “Show the biggest anomalies”
- “Explain what drives revenue”
- “Find the top 3 customer segments”
- “Detect churn risk factors”
- “Generate an executive summary”

**D) “Suggested prompts”**
- context-aware prompts derived from the current dataset and discovered patterns.
- avoid generic prompts; they must reference detected columns/entities.

**E) Action feed**
- Every action produces a logged entry with:
  - what was run
  - when
  - outputs (table/chart/summary)
  - a “Re-run” button

---

## 4) Dataset UX (Upload, Preview, Manage)

### 4.1 Upload UX requirements
User must be able to:
- upload CSV, Excel, JSON, Parquet
- see upload progress and file size limits
- optionally tag the dataset and describe it

After upload, the UI must not show dummy values. It must:
- redirect to a dataset detail page that shows the real dataset summary.

### 4.2 Dataset detail page
Must contain:
- dataset metadata (name, tags, source, size)
- preview table (first N rows, configurable)
- schema table (column name, type, missingness, unique, example values)
- key risks (PII warnings, missingness > threshold, duplicates)
- primary CTAs:
  - “Talk to your data”
  - “Make the data speak”
  - “Run quality check”

### 4.3 Dataset list page
Must support:
- search
- filters by tag, source, owner
- status (processing/ready/error)
- quick actions (talk, delete, export)

### 4.4 Connectors import flow (database → dataset)
User can:
- create a connection (host/user/pass/etc)
- test connection
- browse tables
- import table(s) as dataset(s)
- optionally “join multiple tables” into an analysis workspace

---

## 5) Chat UX (“Talk to your data” page)

### 5.1 Core behaviors
The chat is not just text. It must support rich message types:
- plain language answer
- “evidence card” (table preview)
- “chart card”
- “assumptions / caveats” callout
- “next actions” callout

### 5.2 Grounding rules (user-visible)
The assistant should communicate trust levels:
- “Computed” vs “Estimated” vs “Assumption”

If the user asks a question that requires data access, the assistant should:
- run a computation
- show the result (table/metric)
- then provide interpretation.

### 5.3 Streaming responses
For best UX:
- stream partial responses
- stream intermediate progress for long tasks (e.g., model training)

### 5.4 Clarification questions (resolve ambiguity)
The assistant must ask clarifying questions when:
- “best” is undefined (“best by revenue, margin, growth, retention?”)
- time range is undefined
- aggregation level is unclear
- target variable is unclear for ML

Critical requirement:
- The assistant must ask the *minimum* number of questions needed to proceed.
- Provide suggested answers (radio buttons) when possible.

---

## 6) One-Click Actions (Buttons inside chat / data speaks)

### 6.1 Action categories
Actions should be grouped:
- Summary
- Quality
- Relationships
- Segments
- Time
- Modeling
- Export

### 6.2 Required actions (minimum set)
1. Data summary (shape, types, missingness)
2. Correlation / association overview
3. Outlier/anomaly report
4. Top segments (categorical breakdowns; Pareto)
5. Time trends (if time column)
6. Recommend next questions
7. Generate executive summary
8. Generate full report

Each action must:
- run on the selected dataset
- generate evidence artifacts
- be reproducible

---

## 7) Data Quality and Adequacy (Readiness)

You already have a “data adequacy” concept. For the “AI analyst” product it should be framed as:
- “Is this dataset ready to answer my question / train my model / power my assistant?”

### 7.1 Quality dimensions to show
- Completeness (missingness)
- Validity (type/format checks)
- Consistency (cross-field consistency rules)
- Uniqueness (duplicates)
- Timeliness (staleness if time fields exist)
- Bias/representation (skewed segments)
- Privacy risk (PII, sensitive categories)

### 7.2 Readiness outputs
Readiness must produce:
- a readiness label (Ready / Partially Ready / Unsafe / Blocked)
- a composite score (with weight transparency)
- top recommendations
- a remediation plan

### 7.3 UI requirements
Quality UI must allow:
- selecting a dataset (not file paths)
- running validation
- answering clarifying questions
- downloading report

---

## 8) Transformations (No-Code Data Cleaning)

### 8.1 Transformation plan UX
The assistant can propose a “cleaning plan” that includes:
- which columns are affected
- what operation is applied
- what the expected impact is
- how many rows will change

### 8.2 Preview and reversibility
Users must be able to:
- preview before/after deltas
- apply changes to a new dataset version
- roll back

### 8.3 Transformations to support
- type conversions (date parsing, numeric parsing)
- missing value imputation strategies
- outlier handling
- standardization (case trimming, whitespace)
- encoding categoricals
- feature engineering (time features, bins)

---

## 9) Modeling UX (AI Data Scientist layer)

### 9.1 Model creation flow
User flow:
1. Choose dataset
2. Choose task: classification/regression/forecasting/clustering
3. Select target (assistant suggests candidates + confidence)
4. Confirm train/test split strategy
5. Train baseline model
6. Review metrics and explainability
7. Save model to registry

### 9.2 Explainability UX
Must include:
- global importance ranking
- per-segment performance
- warnings (leakage, imbalance)

### 9.3 Prediction UX
User can:
- upload a scoring file (same schema)
- or paste JSON rows
- get predictions and explanations

---

## 10) Reporting and Export UX

### 10.1 Report types
- Executive summary (short, decision-ready)
- Technical report (methods + evidence)
- Data dictionary export
- Model report (metrics + explainability)

### 10.2 Report UX requirements
- Generate report for a specific analysis session
- Download in markdown/HTML
- Copy share link (self-host)
- Include reproducibility metadata (dataset version, steps)

---

## 11) Failure Modes and “Good Errors”

### 11.1 Principles
Users should never see:
- stack traces
- meaningless “500 error” without context

They should see:
- what failed
- why it failed (likely causes)
- what to do next

### 11.2 Common failure cases to design for
- Upload too large
- Unsupported encoding
- Missing required columns for a requested workflow
- Conflicting date formats
- Job timeout
- LLM or external service error

---

## 12) UX Acceptance Checklist (High-level)

The product is “ready” for the flagship button when:
- A user can upload a dataset and immediately see a truthful schema + profile.
- Clicking “Talk to your data” produces a narrative plus evidence panels.
- Asking “How many rows? What’s the top category? Show outliers.” returns computed results.
- The system asks clarification questions only when needed.
- Every insight links to evidence.
- Reports can be exported and re-run.

For engineering details (how to implement safely), see:
- `docs/TALK_TO_YOUR_DATA_ENGINEERING_SPEC.md`

---

## 13) Detailed Screen Specifications (UI/UX, states, and data dependencies)

This section makes the UX “decision-complete” by specifying what each screen must show, what backend data it depends on, and what states it must handle.

### 13.1 Global UI primitives (used everywhere)

**A) Dataset selector**
- Always select by `dataset_id` (never by filename paths).
- Shows dataset name, version label, last updated, and a readiness badge (processing/ready/error).
- Allows switching datasets without losing the current session (creates a new session or prompts the user).

**B) Evidence viewer**
- A unified panel that can display:
  - metric cards (single values + definitions),
  - tables (paginated),
  - charts (with controls),
  - “how computed” provenance (dataset version, filters, group-by).
- Provides “Open in report”, “Copy as CSV”, and “Download artifact” actions where applicable.

**C) Status states**
Every screen must have consistent states:
- Empty (no datasets, no runs, no conversations)
- Loading (with step labels and progress)
- Partial (some artifacts ready, some pending)
- Error (friendly message + remediation + retry)
- Read-only (when dataset is locked or permissions restricted)

**D) Trust labels**
Every assistant output block must show one of:
- Computed (from artifacts)
- Estimated (sample-based; show sampling note)
- Assumption (requires user confirmation)

### 13.2 Home / Dashboard
Purpose: give the “one-click wow” entry point.

Must include:
- Primary CTA: “Make the data speak”
- Secondary CTA: “Talk to your data”
- Recent datasets (with readiness and last-run summary)
- Recent sessions (Data Speaks sessions and chat conversations)

States:
- If no dataset exists, show upload/connect CTA only.

### 13.3 Datasets list
Purpose: manage datasets as first-class assets.

Must include:
- Search (name, tags, description)
- Filters (format, status, tag, owner)
- Quick actions:
  - Make the data speak
  - Run quality check
  - Delete / archive

States:
- Processing: show progress and disable compute-heavy actions.
- Error: show why (parsing error, file missing, schema inference failed) and a “re-process” action.

### 13.4 Dataset detail
Purpose: be the truthful, minimal “data sheet” (data dictionary + profile).

Must include:
- Dataset identity:
  - name, description, tags
  - dataset_id
  - dataset version (hash/label)
  - source (upload vs connector)
- Dataset preview:
  - first N rows (configurable)
  - column list
  - “download sample” (optional)
- Schema + dictionary:
  - columns with inferred types and semantic roles
  - example values (safe redaction for PII)
  - missingness and unique counts
- Flags:
  - PII risk
  - potential keys
  - likely time column(s)
  - likely target column(s) (if any)
- CTAs:
  - Make the data speak (primary)
  - Talk to your data (secondary)
  - Run quality/readiness
  - Create transformation

### 13.5 Upload / Connect flow
Purpose: ensure the user can bring any dataset without friction and without “mystery errors”.

Upload requirements:
- Clear supported formats and max size.
- Encoding detection and human-readable error when decoding fails.
- Column name normalization rules displayed (with ability to view original names).
- Optional: “My dataset represents…” short text field that is saved as dataset metadata and used by the assistant.

Connector requirements:
- Connection wizard:
  - choose type (Postgres/MySQL/SQLite/BigQuery/Snowflake/etc.)
  - credentials
  - test connection
  - browse tables or enter SQL (read-only enforced)
  - import as dataset snapshot OR use “live query mode” (advanced)
- The user must be shown the data scope and potential cost before importing large tables.

### 13.6 Data Speaks (the flagship autopilot story)
Purpose: the “mesmerizing” screen.

Layout (non-negotiable):
- Narrative header (top 3 takeaways + confidence + scope/time range)
- Evidence panels (tabs):
  - Overview
  - Schema & Dictionary
  - Quality & Readiness
  - Outliers & Anomalies
  - Relationships & Drivers
  - Segments & Cohorts (when possible)
  - Time & Seasonality (when a time column exists)
  - Privacy & Risk
- Suggested next actions (one-click)
- Suggested prompts (context-aware)
- Execution log / action feed

Behavior:
- The narrative must be derived only from computed artifacts.
- The narrative must explicitly label sampling.
- The system must avoid “overclaiming”: if a driver analysis is inconclusive, it must say so.

### 13.7 Talk to Your Data (chat + actions)
Purpose: interactive analysis with strict grounding.

Must include:
- Dataset context header (selected dataset + version)
- Message input with:
  - prompt suggestions
  - “attach” menu for selecting columns or artifacts
- Rich assistant messages:
  - answer text
  - evidence cards
  - charts
  - assumptions and clarifications
  - “re-run with full data” option when sampling was used

Critical behaviors:
- The assistant must ask minimal clarifying questions when required.
- The assistant must always show a computation before presenting numeric conclusions.
- The user must be able to “pin” key definitions (metric definitions, time windows) to avoid repeated clarifications.

### 13.8 Quality / Readiness (unified quality + adequacy)
Purpose: answer “Can I trust this data for my goal?”

Must include:
- Goal selection (what do you want to do with this data?)
- Domain selection (optional)
- Readiness score with transparency:
  - component scores
  - evidence
  - recommended fixes
- A remediation plan that can be turned into transformations:
  - “Create cleaning pipeline from this plan”

### 13.9 Transformations (no-code cleaning with provenance)
Purpose: enable safe dataset cleaning without breaking reproducibility.

Must include:
- Proposed transformation plan from assistant (user can edit)
- Preview and diff:
  - how many rows/values change
  - distribution changes for affected columns
- “Apply as new version” (never overwrite original)
- Undo/rollback (by switching versions)

Transformations that must be supported early:
- type conversions and parsing (dates/numbers)
- trimming/normalization for strings
- missing value handling (drop vs impute)
- duplicate handling
- simple feature engineering (time features, bins)

### 13.10 Models (data scientist workflows)
Purpose: make modeling safe and approachable, not magical.

Must include:
- Task selection (classification/regression/forecasting/clustering)
- Target suggestion with confidence and rationale
- Train/test split choices:
  - random split vs time split
  - leakage warnings when a time split is needed
- Baseline training with:
  - metrics
  - per-segment performance
  - explainability
- Save runs to a model registry (artifact)

### 13.11 Reports
Purpose: make outputs shareable and reproducible.

Must include:
- Report builder:
  - select artifacts to include
  - generate executive summary vs technical report
- Exports:
  - markdown/HTML
  - dataset dictionary export
  - model report export
- Provenance metadata:
  - dataset version
  - time range
  - transformations applied
  - analysis steps

### 13.12 Settings
Purpose: let the user control privacy and compute.

Must include:
- LLM provider configuration (if multiple providers are supported)
- Privacy mode toggles:
  - never send raw data to LLM
  - redact PII in all outputs
  - disable external network tools
- Compute caps:
  - max rows scanned per interactive query
  - max job runtime
  - max concurrent jobs per user

---

## 14) Insight Library (what “smart” looks like)

The assistant should maintain a catalog of insight types that can be computed deterministically. Each insight must have:
- a title,
- a short description,
- an evidence artifact (table/metric/chart),
- a confidence score,
- a “why it matters” explanation,
- and one or more suggested next actions.

Minimum insight types (high value and general):
- Missingness hotspots and patterns (including missing-not-at-random signals)
- Duplicate rows and near-duplicates (with key candidates)
- Outlier clusters and extreme values (with slice explanations)
- Skewness and heavy tails (with transformation suggestions)
- High-cardinality categorical warnings (with grouping suggestions)
- Correlation/association highlights (with caveats)
- Time trends, change points, and seasonality (when time exists)
- Segment comparisons (Pareto, top segments by metric)
- Leakage risk indicators (columns that “know the future”)
- PII and sensitive column detection (with redaction mode suggestions)

Domain-specific insights (optional but differentiating):
- RFM / cohort / retention (customer analytics)
- Funnels (product analytics)
- A/B testing summaries (experiment analytics)
- Churn drivers (subscription analytics)

---

## 15) Prompt System (prompts, suggestions, and clarification UX)

### 15.1 Prompt suggestions that feel “real”
Suggested prompts must be derived from:
- detected column roles (time, target, entity key),
- detected segments (top categories),
- and detected data issues.

Examples of prompt patterns (without referencing specific columns here):
- “What drives [target] the most?”
- “Show me anomalies over time”
- “Compare top segments by [metric]”
- “Summarize the dataset like an analyst briefing”

### 15.2 Clarification questions as UI components (not plain text)
When the assistant needs definitions, it should ask with selectable options:
- metric definition options (revenue vs profit vs count)
- time window options (last 7/30/90 days; full range)
- aggregation level options (by user, by day, by category)
- modeling target options (candidate targets)

Rules:
- ask the minimum number of questions required
- never ask a question that the system could infer with high confidence
- always record user selections as pinned context

### 15.3 “Pinned context” controls
Users must be able to pin:
- the primary dataset
- the target column (if modeling)
- the time window
- the key metric definition

This prevents the assistant from re-asking the same question and makes sessions feel professional.

---

## 16) Collaboration and Sharing (optional for OSS, required for SaaS)

Even in open source, collaboration features are useful for adoption.

Minimum share features:
- share a report (read-only link)
- export a report bundle (zip) for offline sharing

Team features (later):
- workspace-level datasets (shared by team)
- comments on artifacts
- role-based permissions

---

## 17) UX telemetry (how you know it’s working)

Instrument the experience with:
- time-to-first-evidence
- time-to-first-insight
- number of clarifying questions per session
- “abandonment points” (where users drop off)
- export rate (how often users export reports)
- rerun rate (how often users rerun with different filters)

Telemetry must not log raw dataset rows or secrets.

---

## 18) Acceptance test scripts (prompt-driven)

These are manual “demo scripts” and can later become automated tests:

**Dataset onboarding**
- Upload a dataset and verify the dataset detail page shows correct row/column counts.
- Trigger Data Speaks and verify each evidence panel shows computed outputs.

**Basic EDA**
- Ask: “How many rows are there?” and verify the assistant shows a computed metric.
- Ask: “What are the top categories?” and verify a computed table.
- Ask: “Show outliers” and verify a table and explanation.

**Time series**
- Ask: “Show the trend over time” and verify:
  - the assistant selects a time column (or asks)
  - the chart matches computed data
  - change points are described with caveats

**Modeling**
- Ask: “Predict the target” and verify:
  - the assistant suggests target candidates and requires confirmation if ambiguous
  - the model run produces metrics and a model report artifact

**Safety**
- Include dataset rows containing prompt injection text and verify:
  - the assistant treats it as data content, not instructions
  - no secrets or internal prompts are exposed

