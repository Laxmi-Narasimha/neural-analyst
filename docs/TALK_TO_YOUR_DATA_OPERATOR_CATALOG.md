# Neural Analyst — Safe Operator Catalog (Compute Primitives for “Talk to Your Data”)

**Date:** 2026‑02‑03  
**Purpose:** Define the allow-listed compute primitives (“operators”) that replace arbitrary code execution while still supporting expert-level analysis on arbitrary datasets.  
**Rule:** No code snippets. Operators are described by behavior, inputs, outputs, and safety constraints.

This catalog is meant to be:
- **developer-ready** (clear contracts and constraints),
- **product-ready** (maps to user-visible actions),
- and **safety-ready** (explicit limits and guardrails).

Related docs:
- `docs/TALK_TO_YOUR_DATA_ENGINEERING_SPEC.md` (architecture and execution model)
- `docs/TALK_TO_YOUR_DATA_SECURITY_EVAL_SPEC.md` (threat model and grounding rules)

---

## 1) Operator philosophy (why this exists)

### 1.1 “Dynamic analysis” without arbitrary execution
The user experience should feel like an expert analyst is dynamically choosing methods and variables. The system can achieve this by:
- letting the LLM generate a structured plan,
- executing only allow-listed operators with validated parameters,
- and generating the narrative strictly from operator outputs (artifacts).

### 1.2 The operator contract (every operator must specify)
Each operator must define:
- **Inputs**
  - dataset reference (dataset_id + version)
  - optional column bindings (by stable column IDs)
  - optional filter scope (time window, segment filter)
- **Parameters**
  - validated and bounded
  - defaults that are deterministic
- **Outputs**
  - artifact types (metric/table/chart/report/model-run)
  - preview data rules
- **Safety constraints**
  - scan limits (rows/bytes)
  - timeouts
  - disallowed expressions and functions
- **Cost model**
  - approximate runtime and scan cost estimation
- **Determinism**
  - sampling rules and fixed seeds where randomness exists

---

## 2) Core operator families (P0)

These operators are required to deliver a truthful “Data Speaks” and grounded chat for general datasets.

### 2.1 Dataset inspection and preview

**Preview Rows**
- Purpose: show a paginated preview table to users.
- Inputs: dataset_version, optional column selection.
- Output: table artifact (preview only) + pagination metadata.
- Safety: hard cap on rows returned per page; redact PII in preview if privacy mode is on.

**Schema Snapshot**
- Purpose: return column list with inferred types and basic flags.
- Output: table artifact (column dictionary).
- Safety: never include full distinct values for high-cardinality columns.

### 2.2 Profiling operators

**Missingness Scan**
- Purpose: compute missing rate by column and missingness patterns.
- Output: table artifact + summary metric(s).
- Safety: bounded computations; supports sampling when full scan is expensive.

**Uniqueness and Cardinality**
- Purpose: compute unique counts, duplicate rates, key candidates.
- Output: table artifact + key candidate list.
- Safety: use approximate distinct counts for very large data if needed (label as estimated).

**Numeric Summary**
- Purpose: compute min/max/mean/median/std/quantiles for numeric columns.
- Output: table artifact and metric artifacts.
- Safety: avoid expensive full quantile computation if needed; label approximations.

**Categorical Summary**
- Purpose: compute top-k value counts for categorical columns.
- Output: table artifact per chosen column (or combined summary).
- Safety: cap k; cap total returned rows; avoid full cardinality dumps.

**Text Summary**
- Purpose: summarize text columns without exposing raw content.
- Output: table artifact including length stats, language hints, top tokens (optional).
- Safety: never return “top tokens” from sensitive columns if privacy mode blocks it.

### 2.3 Relationship operators

**Correlation (numeric-numeric)**
- Purpose: identify linear associations among numeric features.
- Output: correlation matrix artifact + ranked pairs artifact.
- Safety: cap number of numeric columns; optionally select top columns by variance.

**Association (categorical-numeric / categorical-categorical)**
- Purpose: identify meaningful associations beyond simple correlation.
- Output: ranked association table.
- Safety: cap category cardinality; use sampling when needed; label approximations.

### 2.4 Outlier and anomaly operators

**Outlier Scan (IQR/Z-score)**
- Purpose: surface extreme values and potential data errors.
- Output: outlier table with row references or grouped summaries.
- Safety: avoid returning raw rows by default; prefer “outlier groups” and allow drill-down with explicit user action.

**Anomaly Scan (time series)**
- Purpose: find spikes/drops and change points in time series.
- Inputs: time column + metric column.
- Output: anomaly points table + chart artifact.
- Safety: requires a confirmed time column; otherwise ask.

### 2.5 Time series operators (when a time column exists)

**Resample Aggregate**
- Purpose: compute daily/weekly/monthly aggregates.
- Output: table artifact + chart artifact.
- Safety: enforce max number of points returned; summarize large ranges.

**Rolling Window Summary**
- Purpose: compute moving averages and volatility indicators.
- Output: table + chart artifacts.
- Safety: bounded window sizes; avoid expensive multi-window brute force.

---

## 3) Action operators (P0): what users click in “Data Speaks”

These map directly to one-click actions and should be built from core compute primitives.

**Generate “Data Speaks” Overview**
- Produces: overview narrative inputs (not the narrative itself).
- Outputs: a set of artifacts the narrator uses:
  - dataset overview table/metrics
  - top insights table
  - risk/quality flags table

**Recommend Next Prompts**
- Produces: a ranked list of prompts derived from detected roles and insights.
- Output: prompt list artifact (not free-text only; structured).
- Safety: never embed raw data values into suggested prompts; use column/entity names and safe summaries.

**Generate Executive Summary**
- Produces: a report artifact that is grounded in selected evidence artifacts.
- Output: report artifact.
- Safety: must reference artifact IDs and dataset_version.

---

## 4) Transformation operators (P1)

Transformations must be versioned and reversible by switching dataset versions.

**Type Conversion**
- Inputs: column bindings + target type + parsing rules.
- Output: new dataset_version + transformation artifact (provenance).
- Safety: preview first; require user confirmation; cap rows changed display.

**Missing Value Strategy**
- Strategies: drop rows, constant fill, statistical fill, forward fill (time series), etc.
- Safety: preview impact; warn when target leakage or bias risks increase.

**Deduplication**
- Inputs: key candidate columns or user-defined key.
- Output: new dataset_version + “rows removed” metric artifact.
- Safety: require confirmation if dedup rules are uncertain.

**String Normalization**
- Examples: trimming, casing, removing invalid characters.
- Safety: preview and allow rollback.

---

## 5) Modeling operators (P2)

Modeling operators must be opt-in, compute-budgeted, and leakage-aware.

**Task Inference**
- Purpose: infer if the user’s intent implies classification/regression/forecasting/clustering.
- Output: structured “task proposal” artifact with confidence and required confirmations.

**Train Baseline Model**
- Inputs: target column, feature set, split strategy.
- Outputs:
  - model-run artifact (params, metrics, seed, dataset_version)
  - explainability artifact (feature importance)
- Safety:
  - refuse if target is not confirmed
  - warn on leakage and block obvious leakage features by default

**Evaluate by Segment**
- Inputs: segment column(s).
- Output: table artifact with performance per segment.
- Safety: cap number of segments; aggregate small groups to avoid privacy leaks.

---

## 6) Domain pack operators (optional, later)

These are “expert modules” that feel magical when they work, but should be added after the core truth loop is stable.

Examples:
- Customer analytics: RFM, cohort retention, CLV
- Product analytics: funnel analysis, session analysis
- Experiment analytics: A/B tests with effect sizes and confidence intervals
- Finance analytics: ratio analysis, risk scoring

Each domain operator must still obey the same constraints:
- grounded artifacts
- explicit assumptions
- safe previews
- clear limitations and disclaimers

---

## 7) Evaluation requirements per operator (prevent hallucinated math)

Every operator should have:
- unit tests on small synthetic datasets (where answers are known)
- performance tests on medium datasets (latency budgets)
- safety tests:
  - injection strings in text columns
  - PII redaction behavior
  - denial-of-service inputs (very large strings, high cardinality)

Acceptance threshold examples:
- Profiling operators produce stable results across runs given the same dataset_version.
- Sampling-based operators label estimates and include sample size.
- No operator can return an unbounded amount of data to the UI.

