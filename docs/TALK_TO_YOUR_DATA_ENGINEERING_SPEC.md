# Neural Analyst — Engineering Specification (Dataset → Compute → Evidence → Narrative)

**Date:** 2026‑02‑03  
**Purpose:** Implementation-ready architecture and system design for a safe, scalable “Talk to your data” product.  
**Rule:** No code snippets. Describe interfaces and behavior precisely.

This document answers: “How do we engineer the backend and supporting systems so the UI can deliver the promised experience reliably and safely?”

---

## 1) Design Goals

1. **Grounding**: responses and insights must be backed by real computations on the dataset.  
2. **Dynamic behavior**: infer schema/roles/targets, but ask clarifying questions when needed.  
3. **Safety**: no arbitrary code execution; constrain operations and sanitize untrusted inputs.  
4. **Performance**: handle large datasets with sampling + out-of-core compute; keep UI responsive.  
5. **Reproducibility**: every analysis is an auditable recipe with stored artifacts.  
6. **Extensibility**: easy to add new analysis operators and domain packs.

---

## 2) Current Backend Integration Gaps (Engineering View)

This is the engineering-specific view of why the current system cannot deliver end-to-end grounding.

### 2.1 Dataset loading is not centralized
There is no single authoritative module that loads a dataset by `dataset_id` and returns:
- schema + profile
- safe query execution
- sampling helpers

Result: different endpoints invent their own “dataset loaders”, including placeholders.

### 2.2 The orchestrator tool layer is stubbed
The orchestrator agent defines tools but many tool implementations return placeholders instead of performing computations. This blocks tool-driven grounding.

### 2.3 Analytics endpoints are not wired to real datasets
Several analytics routes use a demo dataset generator rather than retrieving the actual uploaded dataset.

### 2.4 Persisted results model is incomplete
Analyses are created and tasks are queued, but there is not a complete model for:
- analysis runs
- steps executed
- artifacts produced
- progress
- final results and downloads

---

## 3) Target Architecture (Conceptual Components)

### 3.1 Components (logical)
1. **Dataset Registry**
   - resolves dataset_id → storage location + schema + version
2. **Compute Engine**
   - runs safe operations (SQL/DuckDB + built-in operators)
   - produces “evidence artifacts”
3. **Planner**
   - converts user intent + dataset context → structured plan
4. **Executor**
   - validates plan → runs via compute engine → produces artifacts
5. **Narrator**
   - converts artifacts into a human-readable story and next actions
6. **Job System**
   - executes heavy work asynchronously with progress updates
7. **Artifact Store**
   - stores tables/charts/reports and links them to analyses

### 3.2 Physical deployment (typical)
- API service (FastAPI) for interactive requests
- Worker service for heavy jobs
- Postgres for metadata and artifacts index
- Redis (or similar) for queues and caching
- Object storage for uploads and generated artifacts (optional for self-host; required for multi-instance)

---

## 4) Data Model (Required Entities)

### 4.1 User and Auth entities
Minimum:
- User
- API Key
- Session/refresh tokens (if needed)

Required behavior:
- stable user identity across requests
- per-user dataset access control

### 4.2 Dataset entities
Recommended entities:
- Dataset (logical dataset)
- DatasetVersion (content hash + timestamps)
- DatasetAsset (file/blob reference; format; storage path; size)
- DatasetProfile (profile snapshot; computed at a version)
- DatasetSchema (schema snapshot; computed at a version)

Required behavior:
- dataset_id must be stable and used everywhere
- multiple versions allow transformations without destroying the original

### 4.3 Analysis entities
Recommended entities:
- Analysis (user-defined “analysis project”)
- AnalysisRun (a specific execution instance)
- AnalysisStep (validated plan step + status + timings)
- Artifact (tables, charts, reports, logs)
- Metric (named computed values, useful for dashboards)

Required behavior:
- every run produces artifacts and can be reproduced
- every artifact has provenance: dataset_version + step_id

### 4.4 Conversation entities (chat)
Recommended entities:
- Conversation
- Message
- MessageArtifactLink (message references one or more artifacts)

Required behavior:
- chat messages can reference computed evidence
- chat can resume with context reliably

---

## 5) Storage Architecture (Uploads and Artifacts)

### 5.1 Upload storage
For self-host:
- local disk is acceptable, but must have:
  - per-user directory isolation
  - cleanup policies

For multi-instance deployments:
- uploads must be stored in object storage (S3-compatible):
  - avoids losing data on container restart
  - allows horizontal scaling

### 5.2 Artifact storage
Artifacts should not be stored only inside the database as large blobs. Recommended:
- store artifact content in object storage
- store artifact metadata + pointers in Postgres

Artifact types to support:
- tables (parquet, csv, or JSON-with-pagination)
- charts (spec + derived image optionally)
- reports (markdown/html/pdf)
- model artifacts (serialized model + metadata)

---

## 6) Compute Engine (Safe Operators)

### 6.1 Why a compute engine
“Talk to your data” cannot be credible unless:
- computations are consistent across chat and one-click actions
- operators are validated and safe
- results are structured artifacts

### 6.2 Operator categories
Minimum operator set:
- Profiling operators (schema, types, missingness, uniqueness, histograms)
- Query operators (filter, group, aggregate, pivot, join)
- Statistical operators (t-tests, correlations, confidence intervals)
- Time operators (resample, rolling windows, decomposition)
- Modeling operators (baseline train/eval; feature importance)
- Visualization operators (chart selection + rendering spec)

### 6.3 Out-of-core strategy
For large datasets:
- DuckDB should be the default for scans and group-bys (works on file-backed tables).
- Polars lazy frames can be used for transformations and fast columnar operations.

### 6.4 Determinism and caching
The compute engine should:
- cache dataset profiles and common aggregations
- reuse results across sessions when dataset_version is unchanged

---

## 7) Structured “Analysis Plan” (The Dynamic Engine Without Unsafe Code)

### 7.1 Plan vs Code
Your “dynamic code” requirement should be implemented as:
- dynamic plan generation (LLM + heuristics),
- deterministic execution (safe operators),
- narrative generation (LLM uses artifacts as sources).

### 7.2 Plan sections (required)
Every plan must include:
- Goal and intent classification (EDA / stats / ML / time series / etc.)
- Dataset scope (dataset_id + version)
- Column bindings (selected columns, inferred roles, target if needed)
- Steps list:
  - operator type
  - parameters
  - expected outputs
  - validation checks
- Outputs list (artifacts to produce)
- Safety constraints (limits on row scans, timeouts, memory)

### 7.3 Validation rules for plans
Before executing:
- column existence check
- type compatibility check
- cost estimation (scan size, cardinality)
- disallow unsafe operations (filesystem, network, arbitrary code)

### 7.4 Evidence artifact contract
Every step produces:
- artifact metadata (type, schema, row count, preview)
- a pointer to full data (if large)
- provenance fields (dataset_version, step_id)

---

## 8) Planner (Intent → Plan)

### 8.1 Intent classification
Given a user prompt, classify:
- ask vs do (question vs action request)
- analysis domain (EDA, stats, ML, time series, text)
- required ambiguity resolution (metric definitions, time window, target)

### 8.2 Schema-aware planning
Planning must use:
- dataset schema and profile
- inferred column roles and candidate targets
- previously agreed definitions in conversation memory

### 8.3 Clarifying questions engine
If required inputs are missing:
- ask only the highest leverage questions
- include suggested answer choices (for UI)
- record the answers as structured context

---

## 9) Executor (Plan → Artifacts)

### 9.1 Execution model
Two execution modes:
- interactive mode (fast, small scans, low latency)
- job mode (async, heavy scans, training)

### 9.2 Progress reporting
Executor must provide:
- step-level progress
- partial artifacts when possible (sample first, then full)
- failure reason with actionable remediation

### 9.3 Cancellation and retries
Required:
- cancel running jobs
- retry idempotent steps
- persist failure state for debugging

---

## 10) Narrator (Artifacts → Story)

### 10.1 Narrative constraints
The narrator must:
- cite artifacts internally (not necessarily as UI “citations”, but as evidence references)
- avoid claiming numbers that are not present in artifacts
- label assumptions

### 10.2 Output structure
Narrative should include:
- top findings (ranked)
- risks and caveats
- recommended next questions
- recommended next actions

---

## 11) UI Integration Contracts (Backend responsibilities)

### 11.1 Unified response shapes
The backend should consistently return:
- success flag/status
- data payload
- pagination metadata when listing
- error objects with user-safe messaging

### 11.2 Streaming
For long responses:
- provide server-sent events (SSE) or websocket streams
- allow UI to show progress and partial results

---

## 12) Mapping Existing Code to the Target Architecture

The repo already contains useful modules. The main engineering task is to connect them correctly.

### 12.1 Modules to reuse and integrate
- File parsing and profiling: `ai-data-analyst/backend/app/services/data_ingestion.py`
- Report generation: `ai-data-analyst/backend/app/services/report_generator.py`
- SQL planning concepts: `ai-data-analyst/backend/app/services/query_optimizer.py`
- Connectors scaffolding: `ai-data-analyst/backend/app/services/data_connectors.py`
- Many analysis engines: `ai-data-analyst/backend/app/ml/*`

### 12.2 Modules that must be rewritten or upgraded
- Orchestrator tool implementations that currently return placeholders
- Analytics routes that currently use mock datasets
- Dataset/user linkage (stable user id and auth integration)
- Adequacy validator integration to use real uploads/datasets instead of path strings

---

## 13) Engineering Roadmap (P0/P1/P2)

### P0 (Core grounding)
- Central dataset registry + loader by dataset_id
- Compute engine with safe operators (DuckDB first)
- Tool-driven chat enforcement (no numeric claims without artifacts)
- Data Speaks workflow (profile → insights → narrative)
- Contract alignment between UI and API

### P1 (Delight and reproducibility)
- Transformations pipeline (versioned datasets)
- Report generation tied to analysis runs
- Artifact browser and re-run
- Quality/adequacy integrated end-to-end

### P2 (Data scientist completeness)
- Target inference + confirmation flow
- AutoML pipelines with leakage checks
- Model registry + prediction endpoints and UI
- Time series workflows

For security and evaluation requirements, see:
- `docs/TALK_TO_YOUR_DATA_SECURITY_EVAL_SPEC.md`

---

## 14) Concrete API & Contract Requirements (what the frontend can rely on)

This section defines what the backend must expose so the UI can be fully grounded and reproducible.

### 14.1 Dataset APIs (minimum)

**A) Upload dataset**
- Input: multipart upload with dataset name + file + optional description/tags.
- Output: dataset_id, initial status, and an ingest job reference (if processing is async).
- Contract requirements:
  - sanitize filename and store as an internal asset reference
  - persist owner_id and enforce access control
  - return a stable dataset_id that works across all other endpoints

**B) Process/profile dataset**
- If profiling is async, provide:
  - a job endpoint to poll
  - a progress stream (optional)
- Output should include:
  - schema snapshot (columns and types)
  - profile snapshot (basic stats)
  - readiness flags (processing/ready/error)

**C) Dataset preview**
- Must support pagination and column selection.
- Must support “sample mode” for large datasets.

**D) Dataset versioning**
- Provide endpoints to:
  - list dataset versions
  - activate a version for a session
  - create a new version from transformations

### 14.2 Data Speaks session APIs (minimum)

**A) Create Data Speaks session**
- Input: dataset_id (+ optional user-provided goal/context).
- Output: session_id, status, progress.
- Must kick off:
  - profile and role inference (if not already cached)
  - insight generation
  - narrative generation tied to evidence artifacts

**B) Get session summary**
- Output:
  - narrative header (computed)
  - ordered list of top insights with evidence links
  - list of evidence panel artifacts
  - suggested prompts and one-click actions

**C) Get session artifacts**
- List artifacts with:
  - artifact_id, type, created_at
  - provenance (dataset_version, operator/step)
  - preview data (small)
  - pointers for download

### 14.3 Chat APIs (grounded chat)

**A) Post a message**
- Input: conversation_id (optional), dataset_id (required for data questions), message text.
- Output: assistant response message plus:
  - message_id
  - referenced artifact ids
  - clarification questions (when needed)

**B) Clarification loop**
- If missing required definitions, return a “clarification required” response with:
  - question_id
  - options (UI-selectable)
  - explanation of why it’s needed

**C) Evidence-first response contract**
- Any response containing numeric claims must include:
  - artifact references (metric/table/chart)
  - the scope (filters, time window, sampling mode)

### 14.4 Artifact APIs

Artifacts are first-class objects. Provide:
- list artifacts by session_id, conversation_id, dataset_version
- fetch artifact preview
- fetch artifact full content (download)
- render-ready chart specs (and optionally images)

### 14.5 Health and readiness APIs
Expose:
- health endpoint (process is alive)
- readiness endpoint (dependencies reachable: DB, queue, storage)
- version endpoint (build info, git SHA)

---

## 15) Dataset Lifecycle (ingestion → profiling → indexing → versioning)

This is the single most important “backbone” missing today. Without this, every other feature becomes fragile.

### 15.1 Ingestion phases (recommended)

**Phase 1: Upload acceptance**
- Validate file type (extension + signature sniffing).
- Enforce size limits and timeouts.
- Store raw blob in upload storage (local disk for dev, object storage for hosted).
- Create Dataset and DatasetAsset records.
- Return dataset_id immediately.

**Phase 2: Parsing and normalization**
- Detect encoding and delimiter.
- Normalize column names:
  - preserve original_name
  - create a stable internal identifier for each column
- Parse into a columnar format (Parquet recommended) for compute efficiency.

**Phase 3: Profiling**
- Compute:
  - row_count, column_count
  - type inference per column (with confidence)
  - basic stats and distributions
  - missingness and uniqueness
  - value count sketches for categoricals
  - text stats (length distribution) for text columns
- Persist DatasetProfile and DatasetSchema snapshots.

**Phase 4: Indexing (optional but valuable)**
- Build:
  - lightweight search index over column names/descriptions
  - embedding index over column descriptions and sample values (only if privacy permits)

### 15.2 Versioning model (required for transformations)
Rules:
- A dataset can have multiple versions.
- A version is immutable once created.
- Transformations always create a new version.
- All analysis artifacts reference a dataset_version.

### 15.3 Large data strategy
For “huge data”:
- Prefer file-backed compute (DuckDB) over loading into memory.
- Maintain a sampling strategy:
  - quick “preview sample”
  - stratified samples for categorical segments
  - time-window samples for time series
- Always label sampling in UI and narrative.

---

## 16) Column Role Inference and Semantic Typing (dynamic behavior without hard-coded schemas)

This is how the system can adapt to arbitrary datasets and still ask fewer questions.

### 16.1 Roles the system should infer
Minimum roles:
- entity key / ID
- time index
- target candidate (label)
- metric/value columns
- categorical segment columns
- free-form text columns
- geographic columns (optional)

### 16.2 Signals for inference (deterministic)
Use a weighted scoring approach that combines:
- column name patterns (e.g., contains “id”, “date”, “target”, “label”, “price”)
- value patterns:
  - UUID-like strings
  - email/phone-like patterns (PII)
  - monotonic increasing sequences (IDs)
  - date parse success rate
- distribution signals:
  - uniqueness ratio (high uniqueness suggests IDs)
  - cardinality vs row count
  - entropy for categoricals
- relational signals:
  - duplicate patterns across columns (composite keys)
  - temporal continuity and gaps (time)

### 16.3 Confidence and ambiguity handling
The system must output:
- a confidence score per inferred role
- a human-readable rationale (“Why we think this is the time column”)
- an “override” UI control

Rule:
- If confidence is high, proceed automatically.
- If confidence is medium, proceed but label as assumption and provide override.
- If confidence is low, ask a clarification question.

### 16.4 Target inference (for modeling workflows)
Target candidate scoring should use:
- name patterns (“churn”, “label”, “converted”, “outcome”)
- type constraints (classification targets often low-cardinality; regression targets numeric)
- leakage heuristics:
  - timestamps after the predicted event
  - columns that directly encode the outcome
  - “future info” indicators

The system must not start training without explicit user confirmation of the target when ambiguous.

---

## 17) Insight Generation Pipeline (compute-first “analyst instincts”)

### 17.1 Candidate insight generators (minimum)
The compute engine should generate candidate insights such as:
- missingness clusters (“these fields are missing together”)
- duplicate patterns and likely keys
- outliers and anomalies (global and per-segment)
- strongest associations and correlations (with caveats)
- biggest segment differences by a chosen metric
- time trends, change points, and seasonal signals
- rare categories and long-tail warnings
- PII and sensitive column flags
- potential leakage signals (for modeling)

### 17.2 Ranking and selection (“what should the user see first?”)
Rank insights by:
- confidence (statistical strength, stability across samples)
- impact (size of effect, magnitude)
- actionability (suggested next step exists)
- relevance to user goal (if provided)
- novelty (avoid repeating similar insights)

### 17.3 Artifact-first insight objects
Every insight must reference:
- a primary evidence artifact
- optional supporting artifacts
- a provenance record (dataset_version, operator, parameters)

The narrator must not introduce new facts not present in these artifacts.

---

## 18) Safe Operator Catalog (the “dynamic code” substitute)

### 18.1 Why operators matter
Operators are the boundary between “LLM creativity” and “truth”. If operators are well-defined and safe, the system can feel unlimited without being unreliable.

### 18.2 Operator contract (generic)
Each operator definition must include:
- name and category
- required inputs (dataset_id/version, column bindings)
- parameters (validated)
- outputs (artifact types)
- cost model (rows scanned, expected runtime)
- safety limits (max rows, disallowed expressions)
- determinism requirements (seed, sampling rules)

### 18.3 Minimum operator set (P0)
Profiling:
- schema inference
- missingness scan
- uniqueness scan
- numeric distribution summary
- categorical value counts summary

Query primitives:
- filter
- group-by aggregation
- top-k
- join (later; initially avoid multi-table unless necessary)

Statistics:
- correlation matrix (numeric)
- association measures for categorical vs numeric
- simple hypothesis tests (with guardrails)

Outliers:
- z-score/IQR outlier scan
- isolation forest (async, optional)

Time:
- resample aggregation
- rolling aggregates
- seasonal decomposition (async, optional)

Visualization:
- chart recommendation (deterministic based on data types)
- chart rendering spec generation

### 18.4 Advanced operators (P1/P2)
- driver analysis (feature importance for a chosen target/metric)
- segmentation/clustering with quality measures
- cohort analysis and retention
- A/B test evaluation
- AutoML with leakage checks

---

## 19) Execution and Job System (how to handle heavy compute and concurrency)

### 19.1 Execution modes
Interactive mode:
- strict limits (row scans, timeouts)
- sampling-first
- fast artifacts only

Job mode:
- asynchronous execution for heavy operations
- progress updates and cancellation
- retries for transient failures

### 19.2 Job model requirements
Each job must have:
- job_id
- owner_id
- dataset_version
- plan reference
- status (queued/running/succeeded/failed/canceled)
- progress (step index + percent)
- logs and failure reason (safe for users)

### 19.3 Cancellation semantics
Cancellation should:
- stop new steps from starting
- attempt to stop current step if safe
- persist partial artifacts with “partial” flags

### 19.4 Concurrency controls (especially for hosted)
Implement:
- per-user max concurrent jobs
- global worker concurrency
- queue prioritization (interactive sessions first)

---

## 20) Artifact Model, Formats, and Rendering Contracts

### 20.1 Artifact types
Minimum artifact types:
- Metric (single value + definition + confidence)
- Table (schema + row_count + preview + pointer to full)
- Chart (spec + pointer to underlying table)
- Report (markdown/HTML + artifact references)
- ModelRun (params + metrics + explainability + pointer to model file)

### 20.2 Artifact preview rules
- Never return unbounded data to the UI.
- Always provide a small preview (first N rows) with pagination for tables.
- For charts, provide:
  - chart spec
  - data pointer (artifact_id of underlying table)

### 20.3 Provenance requirements
Every artifact must store:
- dataset_version
- operator name
- operator parameters (validated)
- created_at
- parent artifacts (if derived)

This enables reproducibility and debugging.

---

## 21) LLM Integration Patterns (how to keep it grounded, cheap, and safe)

### 21.1 Separate “planning” from “narration”
Use different prompts and constraints:
- Planner prompt:
  - produces a structured plan
  - prohibits numeric claims
  - focuses on selecting operators and bindings
- Narrator prompt:
  - consumes artifact summaries
  - must not invent values
  - outputs human-readable story and next actions

### 21.2 Token minimization
Never send:
- raw dataset rows (except tiny samples with explicit privacy mode)
- full tables
- full logs

Instead send:
- profile summaries
- aggregate results
- artifact previews

### 21.3 Conversation memory (what to store)
Store structured memory items:
- pinned metric definitions
- chosen target and time column
- last executed plan steps and artifacts
- user goal and constraints

Avoid storing:
- raw user data in memory
- secrets

---

## 22) Observability and Debuggability (make it maintainable)

Minimum logs:
- request_id correlation across API, compute, and jobs
- dataset_id and dataset_version in log context
- operator execution timings and row scan counts
- LLM usage and cost estimates (without content leakage)

Minimum metrics:
- time-to-first-evidence
- job latency distribution
- error rate by operator
- grounding gate violations

Debug UX:
- “show how computed” panel for each artifact
- “download run bundle” for reproducing issues locally

---

## 23) Migration plan for this repo (how to evolve without rewriting everything)

### 23.1 Replace placeholders with the real compute engine
Current placeholders to address first:
- orchestrator tool methods that return fake results
- analytics routes that load demo data
- quality UI that sends file paths instead of dataset references

### 23.2 Reuse existing engines as operator implementations
Many modules under `ai-data-analyst/backend/app/ml/*` can be wrapped as operators once:
- dataset loading is centralized
- operator inputs/outputs are standardized
- artifacts and provenance are persisted

### 23.3 Keep the UI stable by standardizing contracts
Once the API contracts are stable:
- UI can be built confidently
- backend can evolve internally without breaking UX

The concrete P0/P1/P2 task list (with file mapping) is in:
- `docs/TALK_TO_YOUR_DATA_BACKLOG.md`

The operator allow-list and per-operator evaluation expectations are in:
- `docs/TALK_TO_YOUR_DATA_OPERATOR_CATALOG.md`
