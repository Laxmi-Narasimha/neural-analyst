# Neural Analyst — production readiness matrix

What real users will do with Neural Analyst, how each can go wrong, and where we stand.
Status is evidence-based: **Verified** means an automated test proves it; **Partial** means
it works for some inputs or lacks a test; **Gap** means known broken or missing; **Unreviewed**
means nobody has checked it yet — treat it as unknown, not as working.

Last updated: 2026-11-23 (branch `feature/talk-to-data-scope-and-ml-trust`).

## Product contract

1. Every number shown to a user is computed by deterministic code from the user's data, never written by a language model.
2. Every answer states the population it describes (filters, sheets, sampling) and links to the evidence table.
3. When the system cannot bind part of a question, it asks or refuses; it never answers a different question.
4. Model evaluations are honest: preprocessing fitted on training folds only, selection by cross-validation, a naive baseline, time-ordered tests when dates exist, leakage checks.
5. A saved model scores raw rows of a new file in a fresh process.

## A. Getting data in

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| A1 | CSV, TSV, JSON, Parquet, XLSX, legacy XLS upload | Verified | `test_chat_scope.py`, `test_retail_legacy_excel.py` |
| A2 | Multi-sheet workbooks, sheet selection, cross-sheet analysis | Verified | `test_workbook_scope.py` |
| A3 | Files above memory (1M+ rows) for Talk to Your Data aggregates | Verified | `test_chat_fiscal_disk.py` (batched DuckDB scan) |
| A4 | Non-UTF-8 encodings (Windows-1252, UTF-16), BOM, `;` delimiters (European CSV), decimal commas | Unreviewed | Add fixtures per encoding/locale |
| A5 | Malformed CSV (ragged rows, embedded newlines, quotes) | Unreviewed | Add fixtures; must fail with a row number, not a stack trace |
| A6 | Header not on row 1 (title rows, merged cells in Excel) | Gap | Header-row detection |
| A7 | Numbers stored as text (₹1,200, 15%, (500) negatives) | Partial | ML pipeline parses them (`test_ml_robustness.py`); profiling/chat typing unreviewed; accounting negatives `(500)` not parsed |
| A8 | Day-first dates (13/01/2024), mixed formats, time zones | Partial | ML pipeline verified; chat date filters unreviewed for day-first |
| A9 | Same file re-uploaded / new month's file (schema drift, renamed columns) | Gap | Investigation rerun with column-mapping prompt (roadmap 3) |
| A10 | Database connections (Postgres/MySQL/Snowflake) | Partial | SQLite connection workflow tested; others unreviewed |

## B. Understanding questions (Talk to Your Data)

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| B1 | Exact supported phrasings (sum/average X by Y, trends, counts) | Verified | `test_talk_to_data_user_flow.py` |
| B2 | Free phrasing ("which branch has the best ticket size?") | Verified (planner) | `test_llm_planner.py` with scripted model; real-model eval set still needed (B12) |
| B3 | Filters that keep across follow-ups, AND/OR, fiscal & retail calendars | Verified | `test_chat_scope.py`, `test_chat_fiscal_disk.py` |
| B4 | Natural phrasing misread as a filter ("brings **in the most money**"); restrictions the parser misses ("our western outlet") | Verified | Planner reviews detected filters and adds typed ones, validated by `validate_filters` then `bind_conditions`; rule follow-ups keep them (`test_llm_planner.py`). Applies when the rule planner can't plan the question |
| B5 | Restrictions the system cannot express ("repeat customers", "big-ticket orders") | Verified | Planner returns a question instead of an answer (`test_llm_planner.py`) |
| B6 | Hallucinated columns, wrong column types, sensitive columns in plans | Verified | `validate_plan` tests |
| B7 | Derived metrics: ratios, per-unit, margin %, share of total, growth % | Gap | Needs a typed expression operator (safe formula over columns), not more templates |
| B8 | Period-over-period ("vs last month", YoY) and contribution to change | Gap | New operator; the flagship analyst question |
| B9 | Top-N with ties, bottom-N, "top 10 customers by revenue" | Partial | `segment_summary` ranks; N and ties not parameterised end to end |
| B10 | Multi-dataset questions (joins, lookups) | Gap | Out of scope until single-dataset is solid |
| B11 | Questions in Hindi/other languages, typos | Partial | Planner can handle via LLM; rule layer English-only; untested |
| B12 | Real-model accuracy | Gap | Build an eval set (questions × datasets × independently computed answers) and score each model/prompt change against it |

## C. Calculation correctness

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| C1 | Correct number for the asked population | Verified for supported ops | Exact-value tests across formats |
| C2 | Sampling disclosed when a step uses a sample | Verified | Chat prepends sample notes |
| C3 | Nulls, duplicates, zero matches | Partial | Zero-match verified; duplicate-row semantics not surfaced to user |
| C4 | Mixed currencies / units in one column | Gap | Detect and ask |
| C5 | Two-sided consistency: batched DuckDB path vs pandas path give same numbers | Partial | Both tested separately; add a differential test |

## D. Data science

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| D1 | Leakage-safe preprocessing, CV selection, baseline with 95% interval | Verified | `test_ml_pipeline_trust.py` |
| D2 | Time-ordered evaluation when dates exist; day-first dates | Verified | `test_ml_pipeline_trust.py`, `test_ml_robustness.py` |
| D3 | Target leakage: copies, deterministic encodings, post-outcome dates | Verified | Same |
| D4 | 30 hostile datasets (tiny, wide, imbalanced, text, JSON cells, inf, unicode, duplicates…) | Verified | `test_ml_robustness.py` |
| D5 | Train → save → restart → predict raw rows via API | Verified | `test_api_train_then_predict_with_returned_name_and_version` |
| D6 | Imbalanced targets: metric choice (PR-AUC), threshold selection, class weights | Partial | Metrics reported; selection still ROC AUC; no threshold tuning |
| D7 | Probability calibration | Gap | Reliability curve + calibrated model option |
| D8 | Error analysis by segment ("where does the model fail?") | Gap | Per-segment held-out metrics |
| D9 | "Prediction moment" chosen by the user, not inferred | Gap | Explicit cutoff column in plan; inference is a heuristic today |
| D10 | Batch scoring of next week's file with drift check | Gap | Upload → score → compare feature distributions to training |
| D11 | Explaining a single prediction | Gap | `/ml/explain` returns global importance only for bundled models |
| D12 | Async AutoML jobs honour `time_column` / `split` overrides | Gap | Sync path does; job payload doesn't carry them |
| D13 | Deep learning / NLP beyond TF-IDF | Gap by design | Only where it beats tabular baselines on real tasks; keep optional |
| D14 | Pickle loading of model artifacts | Partial | Only service-written files are loaded; signing artifacts recommended before multi-tenant hosting |

## E. Language-model layer

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| E1 | No key / provider outage / rate limit | Verified | Fails closed to rule planner and a clear reply |
| E2 | Prompt injection in data values or column names | Partial | Values treated as data in prompt; sensitive values excluded; model output still constrained by validator. Add adversarial eval cases |
| E3 | Data privacy: what leaves the server | Partial | Column names, types and ≤6 example values per non-sensitive column. Needs a per-workspace "never send examples" switch and documentation |
| E4 | Cost and latency per question | Unreviewed | Log tokens/latency per plan; cache plans per (dataset version, question) |
| E5 | Model choice | Partial | `LLM_PLANNER_MODEL` / `LLM_MODEL`; local `.env` currently points at a small local model — use an OpenAI model for planning quality |

## F. Security and multi-tenancy — Unreviewed in this pass

Auth and ownership scoping on every dataset/artifact/model route, upload size limits, SQL operator safety (`sql_safety` exists), rate limiting, model registry isolation per owner (registry folder is currently shared — **Gap**), secrets handling.

## G. Operations — Unreviewed in this pass

Background job retries and timeouts, memory limits for training, concurrency on the same conversation (409 lock exists), observability (structured logs exist; metrics unreviewed), database migrations, backups.

## H. Frontend — Unreviewed in this pass

Clarification UI for planner questions, display of "Interpreted as", evidence tables, training report (baseline, split, leakage exclusions), error states.

## Build order

1. ~~**B4** LLM-proposed filters as validated conditions~~ — done.
2. **B12** Real-model evaluation set and scorer; run it against the configured OpenAI model.
3. **B7/B8** Derived metrics and period-over-period with contribution — the flagship analyst answers.
4. **F** Per-owner model registry and an ownership audit of all routes.
5. **D6–D10** Imbalance handling, calibration, segment error analysis, explicit prediction moment, batch scoring with drift.
6. **A4–A6** Encoding/locale/header detection fixtures.
