# Neural Analyst — production readiness matrix

What real users will do with Neural Analyst, how each can go wrong, and where we stand.
Status is evidence-based: **Verified** means an automated test proves it; **Partial** means
it works for some inputs or lacks a test; **Gap** means known broken or missing; **Unreviewed**
means nobody has checked it yet — treat it as unknown, not as working.

Last updated: 2026-11-24 (branch `feature/talk-to-data-scope-and-ml-trust`).

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
| B2 | Free phrasing ("which branch has the best ticket size?") | Verified | Real-model eval (B12) plus `test_llm_planner.py` |
| B3 | Filters that keep across follow-ups, AND/OR, fiscal & retail calendars | Verified | `test_chat_scope.py`, `test_chat_fiscal_disk.py` |
| B4 | Natural phrasing misread as a filter ("brings **in the most money**", "the West Branch"); restrictions the parser misses ("our western outlet") | Verified | Any rule-stage failure without a pending choice goes to the planner (parse, verify, bind, batched bind). Kept detected filters must match real values (`test_planner_value_binding.py`) |
| B5 | Restrictions the system cannot express ("repeat customers", "big-ticket orders") | Verified | Planner returns a question instead of an answer (`test_llm_planner.py`) |
| B6 | Hallucinated columns, wrong column types, sensitive columns in plans | Verified | `validate_plan` tests |
| B7 | Derived metrics: ratios, per-unit, margin %, share of total, growth % | Verified | Typed operators instead of free formulas. `ratio` (sum/sum, optional subtraction, % form; only complete rows), `share_of_total`, `rate`, `count`/`distinct_count`. Pandas and DuckDB paths share the finishing code and are parity-tested (`test_derived_metrics.py`). Arbitrary formulas across 3+ columns are still out of scope |
| B8 | Period-over-period ("vs last month", YoY) and contribution to change | Verified | `period_compare` returns both values, the change and the % change, and with group_by each group's change and its share of the total change. A period still in progress is compared over equal days. A finished period keeps the dates asked for and says when the data ends early. Without the planner, comparison questions are refused, never answered for one period |
| B9 | Top-N with ties, bottom-N, "top 10 customers by revenue" | Partial | `segment_summary` ranks; N and ties not parameterised end to end |
| B10 | Multi-dataset questions (joins, lookups) | Gap | Out of scope until single-dataset is solid |
| B11 | Questions in Hindi/other languages, typos | Partial | Hinglish and typo cases pass the real-model eval; filter values are bound to real values with disclosed typo fixes. Only one non-English case so far |
| B12 | Real-model accuracy | Verified | `evals/talk_eval` covers 42 cases on 3 datasets, with answers computed independently in pandas. See the results table below |
| B13 | Filter value not in the data ("electronics", "apparal", "Toys") | Verified | Values bound to the column's complete value list; typo fixes disclosed; unknown values sent back for repair |
| B14 | Exclusion of a value that never occurs (`Returned ≠ "orders"`) silently matching every row | Verified | Refused in `bind_conditions` (pandas and batched paths), then recovered by the planner |

## C. Calculation correctness

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| C1 | Correct number for the asked population | Verified for supported ops | Exact-value tests across formats |
| C2 | Sampling disclosed when a step uses a sample | Verified | Chat prepends sample notes |
| C3 | Nulls, duplicates, zero matches | Partial | Zero-match verified; rates leave empty values out of the denominator and say how many; duplicate-row semantics not surfaced to user |
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
| D6 | Imbalanced targets: metric choice, threshold, class weights | Verified | Rarer class < 20% (two classes) or < 10% (more): selection on average precision / macro F1. Two-class: cut-off maximising F1 on out-of-fold training rows, stored with the model and used by `predict`; precision/recall at other cut-offs reported. On the 10%-fraud fixture held-out recall goes 19% → 46%. Class weights opt-in (`balance_class_weights`) because they distort probabilities. `test_ml_evaluation_depth.py` |
| D7 | Probability calibration | Verified (two classes) | Sigmoid/isotonic map learned on out-of-fold probabilities, kept only if it lowers the Brier score under its own cross-validation; held-out reliability bins, Brier and ECE before/after. Multiclass: reliability of the predicted class's confidence only, no map |
| D8 | Error analysis by segment | Verified | Held-out error rate (or MAE) per category / numeric quartile, significance by Wilson interval (or z-test), tiny segments and identifiers skipped; the weakest significant segment is named in the chat answer |
| D9 | Prediction moment chosen by the user | Verified | `time_column` + `test_start` (backtest date) on the API, jobs and the chat planner; an inferred time column is labelled as an assumption the user should confirm |
| D10 | Batch scoring of a new file with drift check | Verified | `POST /ml/models/{name}/score`: every row (cap `ML_MAX_SCORING_ROWS`, refused above it, never sampled) → predictions table artifact, PSI per input vs a training profile stored in the model (no rows kept), prediction-mix shift, and realised accuracy/MAE when the file has the outcome. Gap: files above the cap need chunked scoring; no chat operator yet |
| D11 | Explaining a single prediction | Gap | `/ml/explain` returns global importance only for bundled models |
| D12 | Async AutoML jobs honour `time_column` / `split` overrides | Verified | Local and Celery jobs carry `time_column`, `split`, `test_start`, `balance_class_weights` |
| D13 | Deep learning / NLP beyond TF-IDF | Gap by design | Only where it beats tabular baselines on real tasks; keep optional |
| D14 | Pickle loading of model artifacts | Partial | Only files inside the owner's own registry folder are loaded (a tampered entry pointing elsewhere is refused, `test_tenant_isolation.py`); signing artifacts still recommended |

## E. Language-model layer

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| E1 | No key / provider outage / rate limit | Verified | Fails closed to rule planner and a clear reply |
| E2 | Prompt injection in data values or column names | Partial | Values treated as data in prompt; sensitive values excluded; model output still constrained by validator. Add adversarial eval cases |
| E3 | Data privacy: what leaves the server | Partial | Column names, types, date ranges, and for low-cardinality non-sensitive columns their complete value list (≤50 values; otherwise ≤6 examples). Never row data or results. **If the OpenAI org has data sharing on (the free-token program), OpenAI may train on this**: use an org without sharing for customer data. Still needs a per-workspace "never send values" switch |
| E4 | Cost and latency per question | Unreviewed | Log tokens/latency per plan; cache plans per (dataset version, question) |
| E5 | Model choice | Verified | Chosen from the eval (results below). `OPENAI_*` settings now read `.env`; before this fix a key placed in `.env` was silently ignored |
| E6 | What the user is told the system did | Verified | "Interpreted your question as" is written by code from the validated plan, never in the model's words |

## F. Security and multi-tenancy

Audited 2026-11-24: every route in `app/api/routes/` read for auth, owner scoping (followed into the
services' WHERE clauses), file access built from input, and cross-tenant actions. Each fix below has
a test in `tests/test_tenant_isolation.py` that reproduces the attack.

| # | Use case / failure mode | Status | Evidence / next step |
|---|---|---|---|
| F1 | One user lists, loads, scores with or promotes another user's model | Verified | Registry is one folder per owner (`get_model_registry(owner_id)`); another owner's model is a 404 on every `/ml` route. Was: one shared folder for everyone |
| F2 | A column header such as `../../x` chooses where a model is written | Verified | Folder names are sanitised and hashed; artifact paths must resolve inside the registry |
| F3 | Two trainings finish at once and one version is lost or overwritten | Verified | File lock + reload before every registry write; 8 concurrent writers give versions 1–8 |
| F4 | Models in the old shared folder | Verified | Moved to the owner of their dataset at start-up (idempotent); models whose dataset is gone stay in the old folder, which nothing serves |
| F5 | Dataset SQL reads other files on the server (other users' uploads, `.env`, the platform database) | Verified | **Was exploitable** (`glob`, `read_text` passed the pattern check). Now the query runs on a read-only copy of the dataset with DuckDB external access off and settings locked. First query of a dataset version builds the copy (disk + time); purge deletes it |
| F6 | A saved connection opens the platform's own database or internal hosts (cloud metadata, LAN) | Verified | **Was exploitable** (sqlite connector pointed at `dev.db`). Platform database always refused; with sign-in, private/loopback/link-local hosts refused unless `CONNECTIONS_ALLOW_PRIVATE_HOSTS=true` and SQLite only under `CONNECTIONS_SQLITE_ROOT`. Checked on save and before every use. Residual: DNS can change between the check and the connect (pin the resolved IP to close it) |
| F7 | Production or SaaS server started without sign-in (everyone is the same superuser) | Verified | Start-up refuses unless `AUTH_MODE=jwt` |
| F8 | Any analyst deletes every tenant's unreferenced files via storage GC or clears the shared cache | Verified | Maintenance routes are admin-only |
| F9 | API keys with more rights than their creator, or revoked by another user | Verified | Requested permissions must be held by the caller; revoke checks the owner. Gap: keys are in memory and not yet usable for sign-in |
| F10 | Public report links | Verified | 256-bit tokens stored hashed; expiry and revocation enforced; now also stop working when the owner's account is deactivated or deleted |
| F11 | Replayed billing webhooks; checkout return URL sends users to another site | Verified | 5-minute signature window; return URLs must be on `frontend_url` |
| F12 | Owner scoping on dataset, analysis, artifact, chat, job, connection, share, quality and dashboard routes | Partial | Verified by code audit (all scoped); cross-user tests exist for ML, SQL, connections, maintenance and keys only. Add a two-user test per router |
| F13 | Deleting a dataset leaves derived copies | Verified | Purge also deletes models trained on it and its SQL copy |
| F14 | Responses reveal server file paths | Partial | Removed from analysis export and artifact manifest responses; no test |
| F15 | Global metrics visible to every tenant | Verified | Process-wide counters only for admins |
| F16 | Signing secret exposure | Action | Any deployment that ran the old dataset SQL code could have had `.env` read: rotate `SECURITY_SECRET_KEY` and connection encryption keys there |
| F17 | Upload size limits, rate limiting per user, secrets at rest | Unreviewed | |

## G. Operations — Unreviewed in this pass

Background job retries and timeouts, memory limits for training, concurrency on the same conversation (409 lock exists), observability (structured logs exist; metrics unreviewed), database migrations, backups.

## H. Frontend — Unreviewed in this pass

Clarification UI for planner questions, display of "Interpreted as", evidence tables, training report (baseline, split, leakage exclusions), error states.

## Real-model eval results

53 cases on 4 datasets, run end to end through `/api/v1/chat` (2026-11-24). Run with
`python -m evals.talk_eval.run --models <model>` from `ai-data-analyst/backend` (see `evals/talk_eval/README.md`).

| Planner model | Pass | Median latency | Notes |
|---|---|---|---|
| **gpt-5.4-mini** (default) | **100%**: 106/106 over 2 runs of all 53 | 1.7 s | Plus 42/42 on 3 earlier runs of the original set |
| gpt-4.1-mini | 98% (52/53), then 100% on the date cases after the dropped-date guard | 1.9 s | Good fallback |
| gpt-5-mini | 100% on the original 42 (2 runs, low reasoning effort) | 4.0 s | 9 s and 88% at default reasoning effort |
| gpt-5.4-nano | 81% on the original 42 | 1.8 s | Not accurate enough for planning |

History:
- **First pass (42 cases): 64% → 100%.** Fixes:
  - rule-stage failures now recover through the planner;
  - the model sees complete values, date ranges and identifier flags;
  - counts and rates are real operators;
  - filter values are bound to the data;
  - a silent wrong answer (`Returned ≠ "orders"`) is refused;
  - two crashes are fixed.
- **Second pass (+11 cases):** ratios, margins, shares, period-over-period and drivers of change. 51/53 on the first run. Fixes:
  - a "median ticket size" prompt conflict;
  - "this month vs last month" ending today was not treated as in progress;
  - a finished month was wrongly cut to the days the data happened to cover;
  - a weaker model could drop a real "last month" filter, which code now forbids.

Every fix is pinned offline in `tests/test_planner_value_binding.py` and `tests/test_derived_metrics.py`.

Not yet covered by the eval: workbooks, files over the batching threshold, multi-dataset questions,
arbitrary formulas over 3+ columns, and real user phrasing collected from production.

## Build order

1. ~~**B4** LLM-proposed filters as validated conditions~~: done.
2. ~~**B12** Real-model evaluation set and scorer~~: done. Rerun it on every planner change.
3. ~~**B7/B8** Derived metrics and period-over-period with contribution~~: done.
4. ~~**F** Per-owner model registry and an ownership audit of all routes~~: done (F12 tests per router still to add).
5. ~~**D6–D10** Imbalance handling, calibration, segment error analysis, explicit prediction moment, batch scoring with drift~~: done (chunked scoring of very large files and a chat "score this file" operator still open).
6. **A4–A6** Encoding/locale/header detection fixtures.
