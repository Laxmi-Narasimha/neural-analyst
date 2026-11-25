# Demo guide

A 10-minute walkthrough of Neural Analyst that works the same way every time, plus a check you
run before every interview so nothing surprises you.

## Before the interview (5 minutes)

From `ai-data-analyst/backend`, with the backend virtual environment:

```powershell
.\.venv\Scripts\python.exe -m demo check --repeat 2
```

This loads three demo datasets into a separate demo workspace and asks every question below
twice. It then re-computes every number independently with pandas and checks that it appears in
the answer. It also opens the predictions table the scoring step produces. It must end with
`10/10 demo steps passed` and `2/2 full runs passed`. If a step fails, the answer it got is printed
above the summary.

Then start the app:

```powershell
.\.venv\Scripts\python.exe -m demo serve
```

Open http://localhost:3002/app/datasets. `serve` loads the demo data if needed, builds the
frontend once (production build, so pages open instantly), starts both servers and prints whether
the language model is on. Ctrl+C stops everything. Your own data is never touched; the demo lives
in `ai-data-analyst/backend/demo_workspace/` and `python -m demo reset` deletes it.

To check the exact server the browser is using while `serve` runs:

```powershell
.\.venv\Scripts\python.exe -m demo check --url http://127.0.0.1:8000
```

### Requirements

- Python 3.11+, Node 20.9+, backend packages installed (`pip install -r requirements.txt`).
- `OPENAI_API_KEY` in `ai-data-analyst/backend/.env` for the questions marked *LLM* below. Each
  of those costs one small planning call (the numbers never come from the model). Without a key,
  the steps not marked *LLM* still work, and the others answer with a clear request to
  rephrase. Check that case with `python -m demo check --no-llm`.

## The walkthrough

Open **New Analysis**, pick the dataset in the **Dataset** menu at the top of the chat, and type
the question. The talking points are what to say while the answer appears.

### 1. retail_orders (6,000 orders, 2024 to 2025)

| Question | What to point out |
| --- | --- |
| How many orders are there? *(LLM)* | The model only plans; "Interpreted your question as: number of rows" is written by code from the validated plan, and the 6,000 comes from code. |
| Total Sales by Region | No model call at all: the rule planner handles common phrasings, which keeps answers cheap and deterministic. |
| Monthly Sales trend | Highest and lowest month, the latest month versus a year earlier, a chart, and the full table under **Inspect supporting tables**. |
| Return rate by Category *(LLM)* | "Return rate" is bound to `Returned = Yes` using the column's real values, and every rate shows its counts (175 of 1,348). |
| Margin % by Category *(LLM)* | A derived metric: (Sales − Cost) / Sales as a ratio of totals, not an average of per-row ratios. |
| Total Sales in the West region in 2025 *(LLM)* | The filters are stated back exactly, with the number of matching rows, so the user can see what was counted. |
| Why did sales drop in Q3 2025 compared with Q2 2025? *(LLM)* | Driver analysis: code compares the two quarters by every low-cardinality column and leads with where the change is concentrated. South was 22% of sales but explains 129% of the drop (−40%); the other dimensions are listed as checked. |

### 2. customer_churn (4,000 customers with a known outcome)

| Question | What to point out |
| --- | --- |
| What is the churn rate by contract? *(LLM)* | Monthly contracts churn at about 30%, two-year contracts at about 9%. |
| Predict churned | About 10 seconds. Points to cover: leakage screening, a time-ordered split (it says which date it assumed and asks you to confirm), comparison with a naive baseline, ROC AUC and PR AUC, precision and recall for the churners, a cut-off tuned on training folds (0.24 instead of 0.5: recall 61% instead of 16%), calibration error, and the weakest segment with a significance test. The model is saved per user with its preprocessing. |

### 3. churn_new_customers (1,000 newer customers, no outcome yet)

| Question | What to point out |
| --- | --- |
| Score this file with my churn model | Every row is scored (never sampled); the model was picked from your own saved models by code. The drift check flags `monthly_charges` (PSI 0.67): these customers came after a price rise, so the model is being used outside the conditions it was trained on. Open **Predictions · ml score** under the supporting tables to show the scored rows and the Download button. |

## Questions interviewers ask, and where the answer lives

- **"How do you stop the model making up numbers?"** The LLM returns a JSON plan, which is
  validated against the dataset schema (`app/planning/validator.py`). Operators in
  `app/compute/operators/` compute every number; answers are written from templates in
  `app/api/routes/chat.py`.
- **"What if the LLM is down or wrong?"** The rule planner answers common questions without it.
  An invalid plan is repaired once and otherwise refused with a clarifying question. Results
  that depend on which column the model picked (driver analysis) are made deterministic in code.
- **"How do you know the numbers are right?"** More than 430 backend tests, and `python -m demo check`,
  which re-computes every demo number independently.
- **"Is the ML evaluation sound?"** See `app/ml/auto_ml_engine.py` and `app/ml/evaluation.py`:
  choices (metric, cut-off, calibration) are made on training folds only; held-out rows are only
  scored. The rarer class is the positive label, so PR AUC, precision and recall describe it.
- **"Multi-user?"** Each user has a separate model registry and data, and cross-user access
  returns 404 (`tests/test_tenant_isolation.py`).
- **"Cost?"** One small planning call for questions the rules cannot read; none for the rest.
  Training and scoring run on CPU.

## Honest limits to mention if asked

- Files above 500,000 rows are refused for batch scoring (chunked scoring is not built yet).
- The demo data is synthetic, generated by `ai-data-analyst/backend/demo/generate_data.py`.
- The "Data Adequacy" page is hidden from the menu until it is restyled.
