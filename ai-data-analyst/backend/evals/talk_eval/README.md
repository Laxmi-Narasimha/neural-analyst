# Talk to Your Data: real-model eval

Measures how often a question typed the way users type it gets the **right number**, end to end:
upload → profile → `/api/v1/chat` → rule planner or LLM planner → validated plan → operators.

- `datasets.py`: three seeded datasets (retail orders, SaaS accounts, HR export with abbreviated headers).
- `cases.py`: 42 questions with answers computed **independently in pandas**: filters, synonyms,
  relative dates, typos, Hinglish, rankings, comparisons, rates, counts, follow-ups, privacy, prompt
  injection, and questions that must be declined.
- `scoring.py`: checks the number (or top group, or every group) against the expected value. A
  case passes only when the right value is in the answer. A case marked "decline" passes only
  when nothing is computed.

## Run

From `ai-data-analyst/backend`, with `OPENAI_API_KEY` in `.env`:

```bash
.venv/Scripts/python.exe -m evals.talk_eval.run --models gpt-5.4-mini gpt-4.1-mini
```

Subset: `--only r_total_west s_enterprise_churn_rate`. Stability: `--repeat 3`.
Results are written to `evals/talk_eval/results/<timestamp>/` (git-ignored) as `report.json`
(every answer) and `summary.md`.

Run it after every change to the planner prompt, catalog, validator or chat flow, and before
changing `LLM_PLANNER_MODEL`. The offline tests in `tests/test_planner_value_binding.py` pin
each failure the eval has found so far, so they cannot come back without a test failing.
