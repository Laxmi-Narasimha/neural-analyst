---
name: neural-analyst
description: Investigate a dataset already uploaded to Neural Analyst. Discover questions its schema supports, run a question or clarification through its compute API, and inspect the saved evidence behind an answer.
---

Use the bundled `scripts/analyst.py` with Python and `httpx` (available in the analyst backend environment). It defaults to the local API at `http://127.0.0.1:8000/api/v1`. For another instance, use its API base in `NEURAL_ANALYST_API_URL` and its bearer token in `NEURAL_ANALYST_TOKEN` when authentication is enabled.

- Read the dataset ID from the user's dataset link. Run `analyst.py questions DATASET_ID` to see processing status and the questions offered by the app. If processing is incomplete, report the status.
- For an open-ended exploration, show the relevant starter questions. When the user already has a question, run it directly with `analyst.py ask DATASET_ID "question"`. Use the same dataset and `--conversation ID` for a follow-up or a clarification. Surface the returned choices when a column or intent is ambiguous.
- Read a supporting table with `analyst.py rows ARTIFACT_ID`. Quote the computed measure and grouping, link the artifact, and distinguish the analyzed rows from the full dataset when sampling matters. The sample limit is a cap. A refusal or an empty result is a valid outcome.
