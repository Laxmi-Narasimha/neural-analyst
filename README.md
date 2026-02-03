# data_analyst (monorepo)

This folder contains two related apps:

- `ai-data-analyst/` — FastAPI backend + Next.js dashboard for dataset upload, analysis, and chat.
- `ai-data-validator/` — FastAPI backend + Streamlit UI for “data adequacy” validation (uploads + Pinecone/OpenAI checks).

## Quickstart (macOS)

1) Install prerequisites (Node and Python versions are enforced by the apps):
- See `docs/SETUP.md`.

2) Configure environment variables:
- `ai-data-analyst/backend/.env` from `ai-data-analyst/backend/.env.example`
- `ai-data-validator/backend/.env` from `ai-data-validator/backend/.env.template`

3) Run:

```bash
make setup
make dev-analyst
```

Or run the validator:

```bash
make setup
make dev-validator
```

## Docs

- `docs/PROJECT_MAP.md` — what each folder/file is for (source vs generated vs vendor).
- `docs/ARCHITECTURE.md` — end-to-end request flow for both apps.
- `docs/INDUSTRY_GRADE_AUDIT.md` — severity-ranked gaps + concrete fixes/backlog.
- `docs/SETUP.md` — machine setup runbook (Python/Node/Postgres).

