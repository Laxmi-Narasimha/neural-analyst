# Deprecated — merged into Neural Analyst

The standalone `ai-data-validator` app (FastAPI + Streamlit) has been **merged into** `ai-data-analyst`.

## Use instead

| Old (validator) | New (analyst) |
|-----------------|---------------|
| Streamlit UI | Next.js `/app/quality` (Data Adequacy) |
| `POST /api/validate` | `POST /api/v1/quality/validate` |
| In-memory sessions | DB-backed `DataAdequacySession` |
| Raw file upload flow | Dataset-linked adequacy on `dataset_id` |

Canonical code: `ai-data-analyst/backend/app/agents/data_adequacy/`

This folder is kept for reference only and is **not deployed** in production configs.