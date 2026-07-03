# API Keys & Environment — Neural Analyst

Copy `ai-data-analyst/backend/.env.example` to `.env` and set values below.

---

## Minimum local dev (full UI testing)

| Variable | Required? | Example | Purpose |
|----------|-----------|---------|---------|
| `LLM_MODEL` | Yes for chat w/o dataset, Data Adequacy | `gpt-4o-mini` or `ollama/llama3` | LiteLLM model string |
| Provider key | Yes unless Ollama | `OPENAI_API_KEY=sk-...` | See provider table below |
| `AUTH_MODE` | For login flow | `jwt` (prod) or `local` (dev) | App layout requires JWT tokens |
| `SECRET_KEY` | Yes if `AUTH_MODE=jwt` | 64-char random string | JWT signing |

**Frontend:** set `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1` in `frontend/.env.local`.

Start:

```bash
# Backend
cd ai-data-analyst/backend
python -m uvicorn app.main:app --reload --port 8000

# Frontend
cd ai-data-analyst/frontend
npm run dev
```

Register at `/register`, then use the app.

---

## LLM providers (pick one)

| Provider | `LLM_MODEL` | API key variable |
|----------|-------------|------------------|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-1.5-flash` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| Ollama (free, local) | `ollama/llama3` | None — run `ollama serve` |

Docs: https://docs.litellm.ai/docs/providers

**Without any LLM key:** Talk-to-Data with a dataset still works (compute-backed). General chat and Data Adequacy LLM steps will fail.

---

## Production SaaS

| Variable | Required? | Notes |
|----------|-----------|-------|
| `DEPLOYMENT_MODE` | Yes | `saas` for freemium limits |
| `AUTH_MODE` | Yes | `jwt` |
| `SECRET_KEY` | Yes | Random 64+ chars |
| `DB_URL` | Yes | `postgresql+asyncpg://...` |
| `FRONTEND_URL` | Yes | e.g. `https://app.yourdomain.com` |
| `STRIPE_SECRET_KEY` | For paid plans | `sk_live_...` |
| `STRIPE_WEBHOOK_SECRET` | For paid plans | `whsec_...` |
| `STRIPE_PRICE_PRO` | For Pro tier | Stripe Price ID |
| `STRIPE_PRICE_ENTERPRISE` | For Enterprise | Stripe Price ID |
| `SECURITY_CORS_ORIGINS` | Yes | Include your Pages URL |

See `docs/DEPLOY_SAAS.md` for Cloudflare Pages + Render setup.

---

## Optional scale / features

| Variable | When needed |
|----------|-------------|
| `OBJECT_STORE_BACKEND=s3` + `S3_*` | Multi-instance file storage |
| `JOB_EXECUTOR=celery` + `REDIS_*` + `CELERY_*` | Distributed background jobs |
| `VECTOR_STORE=pinecone` + `PINECONE_API_KEY` | Vector search for adequacy |
| `NARRATOR_MODE=llm` | AI narrative in Data Speaks (default: deterministic) |
| `PINECONE_API_KEY` | Only if `VECTOR_STORE=pinecone` |

---

## Self-host (unlimited OSS)

| Variable | Value |
|----------|-------|
| `DEPLOYMENT_MODE` | `self_host` (default) |
| `AUTH_MODE` | `local` or `jwt` |
| Database | SQLite default OK for single user |

Scripts: `scripts/self_host.ps1` / `scripts/self_host.sh`

---

## What you do NOT need for core Talk-to-Data

- Stripe (unless SaaS billing)
- Pinecone (unless vector adequacy)
- S3 (unless multi-node)
- Celery/Redis (unless distributed jobs)

Dataset upload → processing → chat with grounded operators works with **SQLite + no LLM key**.