# SaaS Deployment (Cloudflare Pages + Free Python Backend)

## Architecture

| Layer | Service | Free tier option |
|-------|---------|------------------|
| Frontend (Next.js) | **Cloudflare Pages** | Free, global CDN |
| API (FastAPI) | **Render** or **Fly.io** | Render free web (cold start) |
| Database | **Neon** or Render Postgres | Neon free tier |
| Redis (optional) | **Upstash** | Free tier for rate limits / Celery |
| Object storage | **Cloudflare R2** or MinIO | R2 free egress to Workers |

Python compute runs on the **backend container** (not Cloudflare Workers — Workers cannot run pandas/sklearn).

## 1) Frontend on Cloudflare Pages

```bash
cd ai-data-analyst/frontend
npm ci
npm run build
```

In Cloudflare Dashboard → Pages → Connect GitHub repo `Laxmi-Narasimha/neural-analyst`:

- **Root directory:** `ai-data-analyst/frontend`
- **Build command:** `npm ci && npm run build`
- **Output directory:** `.next` (use Next.js adapter) or static export

Environment variables:

```
NEXT_PUBLIC_API_URL=https://your-api.onrender.com/api/v1
```

For full Next.js on Cloudflare, use `@cloudflare/next-on-pages` (see Cloudflare docs).

## 2) Backend on Render (free tier)

Use `render.yaml` in repo root (analyst backend only).

Set secrets in Render:

```
DEPLOYMENT_MODE=saas
AUTH_MODE=jwt
OPENAI_API_KEY=...
STRIPE_SECRET_KEY=...
STRIPE_WEBHOOK_SECRET=...
STRIPE_PRICE_PRO=price_...
STRIPE_PRICE_ENTERPRISE=price_...
FRONTEND_URL=https://your-pages.pages.dev
DATABASE_URL=postgresql://...
```

Stripe webhook URL: `https://your-api.onrender.com/api/v1/billing/webhook`

## 3) Freemium model (hosted)

| Plan | Talk-to-Data | Compute |
|------|--------------|---------|
| Free | 1 preview session (schema + overview) | No heavy operators |
| Pro / Enterprise | Unlimited | Full operator catalog |
| Self-host | Unlimited | User's keys + compute |

Set `DEPLOYMENT_MODE=self_host` when users run Docker locally (no limits).

## 4) Self-host (recommended for power users)

```bash
git clone https://github.com/Laxmi-Narasimha/neural-analyst.git
cd neural-analyst
docker compose up --build
```

Copy `ai-data-analyst/backend/.env.example` → `.env`, add your LLM keys.

## 5) Alternative free Python hosts

- **Fly.io** — small VM, always-on possible with credit
- **Railway** — limited trial credits
- **Google Cloud Run** — pay-per-request, scale to zero
- **Modal** — serverless Python for heavy jobs (future integration)