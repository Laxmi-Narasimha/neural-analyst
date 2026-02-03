# Deployment

I cannot directly “take access” to your machine to install software, and I can’t deploy to Vercel/Cloudflare without you connecting an account/repo. What I *can* do is make deployment **one-click** once you push this repo to GitHub.

This repo has two deployable backends (FastAPI). The Next.js frontend is best deployed separately.

## Option A (recommended): Vercel (frontend) + Render (backend)

### 1) Push to GitHub

Create a GitHub repo and push the monorepo root (`/Users/laxmi/Downloads/data_analyst`).

Important:
- Do **not** commit `node_modules/`, `.next/`, `.venv/`, `coverage_html/`, `uploads/` (root `.gitignore` already covers these).
- Remove nested git at `ai-data-analyst/frontend/.git/` if it exists before pushing.

### 2) Deploy backends on Render (fastest path)

This repo includes a Render Blueprint at `render.yaml`.

Steps:
1. In Render, choose “New” → “Blueprint”.
2. Select your GitHub repo.
3. Render will provision:
   - `ai-data-analyst-backend` (FastAPI)
   - `ai-data-validator-backend` (FastAPI)
   - `ai-data-analyst-db` (Postgres)
4. Set the required secrets in Render:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY` (and `PINECONE_ENVIRONMENT` for validator)

Notes:
- For first deploys, `ai-data-analyst-backend` sets `DB_AUTO_CREATE_TABLES=true` to create tables on startup.
- The health endpoint is `/health`.

### 3) Deploy analyst frontend on Vercel

1. Create a new Vercel project.
2. Select the same GitHub repo.
3. Set “Root Directory” to `ai-data-analyst/frontend`.
4. Set an environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://<your-render-analyst-backend>/api/v1`
5. Deploy.

Next.js in this repo requires Node `>= 20.9.0` (Vercel Node 20 should satisfy this).

## Option B: Cloudflare Pages (frontend) + Render (backend)

Cloudflare Pages can host Next.js, but it’s typically more setup than Vercel for modern Next.js. If you specifically want Cloudflare, I can add the exact adapter configuration once you confirm Pages runtime requirements.

