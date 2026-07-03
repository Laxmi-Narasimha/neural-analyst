# Production QA Report — Neural Analyst

**Date:** 2026-07-03  
**Branch:** `main`  
**Tester:** Automated QA matrix + manual code audit  
**Verdict:** Ready for beta users (self-host + SaaS deploy) with documented limitations below.

---

## Test execution summary

| Suite | Result | Notes |
|-------|--------|-------|
| Backend unit/integration (`pytest tests/ -m "not slow"`) | **165 passed**, 23 skipped | ML tests skipped unless `RUN_ML_TESTS=1` |
| Production QA matrix (`test_production_qa_matrix.py`) | **25 passed** | Every wired API button/flow |
| Response quality eval (`test_response_quality_eval.py`) | **3 passed** | Grounding, routing, no-dataset guard |
| Frontend production build (`npm run build`) | **Pass** | Next.js 15 compile OK |

Run locally:

```bash
cd ai-data-analyst/backend
pytest tests/test_production_qa_matrix.py tests/test_response_quality_eval.py -v --no-cov

cd ../frontend
npm run build
```

---

## Wired features — PASS (every button tested)

### Auth (`/login`, `/register`, app shell)

| Control | API | Status |
|---------|-----|--------|
| Register | `POST /auth/register` | PASS |
| Login | `POST /auth/login` | PASS |
| Current user | `GET /auth/me` | PASS |
| Sign out | `POST /auth/logout` + `clearTokens()` | PASS (wired in app layout) |

### Dashboard (`/app/dashboard`)

| Control | API | Status |
|---------|-----|--------|
| Summary cards | `GET /dashboard/summary` | PASS |

### Datasets (`/app/datasets`, `/app/datasets/[id]`)

| Control | API | Status |
|---------|-----|--------|
| Upload | `POST /datasets/upload` | PASS |
| List / search | `GET /datasets` | PASS |
| Dataset detail | `GET /datasets/{id}` | PASS |
| SQL query | `POST /datasets/{id}/query` | PASS (`FROM dataset`) |
| Transform suggest | `POST /datasets/{id}/transform/suggest` | PASS |
| Transform preview | `POST /datasets/{id}/transform/preview` | PASS |
| Versions list | `GET /datasets/{id}/versions` | PASS |
| Delete dataset | `DELETE /datasets/{id}` | PASS (API; no list UI button yet) |

### Talk-to-Data (`/app/analysis/new`)

| Button | Backend path | Status |
|--------|--------------|--------|
| Data Speaks (EDA) | `POST /analyses` (default EDA plan) | PASS |
| Schema | `schema_snapshot` operator | PASS |
| Privacy & Risk | `privacy_risk_scan` | PASS |
| Preview Rows | `preview_rows` | PASS |
| Missingness | `missingness_scan` | PASS |
| Missingness Patterns | `missingness_patterns` | PASS |
| Uniqueness | `uniqueness_scan` | PASS |
| Text Summary | `text_summary` | PASS |
| Trend (Auto) | `resample_aggregate` | PASS |
| Time Anomalies | `time_anomaly_scan` | PASS |
| Segments (Auto) | `segment_summary` | PASS |
| Segment Deep Dive | `segment_deep_dive` | PASS |
| Correlation | `correlation_matrix` | PASS |
| Associations | `association_scan` | PASS |
| Outliers | `outlier_scan` | PASS |
| Explain Outliers | `outlier_explain` | PASS |
| Send message | `POST /chat` | PASS |
| Clarification chips | `context.clarification` | PASS (quality eval) |
| Conversation chips | `GET /chat/conversations` | PASS |
| Upload in chat | `POST /datasets/upload` | PASS |
| No-dataset data questions | Guarded (no hallucinated counts) | PASS |

### Analysis detail (`/app/analysis/[id]`)

| Control | API | Status |
|---------|-----|--------|
| Cancel | `POST /analyses/{id}/cancel` | PASS |
| Export markdown | `POST /analyses/{id}/export` | PASS (requires `tabulate`) |
| Refresh / SSE stream | `GET /analyses/{id}/events` | PASS |
| Download evidence | `GET /data-speaks/artifacts/{id}/download` | PASS |
| Action: missingness patterns | `POST /analyses/{id}/actions/run` | PASS |
| Action: outlier explain | same | PASS |
| Action: segment deep dive | same | PASS |
| Action: privacy risk | same | PASS |
| Action: trend | same | PASS |
| Action: relationships scan | same | PASS |
| Action: relationship explain | same | PASS |
| Action: time anomaly scan | same | PASS |

### Data Speaks (`/app/data-speaks`)

| Control | API | Status |
|---------|-----|--------|
| Run EDA | `POST /data-speaks/run` | PASS |

### Jobs (`/app/jobs`)

| Control | API | Status |
|---------|-----|--------|
| List jobs | `GET /jobs` | PASS |

### Reports & artifacts (`/app/reports`)

| Control | API | Status |
|---------|-----|--------|
| List artifacts | `GET /artifacts` | PASS |
| Artifact detail | `GET /artifacts/{id}` | PASS |
| Table rows | `GET /artifacts/{id}/rows` | PASS (table artifacts only) |
| Share report | `POST /shares/reports/{id}` | PASS (report artifacts only) |
| Public share view | `GET /public/reports/{token}` | PASS |

### Connections (`/app/connections`)

| Control | API | Status |
|---------|-----|--------|
| Create connection | `POST /connections` | PASS |
| List | `GET /connections` | PASS |
| Test | `POST /connections/{id}/test` | PASS |
| Delete | `DELETE /connections/{id}` | PASS |

### Data Adequacy (`/app/quality`)

| Control | API | Status |
|---------|-----|--------|
| List domains | `GET /quality/domains` | PASS |
| Start validation | `POST /quality/validate` | Requires LLM key (see API_KEYS.md) |

### Settings / billing (`/app/settings`)

| Control | API | Status |
|---------|-----|--------|
| Subscription status | `GET /billing/status` | PASS |
| Checkout | `POST /billing/checkout` | 503 when Stripe not configured (expected) |
| Portal | `POST /billing/portal` | 503 when Stripe not configured (expected) |

---

## Known limitations (v1 — document, do not block release)

| Area | Issue | Workaround |
|------|-------|------------|
| Global search (top bar) | UI only — no navigation wired | Use per-page search (Datasets, Analysis list) |
| Settings profile save | Read-only display | Edit via register payload / future profile API |
| Forgot password | Mock page | Use admin reset or self-host without JWT |
| `setup-keys`, contact, feedback, status | Marketing/mock pages | Not product flows |
| `register?plan=pro` | Query param ignored | Use Settings → Billing after Stripe setup |
| Delete conversation | API exists; no UI button | Use API or add button in v1.1 |
| Connection ad-hoc query | API exists; no UI | Import table to dataset instead |
| ML train/predict | Backend only | No frontend in v1 |
| Data Adequacy LLM steps | Needs provider key | BYOK or Ollama local |
| JWT logout | Stateless — token valid until expiry | Frontend clears localStorage (implemented) |

---

## Production fixes in this release

1. **No-dataset chat guard** — data questions without a dataset return a deterministic prompt instead of LLM hallucination.
2. **Sign out wired** — user menu in app layout calls `api.logout()` and redirects to `/login`.
3. **`tabulate` dependency** — required for markdown report export (analysis Export button).
4. **`test_production_qa_matrix.py`** — 25 automated tests mapping every wired button to API behavior.

---

## Recommended before public SaaS launch

- [ ] Set `AUTH_MODE=jwt`, strong `SECRET_KEY`, PostgreSQL `DB_URL`
- [ ] Configure Stripe (`STRIPE_*`) and `DEPLOYMENT_MODE=saas`
- [ ] Set `LLM_MODEL` + provider key (or Ollama for self-host)
- [ ] Cloudflare Pages frontend + Render/Railway backend (see `docs/DEPLOY_SAAS.md`)
- [ ] Optional: S3 object store for multi-instance uploads