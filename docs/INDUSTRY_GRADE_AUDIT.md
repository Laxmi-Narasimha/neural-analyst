# Industry-Grade Audit (current state)

Date: 2026-02-04

This is an honest engineering audit of what exists in this repo today, with a focus on reliability, scale, and "enterprise-grade" expectations.

## What is already strong

- Backend has structured config, error types, logging context, and async SQLAlchemy foundations.
- A safe compute layer exists (operators + executor) to ground analysis on real uploaded datasets.
- Dataset ownership checks are present (prevents cross-user access).
- Backend unit test suite is large and currently passes locally.
- Frontend builds cleanly and is now wired to the real backend for auth, datasets, chat, and Data Speaks.

## Most important gaps (ranked)

### P0 (blocks real production scale)

1) Background jobs are not truly distributed yet
- Upload/processing currently uses in-process background tasks for dev convenience.
- For real scale (multiple API instances), you need an out-of-process worker queue (Celery/RQ/Temporal) + Redis/SQS/RabbitMQ.

2) Migrations are not wired as the source of truth
- The backend currently relies on table auto-create in dev.
- Production must use Alembic migrations for controlled schema evolution and safe deploys.

3) Artifact storage is local-disk
- Artifacts and uploads on local disk break with multiple instances and ephemeral containers.
- Production needs S3-compatible object storage and signed URLs for downloads.

4) CI/CD is missing
- No GitHub Actions workflow (lint + tests + build).
- No environment promotion (dev/stage/prod) or release checklist automation.

### P1 (should do before real customers)

- Observability: tracing (OpenTelemetry), metrics (Prometheus), and structured logs shipped to a log backend.
- Security hardening: stronger auth flows (refresh token rotation if you add refresh), rate limiting backed by Redis, CORS tightening, secrets management, audit logs.
- Multi-tenant safety: limits on dataset size/rows, timeouts for compute, per-user quotas, and abuse protection.
- API contracts: generate a typed client (OpenAPI -> TS client) to prevent frontend/backend drift.
- E2E tests: Playwright smoke tests for "upload -> process -> data speaks -> chat".

### P2 (nice-to-have / competitive moat)

- Semantic layer: column semantics, PII detection, and dataset versioning with lineage.
- More operator coverage (time series, cohort, funnels, segmentation) with strict determinism.
- Caching of compute artifacts keyed by dataset version hash.
- Enterprise RBAC, SSO (SAML/OIDC), org/workspace model, and audit trails.

## What we should build next (recommended sequence)

1) Job system (queue + worker) for dataset processing + heavy compute
2) Storage abstraction (local -> S3) for uploads and artifacts
3) Migrations and production deploy story (Postgres + Alembic + migrations on deploy)
4) CI (backend tests + frontend build) + basic E2E
5) Observability (request IDs already exist; add metrics/tracing)

