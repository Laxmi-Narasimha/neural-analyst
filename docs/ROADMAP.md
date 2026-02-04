# Roadmap (practical, buildable)

This roadmap is derived from `spec_review/*` and is ordered to maximize reliability first, then capability, then scale.

## Milestone 0: Grounded "Talk to Your Data" loop (done-ish)

- Dataset upload + processing
- Safe compute operators + artifacts
- Data Speaks (run a plan and return evidence)
- Chat grounded on compute when dataset_id is present

## Milestone 1: Production foundations (next)

1) Job system
- Move dataset processing + heavy operators to an out-of-process worker (Celery/RQ/etc).
- Add job records and progress endpoints.

2) Storage
- Abstract uploads/artifacts to support S3-compatible storage.
- Signed downloads + lifecycle/cleanup.

3) DB migrations
- Use Alembic migrations as the only schema evolution mechanism in prod.
- Add initial baseline migration + deploy-time migration step.

4) CI/CD
- CI: backend tests + frontend build (add lint later).
- CD: staging deploy + smoke tests.

## Milestone 2: Quality and safety

- Stronger auth/session model (refresh rotation if used, revocation)
- Redis-backed rate limiting
- Prompt injection defenses for dataset text
- Observability (metrics + tracing + dashboards)

## Milestone 3: Feature depth (data science moat)

- Expand operator catalog (time series, cohorts, funnels, segmentation)
- Dataset versioning + lineage
- Caching of operator results by dataset version hash
- Typed client generation from OpenAPI to prevent contract drift

