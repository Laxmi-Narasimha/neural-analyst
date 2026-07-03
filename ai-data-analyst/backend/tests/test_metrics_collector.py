from __future__ import annotations

from app.core.metrics import MetricsCollector


def test_metrics_collector_http_and_job_sections():
    collector = MetricsCollector(max_history=32)

    collector.record_http_request(method="GET", route="/api/v1/datasets/{dataset_id}", status_code=200, duration_ms=12.5)
    collector.record_http_request(method="GET", route="/api/v1/datasets/{dataset_id}", status_code=503, duration_ms=30.0)
    collector.record_job_run(job_type="dataset_processing", duration_seconds=8.0, success=True)
    collector.record_job_run(job_type="dataset_processing", duration_seconds=12.0, success=False)

    snap = collector.snapshot()

    route = snap["http_routes"]["GET /api/v1/datasets/{dataset_id}"]
    assert route["requests"] == 2
    assert route["errors"] == 1
    assert route["status_counts"]["200"] == 1
    assert route["status_counts"]["503"] == 1

    jobs = snap["jobs"]["dataset_processing"]
    assert jobs["runs"] == 2
    assert jobs["failures"] == 1
    assert jobs["duration_seconds"]["n"] == 2
