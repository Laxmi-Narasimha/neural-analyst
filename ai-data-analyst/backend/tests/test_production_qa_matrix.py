"""
Production QA matrix — API-level coverage for every wired UI button/flow.

Maps frontend buttons (analysis/new, datasets, connections, etc.) to backend
endpoints and asserts happy-path responses. Run before release:

  pytest tests/test_production_qa_matrix.py -v
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from app.core.config import AuthMode, settings
from app.main import create_application
from app.services.analysis_execution import AnalysisExecutionService
from app.services.database import db_manager
from app.services.dataset_processing import DatasetProcessingService

# Mirrors frontend analysis/new/page.js analysisButtons
ANALYSIS_BUTTONS: list[tuple[str, list[dict] | None]] = [
    ("data_speaks", None),
    ("schema", [{"operator": "schema_snapshot", "params": {}}]),
    ("risk", [{"operator": "privacy_risk_scan", "params": {}}]),
    ("preview", [{"operator": "preview_rows", "params": {"limit": 25}}]),
    ("missing", [{"operator": "missingness_scan", "params": {}}]),
    ("missing_patterns", [{"operator": "missingness_patterns", "params": {}}]),
    ("uniqueness", [{"operator": "uniqueness_scan", "params": {"max_columns": 200}}]),
    ("text", [{"operator": "text_summary", "params": {"max_columns": 25}}]),
    ("trend", [{"operator": "resample_aggregate", "params": {"freq": "M", "max_points": 200}}]),
    ("time_anomalies", [{"operator": "time_anomaly_scan", "params": {"freq": "M", "max_points": 200}}]),
    ("segments", [{"operator": "segment_summary", "params": {"limit": 50}}]),
    ("segment_deep_dive", [{"operator": "segment_deep_dive", "params": {"limit": 10}}]),
    ("correlation", [{"operator": "correlation_matrix", "params": {"max_columns": 25}}]),
    ("associations", [{"operator": "association_scan", "params": {"max_categorical_columns": 20, "max_numeric_columns": 20, "max_pairs": 200}}]),
    ("outliers", [{"operator": "outlier_scan", "params": {"max_columns": 25}}]),
    ("outlier_explain", [{"operator": "outlier_explain", "params": {}}]),
]

# Mirrors analysis detail page action feed buttons (analyses.py _action_defaults)
ANALYSIS_DETAIL_ACTIONS: list[str] = [
    "missingness_patterns",
    "outlier_explain",
    "segment_deep_dive",
    "privacy_risk_scan",
    "trend",
    "relationships_scan",
    "relationship_explain",
    "time_anomaly_scan",
]


def _extract_items(payload: dict) -> list:
    """Normalize list endpoints (PaginatedResponse uses data: list)."""
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("items") or []
    return payload.get("items") or []


def _qa_csv() -> str:
    return "\n".join(
        [
            "id,value,segment,date,notes",
            "1,10,A,2020-01-01,ok",
            "2,20,A,2020-01-02,",
            "3,,B,2020-01-03,delayed",
            "4,9999,B,2020-01-04,ok",
            "5,50,C,2020-01-05,ok",
            "6,30,C,2020-01-06,ok",
            "7,40,A,2020-01-07,ok",
            "8,15,B,2020-01-08,",
        ]
    )


@pytest_asyncio.fixture(scope="module")
async def qa_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("production_qa")
    db_file = base / "test.db"
    uploads_dir = base / "uploads"
    artifacts_dir = base / "artifacts"
    conn_db = base / "conn.db"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    old_db = settings.database.url
    old_upload = settings.upload_directory
    old_artifact = settings.artifact_directory
    old_auth = settings.auth_mode
    old_rate_limit = settings.security.rate_limit_requests

    try:
        settings.database.url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
        settings.upload_directory = uploads_dir
        settings.artifact_directory = artifacts_dir
        settings.auth_mode = AuthMode.JWT
        settings.security.rate_limit_requests = 10_000

        await db_manager.close()
        app = create_application()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                email = "qa-matrix@example.com"
                password = "Str0ngPassw0rd!"
                reg = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": password, "role": "analyst", "full_name": "QA Matrix"},
                )
                assert reg.status_code in {200, 201, 400}, reg.text

                login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
                assert login.status_code == 200, login.text
                token = login.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}

                yield {
                    "client": client,
                    "headers": headers,
                    "base": base,
                    "conn_db": conn_db,
                }
    finally:
        await db_manager.close()
        settings.database.url = old_db
        settings.upload_directory = old_upload
        settings.artifact_directory = old_artifact
        settings.auth_mode = old_auth
        settings.security.rate_limit_requests = old_rate_limit
        try:
            from app.main import RateLimitMiddleware

            RateLimitMiddleware._requests.clear()
        except Exception:
            pass


@pytest_asyncio.fixture(scope="module")
async def qa_dataset(qa_env):
    """Single shared dataset for parametrized button tests (avoids upload storms)."""
    client, headers = qa_env["client"], qa_env["headers"]
    return await _upload_and_process(client, headers, _qa_csv(), "QA Shared Dataset")


async def _upload_and_process(client: httpx.AsyncClient, headers: dict, csv_text: str, name: str) -> UUID:
    files = {"file": ("qa.csv", csv_text.encode("utf-8"), "text/csv")}
    res = await client.post("/api/v1/datasets/upload", headers=headers, files=files, data={"name": name})
    assert res.status_code == 201, res.text
    payload = res.json()["data"]
    dataset_id = UUID(payload["dataset_id"])
    job_id = UUID(payload["job_id"])
    await DatasetProcessingService().process_dataset(dataset_id, job_id)
    return dataset_id


async def _run_analysis_to_completion(
    client: httpx.AsyncClient,
    headers: dict,
    *,
    dataset_id: UUID,
    name: str,
    plan: list[dict] | None,
) -> UUID:
    config: dict = {"sample_rows": 5000}
    if plan is not None:
        config["plan"] = plan
    created = await client.post(
        "/api/v1/analyses",
        headers=headers,
        json={"name": name, "dataset_id": str(dataset_id), "analysis_type": "eda", "config": config},
    )
    assert created.status_code == 201, created.text
    analysis_id = UUID(created.json()["data"]["id"])
    detail = await client.get(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    job_id = UUID(detail.json()["data"]["config"]["job_id"])
    await AnalysisExecutionService().run_analysis(analysis_id, job_id)
    return analysis_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_auth_register_login_me_logout(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]

    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200, me.text
    assert me.json().get("email") == "qa-matrix@example.com"

    logout = await client.post("/api/v1/auth/logout", headers=headers)
    assert logout.status_code == 200, logout.text
    assert logout.json().get("status") == "success"
    # JWT auth is stateless — client must clear tokens (frontend api.logout does this).


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_dashboard_summary(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    res = await client.get("/api/v1/dashboard/summary", headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()["data"]
    assert "datasets_total" in data
    assert "analyses_total" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_datasets_upload_list_get_sql_transform_versions(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    dataset_id = await _upload_and_process(client, headers, _qa_csv(), "QA Dataset")

    listing = await client.get("/api/v1/datasets?page=1&page_size=20", headers=headers)
    assert listing.status_code == 200, listing.text
    items = _extract_items(listing.json())
    assert any(str(it.get("id")) == str(dataset_id) for it in items)

    detail = await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert str(detail.json()["data"]["status"]).lower() == "ready"

    sql = await client.post(
        f"/api/v1/datasets/{dataset_id}/query",
        headers=headers,
        json={"query": "SELECT COUNT(*) AS n FROM dataset", "max_rows": 10},
    )
    assert sql.status_code == 200, sql.text

    suggest = await client.post(
        f"/api/v1/datasets/{dataset_id}/transform/suggest",
        headers=headers,
        json={"max_steps": 4, "include_drop_columns": True, "include_string_normalization": True},
    )
    assert suggest.status_code == 200, suggest.text
    steps = suggest.json()["data"].get("steps") or []
    if steps:
        preview = await client.post(
            f"/api/v1/datasets/{dataset_id}/transform/preview",
            headers=headers,
            json={"steps": steps[:1], "sample_rows": 1000, "preview_rows": 5},
        )
        assert preview.status_code == 200, preview.text

    versions = await client.get(f"/api/v1/datasets/{dataset_id}/versions?page=1&page_size=20", headers=headers)
    assert versions.status_code == 200, versions.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_chat_send_list_get_delete_and_no_dataset_guard(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    dataset_id = await _upload_and_process(client, headers, _qa_csv(), "Chat QA")

    chat = await client.post(
        "/api/v1/chat",
        headers=headers,
        json={"message": "How many rows and columns?", "dataset_id": str(dataset_id)},
    )
    assert chat.status_code == 200, chat.text
    chat_data = chat.json()["data"]
    conv_id = chat_data["conversation_id"]
    assert chat_data.get("content")
    assert any(
        a.get("operator") == "dataset_overview"
        for a in (chat_data.get("agent_actions") or [])
        if isinstance(a, dict)
    )

    no_ds = await client.post(
        "/api/v1/chat",
        headers=headers,
        json={"message": "How many rows are in my dataset?"},
    )
    assert no_ds.status_code == 200, no_ds.text
    no_ds_body = no_ds.json()["data"]
    content = str(no_ds_body.get("content") or "").lower()
    assert (no_ds_body.get("metadata") or {}).get("no_dataset") is True
    assert not any(isinstance(a, dict) and a.get("operator") for a in (no_ds_body.get("agent_actions") or []))
    assert not re.search(r"\b\d{2,}\s+rows\b", content), content
    assert any(w in content for w in ["dataset", "upload", "select", "attach"])

    convs = await client.get("/api/v1/chat/conversations?page=1&page_size=20", headers=headers)
    assert convs.status_code == 200, convs.text

    conv_detail = await client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
    assert conv_detail.status_code == 200, conv_detail.text

    deleted = await client.delete(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
    assert deleted.status_code in {200, 204}, deleted.text


@pytest.mark.integration
@pytest.mark.parametrize("button_id,plan", ANALYSIS_BUTTONS)
@pytest.mark.asyncio
async def test_qa_analysis_new_buttons(qa_env, qa_dataset, button_id: str, plan: list[dict] | None):
    """Each analysis/new toolbar button creates and completes an analysis."""
    client, headers = qa_env["client"], qa_env["headers"]
    dataset_id = qa_dataset

    analysis_id = await _run_analysis_to_completion(
        client,
        headers,
        dataset_id=dataset_id,
        name=f"QA {button_id}",
        plan=plan,
    )

    detail = await client.get(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    body = detail.json()["data"]
    assert str(body.get("status")).lower() == "completed"
    steps = (body.get("results") or {}).get("steps") or []
    assert len(steps) >= 1, f"{button_id}: no steps recorded"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_analysis_detail_cancel_export_actions(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    dataset_id = await _upload_and_process(client, headers, _qa_csv(), "Detail QA")

    parent_id = await _run_analysis_to_completion(
        client,
        headers,
        dataset_id=dataset_id,
        name="Parent EDA",
        plan=None,
    )

    export = await client.post(f"/api/v1/analyses/{parent_id}/export?format=markdown", headers=headers)
    assert export.status_code == 200, export.text
    assert export.json()["data"].get("artifact_id")

    for action_id in ANALYSIS_DETAIL_ACTIONS:
        run = await client.post(
            f"/api/v1/analyses/{parent_id}/actions/run",
            headers=headers,
            json={"action_id": action_id, "params": {}},
        )
        assert run.status_code == 200, f"{action_id}: {run.text}"
        child_id = UUID(run.json()["data"]["analysis_id"])
        child = await client.get(f"/api/v1/analyses/{child_id}", headers=headers)
        assert child.status_code == 200
        child_job = UUID(child.json()["data"]["config"]["job_id"])
        await AnalysisExecutionService().run_analysis(child_id, child_job)

    feed = await client.get(f"/api/v1/analyses/{parent_id}/actions", headers=headers)
    assert feed.status_code == 200, feed.text
    items = feed.json()["data"]
    assert len(items) >= len(ANALYSIS_DETAIL_ACTIONS)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_data_speaks_jobs_artifacts_reports(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    dataset_id = await _upload_and_process(client, headers, _qa_csv(), "Speaks QA")

    speaks = await client.post(
        "/api/v1/data-speaks/run",
        headers=headers,
        json={"dataset_id": str(dataset_id), "sample_rows": 5000},
    )
    assert speaks.status_code == 200, speaks.text
    speaks_data = speaks.json()["data"]
    assert isinstance(speaks_data.get("steps"), list) and len(speaks_data["steps"]) >= 1

    jobs = await client.get("/api/v1/jobs?page=1&page_size=20", headers=headers)
    assert jobs.status_code == 200, jobs.text

    artifacts = await client.get("/api/v1/artifacts?page=1&page_size=20", headers=headers)
    assert artifacts.status_code == 200, artifacts.text
    art_items = _extract_items(artifacts.json())
    table_art = next(
        (it for it in art_items if str(it.get("artifact_type") or "").lower() in {"table", "dataframe", "csv"}),
        art_items[0] if art_items else None,
    )
    if table_art:
        art_id = table_art.get("id") or table_art.get("artifact_id")
        if art_id:
            art = await client.get(f"/api/v1/artifacts/{art_id}", headers=headers)
            assert art.status_code == 200, art.text
            art_type = str(art.json().get("data", {}).get("artifact_type") or table_art.get("artifact_type") or "").lower()
            if art_type in {"table", "dataframe", "csv"}:
                rows = await client.get(f"/api/v1/artifacts/{art_id}/rows?offset=0&limit=5", headers=headers)
                assert rows.status_code == 200, rows.text

    report_art = next(
        (it for it in art_items if str(it.get("artifact_type") or "").lower() == "report"),
        None,
    )
    if report_art:
        report_id = report_art.get("id") or report_art.get("artifact_id")
        share = await client.post(f"/api/v1/shares/reports/{report_id}", headers=headers, json={})
        assert share.status_code == 200, share.text
        token = share.json()["data"].get("share_token")
        if token:
            public = await client.get(f"/api/v1/public/reports/{token}")
            assert public.status_code == 200, public.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_connections_sqlite_lifecycle(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    conn_db: Path = qa_env["conn_db"]
    conn_db.write_text("SELECT 1", encoding="utf-8")

    created = await client.post(
        "/api/v1/connections",
        headers=headers,
        json={
            "name": "QA SQLite",
            "connector_type": "sqlite",
            "host": "localhost",
            "port": 0,
            "database": str(conn_db),
            "username": "",
            "password": "",
        },
    )
    assert created.status_code in {200, 201}, created.text
    conn_id = created.json().get("id") or created.json().get("data", {}).get("id")
    assert conn_id

    listing = await client.get("/api/v1/connections", headers=headers)
    assert listing.status_code == 200, listing.text

    tested = await client.post(f"/api/v1/connections/{conn_id}/test", headers=headers)
    assert tested.status_code == 200, tested.text

    deleted = await client.delete(f"/api/v1/connections/{conn_id}", headers=headers)
    assert deleted.status_code in {200, 204}, deleted.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_quality_domains_and_billing_status(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]

    domains = await client.get("/api/v1/quality/domains", headers=headers)
    assert domains.status_code == 200, domains.text

    billing = await client.get("/api/v1/billing/status", headers=headers)
    assert billing.status_code == 200, billing.text
    plan = billing.json()["data"].get("plan")
    assert plan in {"free", "pro", "enterprise", "self_host", None} or isinstance(plan, str)

    checkout = await client.post(
        "/api/v1/billing/checkout",
        headers=headers,
        json={"plan": "pro"},
    )
    assert checkout.status_code in {503, 400}, checkout.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_analyses_list_and_dataset_delete(qa_env):
    client, headers = qa_env["client"], qa_env["headers"]
    dataset_id = await _upload_and_process(client, headers, _qa_csv(), "Delete QA")

    analyses = await client.get(f"/api/v1/analyses?page=1&page_size=20&dataset_id={dataset_id}", headers=headers)
    assert analyses.status_code == 200, analyses.text

    deleted = await client.delete(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert deleted.status_code in {200, 204}, deleted.text

    gone = await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert gone.status_code == 404