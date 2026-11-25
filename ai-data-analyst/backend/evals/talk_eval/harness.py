"""Run questions end to end through the real API (upload → process → chat)."""

from __future__ import annotations

import contextlib
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

import httpx
import pandas as pd


@contextlib.asynccontextmanager
async def api_client() -> AsyncIterator[tuple[httpx.AsyncClient, dict[str, str]]]:
    from app.core.config import AuthMode, settings
    from app.main import create_application
    from app.services.database import db_manager

    base = Path(tempfile.mkdtemp(prefix="talk_eval_"))
    (base / "uploads").mkdir()
    (base / "artifacts").mkdir()
    old = (settings.database.url, settings.upload_directory, settings.artifact_directory, settings.auth_mode)
    try:
        settings.database.url = f"sqlite+aiosqlite:///{(base / 'eval.db').as_posix()}"
        settings.upload_directory = base / "uploads"
        settings.artifact_directory = base / "artifacts"
        settings.auth_mode = AuthMode.JWT
        await db_manager.close()
        app = create_application()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=300) as client:
                creds = {"email": "eval-runner@example.com", "password": "Str0ngPassw0rd!"}
                await client.post("/api/v1/auth/register", json={**creds, "role": "analyst", "full_name": "Eval Runner"})
                login = await client.post("/api/v1/auth/login", json=creds)
                login.raise_for_status()
                yield client, {"Authorization": f"Bearer {login.json()['access_token']}"}
    finally:
        await db_manager.close()
        settings.database.url, settings.upload_directory, settings.artifact_directory, settings.auth_mode = old


async def upload_frame(client: httpx.AsyncClient, headers: dict[str, str], name: str, df: pd.DataFrame) -> UUID:
    from app.services.dataset_processing import DatasetProcessingService

    files = {"file": (f"{name}.csv", df.to_csv(index=False).encode("utf-8"), "text/csv")}
    res = await client.post("/api/v1/datasets/upload", headers=headers, files=files, data={"name": name})
    res.raise_for_status()
    payload = res.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(payload["dataset_id"]), UUID(payload["job_id"]))
    return UUID(payload["dataset_id"])


async def ask(client: httpx.AsyncClient, headers: dict[str, str], dataset_id: UUID, message: str,
              conversation_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"dataset_id": str(dataset_id), "message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    t0 = time.perf_counter()
    try:
        res = await client.post("/api/v1/chat", headers=headers, json=body)
    except Exception as error:  # an unhandled server error: record it as a failed case, keep going
        return {"http_status": 500, "content": f"server error: {type(error).__name__}: {error}"[:2000],
                "agent_actions": [], "elapsed_s": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    if res.status_code != 200:
        return {"http_status": res.status_code, "content": res.text[:2000], "agent_actions": [], "elapsed_s": elapsed}
    data = res.json()["data"]
    data["http_status"] = 200
    data["elapsed_s"] = elapsed
    return data
