"""Talk to the backend exactly as the web app does: over HTTP, either in-process or to a live server."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import UUID

import httpx


@contextlib.asynccontextmanager
async def open_client(url: Optional[str] = None) -> AsyncIterator[httpx.AsyncClient]:
    """A client for ``url`` (a running backend), or for an in-process app on the demo workspace."""
    if url:
        async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=600) as client:
            yield client
        return
    from app.main import create_application

    app = create_application()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://demo",
                                     timeout=600) as client:
            yield client


def _data(res: httpx.Response) -> Any:
    if res.status_code >= 400:
        raise RuntimeError(f"{res.request.method} {res.request.url.path} -> {res.status_code}: {res.text[:400]}")
    body = res.json()
    return body.get("data", body) if isinstance(body, dict) else body


async def list_datasets(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    body = _data(await client.get("/api/v1/datasets", params={"page_size": 100}))
    items = body.get("items") if isinstance(body, dict) else body
    return list(items or [])


async def upload(client: httpx.AsyncClient, path: Path, name: str, *, in_process: bool) -> str:
    with path.open("rb") as f:
        res = await client.post("/api/v1/datasets/upload", files={"file": (path.name, f.read(), "text/csv")},
                                data={"name": name})
    data = _data(res)
    dataset_id = str(data["dataset_id"])
    for _ in range(240):
        detail = _data(await client.get(f"/api/v1/datasets/{dataset_id}"))
        state = str(detail.get("status") or "").lower()
        if state in {"ready", "completed", "processed"}:
            return dataset_id
        if state in {"error", "failed"}:
            raise RuntimeError(f"Processing {name} failed: {detail.get('error_message') or detail}")
        if in_process and data.get("job_id") and _ == 2:
            # In-process runs have no worker; process directly, as the tests do.
            from app.services.dataset_processing import DatasetProcessingService

            await DatasetProcessingService().process_dataset(UUID(dataset_id), UUID(str(data["job_id"])))
        await asyncio.sleep(0.5)
    raise RuntimeError(f"{name} was not processed within two minutes")


async def chat(client: httpx.AsyncClient, dataset_id: str, message: str,
               conversation_id: Optional[str] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"dataset_id": dataset_id, "message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return _data(await client.post("/api/v1/chat", json=payload))


async def train(client: httpx.AsyncClient, dataset_id: str, target: str, name: str) -> dict[str, Any]:
    res = await client.post("/api/v1/ml/train", json={
        "dataset_id": dataset_id, "target_column": target, "model_name": name, "optimize": False, "max_models": 3})
    return _data(res)


async def list_models(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    body = _data(await client.get("/api/v1/ml/models"))
    if isinstance(body, dict):
        return list(body.get("models") or body.get("items") or [])
    return list(body or [])
