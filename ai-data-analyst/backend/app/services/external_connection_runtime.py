from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.serialization import to_jsonable


class ConnectionRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConnectionRuntimeConfig:
    connector_type: str
    host: str
    port: int
    database: str
    username: str
    password: Optional[str]
    ssl: bool
    timeout_seconds: int


def _sqlite_db_to_url(database: str) -> str:
    db = (database or "").strip()
    if db in (":memory:", "file::memory:"):
        return "sqlite+aiosqlite:///:memory:"
    # Normalize Windows paths to a SQLAlchemy-friendly sqlite URI.
    p = Path(db).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    return f"sqlite+aiosqlite:///{p.as_posix()}"


def build_async_sqlalchemy_url(cfg: ConnectionRuntimeConfig) -> str:
    t = (cfg.connector_type or "").strip().lower()
    if t in ("postgres", "postgresql"):
        query = {}
        if cfg.ssl:
            query["sslmode"] = "require"
        url = URL.create(
            drivername="postgresql+asyncpg",
            username=cfg.username or None,
            password=cfg.password or None,
            host=cfg.host or "localhost",
            port=int(cfg.port or 5432),
            database=cfg.database or None,
            query=query or None,
        )
        return str(url)

    if t in ("sqlite",):
        return _sqlite_db_to_url(cfg.database)

    if t in ("mysql", "mariadb"):
        # Optional dependency: aiomysql/asyncmy
        try:
            import aiomysql  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ConnectionRuntimeError("MySQL async driver not installed (install aiomysql)") from e
        url = URL.create(
            drivername="mysql+aiomysql",
            username=cfg.username or None,
            password=cfg.password or None,
            host=cfg.host or "localhost",
            port=int(cfg.port or 3306),
            database=cfg.database or None,
        )
        return str(url)

    raise ConnectionRuntimeError(f"Unsupported connector_type: {cfg.connector_type}")


def create_engine_for_connection(cfg: ConnectionRuntimeConfig) -> AsyncEngine:
    url = build_async_sqlalchemy_url(cfg)
    connect_args: dict[str, Any] = {}

    t = (cfg.connector_type or "").strip().lower()
    timeout = int(cfg.timeout_seconds)
    if t in ("postgres", "postgresql"):
        # asyncpg supports a connection-level timeout and per-session server settings.
        connect_args["timeout"] = timeout
        connect_args["server_settings"] = {"statement_timeout": str(timeout * 1000)}
    elif t in ("sqlite",):
        connect_args["timeout"] = timeout
    elif t in ("mysql", "mariadb"):
        connect_args["connect_timeout"] = timeout

    return create_async_engine(
        url,
        poolclass=NullPool,
        connect_args=connect_args,
        future=True,
    )


async def test_connection(cfg: ConnectionRuntimeConfig) -> float:
    engine = create_engine_for_connection(cfg)
    try:
        start = time.perf_counter()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return (time.perf_counter() - start) * 1000.0
    finally:
        await engine.dispose()


async def run_query(cfg: ConnectionRuntimeConfig, sql: str, *, jsonable: bool = True) -> dict[str, Any]:
    engine = create_engine_for_connection(cfg)
    try:
        start = time.perf_counter()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.mappings().all()
            cols = list(result.keys())
        duration_ms = (time.perf_counter() - start) * 1000.0
        raw = [dict(r) for r in rows]
        data = [to_jsonable(r) for r in raw] if jsonable else raw
        return {
            "columns": cols,
            "rows": data,
            "row_count": int(len(data)),
            "execution_time_ms": float(duration_ms),
        }
    finally:
        await engine.dispose()


async def list_tables(cfg: ConnectionRuntimeConfig) -> list[str]:
    engine = create_engine_for_connection(cfg)
    try:
        async with engine.connect() as conn:
            def _sync_inspect(sync_conn):
                insp = inspect(sync_conn)
                return insp.get_table_names()

            tables = await conn.run_sync(_sync_inspect)
        return [str(t) for t in (tables or [])]
    finally:
        await engine.dispose()
