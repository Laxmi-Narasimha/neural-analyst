"""Which databases a user's saved connection may reach.

A connection runs on the server, so without limits a user could point it at things only the server
can reach: the platform's own database (every user's data, password hashes), a SQLite file on the
server's disk, cloud metadata endpoints, or other internal services.

Rules:
- The platform's own database is never reachable, in any mode.
- With sign-in (AUTH_MODE=jwt, several users on one server):
  - SQLite files only inside CONNECTIONS_SQLITE_ROOT, and off when that is unset;
  - hosts that resolve to loopback, private, link-local, reserved or multicast addresses are
    refused unless CONNECTIONS_ALLOW_PRIVATE_HOSTS=true (self-hosted teams whose databases are on
    their own network).
- Without sign-in (AUTH_MODE=local) there is a single user who already controls the machine, so
  local files and private hosts are allowed.

The check runs when a connection is saved and again before every use, because a host name can be
re-pointed after it was saved.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import make_url

from app.core.config import AuthMode, settings


class ConnectionNotAllowed(ValueError):
    pass


_MEMORY = {":memory:", "file::memory:", ""}


def _multi_user() -> bool:
    return settings.auth_mode == AuthMode.JWT


def _resolve_path(value: str) -> Path:
    p = Path(value).expanduser()
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


def _platform_database() -> tuple[Optional[Path], Optional[str], Optional[int]]:
    """(sqlite file, host, port) of the platform's own database."""
    try:
        url = make_url(str(settings.database.url))
    except Exception:
        return None, None, None
    if url.get_backend_name() == "sqlite":
        db = url.database or ""
        return (None if db in _MEMORY else _resolve_path(db)), None, None
    return None, url.host, url.port or (5432 if url.get_backend_name().startswith("postgres") else 3306)


def _addresses(host: str, port: int) -> set[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ConnectionNotAllowed(f"Cannot resolve host {host!r}") from e
    return {info[4][0] for info in infos}


def _is_internal(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any((ip.is_loopback, ip.is_private, ip.is_link_local, ip.is_reserved,
                ip.is_multicast, ip.is_unspecified))


def check_sqlite(database: str) -> None:
    db = (database or "").strip()
    if db in _MEMORY:
        return
    if db.lower().startswith("file:"):
        raise ConnectionNotAllowed("Give the SQLite database as a plain file path")
    path = _resolve_path(db)
    platform_file, _, _ = _platform_database()
    if platform_file is not None and path == platform_file:
        raise ConnectionNotAllowed("That is this platform's own database")
    if not _multi_user():
        return
    root = settings.connections.sqlite_root
    if root is None:
        raise ConnectionNotAllowed("SQLite file connections are turned off on this server")
    root = _resolve_path(str(root))
    if path != root and root not in path.parents:
        raise ConnectionNotAllowed(f"SQLite files must be inside {root}")


def check_network(host: str, port: int, default_port: int) -> None:
    host = (host or "localhost").strip()
    port = int(port or default_port)
    addresses = _addresses(host, port)
    _, platform_host, platform_port = _platform_database()
    if platform_host and port == platform_port:
        platform_addresses = _addresses(platform_host, port) if platform_host else set()
        if addresses & platform_addresses:
            raise ConnectionNotAllowed("That is this platform's own database")
    if _multi_user() and not settings.connections.allow_private_hosts:
        internal = sorted(a for a in addresses if _is_internal(a))
        if internal:
            raise ConnectionNotAllowed(
                f"{host} resolves to an internal address ({internal[0]}); connections from this "
                "server may only reach public database hosts"
            )


def assert_connection_allowed(connector_type: str, host: str, port: int, database: str) -> None:
    t = (connector_type or "").strip().lower()
    if t == "sqlite":
        check_sqlite(database)
    elif t in ("postgres", "postgresql"):
        check_network(host, port, 5432)
    elif t in ("mysql", "mariadb"):
        check_network(host, port, 3306)
    else:
        raise ConnectionNotAllowed("Unsupported connector type")
