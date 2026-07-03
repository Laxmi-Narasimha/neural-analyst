from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Deque


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 1:
        return float(max(values))
    xs = sorted(values)
    idx = int(round(p * (len(xs) - 1)))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


@dataclass
class OperatorMetrics:
    runs: int = 0
    errors: int = 0
    durations_ms: Deque[float] | None = None
    scanned_rows: Deque[int] | None = None


@dataclass
class HttpRouteMetrics:
    requests: int = 0
    errors: int = 0
    durations_ms: Deque[float] | None = None
    status_counts: dict[str, int] | None = None


@dataclass
class JobRunMetrics:
    runs: int = 0
    failures: int = 0
    durations_seconds: Deque[float] | None = None


class MetricsCollector:
    """
    Lightweight in-process metrics collector.

    Notes:
    - Intended for dev/single-process usage. For production, replace/augment
      with a proper metrics backend (Prometheus, OTEL, etc).
    - All updates are thread-safe since compute runs in thread pools.
    """

    def __init__(self, *, max_history: int = 512) -> None:
        self._lock = Lock()
        self._max_history = int(max(64, max_history))
        self._started_at = datetime.utcnow()
        self._operators: dict[str, OperatorMetrics] = defaultdict(OperatorMetrics)
        self._http_routes: dict[str, HttpRouteMetrics] = defaultdict(HttpRouteMetrics)
        self._jobs: dict[str, JobRunMetrics] = defaultdict(JobRunMetrics)

    def record_operator(
        self,
        *,
        operator_name: str,
        duration_ms: float,
        scanned_rows: int,
        success: bool,
    ) -> None:
        name = str(operator_name or "").strip() or "unknown"
        dur = float(duration_ms) if duration_ms is not None else 0.0
        rows = int(scanned_rows) if scanned_rows is not None else 0

        with self._lock:
            m = self._operators[name]
            m.runs += 1
            if not success:
                m.errors += 1
            if m.durations_ms is None:
                m.durations_ms = deque(maxlen=self._max_history)
            if m.scanned_rows is None:
                m.scanned_rows = deque(maxlen=self._max_history)
            m.durations_ms.append(dur)
            m.scanned_rows.append(rows)

    def record_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        mtd = str(method or "").strip().upper() or "GET"
        rte = str(route or "").strip() or "/unknown"
        key = f"{mtd} {rte}"
        status_key = str(int(status_code))
        dur = float(duration_ms) if duration_ms is not None else 0.0
        is_error = int(status_code) >= 500

        with self._lock:
            m = self._http_routes[key]
            m.requests += 1
            if is_error:
                m.errors += 1
            if m.durations_ms is None:
                m.durations_ms = deque(maxlen=self._max_history)
            if m.status_counts is None:
                m.status_counts = {}
            m.durations_ms.append(dur)
            m.status_counts[status_key] = int(m.status_counts.get(status_key, 0)) + 1

    def record_job_run(
        self,
        *,
        job_type: str,
        duration_seconds: float,
        success: bool,
    ) -> None:
        jt = str(job_type or "").strip() or "unknown"
        dur = max(0.0, float(duration_seconds or 0.0))

        with self._lock:
            m = self._jobs[jt]
            m.runs += 1
            if not success:
                m.failures += 1
            if m.durations_seconds is None:
                m.durations_seconds = deque(maxlen=self._max_history)
            m.durations_seconds.append(dur)

    def _distribution(self, values: list[float]) -> dict[str, Any]:
        avg = float(sum(values) / len(values)) if values else None
        return {
            "avg": avg,
            "p50": _pct(values, 0.50),
            "p95": _pct(values, 0.95),
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
            "n": int(len(values)),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ops = dict(self._operators)
            http = dict(self._http_routes)
            jobs = dict(self._jobs)
            started_at = self._started_at

        out: dict[str, Any] = {
            "started_at": started_at.isoformat(),
            "operators": {},
            "http_routes": {},
            "jobs": {},
        }

        for name, m in ops.items():
            durs = list(m.durations_ms or [])
            rows = list(m.scanned_rows or [])

            out["operators"][name] = {
                "runs": int(m.runs),
                "errors": int(m.errors),
                "error_rate": float(m.errors) / float(max(m.runs, 1)),
                "duration_ms": self._distribution(durs),
                "scanned_rows": {
                    "avg": float(sum(rows) / len(rows)) if rows else None,
                    "p50": _pct([float(r) for r in rows], 0.50) if rows else None,
                    "p95": _pct([float(r) for r in rows], 0.95) if rows else None,
                    "min": int(min(rows)) if rows else None,
                    "max": int(max(rows)) if rows else None,
                    "n": int(len(rows)),
                },
            }

        for route_key, m in http.items():
            durs = list(m.durations_ms or [])
            out["http_routes"][route_key] = {
                "requests": int(m.requests),
                "errors": int(m.errors),
                "error_rate": float(m.errors) / float(max(m.requests, 1)),
                "duration_ms": self._distribution(durs),
                "status_counts": dict(m.status_counts or {}),
            }

        for jt, m in jobs.items():
            durs = list(m.durations_seconds or [])
            out["jobs"][jt] = {
                "runs": int(m.runs),
                "failures": int(m.failures),
                "failure_rate": float(m.failures) / float(max(m.runs, 1)),
                "duration_seconds": self._distribution(durs),
            }

        return out


metrics_collector = MetricsCollector()
