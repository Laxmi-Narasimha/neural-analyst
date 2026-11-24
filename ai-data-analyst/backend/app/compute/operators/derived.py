"""Derived metrics: ratios, shares of total, and period-over-period change with contribution.

Each operator is split in two so the pandas path and the batched DuckDB path
(app/services/chat_disk.py) cannot drift apart:

1. a *base* aggregation that only produces sums and counts per group
   (pandas here, SQL in chat_disk), and
2. a shared *finish* function that turns the base table into the answer.

Only step 1 differs between the paths, and it is plain sums and counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.compute.operators.base import OperatorContext, OperatorResult

PERIOD_AGGREGATIONS = ("sum", "mean", "count")


def _pii(ctx: OperatorContext) -> set[str]:
    cols = (ctx.schema_info or {}).get("columns") or []
    return {str(c.get("name")) for c in cols if isinstance(c, dict) and c.get("is_potential_pii")}


def _check_columns(ctx: OperatorContext, *names: Optional[str]) -> None:
    pii = _pii(ctx)
    for name in names:
        if name is None:
            continue
        if name in pii:
            raise ValueError("Select a non-sensitive analysis column")
        if name not in ctx.df.columns:
            raise ValueError(f"Column {name} is not in the dataset")


def _numeric(series: pd.Series, name: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{name} is not numeric")
    return series.astype("float64").replace([np.inf, -np.inf], np.nan)


def parse_times(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, errors="coerce")
        return out.dt.tz_convert("UTC").dt.tz_localize(None) if out.dt.tz is not None else out
    return pd.to_datetime(series, errors="coerce", format="mixed", utc=True).dt.tz_localize(None)


def _div(num: Any, den: Any) -> Optional[float]:
    try:
        num, den = float(num), float(den)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Ratio: (sum(numerator) - sum(subtract)) / sum(denominator), overall or per group
# ---------------------------------------------------------------------------

def ratio_base(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    num, den, sub, group = p["numerator_column"], p["denominator_column"], p.get("subtract_column"), p.get("group_by")
    frame = pd.DataFrame({"num": _numeric(df[num], num), "den": _numeric(df[den], den)})
    if sub:
        frame["sub"] = _numeric(df[sub], sub)
    complete = frame.notna().all(axis=1)
    excluded = int((~complete).sum())
    frame = frame[complete]
    if group:
        frame["group"] = df.loc[frame.index, group]
        base = frame.groupby("group", dropna=False, observed=True).agg(
            num=("num", "sum"), den=("den", "sum"), **({"sub": ("sub", "sum")} if sub else {}), rows=("num", "size"))
        base = base.reset_index()
    else:
        base = pd.DataFrame([{"num": frame["num"].sum(), "den": frame["den"].sum(),
                              **({"sub": frame["sub"].sum()} if sub else {}), "rows": len(frame)}])
    return base, excluded


def ratio_finish(base: pd.DataFrame, p: dict[str, Any], excluded: int) -> OperatorResult:
    scale = 100.0 if p.get("as_percent") else 1.0
    out = base.copy()
    numer = out["num"] - (out["sub"] if "sub" in out else 0.0)
    out["numerator"] = numer
    out["denominator"] = out["den"]
    out["value"] = [None if (v := _div(n, d)) is None else v * scale for n, d in zip(numer, out["den"])]
    cols = (["group"] if p.get("group_by") else []) + ["value", "numerator", "denominator", "rows"]
    out = out[cols]
    if p.get("group_by"):
        out = out.sort_values(["value", "rows"], ascending=[bool(p.get("ascending")), False], na_position="last", kind="stable")
    summary = {k: p.get(k) for k in ("numerator_column", "subtract_column", "denominator_column", "group_by", "as_percent")}
    summary["rows_excluded_missing"] = excluded
    return OperatorResult(tables={"result": out.head(1000)}, metrics={}, charts={}, summary=summary)


class RatioOperator:
    name: str = "ratio"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        _check_columns(ctx, params["numerator_column"], params["denominator_column"], params.get("subtract_column"),
                       params.get("group_by"))
        base, excluded = ratio_base(ctx.df, params)
        return ratio_finish(base, params, excluded)


# ---------------------------------------------------------------------------
# Share of total: each group's share of a measure's total (or of the rows)
# ---------------------------------------------------------------------------

def share_base(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    group, value = p["group_by"], p.get("value_column")
    if value:
        values = _numeric(df[value], value)
        base = values.groupby(df[group], dropna=False, observed=True).sum(min_count=1)
    else:
        base = df.groupby(df[group], dropna=False, observed=True).size()
    return base.rename_axis("group").reset_index(name="value")


def share_finish(base: pd.DataFrame, p: dict[str, Any]) -> OperatorResult:
    out = base.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    total = float(out["value"].sum(skipna=True))
    out["share"] = [_div(v, total) for v in out["value"]]
    out["share_pct"] = [None if s is None else s * 100 for s in out["share"]]
    out = out.sort_values("value", ascending=False, na_position="last", kind="stable")
    summary = {"group_by": p["group_by"], "value_column": p.get("value_column"), "total": total,
               "negative_values": bool((out["value"] < 0).any())}
    return OperatorResult(tables={"result": out[["group", "value", "share", "share_pct"]].head(1000)},
                          metrics={}, charts={}, summary=summary)


class ShareOfTotalOperator:
    name: str = "share_of_total"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        _check_columns(ctx, params["group_by"], params.get("value_column"))
        return share_finish(share_base(ctx.df, params), params)


# ---------------------------------------------------------------------------
# Period comparison: current vs previous period, with each group's contribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Periods:
    current: tuple[date, date]   # [start, end)
    previous: tuple[date, date]
    aligned: bool                # both shortened to the days the data covers in the current period
    requested_current_end: date
    requested_previous_end: date
    data_ends: Optional[date] = None  # set when a finished current period runs past the last date in the data


def _today() -> date:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date()


def _iso(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])


def resolve_periods(p: dict[str, Any], data_max: Optional[pd.Timestamp], today: Optional[date] = None) -> Periods:
    """A current period still in progress (it ends after today) is compared like with like: both periods are
    cut to the days the data covers so far. A finished period keeps the dates asked for; if the data stops
    before it ends, that is reported instead of silently changing the question."""
    cs, ce, ps, pe = (_iso(p[k]) for k in ("current_start", "current_end", "previous_start", "previous_end"))
    today = today or _today()
    aligned = False
    data_ends = None
    eff_ce, eff_pe = ce, pe
    if data_max is not None and not pd.isna(data_max):
        covered_end = pd.Timestamp(data_max).date() + timedelta(days=1)
        if cs < covered_end < ce:
            # The period includes today (or later days): it is still in progress.
            if ce > today and p.get("align_partial", True):
                eff_ce = covered_end
                eff_pe = min(pe, ps + (covered_end - cs))
                aligned = True
            else:
                data_ends = covered_end - timedelta(days=1)
    return Periods((cs, eff_ce), (ps, eff_pe), aligned, ce, pe, data_ends)


def period_base(df: pd.DataFrame, p: dict[str, Any], periods: Periods) -> pd.DataFrame:
    times = parse_times(df[p["time_column"]])
    value = p.get("value_column")
    agg = p.get("aggregation", "sum")
    values = _numeric(df[value], value) if (value and agg != "count") else pd.Series(1.0, index=df.index)
    cur = (times >= pd.Timestamp(periods.current[0])) & (times < pd.Timestamp(periods.current[1]))
    prev = (times >= pd.Timestamp(periods.previous[0])) & (times < pd.Timestamp(periods.previous[1]))
    known = values.notna()
    frame = pd.DataFrame({
        "cur_sum": values.where(cur & known, 0.0), "cur_n": (cur & known).astype(int),
        "prev_sum": values.where(prev & known, 0.0), "prev_n": (prev & known).astype(int),
    })
    if p.get("group_by"):
        frame["group"] = df[p["group_by"]]
        return frame.groupby("group", dropna=False, observed=True).sum().reset_index()
    return frame.sum().to_frame().T


def period_finish(base: pd.DataFrame, p: dict[str, Any], periods: Periods) -> OperatorResult:
    agg = p.get("aggregation", "sum")

    def stat(s: float, n: int) -> Optional[float]:
        if agg == "mean":
            return _div(s, n)
        return float(n) if agg == "count" else float(s)

    def row(r: Any) -> dict[str, Any]:
        cur, prev = stat(r["cur_sum"], int(r["cur_n"])), stat(r["prev_sum"], int(r["prev_n"]))
        change = None if cur is None or prev is None else cur - prev
        pct = None if change is None else (None if (d := _div(change, prev)) is None else d * 100)
        return {"current": cur, "previous": prev, "change": change, "change_pct": pct,
                "current_rows": int(r["cur_n"]), "previous_rows": int(r["prev_n"])}

    totals = base[["cur_sum", "cur_n", "prev_sum", "prev_n"]].sum()
    overall = row(totals)
    tables: dict[str, pd.DataFrame] = {"result": pd.DataFrame([overall])}
    if p.get("group_by"):
        rows = []
        for _, r in base.iterrows():
            item = {"group": r["group"], **row(r)}
            # Contributions add up to the total change only for additive measures.
            if agg in {"sum", "count"} and overall["change"]:
                item["contribution_pct"] = None if item["change"] is None else item["change"] / overall["change"] * 100
            rows.append(item)
        by = pd.DataFrame(rows)
        if not by.empty:
            by = by.assign(_abs=by["change"].abs()).sort_values("_abs", ascending=False, na_position="last",
                                                                 kind="stable").drop(columns="_abs")
        tables = {"result": by.head(1000), "overall": tables["result"]}
    summary = {
        "time_column": p["time_column"], "value_column": p.get("value_column"), "aggregation": agg,
        "group_by": p.get("group_by"),
        "current_start": periods.current[0].isoformat(), "current_end": periods.current[1].isoformat(),
        "previous_start": periods.previous[0].isoformat(), "previous_end": periods.previous[1].isoformat(),
        "aligned_to_data": periods.aligned,
        "data_ends": periods.data_ends.isoformat() if periods.data_ends else None,
        "requested_current_end": periods.requested_current_end.isoformat(),
        "requested_previous_end": periods.requested_previous_end.isoformat(),
        **{f"overall_{k}": v for k, v in overall.items()},
    }
    return OperatorResult(tables=tables, metrics={}, charts={}, summary=summary)


class PeriodCompareOperator:
    name: str = "period_compare"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        _check_columns(ctx, params["time_column"], params.get("value_column"), params.get("group_by"))
        if params.get("aggregation", "sum") != "count" and not params.get("value_column"):
            raise ValueError("A numeric value column is required unless counting rows")
        times = parse_times(ctx.df[params["time_column"]])
        periods = resolve_periods(params, times.max() if times.notna().any() else None)
        return period_finish(period_base(ctx.df, params, periods), params, periods)


P0_DERIVED_OPERATORS = [RatioOperator(), ShareOfTotalOperator(), PeriodCompareOperator()]
