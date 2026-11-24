"""Scan uploads in batches, then calculate exact aggregates on disk-backed rows."""
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pandas as pd

from app.compute.operators.base import OperatorResult
from app.services import dataset_batches
from app.services.chat_scope import (ScopeQuestion, bind_conditions, condition_leaves, scoped_dataset,
                                     _DATE_TYPES, _NUMBER_TYPES)
from app.services.dataset_loader import LoadedDataset, _sha256_fileobj
from app.services.object_store import get_object_store
from app.services.retail_calendar import RetailCalendar
from app.utils.hashing import hash_json


@dataclass
class DiskScan:
    loaded: LoadedDataset
    filters: list
    rows: int
    sampled_ops: list


def needs_scan(record):
    profile = record.profile_report or {}
    if ((record.file_size_bytes or 0) >= dataset_batches.LARGE_BYTES or
            (record.row_count or 0) > dataset_batches.LARGE_ROWS or profile.get("sample_rows") or
            profile.get("batched_source")):
        return True
    # Older profiles predate batched_source. A compact archive may expand substantially.
    if record.file_format in {"parquet", "xlsx", "xls"}:
        path = get_object_store().ensure_local_path(record.file_path, expected_size_bytes=record.file_size_bytes,
                                                     filename_hint=record.original_filename)
        return dataset_batches.large_file(Path(path), record.file_format)
    return False


def _result(name, table, summary, metrics=None):
    return OperatorResult({name: table}, metrics or {}, {}, summary)


def _aggregate(con, op, p, fields, numeric, dates):
    value = p.get("value_column")
    group = p.get("group_by")
    time = p.get("time_column")
    if op in {"segment_summary", "resample_aggregate"} and not value:
        value = next(iter(numeric), None)
    v = fields[value] if value in numeric else "NULL::DOUBLE"
    if op == "correlation_matrix":
        names = numeric[:max(2, min(100, int(p.get("max_columns", 25))))]
        pairs = list(combinations(names, 2))
        valid = {n: f"CASE WHEN isfinite({fields[n]}) THEN {fields[n]} END" for n in names}
        rows = []
        for i in range(0, len(pairs), 32):
            part = pairs[i:i + 32]
            expr = ", ".join(f"corr({valid[a]}, {valid[b]})" for a, b in part)
            values = con.execute("SELECT " + expr + " FROM matches").fetchone()
            rows.extend((a, b, val) for (a, b), val in zip(part, values) if pd.notna(val))
        rows.sort(key=lambda r: abs(r[2]), reverse=True)
        table = pd.DataFrame(rows[:200], columns=["column_a", "column_b", "corr"])
        return _result("correlations", table, {"numeric_columns": len(names), "pairs": len(rows)}, {"numeric_columns": len(names)})
    if op == "numeric_summary":
        rows = []
        for name in numeric[:max(1, min(200, int(p.get("max_columns", 25))))]:
            col = f"CASE WHEN isfinite({fields[name]}) THEN {fields[name]} END"
            stats = con.execute(f"SELECT count({col}), avg({col}), stddev_samp({col}), min({col}), "
                                f"quantile_cont({col}, 0.25), median({col}), quantile_cont({col}, 0.75), "
                                f"max({col}) FROM matches").fetchone()
            rows.append((name, *stats))
        table = pd.DataFrame(rows, columns=["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
        return _result("numeric_summary", table, {"numeric_columns": len(rows)}, {"numeric_columns": len(rows)})
    if op in {"ratio", "share_of_total", "period_compare"}:
        return _derived(con, op, p, fields, numeric)
    if op == "rate":
        col, group = p["column"], p.get("group_by")
        c = fields[col]
        kind = (con.execute(f"SELECT typeof({c}) FROM matches LIMIT 1").fetchone() or ("VARCHAR",))[0]
        positive = str(p["positive_value"]).strip()
        args: list = []
        if kind in {"BIGINT", "DOUBLE", "INTEGER", "FLOAT"}:
            try:
                hit = f"{c} = ?"
                args = [float(positive)]
            except ValueError:
                hit = "FALSE"
        elif kind == "BOOLEAN":
            hit = f"{c} = ?"
            args = [positive.casefold() in {"true", "1", "yes", "y"}]
        else:
            hit = f"lower(trim(cast({c} AS VARCHAR))) = ?"
            args = [positive.casefold()]
        select = (f'SELECT {"cast(" + fields[group] + " AS VARCHAR) AS " + chr(34) + "group" + chr(34) + ", " if group else ""}'
                  f"sum(CASE WHEN {c} IS NOT NULL AND {hit} THEN 1 ELSE 0 END) AS matching, count({c}) AS rows FROM matches")
        if group:
            select += f" GROUP BY {fields[group]} HAVING count({c}) > 0"
        table = con.execute(select, args).fetchdf()
        table["matching"] = table["matching"].astype(int)
        table["rows"] = table["rows"].astype(int)
        table["rate"] = table["matching"] / table["rows"].where(table["rows"] > 0)
        if group:
            table = table.sort_values(["rate", "rows"], ascending=[bool(p.get("ascending")), False], kind="stable")
        table["rate_pct"] = table["rate"] * 100
        missing = con.execute(f"SELECT count(*) - count({c}) FROM matches").fetchone()[0]
        cols = (["group"] if group else []) + ["rate", "rate_pct", "matching", "rows"]
        return _result("result", table[cols].head(1000), {"column": col, "positive_value": p["positive_value"],
                                                           "group_by": group, "rows_missing_value": int(missing)})
    if op == "measure_aggregate" and p["aggregation"] in {"count", "distinct_count"} and not p.get("retail_unit"):
        if p["aggregation"] == "distinct_count":
            expr = f"count(DISTINCT {fields[value]})"
        else:
            expr = "count(*)"
        projection = f'{fields[group]} AS "group", ' if group else ""
        order = ' GROUP BY 1 ORDER BY value DESC NULLS LAST, 1 LIMIT 1001' if group else ""
        table = con.execute(f"SELECT {projection}{expr} AS value FROM matches{order}").fetchdf()
        return _result("result", table.head(1000), {**p, "groups_truncated": len(table) > 1000})
    if op == "measure_aggregate":
        fn = {"mean": "avg", "sum": "sum", "min": "min", "max": "max", "median": "median"}[p["aggregation"]]
        if p.get("retail_unit"):
            t = fields[time]
            low, high, missing = con.execute(f"SELECT min({t}), max({t}), count(*)-count({t}) FROM matches").fetchone()
            table = pd.DataFrame(columns=["group", "value"])
            if low is not None:
                cal = RetailCalendar(**p["retail_calendar"])
                bounds = pd.DataFrame(cal.buckets(low.date(), high.date(), p["retail_unit"]), columns=["lo", "hi", "label"])
                con.register("retail_bounds", bounds)
                try:
                    table = con.execute(f'SELECT label AS "group", {fn}({v}) AS value FROM matches '
                                        f'JOIN retail_bounds ON {t} >= lo AND {t} < hi '
                                        'GROUP BY label ORDER BY label LIMIT 1001').fetchdf()
                finally:
                    con.unregister("retail_bounds")
            return _result("result", table.head(1000), {**p, "group_by": "retail " + p["retail_unit"],
                                                        "undated_rows": missing, "groups_truncated": len(table) > 1000})
        projection = f'{fields[group]} AS "group", ' if group else ""
        order = ' GROUP BY 1 ORDER BY value DESC NULLS LAST, 1 LIMIT 1001' if group else ""
        table = con.execute(f"SELECT {projection}{fn}({v}) AS value FROM matches{order}").fetchdf()
        return _result("result", table.head(1000), {**p, "groups_truncated": len(table) > 1000})
    if op == "segment_summary" and group:
        rank = p.get("sort_by", "count")
        if rank not in {"value_mean", "value_sum", "value_median", "count"}:
            rank = "count"
        limit = max(1, min(200, int(p.get("limit", 50))))
        direction = "ASC" if p.get("ascending") else "DESC"
        having = f"HAVING count({v}) > 0" if rank == "value_mean" else ""
        table = con.execute(f'SELECT cast({fields[group]} as varchar) AS "group", count(*) AS count, '
                            f'avg({v}) AS value_mean, sum({v}) AS value_sum, median({v}) AS value_median '
                            f'FROM matches GROUP BY {fields[group]} {having} ORDER BY {rank} {direction} NULLS LAST, '
                            f'count DESC, 1 LIMIT {limit}').fetchdf()
        return _result("segments", table, {**p, "value_column": value, "sort_by": rank, "limit": limit},
                       {"segments": len(table)})
    if op == "resample_aggregate":
        time = time or next(iter(dates), None)
        if not time:
            raise ScopeQuestion("Select a timestamp column for the trend.")
        t = fields[time]
        low, high = con.execute(f"SELECT min({t}), max({t}) FROM matches").fetchone()
        codes = {"D": ("day", "D"), "W": ("week", "W-SUN"), "M": ("month", "ME"),
                 "Q": ("quarter", "QE"), "Y": ("year", "YE")}
        freq = p.get("freq", "M")
        if freq not in codes:
            freq = "M"
        rows = pd.DataFrame(columns=["period", "count", "value_mean", "value_sum"])
        if low is not None:
            periods = {"D": "D", "W": "W-SUN", "M": "M", "Q": "Q", "Y": "Y"}
            for candidate in list(codes)[list(codes).index(freq):]:
                freq = candidate
                first = pd.Period(low, freq=periods[freq])
                last = pd.Period(high, freq=periods[freq])
                if last.ordinal - first.ordinal + 1 <= int(p.get("max_points", 200)):
                    break
            unit = codes[freq][0]
            rows = con.execute(f"SELECT date_trunc('{unit}', {t}) AS period, count(*) AS count, "
                               f"avg({v}) AS value_mean, sum({v}) AS value_sum FROM matches "
                               f"WHERE {t} IS NOT NULL GROUP BY 1 ORDER BY 1").fetchdf()
            rows.index = pd.DatetimeIndex(rows.pop("period")).to_period(periods[freq])
            rows = rows.reindex(pd.period_range(first, last, freq=periods[freq]))
            rows["count"] = rows["count"].fillna(0).astype(int)
            if value:
                rows["value_sum"] = rows["value_sum"].fillna(0)
            rows.insert(0, "period", rows.index.to_timestamp(how="end").normalize().strftime("%Y-%m-%d %H:%M:%S"))
            rows.reset_index(drop=True, inplace=True)
        return _result("resample", rows, {"time_column": time, "value_column": value, "freq": freq, "points": len(rows)},
                       {"points": len(rows)})
    return None


def _derived(con, op, p, fields, numeric):
    """SQL for the base sums and counts; the answer itself comes from the shared finish functions."""
    from app.compute.operators import derived as d

    group = p.get("group_by")
    g_sel = f'cast({fields[group]} AS VARCHAR) AS "group", ' if group else ""
    g_by = f" GROUP BY {fields[group]}" if group else ""

    def num(name):
        if name not in numeric:
            raise ValueError(f"{name} is not numeric")
        return f"CASE WHEN isfinite({fields[name]}) THEN {fields[name]} END"

    if op == "ratio":
        parts = {"num": num(p["numerator_column"]), "den": num(p["denominator_column"])}
        if p.get("subtract_column"):
            parts["sub"] = num(p["subtract_column"])
        complete = " AND ".join(f"({e}) IS NOT NULL" for e in parts.values())
        sums = ", ".join(f"coalesce(sum({e}), 0) AS {k}" for k, e in parts.items())
        base = con.execute(f"SELECT {g_sel}{sums}, count(*) AS rows FROM matches WHERE {complete}{g_by}").fetchdf()
        excluded = con.execute(f"SELECT count(*) FROM matches WHERE NOT ({complete})").fetchone()[0]
        if not group and base.empty:
            base = pd.DataFrame([{**{k: 0.0 for k in parts}, "rows": 0}])
        return d.ratio_finish(base, p, int(excluded))
    if op == "share_of_total":
        expr = f"sum({num(p['value_column'])})" if p.get("value_column") else "count(*)"
        base = con.execute(f'SELECT cast({fields[group]} AS VARCHAR) AS "group", {expr} AS value FROM matches '
                           f"GROUP BY {fields[group]}").fetchdf()
        return d.share_finish(base, p)
    # period_compare
    t = fields[p["time_column"]]
    data_max = con.execute(f"SELECT max({t}) FROM matches").fetchone()[0]
    periods = d.resolve_periods(p, pd.Timestamp(data_max) if data_max is not None else None)
    agg = p.get("aggregation", "sum")
    v = num(p["value_column"]) if (p.get("value_column") and agg != "count") else "1.0"
    (cs, ce), (ps, pe) = periods.current, periods.previous
    cur = f"{t} >= TIMESTAMP '{cs.isoformat()}' AND {t} < TIMESTAMP '{ce.isoformat()}' AND ({v}) IS NOT NULL"
    prev = f"{t} >= TIMESTAMP '{ps.isoformat()}' AND {t} < TIMESTAMP '{pe.isoformat()}' AND ({v}) IS NOT NULL"
    base = con.execute(
        f"SELECT {g_sel}coalesce(sum(CASE WHEN {cur} THEN {v} END), 0) AS cur_sum, "
        f"count(CASE WHEN {cur} THEN 1 END) AS cur_n, "
        f"coalesce(sum(CASE WHEN {prev} THEN {v} END), 0) AS prev_sum, "
        f"count(CASE WHEN {prev} THEN 1 END) AS prev_n FROM matches{g_by}").fetchdf()
    return d.period_finish(base, p, periods)


def prepare_scan(record, edit, previous, plan, bindings):
    try:
        return _scan(record, edit, previous, plan, bindings)
    except ScopeQuestion:
        raise
    except (duckdb.OutOfMemoryException, OSError) as error:
        raise ScopeQuestion("The disk-backed scan could not finish with the available memory or temporary disk space. Retry after making more resources available.") from error
    except (duckdb.Error, ValueError, OverflowError) as error:
        raise ScopeQuestion("The bounded reader could not calculate this result using the stored column types. Check the source data types and reprocess the file.") from error


def _scan(record, edit, previous, plan, bindings):
    path = Path(get_object_store().ensure_local_path(record.file_path, expected_size_bytes=record.file_size_bytes,
                                                     filename_hint=record.original_filename))
    version = (record.profile_report or {}).get("file_hash")
    if not version:
        with path.open("rb") as f:
            version = _sha256_fileobj(f)
    schema = record.schema_info or {}
    cols = {c["name"]: c for c in schema.get("columns", [])}
    empty = LoadedDataset(record.id, record.name, version, "", record.file_format,
                          pd.DataFrame(columns=list(cols)), {}, schema)
    wanted = {str(c["value"]).casefold() for c in condition_leaves(edit.conditions)
              if not c.get("column") and c["op"] != "date"}
    # Exclusions are checked against the data too: excluding a value that never occurs changes nothing.
    for c in condition_leaves(edit.conditions):
        if c.get("column") and c["op"] in {"ne", "not_in"}:
            wanted.update(str(v).casefold() for v in (c["value"] if isinstance(c.get("value"), list) else [c.get("value")]))
    hits = {v: set() for v in wanted}
    def batches():
        return dataset_batches.frames(path, record.file_format, schema)
    if wanted:
        for frame in batches():
            for name in cols:
                if cols[name].get("is_potential_pii") or cols[name].get("inferred_type") in _DATE_TYPES:
                    continue
                values = frame[name].astype("string").str.casefold()
                for v in wanted:
                    if name not in hits[v] and values.eq(v).any():
                        hits[v].add(name)
    conditions = bind_conditions(edit, previous, empty, plan, bindings, lambda n, v: n in hits.get(str(v).casefold(), ()))
    if conditions and any(s["operator"] in {"ml_train_baseline", "ml_task_inference"} for s in plan):
        raise ScopeQuestion("Save the filtered rows as a versioned dataset before training.")
    fields = {name: f"c{i}" for i, name in enumerate(cols)}
    types = {}
    numeric = []
    dates = []
    for name, meta in cols.items():
        kind = meta["inferred_type"]
        if kind in _DATE_TYPES:
            types[name] = "TIMESTAMP"
            dates.append(name)
        elif kind in _NUMBER_TYPES:
            types[name] = "BIGINT" if kind in {"integer", "int"} else "DOUBLE"
            if not meta.get("is_potential_pii"):
                numeric.append(name)
        elif kind == "boolean":
            types[name] = "BOOLEAN"
        else:
            types[name] = "VARCHAR"
    limit = dataset_batches.BATCH_ROWS
    aggregates = {"measure_aggregate", "resample_aggregate", "segment_summary", "numeric_summary", "correlation_matrix", "rate",
                  "ratio", "share_of_total", "period_compare"}
    exact = {"dataset_overview", "preview_rows", "schema_snapshot", "duckdb_profile", "missingness_scan"}
    needs_all = any(s["operator"] not in aggregates | exact or
                    (s["operator"] == "segment_summary" and not s.get("params", {}).get("group_by")) for s in plan)
    needed = set()
    for step in plan:
        op, params = step["operator"], step.get("params", {})
        if op in aggregates:
            needed.update(params.get(k) for k in ("value_column", "time_column", "group_by", "column",
                                                   "numerator_column", "denominator_column", "subtract_column"))
            if op in {"numeric_summary", "correlation_matrix"}:
                needed.update(numeric[:max(2, min(200, int(params.get("max_columns", 25))))])
            elif (not params.get("value_column") and numeric and op not in {"rate", "ratio", "share_of_total", "period_compare"}
                  and params.get("aggregation") not in {"count", "distinct_count"}):
                needed.add(numeric[0])
            if op == "resample_aggregate" and not params.get("time_column") and dates:
                needed.add(dates[0])
    projected = [n for n in cols if needs_all or n in needed]
    if not projected and cols and any(s["operator"] in aggregates for s in plan):
        projected = [next(iter(cols))]  # e.g. a plain row count still needs rows to count
    with TemporaryDirectory(prefix="neural-analyst-") as work:
        with duckdb.connect(str(Path(work) / "scan.duckdb"), config={"memory_limit": "128MB", "threads": "1"}) as con:
            if projected:
                con.execute("CREATE TABLE matches (" + ", ".join(f"{fields[n]} {types[n]}" for n in projected) + ")")
            count = 0
            head = []
            nulls = pd.Series(0, index=list(cols), dtype="int64")
            for batch in batches():
                if conditions:
                    batch = scoped_dataset(replace(empty, df=batch), conditions).df
                if batch.empty:
                    continue
                if record.file_format == "xls":
                    limit = dataset_batches.xls_sample_limit(batch, limit)
                head_limit = min(200, limit) if record.file_format == "xls" else 200
                if count < head_limit:
                    head.append(batch.iloc[:head_limit - count].copy())
                count += len(batch)
                nulls += batch.isna().sum()
                if not projected:
                    continue
                batch = batch.loc[:, projected].copy()
                for name in set(dates).intersection(projected):
                    batch[name] = pd.to_datetime(batch[name], utc=True, format="mixed", errors="coerce").dt.tz_localize(None)
                con.register("incoming", batch)
                con.execute("INSERT INTO matches SELECT * FROM incoming")
                con.unregister("incoming")
            prepared = {}
            sampled = []
            for step in plan:
                op, params = step["operator"], step.get("params", {})
                result = None
                if count:
                    result = _aggregate(con, op, params, fields, numeric, dates)
                    if op == "missingness_scan":
                        table = nulls.rename("null_count").rename_axis("column").reset_index()
                        table["null_pct"] = table["null_count"] / count
                        pct = float(nulls.sum() / max(1, count * len(cols)))
                        result = _result("missingness", table.sort_values("null_pct", ascending=False),
                                         {"rows": count, "columns": len(cols), "missing_pct": pct}, {"missing_pct": pct})
                    elif op == "duckdb_profile":
                        table = pd.DataFrame([{"row_count": count}])
                        result = _result("profile", table, {"row_count": count, "column_count": len(cols), "engine": "duckdb"})
                if result is not None:
                    prepared[op, hash_json(params)] = result
                elif op not in exact and count > limit:
                    sampled.append(op)
            first = pd.concat(head, ignore_index=True) if head else empty.df
            frame = first
            if needs_all:
                sample_sql = f"USING SAMPLE reservoir({limit} ROWS) REPEATABLE(42)" if sampled else f"LIMIT {limit}"
                frame = con.execute("SELECT * FROM matches " + sample_sql).fetchdf()
                frame.columns = list(cols)
            if "preview_rows" in {s["operator"] for s in plan} and sampled:
                from app.compute.operators.eda import PreviewRowsOperator
                from app.compute.operators.base import OperatorContext
                ctx = OperatorContext(record.id, version, first, {}, schema)
                params = next(s["params"] for s in plan if s["operator"] == "preview_rows")
                prepared["preview_rows", hash_json(params)] = PreviewRowsOperator().run(ctx, params)
    loaded = replace(empty, df=frame)
    # Worksheet profile statistics describe individual samples, not the selected rows.
    if conditions or schema.get("worksheets"):
        loaded = scoped_dataset(loaded, [])
    return DiskScan(replace(loaded, profile_report={"row_count": count, "column_count": len(cols)},
                            prepared=prepared, disk_scan=True), conditions, count, sampled)
