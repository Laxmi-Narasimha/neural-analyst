from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass(frozen=True)
class TransformOutput:
    df: pd.DataFrame
    metrics: dict[str, Any]
    warnings: list[str]


class DatasetTransformError(ValueError):
    pass


ALLOWED_TRANSFORM_OPS: set[str] = {
    "type_convert",
    "fill_missing",
    "drop_missing",
    "deduplicate",
    "string_normalize",
    "drop_columns",
    "rename_columns",
    "filter_rows",
    "sort_rows",
    "limit_rows",
    "time_features",
    "bin_numeric",
    "clip_outliers",
    "encode_categorical",
}


def _require_columns(df: pd.DataFrame, cols: list[str], *, op: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DatasetTransformError(f"{op}: missing columns: {missing}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_step(step: Any) -> dict[str, Any]:
    if hasattr(step, "model_dump"):
        step = step.model_dump()
    if not isinstance(step, dict):
        raise DatasetTransformError("step must be an object")

    op = step.get("op") or step.get("operator") or step.get("type")
    params = step.get("params") or {}
    if not isinstance(params, dict):
        raise DatasetTransformError("step.params must be an object")

    op = str(op or "").strip().lower()
    if not op:
        raise DatasetTransformError("step.op is required")
    if op not in ALLOWED_TRANSFORM_OPS:
        raise DatasetTransformError(f"unsupported transform op: {op}")

    return {"op": op, "params": params}


def _coerce_like_series(series: pd.Series, value: Any) -> Any:
    if value is None:
        return None
    if is_datetime64_any_dtype(series.dtype):
        try:
            return pd.to_datetime(value, errors="coerce")
        except Exception:
            return value
    if is_numeric_dtype(series.dtype):
        try:
            coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            return None if pd.isna(coerced) else coerced
        except Exception:
            return value
    return value


def _filter_condition_mask(df: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    col = str(condition.get("column") or "").strip()
    if not col:
        raise DatasetTransformError("filter_rows: each condition must include column")
    _require_columns(df, [col], op="filter_rows")

    s = df[col]
    op = str(condition.get("op") or condition.get("operator") or "eq").strip().lower()
    case_sensitive = bool(condition.get("case_sensitive", False))
    regex = bool(condition.get("regex", False))
    raw_value = condition.get("value")
    value = _coerce_like_series(s, raw_value)

    if op in {"eq", "=="}:
        return s == value
    if op in {"ne", "!=", "<>"}:
        return s != value
    if op in {"gt", ">"}:
        return s > value
    if op in {"gte", ">="}:
        return s >= value
    if op in {"lt", "<"}:
        return s < value
    if op in {"lte", "<="}:
        return s <= value
    if op in {"in", "not_in"}:
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        vals = [_coerce_like_series(s, v) for v in values]
        m = s.isin(vals)
        return ~m if op == "not_in" else m
    if op == "between":
        if isinstance(raw_value, list) and len(raw_value) >= 2:
            lo = _coerce_like_series(s, raw_value[0])
            hi = _coerce_like_series(s, raw_value[1])
        else:
            lo = _coerce_like_series(s, condition.get("min"))
            hi = _coerce_like_series(s, condition.get("max"))
        if lo is None or hi is None:
            raise DatasetTransformError("filter_rows: between requires value=[min,max] or min/max")
        return s.between(lo, hi, inclusive="both")
    if op == "contains":
        pat = str(raw_value if raw_value is not None else "")
        return s.astype("string").str.contains(pat, case=case_sensitive, na=False, regex=regex)
    if op == "startswith":
        pat = str(raw_value if raw_value is not None else "")
        ss = s.astype("string")
        if not case_sensitive:
            return ss.str.lower().str.startswith(pat.lower(), na=False)
        return ss.str.startswith(pat, na=False)
    if op == "endswith":
        pat = str(raw_value if raw_value is not None else "")
        ss = s.astype("string")
        if not case_sensitive:
            return ss.str.lower().str.endswith(pat.lower(), na=False)
        return ss.str.endswith(pat, na=False)
    if op == "is_null":
        return s.isna()
    if op in {"not_null", "is_not_null"}:
        return ~s.isna()
    raise DatasetTransformError(f"filter_rows: unsupported condition operator: {op}")


def apply_transform_steps(df: pd.DataFrame, steps: list[Any]) -> TransformOutput:
    if df is None or not isinstance(df, pd.DataFrame):
        raise DatasetTransformError("df must be a pandas DataFrame")
    if not isinstance(steps, list) or not steps:
        raise DatasetTransformError("steps must be a non-empty list")

    out = df.copy()
    warnings: list[str] = []
    metrics: dict[str, Any] = {"steps": []}

    normalized = [_normalize_step(s) for s in steps]

    for idx, step in enumerate(normalized):
        op = step["op"]
        params = step["params"]

        step_metrics: dict[str, Any] = {"op": op}
        if params:
            step_metrics["params"] = params

        if op == "type_convert":
            column = params.get("column")
            if not column:
                raise DatasetTransformError("type_convert: params.column is required")
            column = str(column)
            _require_columns(out, [column], op=op)

            target = params.get("to") or params.get("target_type") or params.get("dtype")
            if not target:
                raise DatasetTransformError("type_convert: params.to is required")
            target = str(target).strip().lower()

            errors = str(params.get("errors") or "raise").strip().lower()
            if errors not in {"raise", "coerce"}:
                raise DatasetTransformError("type_convert: params.errors must be 'raise' or 'coerce'")

            before_dtype = str(out[column].dtype)
            before_nulls = int(out[column].isna().sum())

            if target in {"int", "int64", "integer"}:
                out[column] = pd.to_numeric(out[column], errors=errors).astype("Int64")
            elif target in {"float", "float64", "double", "number"}:
                out[column] = pd.to_numeric(out[column], errors=errors).astype("float64")
            elif target in {"str", "string", "text"}:
                out[column] = out[column].astype("string")
            elif target in {"bool", "boolean"}:
                s = out[column]
                if s.dtype == bool:
                    out[column] = s
                else:
                    lowered = s.astype("string").str.strip().str.lower()
                    out[column] = lowered.isin(["1", "true", "t", "yes", "y"])
            elif target in {"datetime", "timestamp"}:
                fmt = params.get("format") or params.get("datetime_format")
                fmt = str(fmt) if fmt else None
                out[column] = pd.to_datetime(out[column], errors=errors, format=fmt)
            elif target in {"date"}:
                fmt = params.get("format") or params.get("datetime_format")
                fmt = str(fmt) if fmt else None
                out[column] = pd.to_datetime(out[column], errors=errors, format=fmt).dt.date
            elif target in {"category", "categorical"}:
                out[column] = out[column].astype("category")
            else:
                raise DatasetTransformError(f"type_convert: unsupported target type: {target}")

            after_nulls = int(out[column].isna().sum())
            step_metrics.update(
                {
                    "column": column,
                    "before_dtype": before_dtype,
                    "after_dtype": str(out[column].dtype),
                    "before_nulls": before_nulls,
                    "after_nulls": after_nulls,
                }
            )

        elif op == "fill_missing":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            if not columns:
                raise DatasetTransformError("fill_missing: params.columns is required")
            _require_columns(out, columns, op=op)

            strategy = str(params.get("strategy") or params.get("method") or "constant").strip().lower()
            if strategy not in {"constant", "mean", "median", "mode", "ffill", "bfill"}:
                raise DatasetTransformError(
                    "fill_missing: params.strategy must be one of constant|mean|median|mode|ffill|bfill"
                )

            filled_total = 0
            per_col: list[dict[str, Any]] = []
            for col in columns:
                s = out[col]
                before = int(s.isna().sum())
                if before == 0:
                    per_col.append({"column": col, "filled": 0})
                    continue

                if strategy == "constant":
                    if "value" not in params:
                        raise DatasetTransformError("fill_missing: params.value is required for constant strategy")
                    out[col] = s.fillna(params.get("value"))
                elif strategy == "mean":
                    out[col] = s.fillna(pd.to_numeric(s, errors="coerce").mean())
                elif strategy == "median":
                    out[col] = s.fillna(pd.to_numeric(s, errors="coerce").median())
                elif strategy == "mode":
                    mode = s.mode(dropna=True)
                    if mode.empty:
                        warnings.append(f"fill_missing: no mode for column '{col}', leaving nulls unchanged")
                        per_col.append({"column": col, "filled": 0})
                        continue
                    out[col] = s.fillna(mode.iloc[0])
                elif strategy == "ffill":
                    out[col] = s.ffill()
                else:
                    out[col] = s.bfill()

                after = int(out[col].isna().sum())
                filled = max(before - after, 0)
                filled_total += filled
                per_col.append({"column": col, "filled": filled, "before_nulls": before, "after_nulls": after})

            step_metrics.update({"columns": columns, "strategy": strategy, "filled": filled_total, "per_column": per_col})

        elif op == "drop_missing":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            how = str(params.get("how") or "any").strip().lower()
            if how not in {"any", "all"}:
                raise DatasetTransformError("drop_missing: params.how must be 'any' or 'all'")

            before = int(out.shape[0])
            if columns:
                _require_columns(out, columns, op=op)
                out = out.dropna(subset=columns, how=how)
            else:
                out = out.dropna(how=how)
            after = int(out.shape[0])
            step_metrics.update({"how": how, "columns": columns, "rows_dropped": max(before - after, 0)})

        elif op == "deduplicate":
            subset = _as_list(params.get("subset") or params.get("columns"))
            subset = [str(c) for c in subset if str(c)]
            keep = str(params.get("keep") or "first").strip().lower()
            if keep not in {"first", "last"}:
                raise DatasetTransformError("deduplicate: params.keep must be 'first' or 'last'")
            if subset:
                _require_columns(out, subset, op=op)

            before = int(out.shape[0])
            out = out.drop_duplicates(subset=subset or None, keep=keep)
            after = int(out.shape[0])
            step_metrics.update({"subset": subset, "keep": keep, "rows_removed": max(before - after, 0)})

        elif op == "string_normalize":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            if not columns:
                raise DatasetTransformError("string_normalize: params.columns is required")
            _require_columns(out, columns, op=op)

            trim = bool(params.get("trim", True))
            lowercase = bool(params.get("lowercase", False))
            uppercase = bool(params.get("uppercase", False))
            if lowercase and uppercase:
                raise DatasetTransformError("string_normalize: lowercase and uppercase are mutually exclusive")
            strip_chars = params.get("strip_chars")
            strip_chars = str(strip_chars) if strip_chars is not None else None

            regex_replacements = params.get("regex_replace") or []
            if regex_replacements is None:
                regex_replacements = []
            if not isinstance(regex_replacements, list):
                raise DatasetTransformError("string_normalize: regex_replace must be a list")

            for col in columns:
                s = out[col].astype("string")
                if trim:
                    s = s.str.strip(strip_chars)
                if lowercase:
                    s = s.str.lower()
                if uppercase:
                    s = s.str.upper()
                for rep in regex_replacements:
                    if not isinstance(rep, dict):
                        raise DatasetTransformError("string_normalize: regex_replace items must be objects")
                    pattern = rep.get("pattern")
                    repl = rep.get("repl", "")
                    if pattern is None:
                        raise DatasetTransformError("string_normalize: regex_replace.pattern is required")
                    s = s.str.replace(str(pattern), str(repl), regex=True)
                out[col] = s

            step_metrics.update({"columns": columns, "trim": trim, "lowercase": lowercase, "uppercase": uppercase})

        elif op == "drop_columns":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            if not columns:
                raise DatasetTransformError("drop_columns: params.columns is required")
            _require_columns(out, columns, op=op)
            out = out.drop(columns=columns)
            step_metrics.update({"columns": columns})

        elif op == "rename_columns":
            mapping = params.get("mapping") or params.get("columns") or params.get("rename")
            if not isinstance(mapping, dict) or not mapping:
                raise DatasetTransformError("rename_columns: params.mapping must be a non-empty object")

            src_cols = [str(k) for k in mapping.keys()]
            _require_columns(out, src_cols, op=op)

            dst_cols = [str(v) for v in mapping.values()]
            if any(not c for c in dst_cols):
                raise DatasetTransformError("rename_columns: target column names must be non-empty")
            if len(set(dst_cols)) != len(dst_cols):
                raise DatasetTransformError("rename_columns: target column names must be unique")

            out = out.rename(columns={str(k): str(v) for k, v in mapping.items()})
            step_metrics.update({"mapping": {str(k): str(v) for k, v in mapping.items()}})

        elif op == "filter_rows":
            conditions = params.get("conditions")
            if conditions is None:
                conditions = [
                    {
                        "column": params.get("column"),
                        "op": params.get("op") or params.get("operator") or "eq",
                        "value": params.get("value"),
                        "case_sensitive": params.get("case_sensitive", False),
                        "regex": params.get("regex", False),
                    }
                ]
            if not isinstance(conditions, list) or not conditions:
                raise DatasetTransformError("filter_rows: params.conditions must be a non-empty list")

            combine = str(params.get("combine") or "all").strip().lower()
            if combine not in {"all", "any"}:
                raise DatasetTransformError("filter_rows: params.combine must be all|any")
            keep = bool(params.get("keep", True))

            masks: list[pd.Series] = []
            for cond in conditions:
                if not isinstance(cond, dict):
                    raise DatasetTransformError("filter_rows: each condition must be an object")
                masks.append(_filter_condition_mask(out, cond).fillna(False))

            combined = masks[0]
            for m in masks[1:]:
                combined = combined & m if combine == "all" else combined | m
            if not keep:
                combined = ~combined

            before = int(out.shape[0])
            out = out.loc[combined]
            after = int(out.shape[0])
            step_metrics.update(
                {
                    "conditions": conditions,
                    "combine": combine,
                    "keep": keep,
                    "rows_before": before,
                    "rows_after": after,
                    "rows_filtered": max(before - after, 0),
                }
            )

        elif op == "sort_rows":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            if not columns:
                raise DatasetTransformError("sort_rows: params.columns is required")
            _require_columns(out, columns, op=op)

            ascending_raw = params.get("ascending", True)
            if isinstance(ascending_raw, list):
                if len(ascending_raw) != len(columns):
                    raise DatasetTransformError("sort_rows: ascending list must match columns length")
                ascending = [bool(v) for v in ascending_raw]
            else:
                ascending = bool(ascending_raw)

            na_position = str(params.get("na_position") or "last").strip().lower()
            if na_position not in {"first", "last"}:
                raise DatasetTransformError("sort_rows: na_position must be first|last")

            out = out.sort_values(by=columns, ascending=ascending, na_position=na_position, kind="mergesort")
            if bool(params.get("reset_index", False)):
                out = out.reset_index(drop=True)

            step_metrics.update({"columns": columns, "ascending": ascending, "na_position": na_position})

        elif op == "limit_rows":
            n = params.get("n")
            if n is None:
                n = params.get("limit")
            try:
                n_int = int(n)
            except Exception:
                raise DatasetTransformError("limit_rows: params.n must be a positive integer")
            if n_int <= 0:
                raise DatasetTransformError("limit_rows: params.n must be a positive integer")
            n_int = min(n_int, 5_000_000)

            before = int(out.shape[0])
            from_end = bool(params.get("from_end", False))
            out = out.tail(n_int) if from_end else out.head(n_int)
            after = int(out.shape[0])
            step_metrics.update(
                {
                    "n": n_int,
                    "from_end": from_end,
                    "rows_before": before,
                    "rows_after": after,
                    "rows_removed": max(before - after, 0),
                }
            )

        elif op == "time_features":
            column = str(params.get("column") or "").strip()
            if not column:
                raise DatasetTransformError("time_features: params.column is required")
            _require_columns(out, [column], op=op)

            fmt = params.get("format")
            fmt = str(fmt) if fmt else None
            ts = pd.to_datetime(out[column], errors="coerce", format=fmt)
            if ts.notna().sum() == 0:
                raise DatasetTransformError("time_features: no parseable datetime values in source column")

            feature_set = params.get("features")
            if not isinstance(feature_set, list) or not feature_set:
                feature_set = ["year", "quarter", "month", "week", "day", "day_of_week", "is_weekend", "hour"]
            features = [str(f).strip().lower() for f in feature_set if str(f).strip()]
            if not features:
                raise DatasetTransformError("time_features: params.features must include at least one feature")

            prefix = params.get("prefix")
            if prefix is None:
                prefix = f"{column}_"
            prefix = str(prefix)
            if prefix and not prefix.endswith("_"):
                prefix = f"{prefix}_"

            added: list[str] = []
            for f in features:
                if f == "year":
                    out_col = f"{prefix}year"
                    out[out_col] = ts.dt.year.astype("Int64")
                elif f == "quarter":
                    out_col = f"{prefix}quarter"
                    out[out_col] = ts.dt.quarter.astype("Int64")
                elif f == "month":
                    out_col = f"{prefix}month"
                    out[out_col] = ts.dt.month.astype("Int64")
                elif f in {"week", "iso_week"}:
                    out_col = f"{prefix}week"
                    out[out_col] = ts.dt.isocalendar().week.astype("Int64")
                elif f == "day":
                    out_col = f"{prefix}day"
                    out[out_col] = ts.dt.day.astype("Int64")
                elif f in {"day_of_week", "weekday"}:
                    out_col = f"{prefix}day_of_week"
                    out[out_col] = ts.dt.dayofweek.astype("Int64")
                elif f == "hour":
                    out_col = f"{prefix}hour"
                    out[out_col] = ts.dt.hour.astype("Int64")
                elif f == "is_weekend":
                    out_col = f"{prefix}is_weekend"
                    out[out_col] = ts.dt.dayofweek.isin([5, 6]).astype("boolean")
                else:
                    raise DatasetTransformError(f"time_features: unsupported feature: {f}")
                added.append(out_col)

            if bool(params.get("drop_source", False)):
                out = out.drop(columns=[column])

            step_metrics.update(
                {
                    "column": column,
                    "features": features,
                    "added_columns": added,
                    "drop_source": bool(params.get("drop_source", False)),
                    "parsed_non_null_rows": int(ts.notna().sum()),
                }
            )

        elif op == "bin_numeric":
            column = str(params.get("column") or "").strip()
            if not column:
                raise DatasetTransformError("bin_numeric: params.column is required")
            _require_columns(out, [column], op=op)

            bins_raw = params.get("bins", 10)
            try:
                bins = int(bins_raw)
            except Exception:
                raise DatasetTransformError("bin_numeric: params.bins must be an integer")
            if bins < 2 or bins > 100:
                raise DatasetTransformError("bin_numeric: params.bins must be between 2 and 100")

            strategy = str(params.get("strategy") or "quantile").strip().lower()
            if strategy not in {"quantile", "uniform"}:
                raise DatasetTransformError("bin_numeric: params.strategy must be quantile|uniform")

            out_col = str(params.get("output_column") or f"{column}_bin").strip()
            if not out_col:
                raise DatasetTransformError("bin_numeric: output column name cannot be empty")

            s_num = pd.to_numeric(out[column], errors="coerce")
            if s_num.notna().sum() == 0:
                raise DatasetTransformError("bin_numeric: source column has no numeric values")

            if strategy == "quantile":
                binned = pd.qcut(s_num, q=bins, duplicates="drop")
            else:
                binned = pd.cut(s_num, bins=bins, include_lowest=bool(params.get("include_lowest", True)))

            output_mode = str(params.get("output") or "labels").strip().lower()
            if output_mode == "codes":
                codes = binned.cat.codes
                out[out_col] = codes.where(codes >= 0, pd.NA).astype("Int64")
            elif output_mode == "labels":
                out[out_col] = binned.astype("string")
            else:
                raise DatasetTransformError("bin_numeric: params.output must be labels|codes")

            step_metrics.update(
                {
                    "column": column,
                    "output_column": out_col,
                    "bins_requested": bins,
                    "bins_actual": int(len(binned.cat.categories)),
                    "strategy": strategy,
                    "output": output_mode,
                }
            )

        elif op == "clip_outliers":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            if not columns:
                raise DatasetTransformError("clip_outliers: params.columns is required")
            _require_columns(out, columns, op=op)

            method = str(params.get("method") or "iqr").strip().lower()
            if method not in {"iqr", "quantile"}:
                raise DatasetTransformError("clip_outliers: params.method must be iqr|quantile")

            action = str(params.get("action") or "clip").strip().lower()
            if action not in {"clip", "drop"}:
                raise DatasetTransformError("clip_outliers: params.action must be clip|drop")

            outlier_masks: list[pd.Series] = []
            per_col: list[dict[str, Any]] = []
            for col in columns:
                s_num = pd.to_numeric(out[col], errors="coerce")
                non_null = s_num.dropna()
                if non_null.empty:
                    warnings.append(f"clip_outliers: column '{col}' has no numeric values; skipped")
                    continue

                if method == "iqr":
                    q1 = float(non_null.quantile(0.25))
                    q3 = float(non_null.quantile(0.75))
                    iqr = q3 - q1
                    k = float(params.get("iqr_multiplier", 1.5))
                    lower = q1 - (k * iqr)
                    upper = q3 + (k * iqr)
                else:
                    lower_q = float(params.get("lower_quantile", 0.01))
                    upper_q = float(params.get("upper_quantile", 0.99))
                    if not (0.0 <= lower_q < upper_q <= 1.0):
                        raise DatasetTransformError("clip_outliers: quantiles must satisfy 0<=lower<upper<=1")
                    lower = float(non_null.quantile(lower_q))
                    upper = float(non_null.quantile(upper_q))

                mask = (s_num < lower) | (s_num > upper)
                outlier_masks.append(mask.fillna(False))
                outlier_count = int(mask.fillna(False).sum())
                per_col.append(
                    {
                        "column": col,
                        "lower": float(lower),
                        "upper": float(upper),
                        "outliers": outlier_count,
                    }
                )

                if action == "clip":
                    clipped = s_num.clip(lower=lower, upper=upper)
                    out[col] = clipped

            rows_dropped = 0
            if action == "drop" and outlier_masks:
                combine = str(params.get("combine") or "any").strip().lower()
                if combine not in {"any", "all"}:
                    raise DatasetTransformError("clip_outliers: params.combine must be any|all")
                combined = outlier_masks[0]
                for m in outlier_masks[1:]:
                    combined = combined | m if combine == "any" else combined & m
                before = int(out.shape[0])
                out = out.loc[~combined]
                rows_dropped = max(before - int(out.shape[0]), 0)

            step_metrics.update(
                {
                    "columns": columns,
                    "method": method,
                    "action": action,
                    "rows_dropped": rows_dropped,
                    "per_column": per_col,
                }
            )

        elif op == "encode_categorical":
            columns = _as_list(params.get("columns") or params.get("column"))
            columns = [str(c) for c in columns if str(c)]
            if not columns:
                raise DatasetTransformError("encode_categorical: params.columns is required")
            _require_columns(out, columns, op=op)

            strategy = str(params.get("strategy") or "label").strip().lower()
            if strategy not in {"label", "one_hot"}:
                raise DatasetTransformError("encode_categorical: params.strategy must be label|one_hot")

            max_categories = int(params.get("max_categories", 100))
            if max_categories < 2 or max_categories > 1000:
                raise DatasetTransformError("encode_categorical: params.max_categories must be between 2 and 1000")

            drop_original = bool(params.get("drop_original", False))
            encoded_suffix = str(params.get("encoded_suffix") or "_encoded")
            drop_first = bool(params.get("drop_first", False))
            dummy_na = bool(params.get("dummy_na", False))

            added_columns: list[str] = []
            per_col: list[dict[str, Any]] = []

            for col in columns:
                s = out[col].astype("string")
                categories = [str(v) for v in s.dropna().unique().tolist()]
                if len(categories) > max_categories:
                    raise DatasetTransformError(
                        f"encode_categorical: column '{col}' has {len(categories)} categories (max {max_categories})"
                    )

                if strategy == "label":
                    ordered = sorted(categories)
                    mapping = {v: i for i, v in enumerate(ordered)}
                    encoded = s.map(mapping).astype("Int64")
                    target_col = col if drop_original else f"{col}{encoded_suffix}"
                    if not target_col:
                        raise DatasetTransformError("encode_categorical: output column name cannot be empty")
                    out[target_col] = encoded
                    if drop_original and target_col != col:
                        out = out.drop(columns=[col])
                    added_columns.append(target_col)
                    per_col.append(
                        {
                            "column": col,
                            "output_column": target_col,
                            "categories": len(ordered),
                            "mapping": mapping,
                        }
                    )
                else:
                    dummies = pd.get_dummies(
                        s,
                        prefix=col,
                        prefix_sep="__",
                        drop_first=drop_first,
                        dummy_na=dummy_na,
                        dtype="Int64",
                    )
                    if dummies.shape[1] > max_categories:
                        raise DatasetTransformError(
                            f"encode_categorical: one_hot output for '{col}' has {dummies.shape[1]} columns (max {max_categories})"
                        )
                    out = pd.concat([out, dummies], axis=1)
                    if drop_original:
                        out = out.drop(columns=[col])
                    out_cols = [str(c) for c in dummies.columns.tolist()]
                    added_columns.extend(out_cols)
                    per_col.append(
                        {
                            "column": col,
                            "output_columns": out_cols,
                            "categories": len(categories),
                        }
                    )

            step_metrics.update(
                {
                    "columns": columns,
                    "strategy": strategy,
                    "drop_original": drop_original,
                    "added_columns": added_columns,
                    "per_column": per_col,
                }
            )

        else:
            raise DatasetTransformError(f"unsupported transform op: {op}")

        metrics["steps"].append(step_metrics)

    return TransformOutput(df=out, metrics=metrics, warnings=warnings)


def build_preview_diff(*, before: pd.DataFrame, after: pd.DataFrame, preview_rows: int) -> dict[str, Any]:
    before_cols = [str(c) for c in before.columns.tolist()]
    after_cols = [str(c) for c in after.columns.tolist()]

    added = [c for c in after_cols if c not in before_cols]
    removed = [c for c in before_cols if c not in after_cols]

    changed: list[dict[str, Any]] = []
    shared = [c for c in before_cols if c in set(after_cols)]
    for c in shared:
        bd = str(before[c].dtype)
        ad = str(after[c].dtype)
        if bd != ad:
            changed.append({"column": c, "before": bd, "after": ad})

    pr = max(1, min(int(preview_rows), 200))
    preview = after.head(pr).to_dict(orient="records")

    return {
        "input_rows": int(before.shape[0]),
        "output_rows": int(after.shape[0]),
        "input_columns": int(before.shape[1]),
        "output_columns": int(after.shape[1]),
        "added_columns": added,
        "removed_columns": removed,
        "changed_dtypes": changed,
        "preview_rows": preview,
    }
