from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.compute.chart_helpers import build_chart
from app.compute.operators.base import OperatorContext, OperatorResult
from app.ml.visualization import ChartType
from app.services.sql_safety import enforce_row_limit, validate_dataset_sql


def _duckdb_read_sql(file_path: str, file_format: str, sql: str) -> pd.DataFrame:
    import duckdb

    fmt = str(file_format or "").lower()
    con = duckdb.connect(database=":memory:")
    try:
        if fmt in {"parquet", "pq"}:
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM read_parquet('{file_path}')")
        elif fmt in {"csv", "tsv", "txt"}:
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM read_csv_auto('{file_path}')")
        else:
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM read_csv_auto('{file_path}')")
        return con.execute(sql).fetchdf()
    finally:
        con.close()


@dataclass(frozen=True)
class DuckDBProfileOperator:
    """Full-dataset aggregates via DuckDB without loading all rows into pandas."""

    name: str = "duckdb_profile"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        path = str(ctx.source_file_path or "").strip()
        if not path:
            rows = int(ctx.df.shape[0]) if ctx.df is not None else 0
            cols = int(ctx.df.shape[1]) if ctx.df is not None else 0
            out = pd.DataFrame([{"rows": rows, "columns": cols, "engine": "pandas_sample"}])
            return OperatorResult(
                tables={"profile": out},
                metrics={"rows": rows, "columns": cols},
                charts={},
                summary={"rows": rows, "columns": cols, "engine": "pandas_sample"},
            )

        sql = str(params.get("sql") or "SELECT COUNT(*) AS row_count FROM dataset")
        normalized = validate_dataset_sql(sql)
        limited = enforce_row_limit(normalized, int(params.get("max_rows") or 1000))
        df = _duckdb_read_sql(path, str(ctx.source_file_format or ""), limited)

        summary = df.iloc[0].to_dict() if len(df) else {}
        cols = ctx.schema_info.get("columns") if isinstance(ctx.schema_info, dict) else None
        if isinstance(cols, list):
            summary["column_count"] = len(cols)
        pr = ctx.profile_report if isinstance(ctx.profile_report, dict) else {}
        if pr.get("row_count") is not None and "row_count" not in summary:
            summary["row_count"] = int(pr.get("row_count"))
        summary["engine"] = "duckdb_pushdown"
        return OperatorResult(
            tables={"profile": df},
            metrics=summary,
            charts={},
            summary=summary,
        )


@dataclass(frozen=True)
class SentimentScanOperator:
    name: str = "sentiment_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        from app.ml.sentiment_analysis import SentimentAnalysisEngine

        max_cols = int(params.get("max_columns") or 3)
        sample_rows = int(params.get("sample_rows") or 500)
        df = ctx.df.head(sample_rows) if ctx.df is not None else pd.DataFrame()

        text_cols = []
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype) == "string":
                non_null = df[c].dropna().astype(str)
                if len(non_null) and non_null.str.len().mean() > 8:
                    text_cols.append(str(c))
            if len(text_cols) >= max_cols:
                break

        if not text_cols:
            raise ValueError("No text columns found for sentiment analysis")

        engine = SentimentAnalysisEngine(verbose=False)
        rows = []
        for col in text_cols:
            result = engine.analyze(df[[col]].head(200), text_col=col)
            dist = result.distribution_pct if hasattr(result, "distribution_pct") else {}
            rows.append(
                {
                    "column": col,
                    "samples": int(getattr(result, "n_documents", 0)),
                    "positive_pct": round(float(dist.get("positive", 0) + dist.get("very_positive", 0)), 2),
                    "negative_pct": round(float(dist.get("negative", 0) + dist.get("very_negative", 0)), 2),
                    "neutral_pct": round(float(dist.get("neutral", 0)), 2),
                    "avg_score": round(float(getattr(result, "avg_polarity", 0)), 4),
                }
            )

        table = pd.DataFrame(rows)
        return OperatorResult(
            tables={"sentiment": table},
            metrics={"columns_analyzed": len(rows)},
            charts={},
            summary={"columns": rows},
        )


@dataclass(frozen=True)
class ForecastScanOperator:
    name: str = "forecast_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        from app.ml.forecasting import ForecastConfig, ForecastEngine, ForecastMethod

        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        time_col = params.get("time_column")
        value_col = params.get("value_column")
        if not time_col or not value_col:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]) and not time_col:
                    time_col = str(c)
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            value_col = value_col or (numeric[-1] if numeric else None)

        if not time_col or not value_col or time_col not in df.columns or value_col not in df.columns:
            raise ValueError("Could not infer time_column and value_column for forecasting")

        method_name = str(params.get("method") or "ensemble").lower()
        method_map = {
            "ensemble": ForecastMethod.ENSEMBLE,
            "prophet": ForecastMethod.PROPHET,
            "linear": ForecastMethod.LINEAR,
            "exponential": ForecastMethod.EXPONENTIAL_SMOOTHING,
            "arima": ForecastMethod.ARIMA,
        }
        method = method_map.get(method_name, ForecastMethod.ENSEMBLE)
        engine = ForecastEngine()
        config = ForecastConfig(method=method, horizon=int(params.get("horizon") or 8))
        result = engine.forecast(df, str(time_col), str(value_col), config=config)

        forecast_rows = []
        fdf = result.forecast if hasattr(result, "forecast") else None
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            for _, row in fdf.head(12).iterrows():
                forecast_rows.append(row.to_dict())

        table = pd.DataFrame(forecast_rows) if forecast_rows else pd.DataFrame(
            [{"note": "forecast generated", "method": str(config.method.value)}]
        )
        metrics = result.metrics if hasattr(result, "metrics") else {}
        charts: dict[str, Any] = {}
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            plot_df = fdf.copy()
            date_key = "date" if "date" in plot_df.columns else plot_df.columns[0]
            value_key = "yhat" if "yhat" in plot_df.columns else plot_df.columns[-1]
            try:
                charts["forecast_line"] = build_chart(
                    plot_df,
                    chart_type=ChartType.LINE,
                    x=str(date_key),
                    y=str(value_key),
                    title=f"Forecast ({config.method.value})",
                )
            except Exception:
                pass

        return OperatorResult(
            tables={"forecast": table},
            metrics={"method": str(config.method.value), **(metrics or {})},
            charts=charts,
            summary={
                "time_column": time_col,
                "value_column": value_col,
                "method": str(config.method.value),
                "horizon": int(config.horizon),
                "metrics": metrics,
            },
        )


@dataclass(frozen=True)
class ABTestScanOperator:
    name: str = "ab_test_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        from app.ml.ab_testing import ABTestingEngine, TestType

        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        variant_col = str(params.get("variant_column") or params.get("group_column") or "").strip()
        metric_col = str(params.get("metric_column") or params.get("value_column") or "").strip()

        if not variant_col:
            for c in df.columns:
                if df[c].dtype == object and 2 <= int(df[c].nunique()) <= 10:
                    variant_col = str(c)
                    break
        if not metric_col:
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]) and c != variant_col:
                    metric_col = str(c)
                    break

        if not variant_col or not metric_col:
            raise ValueError("Need variant_column and metric_column for A/B test")

        variants = [str(v) for v in df[variant_col].dropna().unique()[:4]]
        if len(variants) < 2:
            raise ValueError("Need at least two variants in variant column")

        engine = ABTestingEngine()
        result = engine.analyze_experiment(
            df,
            variant_col=variant_col,
            metric_col=metric_col,
            control_name=variants[0],
            treatment_name=variants[1],
            test_type=TestType.FREQUENTIST,
            is_binary=bool(params.get("is_binary", False)),
        )

        row = {
            "variant_a": variants[0],
            "variant_b": variants[1],
            "metric": metric_col,
            "p_value": round(float(getattr(result, "p_value", 0) or 0), 6),
            "significant": bool(getattr(result, "is_significant", False)),
            "lift_pct": round(float(getattr(result, "relative_lift", 0) or 0) * 100, 2),
        }
        table = pd.DataFrame([row])
        return OperatorResult(
            tables={"ab_test": table},
            metrics=row,
            charts={},
            summary=row,
        )


P0_ANALYTICS_OPERATORS = [
    DuckDBProfileOperator(),
    SentimentScanOperator(),
    ForecastScanOperator(),
    ABTestScanOperator(),
]