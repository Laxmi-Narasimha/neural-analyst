"""Machine-readable catalog of the operators a planner may use.

Each parameter declares its kind so a plan can be validated against the
dataset before anything runs:

* ``column``  - a dataset column; ``column_types`` restricts which kinds fit
* ``enum``    - one of ``choices``
* ``int``     - integer within [min, max]
* ``bool``    - true/false
* ``date``    - an ISO calendar date, YYYY-MM-DD
* ``value``   - a value of the column named by the operator's ``column`` parameter,
                bound to the real values in the data by the validator

Only operators listed here can be planned. Their implementations live in
app/compute/operators and are the only code that produces numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
TEXT = "text"
ANY = "any"


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # column | enum | int | bool | value | date
    description: str
    required: bool = False
    column_types: tuple[str, ...] = (ANY,)
    choices: tuple[str, ...] = ()
    min: Optional[int] = None
    max: Optional[int] = None
    default: Any = None


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    description: str
    use_when: str
    params: tuple[Param, ...] = field(default_factory=tuple)

    def param(self, name: str) -> Optional[Param]:
        return next((p for p in self.params if p.name == name), None)


AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count", "distinct_count")
NUMERIC_AGGREGATIONS = ("sum", "mean", "median", "min", "max")
FREQUENCIES = ("D", "W", "M", "Q", "Y")

CATALOG: dict[str, OperatorSpec] = {
    spec.name: spec
    for spec in [
        OperatorSpec(
            "dataset_overview",
            "Row and column counts and basic shape of the dataset (or the filtered rows).",
            "How many rows/records, dataset size, what is in this file.",
        ),
        OperatorSpec(
            "schema_snapshot",
            "Column names with their inferred types.",
            "Which columns exist, what types they are.",
        ),
        OperatorSpec(
            "preview_rows",
            "Show example rows.",
            "Show me some rows / what does the data look like.",
            (Param("limit", "int", "Rows to show", min=1, max=200, default=25),),
        ),
        OperatorSpec(
            "measure_aggregate",
            "One exact aggregate, optionally per group: sum, mean, median, min or max of a numeric column; "
            "count = number of rows; distinct_count = number of distinct values of a column.",
            "Total / average / median / min / max of a measure, overall or by a category; 'how many orders/"
            "employees/accounts' (count, with filters for the restriction); 'how many unique customers' "
            "(distinct_count). Prefer this for any single-number or per-group number question.",
            (
                Param("value_column", "column", "Numeric measure (sum/mean/median/min/max), the column whose "
                      "distinct values are counted (distinct_count), or omitted (count)", column_types=(ANY,)),
                Param("aggregation", "enum", "Aggregation", required=True, choices=AGGREGATIONS),
                Param("group_by", "column", "Optional grouping column", column_types=(CATEGORICAL, ANY)),
            ),
        ),
        OperatorSpec(
            "rate",
            "Share of rows where a column equals one value (e.g. churned = 1, Returned = Yes), overall or per "
            "group, with the matching and total row counts.",
            "Churn rate, return rate, attrition rate, conversion rate, win rate, '% of orders that ...', "
            "'what share of ...'. Rows where the column is empty are left out of the denominator.",
            (
                Param("column", "column", "Column that marks the outcome", required=True, column_types=(ANY,)),
                Param("positive_value", "value", "The value of that column that counts as the outcome, as it "
                      "appears in the column's values", required=True),
                Param("group_by", "column", "Optional grouping column", column_types=(CATEGORICAL, ANY)),
                Param("ascending", "bool", "True ranks the lowest rate first", default=False),
            ),
        ),
        OperatorSpec(
            "ratio",
            "Ratio of two totals, overall or per group: (sum of numerator - sum of subtract) / sum of "
            "denominator, optionally as a percent. Only rows with all the columns present are used.",
            "Revenue per unit, cost per order, price per seat, margin % = (revenue - cost) / revenue "
            "(numerator revenue, subtract cost, denominator revenue, as_percent), discount rate = discount / "
            "gross, conversion = orders / visits. Use when the metric divides one column by another.",
            (
                Param("numerator_column", "column", "Numeric column summed on top", required=True, column_types=(NUMERIC,)),
                Param("denominator_column", "column", "Numeric column summed below", required=True, column_types=(NUMERIC,)),
                Param("subtract_column", "column", "Optional numeric column subtracted from the numerator total "
                      "(margin: cost)", column_types=(NUMERIC,)),
                Param("group_by", "column", "Optional grouping column", column_types=(CATEGORICAL, ANY)),
                Param("as_percent", "bool", "Multiply by 100", default=False),
                Param("ascending", "bool", "True ranks the lowest first", default=False),
            ),
        ),
        OperatorSpec(
            "share_of_total",
            "Each group's share of the total of a numeric column (or of the row count), in percent.",
            "What share / percentage / mix / split of revenue (or orders) comes from each region, product, "
            "channel; contribution of each group to the total.",
            (
                Param("group_by", "column", "Grouping column", required=True, column_types=(CATEGORICAL, ANY)),
                Param("value_column", "column", "Numeric measure; omit to share the row count", column_types=(NUMERIC,)),
            ),
        ),
        OperatorSpec(
            "period_compare",
            "Compare a measure between two date ranges (current vs previous): both values, the change and the "
            "% change; with group_by, every group's change and its contribution to the total change. If the "
            "current range runs past the last date in the data, both ranges are cut to the same number of days.",
            "This month vs last month, Q3 vs Q2, October 2026 vs October 2025 (year over year), growth or "
            "decline between periods, 'why did sales drop', 'what drove the increase' (use group_by on the "
            "dimension to explain). The two ranges are parameters: never also put them in filters.",
            (
                Param("time_column", "column", "Date column that places rows in a period", required=True,
                      column_types=(DATETIME,)),
                Param("value_column", "column", "Numeric measure; omit when counting rows", column_types=(NUMERIC,)),
                Param("aggregation", "enum", "sum, mean or count of rows", choices=("sum", "mean", "count"), default="sum"),
                Param("current_start", "date", "First day of the current period", required=True),
                Param("current_end", "date", "Day after the last day of the current period (exclusive)", required=True),
                Param("previous_start", "date", "First day of the comparison period", required=True),
                Param("previous_end", "date", "Day after the last day of the comparison period (exclusive)", required=True),
                Param("group_by", "column", "Optional dimension to explain the change", column_types=(CATEGORICAL, ANY)),
            ),
        ),
        OperatorSpec(
            "segment_summary",
            "Per-group count, mean, sum and median of a numeric column, ranked.",
            "Compare groups: which region/product/customer type has the highest or lowest average, total or "
            "count; rankings; 'top N by'.",
            (
                Param("group_by", "column", "Grouping column", required=True, column_types=(CATEGORICAL, ANY)),
                Param("value_column", "column", "Numeric measure", column_types=(NUMERIC,)),
                Param("sort_by", "enum", "Ranking metric", choices=("value_mean", "value_sum", "value_median", "count"),
                      default="value_sum"),
                Param("ascending", "bool", "True ranks lowest first", default=False),
                Param("limit", "int", "Groups to return", min=5, max=200, default=50),
            ),
        ),
        OperatorSpec(
            "resample_aggregate",
            "Time series of a numeric column (count, mean, sum per period).",
            "Trends over time, monthly/weekly/daily totals or averages, growth over time.",
            (
                Param("time_column", "column", "Date column", required=True, column_types=(DATETIME,)),
                Param("value_column", "column", "Numeric measure", column_types=(NUMERIC,)),
                Param("freq", "enum", "Period: D day, W week, M month, Q quarter, Y year", choices=FREQUENCIES,
                      default="M"),
                Param("max_points", "int", "Maximum periods", min=10, max=1000, default=200),
            ),
        ),
        OperatorSpec(
            "time_anomaly_scan",
            "Periods whose value is unusually high or low compared with the rest of the series.",
            "Spikes, drops, unusual days/weeks/months.",
            (
                Param("time_column", "column", "Date column", required=True, column_types=(DATETIME,)),
                Param("value_column", "column", "Numeric measure", column_types=(NUMERIC,)),
                Param("freq", "enum", "Period", choices=FREQUENCIES, default="M"),
            ),
        ),
        OperatorSpec(
            "forecast_scan",
            "Forecast future periods of a numeric series with backtested methods.",
            "Predict next months/weeks of a measure over time.",
            (
                Param("time_column", "column", "Date column", required=True, column_types=(DATETIME,)),
                Param("value_column", "column", "Numeric measure", required=True, column_types=(NUMERIC,)),
                Param("horizon", "int", "Periods to forecast", min=1, max=104, default=8),
            ),
        ),
        OperatorSpec(
            "numeric_summary",
            "Count, mean, std, min, quartiles and max of numeric columns.",
            "Describe numeric columns, distributions, ranges, typical values.",
            (Param("max_columns", "int", "Columns to summarise", min=1, max=200, default=25),),
        ),
        OperatorSpec(
            "categorical_topk",
            "Most frequent values of categorical columns.",
            "Most common categories, value counts, top values.",
            (
                Param("k", "int", "Top values per column", min=1, max=100, default=10),
                Param("max_columns", "int", "Columns", min=1, max=100, default=10),
            ),
        ),
        OperatorSpec(
            "missingness_scan",
            "Missing-value counts and percentages per column.",
            "Missing data, nulls, completeness, data quality.",
        ),
        OperatorSpec(
            "uniqueness_scan",
            "Distinct counts, duplicate and key candidates per column.",
            "Duplicates, unique identifiers, cardinality.",
            (Param("max_columns", "int", "Columns", min=1, max=500, default=200),),
        ),
        OperatorSpec(
            "correlation_matrix",
            "Pairwise correlations between numeric columns.",
            "Which numeric variables move together, strongest correlations.",
            (Param("max_columns", "int", "Columns", min=2, max=100, default=25),),
        ),
        OperatorSpec(
            "association_scan",
            "Strength of relationships between categorical and numeric columns (all pairs).",
            "What drives / is associated with a measure across mixed column types.",
        ),
        OperatorSpec(
            "relationship_explain",
            "How one column relates to another, with binned or per-category detail.",
            "Does A affect B, how does B change with A, relationship between two named columns.",
            (
                Param("column_a", "column", "Driver / explanatory column", required=True),
                Param("column_b", "column", "Outcome column", required=True),
            ),
        ),
        OperatorSpec(
            "outlier_scan",
            "Unusual values in numeric columns (IQR based).",
            "Outliers, extreme values, anomalies in values (not over time).",
            (Param("max_columns", "int", "Columns", min=1, max=100, default=25),),
        ),
        OperatorSpec(
            "distribution_chart",
            "Histogram / distribution of one column.",
            "Show the distribution or spread of a column.",
            (Param("column", "column", "Column to chart", required=True),),
        ),
        OperatorSpec(
            "ab_test_scan",
            "Statistical comparison of a metric between variants of a group column.",
            "Only when the user describes an experiment, test/control or variants that were assigned.",
            (
                Param("variant_column", "column", "Variant / group column", required=True, column_types=(CATEGORICAL, ANY)),
                Param("metric_column", "column", "Outcome metric", required=True, column_types=(NUMERIC, ANY)),
                Param("is_binary", "bool", "Metric is a 0/1 conversion", default=False),
            ),
        ),
        OperatorSpec(
            "sentiment_scan",
            "Positive/negative sentiment of free-text columns.",
            "Sentiment, tone or opinion in reviews, comments, feedback text.",
        ),
        OperatorSpec(
            "ml_task_inference",
            "Suggest which column could be predicted and whether it is classification or regression.",
            "The user wants a prediction or model but has not said what to predict.",
            (Param("preferred_target", "column", "Column the user mentioned as the outcome"),),
        ),
        OperatorSpec(
            "ml_train_baseline",
            "Train and cross-validate models to predict a column, compared with a naive baseline, "
            "with leakage checks and a time-ordered split when dates exist.",
            "Predict / classify / model / what drives an outcome column, when the outcome is known.",
            (
                Param("target_column", "column", "Column to predict", required=True),
                Param("time_column", "column", "Date that orders rows for evaluation", column_types=(DATETIME,)),
                Param("optimize", "bool", "Tune hyperparameters (slower)", default=False),
            ),
        ),
    ]
}


def column_kind(inferred_type: str) -> str:
    t = str(inferred_type or "").lower()
    if t in {"integer", "float", "number", "numeric", "int", "double", "decimal"}:
        return NUMERIC
    if t in {"date", "datetime", "timestamp", "time"}:
        return DATETIME
    if t in {"text"}:
        return TEXT
    if t in {"boolean", "bool", "categorical", "string", "email", "url", "phone"}:
        return CATEGORICAL
    return ANY


def catalog_for_prompt() -> list[dict[str, Any]]:
    """Compact JSON description of the catalog for the planner prompt."""
    out = []
    for spec in CATALOG.values():
        params = []
        for p in spec.params:
            item: dict[str, Any] = {"name": p.name, "kind": p.kind, "required": p.required, "description": p.description}
            if p.kind == "column" and p.column_types != (ANY,):
                item["column_types"] = list(p.column_types)
            if p.choices:
                item["choices"] = list(p.choices)
            if p.min is not None or p.max is not None:
                item["range"] = [p.min, p.max]
            params.append(item)
        out.append({"operator": spec.name, "description": spec.description, "use_when": spec.use_when, "params": params})
    return out
