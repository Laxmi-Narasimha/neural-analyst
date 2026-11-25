"""The interview demo: the questions, in order, and how each answer is checked.

Every expected number is re-computed here with pandas straight from the CSV, independently of the
app, and must appear in the answer exactly as the user sees it. ``needs_llm`` marks questions whose
wording needs the language-model planner; the rest are answered by the rule planner alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from demo.workspace import DATA


def money(v: float) -> str:
    """Formatted the way answers show numbers (thousands separators, 2 decimals unless whole)."""
    v = float(v)
    return f"{int(v):,}" if v == int(v) else f"{v:,.2f}"


def pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.2f}%"


@dataclass
class Step:
    dataset: str
    question: str
    operator: str
    expect: Callable[[], list[str]] = field(default=lambda: [])
    needs_llm: bool = False
    talking_point: str = ""


def _retail() -> pd.DataFrame:
    df = pd.read_csv(DATA / "retail_orders.csv", parse_dates=["Order Date"])
    return df


def _churn() -> pd.DataFrame:
    return pd.read_csv(DATA / "customer_churn.csv")


def _region_totals() -> list[str]:
    s = _retail().groupby("Region")["Sales"].sum().round(2)
    return [f"{k}: {money(v)}" for k, v in s.items()]


def _trend() -> list[str]:
    df = _retail()
    m = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().round(2)
    hi, lo = m.idxmax(), m.idxmin()
    return [f"**{hi.strftime('%b %Y')}** ({money(m[hi])})", f"**{lo.strftime('%b %Y')}** ({money(m[lo])})"]


def _return_rates() -> list[str]:
    df = _retail()
    out = []
    for cat, g in df.groupby("Category"):
        out.append(f"{cat}: {pct(int((g['Returned'] == 'Yes').sum()), len(g))}")
    return out


def _margins() -> list[str]:
    df = _retail()
    out = []
    for cat, g in df.groupby("Category"):
        out.append(f"{cat}: {(g['Sales'].sum() - g['Cost'].sum()) / g['Sales'].sum() * 100:.2f}%")
    return out


def _west_2025() -> list[str]:
    df = _retail()
    m = (df["Region"] == "West") & (df["Order Date"].dt.year == 2025)
    return [money(round(df.loc[m, "Sales"].sum(), 2)), f"{int(m.sum()):,} rows"]


def _q3_drop() -> list[str]:
    df = _retail()
    q = df["Order Date"].dt.to_period("Q")
    south = df["Region"] == "South"
    q2 = df.loc[south & (q == pd.Period("2025Q2")), "Sales"].sum()
    q3 = df.loc[south & (q == pd.Period("2025Q3")), "Sales"].sum()
    return ["South", f"{(q3 - q2) / q2 * 100:.2f}%"]


def _churn_by_contract() -> list[str]:
    df = _churn()
    out = []
    for c, g in df.groupby("contract"):
        out.append(f"{c}: {pct(int((g['churned'] == 'Yes').sum()), len(g))}")
    return out


DEMO: list[Step] = [
    Step("retail_orders", "How many orders are there?", "measure_aggregate",
         lambda: [f"{len(_retail()):,}"], needs_llm=True,
         talking_point="Plain language becomes a validated plan; the number comes from code, not the model."),
    Step("retail_orders", "Total Sales by Region", "measure_aggregate", _region_totals,
         talking_point="Rule planner, no LLM call: cheap and deterministic."),
    Step("retail_orders", "Monthly Sales trend", "resample_aggregate", _trend,
         talking_point="Highest and lowest month, year-over-year, chart and the full table as evidence."),
    Step("retail_orders", "Return rate by Category", "rate", _return_rates, needs_llm=True,
         talking_point="'Return rate' is bound to Returned = Yes from the column's real values."),
    Step("retail_orders", "Margin % by Category", "ratio", _margins, needs_llm=True,
         talking_point="Derived metric: (Sales - Cost) / Sales, computed as a ratio of totals, not an average of ratios."),
    Step("retail_orders", "Total Sales in the West region in 2025", "measure_aggregate", _west_2025, needs_llm=True,
         talking_point="Filters are stated back exactly (Region = West, dates in 2025) with the matched row count."),
    Step("retail_orders", "Why did sales drop in Q3 2025 compared with Q2 2025?", "period_compare", _q3_drop,
         needs_llm=True,
         talking_point="Driver analysis: the change is decomposed by region and South explains it (-40%)."),
    Step("customer_churn", "What is the churn rate by contract?", "rate", _churn_by_contract, needs_llm=True,
         talking_point="Monthly contracts churn about three times as often as two-year ones."),
    Step("customer_churn", "Predict churned", "ml_train_baseline", lambda: ["beats it", "**Decision cut-off:**"],
         talking_point="Leakage checks, time-ordered split, baseline comparison, tuned cut-off, calibration, "
                       "weakest segment; the model is saved per user."),
    Step("churn_new_customers", "Score this file with my churn model", "ml_score",
         lambda: ["all 1,000 rows", "monthly_charges"],
         talking_point="Batch scoring of every row with a drift check: the price rise shows up as major drift."),
]

DATASETS = {
    "retail_orders": "retail_orders.csv",
    "customer_churn": "customer_churn.csv",
    "churn_new_customers": "churn_new_customers.csv",
}
