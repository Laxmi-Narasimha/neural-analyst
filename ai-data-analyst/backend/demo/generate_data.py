"""Generate the demo datasets (deterministic: same seed, same files).

    python demo/generate_data.py

Writes three CSVs next to this file, in demo/data/:

retail_orders.csv         6,000 orders, Jan 2024 - Dec 2025, four regions and categories.
                          Built-in stories: the West grows fastest; Furniture sales in the South
                          fall sharply in Q3 2025 (a discount programme ended), so "why did sales
                          drop in Q3 2025" has a real answer; Apparel is returned most often.
customer_churn.csv        4,000 subscription customers with a known outcome (churned Yes/No, ~20%).
                          Churn rises with monthly contracts, support tickets, late payments and
                          days since last login, and falls with tenure.
churn_new_customers.csv   1,000 newer customers to score, without the outcome. A price rise shifted
                          monthly_charges up, so the drift check has something real to report.

The data is synthetic. No row describes a real person or company.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"
REGIONS = ["North", "South", "East", "West"]
CATEGORIES = {"Electronics": 420.0, "Furniture": 610.0, "Office Supplies": 85.0, "Apparel": 120.0}
MARGIN = {"Electronics": 0.22, "Furniture": 0.31, "Office Supplies": 0.38, "Apparel": 0.47}
RETURN_RATE = {"Electronics": 0.06, "Furniture": 0.04, "Office Supplies": 0.02, "Apparel": 0.14}


def retail_orders(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    # More orders later (growth) and in November/December (season).
    weight = np.linspace(1.0, 1.5, len(days)) * np.where(days.month.isin([11, 12]), 1.6, 1.0)
    dates = rng.choice(days, size=n, p=weight / weight.sum())
    dates = np.sort(dates)
    region_p = np.array([0.24, 0.26, 0.24, 0.26])
    regions = rng.choice(REGIONS, size=n, p=region_p)
    cats = rng.choice(list(CATEGORIES), size=n, p=[0.27, 0.18, 0.33, 0.22])
    ts = pd.to_datetime(dates)
    # The West grows over time: more West orders in 2025.
    late = ts >= pd.Timestamp("2025-01-01")
    flip = late & (rng.random(n) < 0.18)
    regions = np.where(flip, "West", regions)
    units = rng.integers(1, 6, n)
    base = np.array([CATEGORIES[c] for c in cats])
    price = base * rng.lognormal(0.0, 0.35, n)
    # South furniture lost its discount programme in Q3 2025: far fewer, smaller orders.
    q3 = (ts >= pd.Timestamp("2025-07-01")) & (ts < pd.Timestamp("2025-10-01"))
    hit = q3 & (regions == "South") & (cats == "Furniture")
    units = np.where(hit, np.maximum(1, units // 3), units)
    sales = np.round(price * units, 2)
    sales = np.where(hit & (rng.random(n) < 0.6), np.round(sales * 0.35, 2), sales)
    margin = np.array([MARGIN[c] for c in cats]) + rng.normal(0, 0.04, n)
    cost = np.round(sales * (1 - np.clip(margin, 0.05, 0.7)), 2)
    returned = np.where(rng.random(n) < np.array([RETURN_RATE[c] for c in cats]), "Yes", "No")
    channel = rng.choice(["Online", "Store"], size=n, p=[0.58, 0.42])
    segment = rng.choice(["Consumer", "Small Business", "Enterprise"], size=n, p=[0.55, 0.30, 0.15])
    return pd.DataFrame({
        "Order ID": [f"ORD-{100000 + i}" for i in range(n)],
        "Order Date": ts.strftime("%Y-%m-%d"),
        "Region": regions,
        "Category": cats,
        "Channel": channel,
        "Customer Segment": segment,
        "Units": units,
        "Sales": sales,
        "Cost": cost,
        "Returned": returned,
    })


def _customers(n: int, seed: int, start: str, end: str, price_rise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Monthly", "Annual", "Two-year"], size=n, p=[0.52, 0.32, 0.16])
    plan = rng.choice(["Basic", "Standard", "Premium"], size=n, p=[0.40, 0.40, 0.20])
    plan_price = pd.Series(plan).map({"Basic": 29.0, "Standard": 59.0, "Premium": 99.0}).to_numpy()
    monthly = np.round(plan_price * (1 + price_rise) * rng.lognormal(0, 0.12, n), 2)
    tenure = np.clip(np.round(rng.gamma(2.0, 9.0, n)), 1, 72).astype(int)
    tickets = rng.poisson(0.8 + (plan == "Basic") * 0.6, n)
    late = rng.poisson(0.35, n)
    last_login = np.clip(np.round(rng.exponential(9, n)), 0, 120).astype(int)
    region = rng.choice(REGIONS, size=n)
    signup = pd.to_datetime(rng.choice(pd.date_range(start, end, freq="D"), size=n)).strftime("%Y-%m-%d")
    logit = (-2.3 + 1.25 * (contract == "Monthly") - 0.6 * (contract == "Two-year") - 0.045 * tenure
             + 0.42 * tickets + 0.55 * late + 0.035 * last_login + 0.008 * (monthly - 55))
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return pd.DataFrame({
        "customer_id": [f"C{seed:02d}{i:05d}" for i in range(n)],
        "signup_date": signup,
        "region": region,
        "plan": plan,
        "contract": contract,
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_tickets_90d": tickets,
        "late_payments_12m": late,
        "days_since_last_login": last_login,
        "churned": churned,
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    retail_orders().to_csv(OUT / "retail_orders.csv", index=False)
    _customers(4000, seed=11, start="2022-01-01", end="2025-06-30").to_csv(OUT / "customer_churn.csv", index=False)
    new = _customers(1000, seed=23, start="2025-07-01", end="2025-12-31", price_rise=0.18).drop(columns=["churned"])
    new.to_csv(OUT / "churn_new_customers.csv", index=False)
    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name}: {sum(1 for _ in f.open()) - 1:,} rows")


if __name__ == "__main__":
    main()
