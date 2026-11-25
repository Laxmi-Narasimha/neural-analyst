"""Deterministic evaluation datasets for Talk to Your Data.

Each builder returns a pandas DataFrame generated from a fixed seed, so expected
answers can be computed independently with pandas and compared with what the
product reports. Values are deliberately non-round so a correct number is hard
to hit by accident.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def retail_orders(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1200
    start = date(2025, 1, 1)
    days = rng.integers(0, 690, n)  # 2025-01-01 .. 2026-11-21
    branches = np.array(["North", "South", "East", "West"])
    channels = np.array(["Online", "Store"])
    categories = np.array(["Grocery", "Electronics", "Apparel", "Home"])
    branch = branches[rng.choice(4, n, p=[0.3, 0.25, 0.25, 0.2])]
    category = categories[rng.choice(4, n, p=[0.4, 0.15, 0.25, 0.2])]
    base = {"Grocery": 38.0, "Electronics": 410.0, "Apparel": 72.0, "Home": 145.0}
    branch_lift = {"North": 1.0, "South": 0.92, "East": 1.07, "West": 1.21}
    net = np.array([base[c] * branch_lift[b] for c, b in zip(category, branch)]) * rng.lognormal(0, 0.35, n)
    units = rng.integers(1, 7, n)
    discount = np.round(rng.choice([0, 0, 0, 5, 10, 15, 20], n) + rng.random(n), 2)
    return pd.DataFrame({
        "Order ID": np.arange(10001, 10001 + n),
        "Order Date": [(start + timedelta(days=int(d))).isoformat() for d in days],
        "Branch": branch,
        "Channel": channels[rng.choice(2, n, p=[0.45, 0.55])],
        "Category": category,
        "Net Sales": np.round(net, 2),
        "Units": units,
        "Discount %": discount,
        "Customer Email": [f"cust{int(i)}@example.com" for i in rng.integers(1, 400, n)],
        "Returned": np.where(rng.random(n) < 0.08, "Yes", "No"),
    })


def saas_accounts(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    start = date(2024, 6, 1)
    plans = np.array(["Basic", "Pro", "Enterprise"])
    plan = plans[rng.choice(3, n, p=[0.5, 0.35, 0.15])]
    mrr_base = {"Basic": 29.0, "Pro": 119.0, "Enterprise": 940.0}
    seats = np.array([int(rng.integers(1, 4)) if p == "Basic" else int(rng.integers(3, 25)) if p == "Pro" else int(rng.integers(20, 300)) for p in plan])
    mrr = np.round(np.array([mrr_base[p] for p in plan]) * (1 + seats / 50) * rng.uniform(0.9, 1.1, n), 2)
    signup = [start + timedelta(days=int(d)) for d in rng.integers(0, 880, n)]
    churn_p = np.array([{"Basic": 0.32, "Pro": 0.18, "Enterprise": 0.07}[p] for p in plan])
    churned = rng.random(n) < churn_p
    return pd.DataFrame({
        "account_id": [f"A-{i:05d}" for i in range(n)],
        "signup_date": [d.isoformat() for d in signup],
        "plan": plan,
        "country": np.array(["India", "United States", "Germany", "Brazil", "Japan"])[rng.choice(5, n, p=[0.3, 0.3, 0.15, 0.15, 0.1])],
        "seats": seats,
        "mrr_usd": mrr,
        "churned": np.where(churned, 1, 0),
        "nps_score": np.where(rng.random(n) < 0.15, np.nan, rng.integers(-100, 101, n)).astype("float"),
    })


def hr_people(seed: int = 3) -> pd.DataFrame:
    """Messy, abbreviated headers like real HR exports."""
    rng = np.random.default_rng(seed)
    n = 420
    depts = np.array(["Engineering", "Sales", "Support", "Finance", "Marketing"])
    dept = depts[rng.choice(5, n, p=[0.35, 0.25, 0.2, 0.1, 0.1])]
    pay = {"Engineering": 1850000, "Sales": 1120000, "Support": 640000, "Finance": 1320000, "Marketing": 980000}
    salary = np.round(np.array([pay[d] for d in dept]) * rng.lognormal(0, 0.18, n), -2)
    hire = [date(2016, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 3800, n)]
    return pd.DataFrame({
        "Emp #": np.arange(1, n + 1),
        "Dept": dept,
        "Loc": np.array(["Bengaluru", "Hyderabad", "Pune", "Remote"])[rng.choice(4, n)],
        "DOJ": [d.isoformat() for d in hire],
        "CTC (INR)": salary,
        "Perf Rtg": rng.choice([1, 2, 3, 3, 4, 4, 4, 5], n),
        "Left Co?": np.where(rng.random(n) < 0.14, "Y", "N"),
    })


PNL_LAST_DAY = date(2026, 11, 20)


def store_pnl(seed: int = 5) -> pd.DataFrame:
    """Daily revenue, cost and units per region and product line, ending on PNL_LAST_DAY."""
    rng = np.random.default_rng(seed)
    rows = []
    day = date(2025, 1, 1)
    regions = {"North": 1.0, "South": 0.8, "East": 1.15, "West": 0.95}
    lines = {"Core": (40.0, 0.62), "Premium": (120.0, 0.48), "Accessories": (15.0, 0.35)}
    while day <= PNL_LAST_DAY:
        season = 1 + 0.15 * np.sin(day.timetuple().tm_yday / 365 * 2 * np.pi)
        for region, lift in regions.items():
            for line, (price, cost_share) in lines.items():
                if rng.random() < 0.35:
                    continue
                units = int(rng.poisson(6 * lift * season)) + 1
                revenue = round(units * price * rng.uniform(0.9, 1.1), 2)
                cost = round(revenue * cost_share * rng.uniform(0.92, 1.08), 2)
                rows.append((day.isoformat(), region, line, units, revenue, cost))
        day += timedelta(days=1)
    return pd.DataFrame(rows, columns=["Date", "Region", "Product Line", "Units", "Revenue", "Cost"])


DATASETS = {
    "retail_orders": retail_orders,
    "saas_accounts": saas_accounts,
    "hr_people": hr_people,
    "store_pnl": store_pnl,
}
