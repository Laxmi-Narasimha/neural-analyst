"""Hostile datasets the ML pipeline must handle: train and score, or refuse clearly."""
import numpy as np, pandas as pd

R = np.random.default_rng(0)  # regenerated per test via reseed()


def sig(n, k=3):
    X = R.normal(size=(n, k))
    y = (X[:, 0] + 0.5 * X[:, 1] + R.normal(scale=.7, size=n)) > 0
    return X, y


cases = {}


def case(name, expect="train"):
    def deco(f):
        cases[name] = (f, expect)
        return f
    return deco


@case("tiny_25_rows")
def _():
    X, y = sig(25); return pd.DataFrame(X, columns=list("abc")).assign(t=y), "t"

@case("too_small_19_rows", "refuse")
def _():
    X, y = sig(19); return pd.DataFrame(X, columns=list("abc")).assign(t=y), "t"

@case("all_categorical")
def _():
    n = 800; a = R.choice(list("pqrs"), n); b = R.choice(["x", "y"], n)
    t = np.where((a == "p") | (b == "x"), "yes", "no")
    t = np.where(R.random(n) < .1, np.where(t == "yes", "no", "yes"), t)
    return pd.DataFrame({"a": a, "b": b, "t": t}), "t"

@case("only_date_features")
def _():
    n = 700; d = pd.date_range("2024-01-01", periods=n, freq="D"); t = (d.dayofweek >= 5).astype(int)
    return pd.DataFrame({"event_date": d.strftime("%d/%m/%Y"), "t": t}), "t"

@case("wide_300_cols")
def _():
    X, y = sig(500, 300); return pd.DataFrame(X, columns=[f"f{i}" for i in range(300)]).assign(t=y), "t"

@case("high_cardinality_city")
def _():
    n = 6000; X, y = sig(n)
    return pd.DataFrame({"city": [f"city_{i}" for i in R.integers(0, 3000, n)], "x": X[:, 0], "t": y}), "t"

@case("80pct_missing")
def _():
    X, y = sig(1000); df = pd.DataFrame(X, columns=list("abc")); df = df.mask(R.random(df.shape) < .8)
    return df.assign(t=y), "t"

@case("imbalanced_1pct")
def _():
    n = 5000; X = R.normal(size=(n, 3)); y = (X[:, 0] > 2.33).astype(int)
    return pd.DataFrame(X, columns=list("abc")).assign(t=y), "t"

@case("multiclass_with_singleton_class")
def _():
    n = 600; x = R.normal(size=n); t = np.select([x < -.5, x > .5], ["lo", "hi"], "mid").astype(object); t[0] = "ultra_rare"
    return pd.DataFrame({"x": x, "z": R.normal(size=n), "t": t}), "t"

@case("currency_text_regression_target")
def _():
    n = 600; x = R.uniform(0, 100, n); y = 50 * x + R.normal(scale=50, size=n)
    return pd.DataFrame({"x": x, "t": [f"₹{v:,.0f}" for v in y]}), "t"

@case("boolean_target")
def _():
    X, y = sig(500); return pd.DataFrame(X, columns=list("abc")).assign(t=y.astype(bool)), "t"

@case("messy_yes_no_target")
def _():
    X, y = sig(600)
    t = np.where(y, R.choice([" yes", "YES", "Yes"], 600), R.choice(["no", "No ", "NO"], 600))
    return pd.DataFrame(X, columns=list("abc")).assign(t=t), "t"

@case("mixed_type_column")
def _():
    X, y = sig(600); m = X[:, 0].astype(object); m[::5] = "N/A"; m[1::7] = "unknown"
    return pd.DataFrame({"m": m, "b": X[:, 1], "t": y}), "t"

@case("unicode_and_spaces_in_names")
def _():
    X, y = sig(500)
    return pd.DataFrame(X, columns=["Net Sales (₹)", "Région", "客户 年龄"]).assign(**{"Churned?": y}), "Churned?"

@case("duplicate_column_names")
def _():
    X, y = sig(500); df = pd.DataFrame(X, columns=["a", "a", "b"]); df["t"] = y; return df, "t"

@case("integer_column_names")
def _():
    X, y = sig(500); df = pd.DataFrame(X); df["t"] = y; return df, "t"

@case("infinities_in_numeric")
def _():
    X, y = sig(500); X[::9, 0] = np.inf; X[::11, 1] = -np.inf
    return pd.DataFrame(X, columns=list("abc")).assign(t=y), "t"

@case("tz_aware_and_mixed_format_dates")
def _():
    n = 500; d = pd.date_range("2025-01-01", periods=n, freq="6h", tz="Asia/Kolkata"); X, y = sig(n)
    s = d.astype(str).to_numpy(dtype=object); s[::3] = d[::3].strftime("%d %b %Y %H:%M")
    return pd.DataFrame({"created_at": s, "a": X[:, 0], "t": y}), "t"

@case("constant_target", "refuse")
def _():
    X, _ = sig(300); return pd.DataFrame(X, columns=list("abc")).assign(t=1), "t"

@case("huge_regression_values")
def _():
    n = 800; x = R.lognormal(10, 3, n)
    return pd.DataFrame({"x": x, "seg": R.choice(list("ab"), n), "t": x * 1e6 + R.normal(size=n) * 1e9}), "t"

@case("free_text_reviews")
def _():
    n = 600; good = R.random(n) < .5; words = np.where(good, "great fast lovely", "awful slow broken")
    txt = [f"{w} order {i} {R.integers(1e6)}" for i, w in enumerate(words)]
    return pd.DataFrame({"review": txt, "stars": np.where(good, R.integers(4, 6, n), R.integers(1, 3, n)), "t": good}), "t"

@case("only_identifier_features", "refuse")
def _():
    n = 300
    return pd.DataFrame({"customer_id": range(n), "uuid": [f"u-{i:08d}" for i in range(n)], "t": R.choice([0, 1], n)}), "t"

@case("time_split_with_many_NaT")
def _():
    n = 900; X, y = sig(n); d = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D").astype(str))
    d[R.random(n) < .4] = None
    return pd.DataFrame({"order_date": d, "a": X[:, 0], "b": X[:, 1], "t": y}), "t"

@case("pandas_category_dtype")
def _():
    X, y = sig(600)
    return pd.DataFrame({"c": pd.Categorical(R.choice(["lo", "hi"], 600)), "a": X[:, 0],
                         "t": pd.Categorical(np.where(y, "A", "B"))}), "t"

@case("json_nested_cells")
def _():
    X, y = sig(400)
    return pd.DataFrame({"meta": [{"k": int(i % 3)} for i in range(400)], "tags": [["a", "b"][: i % 3] for i in range(400)],
                         "a": X[:, 0], "t": y}), "t"

@case("single_feature")
def _():
    X, y = sig(400); return pd.DataFrame({"a": X[:, 0], "t": y}), "t"

@case("numeric_codes_as_categories")
def _():
    n = 800; z = R.choice([101, 202, 303, 404], n); t = np.isin(z, [101, 303]); t = np.where(R.random(n) < .1, ~t, t)
    return pd.DataFrame({"zone_code": z, "x": R.normal(size=n), "t": t}), "t"

@case("regression_negative_and_zero_target")
def _():
    n = 500; x = R.normal(size=n); return pd.DataFrame({"x": x, "t": np.round(3 * x, 0) * 100}), "t"

@case("200k_rows")
def _():
    n = 200_000; X, y = sig(n); df = pd.DataFrame(X, columns=list("abc")); df["seg"] = R.choice(list("pqrstu"), n)
    return df.assign(t=y), "t"

@case("leaky_future_date_column")
def _():
    n = 800; od = pd.date_range("2025-01-01", periods=n, freq="12h"); late = R.random(n) < .4
    dd = od + pd.to_timedelta(np.where(late, R.integers(6, 10, n), R.integers(1, 4, n)), unit="D")
    return pd.DataFrame({"order_date": od.astype(str), "delivered_date": dd.astype(str), "dist": R.normal(size=n),
                         "late": late}), "late"



def reseed(seed: int = 0) -> None:
    global R
    R = np.random.default_rng(seed)
