"""Evaluation that holds up for real targets: imbalance, calibration, thresholds, error by segment.

Every decision here is made on the training split only — out-of-fold predictions from the same
cross-validation folds used for model selection. The held-out split is scored once afterwards, so
the reported test numbers are never used to choose anything.

- Imbalance: when the rarer class is under 20% of rows, accuracy and ROC AUC flatter a model that
  ignores it. Selection then uses average precision (PR AUC) for two classes and macro F1 for more.
- Threshold: for an imbalanced two-class target the default 0.5 cut-off usually predicts the
  majority class for everyone. The cut-off that maximises F1 on out-of-fold probabilities is used
  instead and stored with the model; precision and recall at other cut-offs are reported so the
  user can pick a different trade-off.
- Calibration: a probability of 0.8 should come true about 80% of the time. A calibration map is
  learned on out-of-fold probabilities and kept only if it improves the out-of-fold Brier score
  under its own cross-validation. Reliability (predicted vs observed, per bin) is reported on the
  held-out split before and after.
- Error by segment: where the model is worse than average on the held-out rows, with a
  significance check so small groups do not raise false alarms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

IMBALANCE_SHARE = 0.20
THRESHOLD_TUNING_SHARE = 0.40
MULTICLASS_IMBALANCE_SHARE = 0.10


# ---------------------------------------------------------------------------
# Imbalance and the selection metric
# ---------------------------------------------------------------------------

def class_balance(y: np.ndarray, n_classes: int) -> dict[str, Any]:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes)
    total = int(counts.sum()) or 1
    shares = counts / total
    minority = int(np.argmin(counts))
    imbalanced = bool(
        shares[minority] < (IMBALANCE_SHARE if n_classes == 2 else MULTICLASS_IMBALANCE_SHARE)
    )
    return {
        "counts": [int(c) for c in counts],
        "minority_class_index": minority,
        "minority_share": float(shares[minority]),
        "imbalanced": imbalanced,
        # A 0.5 cut-off under-predicts any clearly rarer outcome (a 21% churn rate is typical), so the
        # cut-off is tuned on training folds whenever the rarer class is under 40%, not only when the
        # model selection metric changes (under 20%).
        "tune_threshold": bool(n_classes == 2 and shares[minority] < THRESHOLD_TUNING_SHARE),
    }


def selection_metric(n_classes: int, imbalanced: bool) -> tuple[str, str]:
    """(scikit-learn scoring name, why) for model selection."""
    if n_classes == 2:
        if imbalanced:
            return "average_precision", (
                "The rarer class is under 20% of rows, so models are compared on average precision "
                "(PR AUC), which rewards finding the rare class; ROC AUC and accuracy look good even "
                "for models that miss it."
            )
        return "roc_auc", "Two balanced classes: models are compared on ROC AUC."
    if imbalanced:
        return "f1_macro", (
            "At least one class is under 10% of rows, so models are compared on macro F1, which "
            "weighs every class equally; weighted F1 would let the large classes hide failures."
        )
    return "f1_weighted", "Models are compared on weighted F1."


# ---------------------------------------------------------------------------
# Out-of-fold probabilities
# ---------------------------------------------------------------------------

def out_of_fold_proba(pipe: Any, X: pd.DataFrame, y: np.ndarray, cv: Any) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities for each training row from a model that did not see it.

    Returns (proba, mask): rows never in a validation fold (the first block of a forward-chaining
    split) have mask False.
    """
    n = len(X)
    proba: Optional[np.ndarray] = None
    mask = np.zeros(n, dtype=bool)
    for train_idx, val_idx in cv.split(X, y):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = clone(pipe).fit(X.iloc[train_idx], y[train_idx])
        p = np.asarray(model.predict_proba(X.iloc[val_idx]), dtype=float)
        if proba is None:
            proba = np.full((n, p.shape[1]), np.nan)
        if p.shape[1] != proba.shape[1]:
            continue  # a fold missed a class entirely; skip rather than misalign columns
        proba[val_idx] = p
        mask[val_idx] = True
    if proba is None:
        raise ValueError("no cross-validation fold could produce probabilities")
    return proba, mask


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibrator:
    """Maps a raw positive-class probability to a calibrated one (two-class targets)."""

    method: str  # "sigmoid" or "isotonic"
    model: Any

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            return self.model.predict_proba(logit)[:, 1]
        return np.clip(self.model.predict(p), 0.0, 1.0)


def _fit_calibrator(p: np.ndarray, y: np.ndarray, method: str) -> Calibrator:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if method == "sigmoid":
        model = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    else:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return Calibrator(method, model)


def choose_calibration(p_oof: np.ndarray, y: np.ndarray, random_state: int = 0) -> tuple[Optional[Calibrator], dict[str, Any]]:
    """Fit a calibration map on out-of-fold probabilities; keep it only if it helps out of sample."""
    y = np.asarray(y, dtype=int)
    method = "isotonic" if len(y) >= 1000 else "sigmoid"
    raw_brier = float(brier_score_loss(y, p_oof))
    if len(y) < 50 or len(np.unique(y)) < 2:
        return None, {"applied": False, "method": None, "reason": "too few rows to learn a calibration map",
                      "oof_brier_raw": raw_brier}
    cal_oof = np.full(len(y), np.nan)
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=random_state).split(p_oof):
        if len(np.unique(y[tr])) < 2:
            continue
        cal_oof[va] = _fit_calibrator(p_oof[tr], y[tr], method)(p_oof[va])
    ok = np.isfinite(cal_oof)
    cal_brier = float(brier_score_loss(y[ok], cal_oof[ok])) if ok.any() else raw_brier
    raw_on_same = float(brier_score_loss(y[ok], p_oof[ok])) if ok.any() else raw_brier
    helps = ok.mean() > 0.9 and cal_brier < raw_on_same * 0.99
    info = {
        "applied": bool(helps),
        "method": method if helps else None,
        "oof_brier_raw": raw_on_same,
        "oof_brier_calibrated": cal_brier,
        "reason": (
            f"{method} calibration lowered the out-of-fold Brier score from {raw_on_same:.4f} to {cal_brier:.4f}"
            if helps else "the model's probabilities were already as well calibrated as a learned map could make them"
        ),
    }
    return (_fit_calibrator(p_oof, y, method) if helps else None), info


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Predicted vs observed rate in equal-count bins, Brier score and expected calibration error."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return {"bins": [], "brier": None, "ece": None}
    k = max(1, min(bins, n // 10 or 1))
    order = np.argsort(p, kind="stable")
    rows = []
    ece = 0.0
    for chunk in np.array_split(order, k):
        if len(chunk) == 0:
            continue
        mp, obs = float(p[chunk].mean()), float(y[chunk].mean())
        ece += len(chunk) / n * abs(mp - obs)
        rows.append({"n": int(len(chunk)), "mean_predicted": round(mp, 4), "observed_rate": round(obs, 4)})
    return {"bins": rows, "brier": round(float(np.mean((p - y) ** 2)), 5), "ece": round(float(ece), 4)}


# ---------------------------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------------------------

def _scores_at(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = (p >= t).astype(int)
    return {
        "threshold": round(float(t), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "predicted_positive_share": round(float(pred.mean()), 4),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Cut-off on the positive-class probability that maximises F1 on out-of-fold rows."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f1 + 1e-12:
            best_t, best_f1 = float(t), float(f)
    table = [_scores_at(y, p, t) for t in sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, best_t})]
    return {
        "threshold": round(best_t, 4),
        "objective": "F1 of the positive (rarer) class on out-of-fold training rows",
        "at_default_0_5": _scores_at(y, p, 0.5),
        "at_chosen": _scores_at(y, p, best_t),
        "trade_offs": table,
    }


def scores_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    return _scores_at(np.asarray(y, dtype=int), np.asarray(p, dtype=float), threshold)


# ---------------------------------------------------------------------------
# Error by segment
# ---------------------------------------------------------------------------

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _segments(X: pd.DataFrame, max_levels: int = 20) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = []
    n = len(X)
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.nunique() <= max_levels:
                groups = values.astype("object").where(values.notna(), "(missing)")
            else:
                try:
                    groups = pd.qcut(values, 4, duplicates="drop").astype(str).where(values.notna(), "(missing)")
                except ValueError:
                    continue
        else:
            text = s.astype("string").str.strip()
            if text.nunique(dropna=True) > max_levels or text.nunique(dropna=True) > 0.5 * n:
                continue  # identifiers and free text are not segments
            groups = text.fillna("(missing)")
        for level in pd.unique(groups):
            out.append((str(col), str(level), (groups == level).to_numpy()))
    return out


def error_by_segment(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str,
    min_rows: int = 20,
    top: int = 10,
) -> dict[str, Any]:
    """Held-out segments where the model is clearly worse than on all held-out rows."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    classification = task != "regression"
    if classification:
        wrong = (y_true != y_pred).astype(float)
        overall = float(wrong.mean()) if n else 0.0
        metric = "error_rate"
    else:
        err = np.abs(y_true.astype(float) - y_pred.astype(float))
        overall = float(err.mean()) if n else 0.0
        metric = "mean_absolute_error"
    rows = []
    floor = max(min_rows, int(0.05 * n))
    for col, level, mask in _segments(X):
        m = int(mask.sum())
        if m < floor or m == n:
            continue
        if classification:
            k = int(wrong[mask].sum())
            value = k / m
            lo, _ = _wilson(k, m)
            significant = lo > overall
        else:
            seg = err[mask]
            rest = err[~mask]
            value = float(seg.mean())
            se = math.sqrt(seg.var(ddof=1) / m + rest.var(ddof=1) / len(rest)) if len(rest) > 1 and m > 1 else float("inf")
            significant = se > 0 and (value - float(rest.mean())) / se > 1.96
        if value <= overall:
            continue
        rows.append({
            "column": col, "segment": level, "rows": m, metric: round(value, 4),
            "overall": round(overall, 4), "ratio_to_overall": round(value / overall, 3) if overall else None,
            "significant": bool(significant),
        })
    rows.sort(key=lambda r: (not r["significant"], -(r["ratio_to_overall"] or 0), -r["rows"]))
    return {
        "metric": metric,
        "overall": round(overall, 4),
        "held_out_rows": n,
        "min_segment_rows": floor,
        "worst_segments": rows[:top],
        "method": (
            "Held-out rows grouped by each input column (categories, or quartiles for numbers). A segment "
            "is significant when it is worse than the overall rate beyond its 95% confidence interval."
        ),
    }
