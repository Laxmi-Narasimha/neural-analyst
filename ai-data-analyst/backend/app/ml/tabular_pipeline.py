"""Leakage-safe tabular preprocessing and a self-contained model bundle.

Everything that learns from data (column roles, imputation values, scaling
statistics, category vocabularies) is learned in ``TabularPreprocessor.fit``.
Because the preprocessor is the first step of a scikit-learn ``Pipeline``,
cross-validation refits it on every training fold and the held-out rows never
influence it.

``ModelBundle`` is what gets saved to the registry: the fitted pipeline plus
everything needed to score *raw* rows later (input schema, class labels and a
record of how the model was trained and evaluated).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_ID_NAME_RE = re.compile(r"(^|[_\s-])(id|uuid|guid|key|code|number|no)$|^id[_\s-]", re.IGNORECASE)
_TIME_NAME_RE = re.compile(r"(date|time|timestamp|created|updated|_at$|_on$)", re.IGNORECASE)
_DATE_HINT_RE = re.compile(r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE)
_NUMERIC_JUNK_RE = re.compile(r"[,\s$€£₹%]")
_MISSING = "__missing__"

ROLE_NUMERIC = "numeric"
ROLE_NUMERIC_TEXT = "numeric_text"
ROLE_BOOLEAN = "boolean"
ROLE_CATEGORICAL = "categorical"
ROLE_DATETIME = "datetime"
ROLE_TEXT = "text"

_DMY_RE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
_NESTED = (dict, list, tuple, set)


def detect_dayfirst(series: pd.Series) -> bool:
    """True when slash/dot dates in the column are day-first (e.g. 13/01/2024)."""
    sample = series.dropna().astype(str).head(2000)
    parts = sample.str.extract(_DMY_RE)
    if parts.empty or parts[0].isna().all():
        return False
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    if bool((first > 12).any()):
        return True
    if bool((second > 12).any()):
        return False
    return False


def parse_datetime(series: pd.Series, dayfirst: Optional[bool] = None) -> pd.Series:
    """Parse to naive UTC datetimes with one consistent day/month order per column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        out = series
    else:
        if dayfirst is None:
            dayfirst = detect_dayfirst(series)
        out = pd.to_datetime(
            series.astype("object").where(series.notna(), None),
            errors="coerce",
            utc=True,
            format="mixed",
            dayfirst=bool(dayfirst),
        )
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def normalise_cells(series: pd.Series) -> pd.Series:
    """Turn nested values (dicts, lists, sets from JSON sources) into stable strings."""
    if series.dtype != object or not series.map(lambda v: isinstance(v, _NESTED)).any():
        return series

    def conv(v: Any) -> Any:
        if isinstance(v, _NESTED):
            try:
                return json.dumps(sorted(v) if isinstance(v, set) else v, sort_keys=True, default=str)
            except Exception:
                return str(v)
        return v

    return series.map(conv)


def dedupe_column_names(columns: Any) -> list[str]:
    """String column names, made unique the same way at training and scoring time."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for c in columns:
        base = str(c).strip() or "column"
        name = base
        while name in seen:
            seen[base] = seen.get(base, 0) + 1
            name = f"{base}__{seen[base]}"
        seen.setdefault(name, 0)
        names.append(name)
    return names


def _to_datetime(series: pd.Series, dayfirst: Optional[bool] = None) -> pd.Series:
    """Parse a column to naive UTC datetimes; unparseable values become NaT."""
    return parse_datetime(series, dayfirst=dayfirst)


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values), np.nan)


def _clean_numeric_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(_NUMERIC_JUNK_RE, "", regex=True)
    return _finite(pd.to_numeric(cleaned, errors="coerce"))


def _looks_like_dates(non_null: pd.Series, column_name: str) -> bool:
    sample = non_null.astype(str).head(300)
    if sample.empty:
        return False
    if not (_TIME_NAME_RE.search(column_name) or sample.str.contains(_DATE_HINT_RE).mean() >= 0.9):
        return False
    try:
        parsed = parse_datetime(sample)
    except Exception:
        return False
    return float(parsed.notna().mean()) >= 0.9


class TabularPreprocessor(TransformerMixin, BaseEstimator):
    """Turn a raw DataFrame of mixed columns into a numeric feature matrix.

    Column roles are decided from the data passed to ``fit`` only:

    * numeric / boolean columns -> median imputation (+ missing indicator) and scaling
    * numeric-looking text ("$1,200", "15%") -> parsed to numbers, then as numeric
    * date columns -> year, month, day, weekday, hour (if present) and elapsed days
    * categorical text -> one-hot encoding; rare and unseen values share one bucket
    * free text (several words per value) -> TF-IDF word and bigram features
    * infinities are treated as missing; nested JSON values become strings
    * constant, mostly-empty, identifier-like and free-text columns are dropped,
      each with a recorded reason
    """

    def __init__(
        self,
        max_onehot_categories: int = 20,
        max_missing_fraction: float = 0.95,
        identifier_unique_ratio: float = 0.95,
        max_text_features: int = 300,
    ) -> None:
        self.max_onehot_categories = max_onehot_categories
        self.max_missing_fraction = max_missing_fraction
        self.identifier_unique_ratio = identifier_unique_ratio
        self.max_text_features = max_text_features

    # -- fitting -----------------------------------------------------------------
    def _role_for(self, name: str, series: pd.Series) -> tuple[Optional[str], str]:
        n = int(series.shape[0])
        non_null = series.dropna()
        if non_null.empty:
            return None, "every value is missing"
        missing_fraction = 1.0 - non_null.shape[0] / max(n, 1)
        if missing_fraction > self.max_missing_fraction:
            return None, f"{missing_fraction:.0%} of values are missing"
        n_unique = int(non_null.astype(str).nunique()) if non_null.dtype == object else int(non_null.nunique())
        if n_unique <= 1:
            return None, "the column has a single constant value"
        unique_ratio = n_unique / max(non_null.shape[0], 1)

        if pd.api.types.is_bool_dtype(series):
            return ROLE_BOOLEAN, ""
        if pd.api.types.is_datetime64_any_dtype(series):
            return ROLE_DATETIME, ""
        if pd.api.types.is_numeric_dtype(series):
            is_integer_like = bool(np.all(np.mod(non_null.to_numpy(dtype=float), 1) == 0))
            if _ID_NAME_RE.search(name) and is_integer_like and unique_ratio >= self.identifier_unique_ratio:
                return None, "looks like a row identifier"
            return ROLE_NUMERIC, ""

        # Text-like column.
        if _looks_like_dates(non_null, name):
            return ROLE_DATETIME, ""
        parsed_numbers = _clean_numeric_text(non_null)
        if float(parsed_numbers.notna().mean()) >= 0.95:
            return ROLE_NUMERIC_TEXT, ""
        words = non_null.astype(str).head(500).str.split().str.len()
        if float(words.median()) >= 3 and unique_ratio >= 0.3:
            return ROLE_TEXT, ""
        if unique_ratio >= self.identifier_unique_ratio and n_unique > 20:
            return None, "almost every value is unique (looks like an identifier)"
        return ROLE_CATEGORICAL, ""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TabularPreprocessor":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.input_columns_ = [str(c) for c in X.columns]
        X = X.copy()
        X.columns = self.input_columns_

        self.column_roles_: dict[str, str] = {}
        self.dropped_columns_: dict[str, str] = {}
        self.datetime_parts_: dict[str, list[str]] = {}
        self.date_dayfirst_: dict[str, bool] = {}
        for col in self.input_columns_:
            series = normalise_cells(X[col])
            role, reason = self._role_for(col, series)
            if role is None:
                self.dropped_columns_[col] = reason
                continue
            self.column_roles_[col] = role
            if role == ROLE_DATETIME:
                self.date_dayfirst_[col] = detect_dayfirst(series)
                parsed = _to_datetime(series, dayfirst=self.date_dayfirst_[col])
                parts = ["year", "month", "day", "dayofweek", "elapsed_days"]
                hours = parsed.dt.hour.dropna()
                if not hours.empty and int(hours.nunique()) > 1:
                    parts.append("hour")
                self.datetime_parts_[col] = parts

        if not self.column_roles_:
            reasons = "; ".join(f"{c}: {r}" for c, r in self.dropped_columns_.items()) or "no columns supplied"
            raise ValueError(f"No usable feature columns ({reasons})")

        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.text_features_: list[str] = []
        for col, role in self.column_roles_.items():
            if role == ROLE_CATEGORICAL:
                self.categorical_features_.append(col)
            elif role == ROLE_TEXT:
                self.text_features_.append(col)
            elif role == ROLE_DATETIME:
                self.numeric_features_.extend(f"{col}__{p}" for p in self.datetime_parts_[col])
            else:
                self.numeric_features_.append(col)

        prepared = self._prepare(X)
        transformers = []
        if self.numeric_features_:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    self.numeric_features_,
                )
            )
        if self.categorical_features_:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        max_categories=int(self.max_onehot_categories),
                        sparse_output=False,
                    ),
                    self.categorical_features_,
                )
            )
        for i, col in enumerate(self.text_features_):
            transformers.append(
                (
                    f"text{i}",
                    TfidfVectorizer(
                        max_features=int(self.max_text_features),
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                    col,
                )
            )
        self.column_transformer_ = ColumnTransformer(
            transformers, remainder="drop", verbose_feature_names_out=True, sparse_threshold=0.0
        )
        self.column_transformer_.fit(prepared)
        self.feature_names_out_ = [str(n) for n in self.column_transformer_.get_feature_names_out()]
        self.n_features_in_ = len(self.input_columns_)
        return self

    # -- transforming --------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        out: dict[str, pd.Series] = {}
        for col, role in self.column_roles_.items():
            if col in X.columns:
                series = normalise_cells(X[col])
            else:
                series = pd.Series(np.nan, index=X.index, dtype=object)
            if role == ROLE_NUMERIC:
                out[col] = _finite(series)
            elif role == ROLE_BOOLEAN:
                mapped = series.map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0})
                out[col] = _finite(mapped)
            elif role == ROLE_NUMERIC_TEXT:
                out[col] = _clean_numeric_text(series)
            elif role == ROLE_TEXT:
                out[col] = series.astype("object").where(series.notna(), "").astype(str)
            elif role == ROLE_DATETIME:
                parsed = _to_datetime(series, dayfirst=self.date_dayfirst_.get(col))
                for part in self.datetime_parts_[col]:
                    if part == "elapsed_days":
                        values = (parsed - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
                    else:
                        values = getattr(parsed.dt, part)
                    out[f"{col}__{part}"] = pd.to_numeric(values, errors="coerce").astype(float)
            else:  # categorical
                as_text = series.astype("object").where(series.notna(), _MISSING).astype(str).str.strip()
                out[col] = as_text.where(as_text != "", _MISSING)
        return pd.DataFrame(out, index=X.index)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "column_transformer_"):
            raise ValueError("TabularPreprocessor is not fitted")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.input_columns_)
        return self.column_transformer_.transform(self._prepare(X))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def summary(self) -> dict[str, Any]:
        return {
            "input_columns": list(self.input_columns_),
            "column_roles": dict(self.column_roles_),
            "dropped_columns": dict(self.dropped_columns_),
            "n_output_features": len(self.feature_names_out_),
        }


@dataclass
class ModelBundle:
    """A trained pipeline that can score raw rows in a fresh process."""

    pipeline: Any
    task: str
    target_column: str
    model_name: str
    input_columns: list[str]
    classes: Optional[list[Any]] = None
    # Input columns the preprocessing actually uses (identifiers, constants etc. are dropped).
    used_columns: Optional[list[str]] = None
    training_info: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 2
    # Version 2. Bundles saved before have none of these; every use goes through getattr.
    calibrator: Any = None                       # {"positive_index", "map"} for two-class targets
    decision_threshold: Optional[dict[str, Any]] = None  # {"positive_index", "threshold"}
    reference_profile: Optional[dict[str, Any]] = None   # training-input profile for drift checks
    prediction_reference: Optional[dict[str, Any]] = None  # held-out prediction mix at training time

    @property
    def is_classifier(self) -> bool:
        return self.classes is not None

    def check_input(self, df: pd.DataFrame) -> list[str]:
        """Return warnings about the input; raise if it cannot be scored at all."""
        present = set(dedupe_column_names(df.columns))
        needed = list(self.used_columns or self.input_columns)
        missing = [c for c in needed if c not in present]
        if missing and len(missing) == len(needed):
            raise ValueError(
                "None of the columns this model was trained on are present. Expected: "
                + ", ".join(needed)
            )
        warnings: list[str] = []
        if missing:
            warnings.append(
                "Missing input columns were treated as unknown values: " + ", ".join(missing)
            )
        if self.target_column in present:
            warnings.append(f"Column '{self.target_column}' is the training target and was ignored for scoring.")
        return warnings

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = dedupe_column_names(frame.columns)
        return frame.reindex(columns=self.input_columns)

    def _proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
        cal = getattr(self, "calibrator", None)
        if cal and proba.shape[1] == 2:
            pos = int(cal["positive_index"])
            p = np.asarray(cal["map"](proba[:, pos]), dtype=float)
            proba[:, pos], proba[:, 1 - pos] = p, 1.0 - p
        return proba

    def predict(self, df: pd.DataFrame) -> list[Any]:
        frame = self._frame(df)
        thr = getattr(self, "decision_threshold", None)
        if self.is_classifier and thr and hasattr(self.pipeline, "predict_proba"):
            pos = int(thr["positive_index"])
            p = self._proba(frame)[:, pos]
            raw = np.where(p >= float(thr["threshold"]), pos, 1 - pos)
        else:
            raw = self.pipeline.predict(frame)
        if self.is_classifier:
            return [self.classes[int(i)] for i in raw]
        return [float(v) for v in raw]

    def predict_proba(self, df: pd.DataFrame) -> Optional[list[list[float]]]:
        if not self.is_classifier or not hasattr(self.pipeline, "predict_proba"):
            return None
        return self._proba(self._frame(df)).tolist()

    def drift_report(self, df: pd.DataFrame) -> Optional[dict[str, Any]]:
        profile = getattr(self, "reference_profile", None)
        if not profile:
            return None
        from app.ml.drift import compare

        frame = df.copy()
        frame.columns = dedupe_column_names(frame.columns)
        return compare(profile, frame)

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "target_column": self.target_column,
            "input_columns": list(self.input_columns),
            "used_columns": list(self.used_columns or self.input_columns),
            "classes": list(self.classes) if self.classes is not None else None,
            "training_info": self.training_info,
            "created_at": self.created_at,
            "schema_version": getattr(self, "schema_version", 1),
            "calibrated": bool(getattr(self, "calibrator", None)),
            "decision_threshold": getattr(self, "decision_threshold", None),
            "has_drift_reference": bool(getattr(self, "reference_profile", None)),
        }
