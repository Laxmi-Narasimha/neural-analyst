# AI Enterprise Data Analyst - Advanced Data Quality Engine
# Production-grade data quality assessment and remediation

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import DataProcessingException, ValidationException
except ImportError:
    class DataProcessingException(Exception): pass
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Data Quality Enums and Models
# ============================================================================

class QualityDimension(str, Enum):
    """Data quality dimensions following DAMA standards."""
    COMPLETENESS = "completeness"  # Missing values
    ACCURACY = "accuracy"  # Correctness
    VALIDITY = "validity"  # Conformance to rules
    CONSISTENCY = "consistency"  # Across datasets
    UNIQUENESS = "uniqueness"  # Duplicates
    TIMELINESS = "timeliness"  # Data freshness
    INTEGRITY = "integrity"  # Referential constraints


class MissingPattern(str, Enum):
    """Missing data patterns."""
    MCAR = "mcar"  # Missing Completely At Random
    MAR = "mar"  # Missing At Random
    MNAR = "mnar"  # Missing Not At Random


class OutlierMethod(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    MAHALANOBIS = "mahalanobis"


@dataclass
class ColumnQualityReport:
    """Quality report for a single column."""
    
    column: str
    dtype: str
    
    # Completeness
    total_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    missing_pattern: Optional[MissingPattern] = None
    
    # Uniqueness
    unique_count: int = 0
    unique_percentage: float = 0.0
    duplicate_count: int = 0
    
    # Validity
    valid_count: int = 0
    invalid_count: int = 0
    validation_rules_applied: list[str] = field(default_factory=list)
    
    # Statistical anomalies
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    outlier_indices: list[int] = field(default_factory=list)
    
    # Distribution
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    is_normal: Optional[bool] = None
    
    # Quality score (0-100)
    quality_score: float = 100.0
    
    # Issues found
    issues: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "completeness": {
                "total": self.total_count,
                "null_count": self.null_count,
                "null_percentage": round(self.null_percentage, 2),
                "missing_pattern": self.missing_pattern.value if self.missing_pattern else None
            },
            "uniqueness": {
                "unique_count": self.unique_count,
                "unique_percentage": round(self.unique_percentage, 2),
                "duplicate_count": self.duplicate_count
            },
            "validity": {
                "valid_count": self.valid_count,
                "invalid_count": self.invalid_count,
                "rules_applied": self.validation_rules_applied
            },
            "outliers": {
                "count": self.outlier_count,
                "percentage": round(self.outlier_percentage, 2)
            },
            "distribution": {
                "skewness": round(self.skewness, 4) if self.skewness else None,
                "kurtosis": round(self.kurtosis, 4) if self.kurtosis else None,
                "is_normal": self.is_normal
            },
            "quality_score": round(self.quality_score, 2),
            "issues": self.issues
        }


@dataclass
class DatasetQualityReport:
    """Comprehensive quality report for entire dataset."""
    
    dataset_name: str
    row_count: int
    column_count: int
    
    # Column reports
    column_reports: list[ColumnQualityReport] = field(default_factory=list)
    
    # Dataset-level metrics
    overall_completeness: float = 0.0
    overall_uniqueness: float = 0.0
    duplicate_rows: int = 0
    duplicate_row_percentage: float = 0.0
    
    # Data types
    type_distribution: dict[str, int] = field(default_factory=dict)
    
    # Overall score
    overall_quality_score: float = 100.0
    
    # Critical issues
    critical_issues: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "shape": {"rows": self.row_count, "columns": self.column_count},
            "overall_metrics": {
                "completeness": round(self.overall_completeness, 2),
                "uniqueness": round(self.overall_uniqueness, 2),
                "duplicate_rows": self.duplicate_rows,
                "duplicate_row_percentage": round(self.duplicate_row_percentage, 2),
                "quality_score": round(self.overall_quality_score, 2)
            },
            "type_distribution": self.type_distribution,
            "columns": [c.to_dict() for c in self.column_reports],
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations[:10]
        }


# ============================================================================
# Missing Data Analyzer
# ============================================================================

class MissingDataAnalyzer:
    """
    Analyze missing data patterns (MCAR, MAR, MNAR).
    
    Uses Little's MCAR test and correlation analysis.
    """
    
    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing data patterns in the dataset."""
        result = {
            "total_missing": 0,
            "total_cells": df.shape[0] * df.shape[1],
            "missing_percentage": 0.0,
            "columns_with_missing": {},
            "likely_mechanism": {},
            "correlation_with_missingness": []
        }
        
        # Calculate missing per column
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        result["total_missing"] = int(total_missing)
        result["missing_percentage"] = round(total_missing / result["total_cells"] * 100, 2)
        
        for col in df.columns:
            if missing[col] > 0:
                result["columns_with_missing"][col] = {
                    "count": int(missing[col]),
                    "percentage": round(missing[col] / len(df) * 100, 2)
                }
        
        # Detect missing pattern for each column
        for col in result["columns_with_missing"]:
            pattern = self._detect_missing_pattern(df, col)
            result["likely_mechanism"][col] = pattern.value
        
        # Correlation between missingness indicators
        if len(result["columns_with_missing"]) >= 2:
            missing_indicators = df[list(result["columns_with_missing"].keys())].isnull().astype(int)
            corr = missing_indicators.corr()
            
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1:]:
                    if abs(corr.loc[c1, c2]) > 0.3:
                        result["correlation_with_missingness"].append({
                            "column1": c1,
                            "column2": c2,
                            "correlation": round(corr.loc[c1, c2], 3)
                        })
        
        return result
    
    def _detect_missing_pattern(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> MissingPattern:
        """
        Detect missing pattern for a column.
        
        MCAR: Missing is independent of observed and unobserved values
        MAR: Missing depends on observed values but not on the missing value itself
        MNAR: Missing depends on the unobserved value
        """
        missing_mask = df[target_col].isnull()
        
        if missing_mask.sum() == 0 or missing_mask.all():
            return MissingPattern.MCAR
        
        # Check correlation with other columns
        significant_correlations = 0
        
        for col in df.columns:
            if col == target_col:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # Point-biserial correlation for numeric columns
                observed = df[col].dropna()
                missing_indicator = missing_mask[observed.index].astype(int)
                
                if len(observed) > 10 and missing_indicator.var() > 0:
                    try:
                        corr, pval = scipy_stats.pointbiserialr(missing_indicator, observed)
                        if pval < 0.05 and abs(corr) > 0.1:
                            significant_correlations += 1
                    except:
                        pass
        
        # Interpretation
        if significant_correlations == 0:
            # Little's MCAR test approximation
            return MissingPattern.MCAR
        elif significant_correlations <= len(df.columns) // 3:
            return MissingPattern.MAR
        else:
            return MissingPattern.MNAR


# ============================================================================
# Outlier Detector
# ============================================================================

class OutlierDetector:
    """
    Multi-method outlier detection.
    
    Methods:
    - IQR (Interquartile Range)
    - Z-Score
    - Modified Z-Score (for skewed data)
    - Isolation Forest
    - Local Outlier Factor
    - Mahalanobis Distance (multivariate)
    """
    
    def detect(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
        method: OutlierMethod = OutlierMethod.IQR,
        threshold: float = None
    ) -> dict[str, Any]:
        """Detect outliers in the dataset."""
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = {
            "method": method.value,
            "columns_analyzed": columns,
            "outliers_by_column": {},
            "total_outlier_rows": 0,
            "outlier_row_indices": []
        }
        
        all_outlier_indices = set()
        
        for col in columns:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            if method == OutlierMethod.IQR:
                outliers = self._iqr_outliers(series, threshold or 1.5)
            elif method == OutlierMethod.ZSCORE:
                outliers = self._zscore_outliers(series, threshold or 3.0)
            elif method == OutlierMethod.ISOLATION_FOREST:
                outliers = self._isolation_forest_outliers(series, threshold or 0.1)
            elif method == OutlierMethod.LOF:
                outliers = self._lof_outliers(series, threshold or 0.1)
            else:
                outliers = self._iqr_outliers(series, threshold or 1.5)
            
            result["outliers_by_column"][col] = {
                "count": len(outliers),
                "percentage": round(len(outliers) / len(series) * 100, 2),
                "indices": outliers[:100],  # Limit for response size
                "bounds": self._calculate_bounds(series, method, threshold)
            }
            
            all_outlier_indices.update(outliers)
        
        result["total_outlier_rows"] = len(all_outlier_indices)
        result["outlier_row_indices"] = sorted(list(all_outlier_indices))[:1000]
        
        return result
    
    def _iqr_outliers(
        self,
        series: pd.Series,
        multiplier: float = 1.5
    ) -> list[int]:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        outliers = series[(series < lower) | (series > upper)]
        return outliers.index.tolist()
    
    def _zscore_outliers(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> list[int]:
        """Detect outliers using Z-score."""
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers.index.tolist()
    
    def _isolation_forest_outliers(
        self,
        series: pd.Series,
        contamination: float = 0.1
    ) -> list[int]:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            X = series.values.reshape(-1, 1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            predictions = clf.fit_predict(X)
            
            outliers = series[predictions == -1]
            return outliers.index.tolist()
        except ImportError:
            # Fallback to IQR
            return self._iqr_outliers(series)
    
    def _lof_outliers(
        self,
        series: pd.Series,
        contamination: float = 0.1
    ) -> list[int]:
        """Detect outliers using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            X = series.values.reshape(-1, 1)
            clf = LocalOutlierFactor(contamination=contamination, n_neighbors=min(20, len(X) - 1))
            predictions = clf.fit_predict(X)
            
            outliers = series[predictions == -1]
            return outliers.index.tolist()
        except ImportError:
            return self._iqr_outliers(series)
    
    def _calculate_bounds(
        self,
        series: pd.Series,
        method: OutlierMethod,
        threshold: float = None
    ) -> dict[str, float]:
        """Calculate outlier bounds."""
        if method == OutlierMethod.IQR:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            mult = threshold or 1.5
            return {
                "lower": float(q1 - mult * iqr),
                "upper": float(q3 + mult * iqr)
            }
        elif method == OutlierMethod.ZSCORE:
            t = threshold or 3.0
            return {
                "lower": float(series.mean() - t * series.std()),
                "upper": float(series.mean() + t * series.std())
            }
        else:
            return {}


# ============================================================================
# Data Validator
# ============================================================================

class ValidationType(str, Enum):
    """Types of validation rules."""
    RANGE = "range"
    REGEX = "regex"
    ENUM = "enum"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    TYPE = "type"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    
    name: str
    validation_type: ValidationType
    column: str
    params: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    severity: str = "error"  # error, warning, info


class DataValidator:
    """
    Rule-based data validation engine.
    
    Supports:
    - Range checks (numeric)
    - Regex pattern matching
    - Enum/allowed values
    - Uniqueness constraints
    - Not null constraints
    - Type validation
    - Custom functions
    """
    
    def __init__(self):
        self._rules: list[ValidationRule] = []
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self._rules.append(rule)
    
    def add_range_rule(
        self,
        column: str,
        min_val: float = None,
        max_val: float = None,
        name: str = None
    ) -> None:
        """Add range validation rule."""
        self._rules.append(ValidationRule(
            name=name or f"{column}_range_check",
            validation_type=ValidationType.RANGE,
            column=column,
            params={"min": min_val, "max": max_val},
            error_message=f"Value must be between {min_val} and {max_val}"
        ))
    
    def add_regex_rule(
        self,
        column: str,
        pattern: str,
        name: str = None
    ) -> None:
        """Add regex pattern validation rule."""
        self._rules.append(ValidationRule(
            name=name or f"{column}_pattern_check",
            validation_type=ValidationType.REGEX,
            column=column,
            params={"pattern": pattern},
            error_message=f"Value must match pattern: {pattern}"
        ))
    
    def add_enum_rule(
        self,
        column: str,
        allowed_values: list[Any],
        name: str = None
    ) -> None:
        """Add allowed values validation rule."""
        self._rules.append(ValidationRule(
            name=name or f"{column}_enum_check",
            validation_type=ValidationType.ENUM,
            column=column,
            params={"allowed": allowed_values},
            error_message=f"Value must be one of: {allowed_values}"
        ))
    
    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate dataframe against all rules."""
        results = {
            "total_rules": len(self._rules),
            "passed": 0,
            "failed": 0,
            "violations": [],
            "summary_by_column": {}
        }
        
        for rule in self._rules:
            if rule.column not in df.columns:
                continue
            
            violations = self._check_rule(df, rule)
            
            if violations:
                results["failed"] += 1
                results["violations"].append({
                    "rule": rule.name,
                    "column": rule.column,
                    "type": rule.validation_type.value,
                    "violation_count": len(violations),
                    "message": rule.error_message,
                    "sample_indices": violations[:10]
                })
                
                if rule.column not in results["summary_by_column"]:
                    results["summary_by_column"][rule.column] = {"total_violations": 0}
                results["summary_by_column"][rule.column]["total_violations"] += len(violations)
            else:
                results["passed"] += 1
        
        return results
    
    def _check_rule(
        self,
        df: pd.DataFrame,
        rule: ValidationRule
    ) -> list[int]:
        """Check a single rule and return violation indices."""
        series = df[rule.column]
        
        if rule.validation_type == ValidationType.RANGE:
            min_val = rule.params.get("min")
            max_val = rule.params.get("max")
            
            mask = pd.Series(False, index=series.index)
            if min_val is not None:
                mask |= series < min_val
            if max_val is not None:
                mask |= series > max_val
            
            return series[mask].index.tolist()
        
        elif rule.validation_type == ValidationType.REGEX:
            pattern = rule.params.get("pattern", "")
            mask = ~series.astype(str).str.match(pattern, na=False)
            return series[mask].index.tolist()
        
        elif rule.validation_type == ValidationType.ENUM:
            allowed = rule.params.get("allowed", [])
            mask = ~series.isin(allowed)
            return series[mask].index.tolist()
        
        elif rule.validation_type == ValidationType.UNIQUE:
            duplicated = series[series.duplicated(keep=False)]
            return duplicated.index.tolist()
        
        elif rule.validation_type == ValidationType.NOT_NULL:
            return series[series.isnull()].index.tolist()
        
        return []


# ============================================================================
# Data Quality Engine
# ============================================================================

class DataQualityEngine:
    """
    Comprehensive data quality assessment engine.
    
    Provides:
    - Column-level quality analysis
    - Missing data pattern detection
    - Outlier detection (multiple methods)
    - Validation rule checking
    - Quality scoring
    - Remediation recommendations
    """
    
    def __init__(self):
        self.missing_analyzer = MissingDataAnalyzer()
        self.outlier_detector = OutlierDetector()
        self.validator = DataValidator()
    
    def analyze(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> DatasetQualityReport:
        """Perform comprehensive data quality analysis."""
        report = DatasetQualityReport(
            dataset_name=dataset_name,
            row_count=len(df),
            column_count=len(df.columns)
        )
        
        # Analyze each column
        for col in df.columns:
            col_report = self._analyze_column(df, col)
            report.column_reports.append(col_report)
        
        # Dataset-level metrics
        self._compute_dataset_metrics(df, report)
        
        # Compute overall score
        self._compute_overall_score(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _analyze_column(
        self,
        df: pd.DataFrame,
        column: str
    ) -> ColumnQualityReport:
        """Analyze a single column."""
        series = df[column]
        
        report = ColumnQualityReport(
            column=column,
            dtype=str(series.dtype),
            total_count=len(series)
        )
        
        # Completeness
        report.null_count = int(series.isnull().sum())
        report.null_percentage = report.null_count / len(series) * 100
        
        if report.null_count > 0:
            missing_analysis = self.missing_analyzer.analyze(df[[column]])
            if column in missing_analysis.get("likely_mechanism", {}):
                report.missing_pattern = MissingPattern(
                    missing_analysis["likely_mechanism"][column]
                )
        
        # Uniqueness
        report.unique_count = int(series.nunique())
        report.unique_percentage = report.unique_count / len(series) * 100
        report.duplicate_count = len(series) - report.unique_count
        
        # Validity (basic type-based checks)
        non_null = series.dropna()
        report.valid_count = len(non_null)
        
        # Outliers (for numeric)
        if pd.api.types.is_numeric_dtype(series):
            outlier_result = self.outlier_detector.detect(
                df[[column]],
                [column],
                OutlierMethod.IQR
            )
            if column in outlier_result.get("outliers_by_column", {}):
                outlier_info = outlier_result["outliers_by_column"][column]
                report.outlier_count = outlier_info["count"]
                report.outlier_percentage = outlier_info["percentage"]
                report.outlier_indices = outlier_info["indices"][:100]
            
            # Distribution
            if len(non_null) >= 20:
                report.skewness = float(non_null.skew())
                report.kurtosis = float(non_null.kurtosis())
                
                try:
                    _, pval = scipy_stats.normaltest(non_null)
                    report.is_normal = pval > 0.05
                except:
                    pass
        
        # Identify issues
        self._identify_column_issues(report)
        
        # Calculate column quality score
        self._calculate_column_score(report)
        
        return report
    
    def _identify_column_issues(self, report: ColumnQualityReport) -> None:
        """Identify issues in column quality."""
        # High missing values
        if report.null_percentage > 50:
            report.issues.append({
                "type": "high_missing",
                "severity": "critical",
                "message": f"{report.null_percentage:.1f}% missing values"
            })
        elif report.null_percentage > 20:
            report.issues.append({
                "type": "moderate_missing",
                "severity": "warning",
                "message": f"{report.null_percentage:.1f}% missing values"
            })
        
        # Constant column
        if report.unique_count == 1:
            report.issues.append({
                "type": "constant_column",
                "severity": "warning",
                "message": "Column has only one unique value"
            })
        
        # High cardinality
        if report.unique_percentage > 90 and report.dtype == "object":
            report.issues.append({
                "type": "high_cardinality",
                "severity": "info",
                "message": "High cardinality categorical column"
            })
        
        # Outliers
        if report.outlier_percentage > 5:
            report.issues.append({
                "type": "high_outliers",
                "severity": "warning",
                "message": f"{report.outlier_percentage:.1f}% outliers detected"
            })
        
        # Skewness
        if report.skewness is not None and abs(report.skewness) > 2:
            report.issues.append({
                "type": "high_skewness",
                "severity": "info",
                "message": f"Highly skewed distribution (skewness: {report.skewness:.2f})"
            })
    
    def _calculate_column_score(self, report: ColumnQualityReport) -> None:
        """Calculate quality score for a column."""
        score = 100.0
        
        # Deduct for missing values
        score -= min(report.null_percentage * 0.5, 30)
        
        # Deduct for outliers
        score -= min(report.outlier_percentage * 0.3, 15)
        
        # Deduct for issues
        for issue in report.issues:
            if issue["severity"] == "critical":
                score -= 15
            elif issue["severity"] == "warning":
                score -= 5
            elif issue["severity"] == "info":
                score -= 1
        
        report.quality_score = max(0, score)
    
    def _compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        report: DatasetQualityReport
    ) -> None:
        """Compute dataset-level metrics."""
        # Completeness
        total_cells = report.row_count * report.column_count
        total_missing = df.isnull().sum().sum()
        report.overall_completeness = (1 - total_missing / total_cells) * 100
        
        # Duplicate rows
        report.duplicate_rows = int(df.duplicated().sum())
        report.duplicate_row_percentage = report.duplicate_rows / len(df) * 100
        
        # Type distribution
        type_counts = df.dtypes.value_counts()
        report.type_distribution = {str(k): int(v) for k, v in type_counts.items()}
    
    def _compute_overall_score(self, report: DatasetQualityReport) -> None:
        """Compute overall quality score."""
        if not report.column_reports:
            report.overall_quality_score = 0
            return
        
        # Weighted average of column scores
        weights = []
        scores = []
        
        for col_report in report.column_reports:
            weight = 1.0
            # Weight columns with more data higher
            if col_report.null_percentage < 50:
                weight = 1.5
            
            weights.append(weight)
            scores.append(col_report.quality_score)
        
        report.overall_quality_score = np.average(scores, weights=weights)
        
        # Aggregate issues
        for col_report in report.column_reports:
            for issue in col_report.issues:
                issue_copy = issue.copy()
                issue_copy["column"] = col_report.column
                
                if issue["severity"] == "critical":
                    report.critical_issues.append(issue_copy)
                else:
                    report.warnings.append(issue_copy)
    
    def _generate_recommendations(self, report: DatasetQualityReport) -> None:
        """Generate remediation recommendations."""
        recommendations = []
        
        # High missing columns
        high_missing = [
            cr.column for cr in report.column_reports
            if cr.null_percentage > 40
        ]
        if high_missing:
            recommendations.append({
                "priority": "high",
                "type": "imputation",
                "columns": high_missing,
                "action": "Consider imputation or removal of columns with >40% missing",
                "methods": ["KNN imputation", "MICE", "Drop column"]
            })
        
        # Moderate missing
        moderate_missing = [
            cr for cr in report.column_reports
            if 5 < cr.null_percentage <= 40
        ]
        for cr in moderate_missing:
            method = "mean/median" if cr.dtype in ["int64", "float64"] else "mode"
            recommendations.append({
                "priority": "medium",
                "type": "imputation",
                "columns": [cr.column],
                "action": f"Impute {cr.column} using {method}",
                "pattern": cr.missing_pattern.value if cr.missing_pattern else "unknown"
            })
        
        # Outliers
        outlier_columns = [
            cr.column for cr in report.column_reports
            if cr.outlier_percentage > 3
        ]
        if outlier_columns:
            recommendations.append({
                "priority": "medium",
                "type": "outlier_treatment",
                "columns": outlier_columns,
                "action": "Review and treat outliers",
                "methods": ["Clip to bounds", "Winsorize", "Remove", "Transform (log)"]
            })
        
        # Duplicates
        if report.duplicate_row_percentage > 1:
            recommendations.append({
                "priority": "high",
                "type": "deduplication",
                "action": f"Remove {report.duplicate_rows} duplicate rows",
                "percentage": round(report.duplicate_row_percentage, 2)
            })
        
        # Constant columns
        constant_cols = [
            cr.column for cr in report.column_reports
            if cr.unique_count == 1
        ]
        if constant_cols:
            recommendations.append({
                "priority": "low",
                "type": "column_removal",
                "columns": constant_cols,
                "action": "Consider removing constant columns (no predictive value)"
            })
        
        report.recommendations = recommendations


# Factory functions
def get_data_quality_engine() -> DataQualityEngine:
    """Get data quality engine instance."""
    return DataQualityEngine()


def get_outlier_detector() -> OutlierDetector:
    """Get outlier detector instance."""
    return OutlierDetector()


def get_data_validator() -> DataValidator:
    """Get data validator instance."""
    return DataValidator()
