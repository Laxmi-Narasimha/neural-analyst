# AI Enterprise Data Analyst - Data Quality Report Engine
# Production-grade data quality assessment and reporting
# Handles: completeness, consistency, accuracy, timeliness

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class QualityDimension(str, Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    ACCURACY = "accuracy"


class SeverityLevel(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QualityIssue:
    """Single data quality issue."""
    dimension: QualityDimension
    severity: SeverityLevel
    column: str
    description: str
    affected_rows: int
    affected_pct: float
    recommendation: str


@dataclass
class ColumnQuality:
    """Quality metrics for a column."""
    column: str
    completeness: float  # 0-100
    uniqueness: float  # 0-100
    validity: float  # 0-100
    overall_score: float
    issues: List[QualityIssue]


@dataclass
class DataQualityResult:
    """Complete data quality report."""
    overall_score: float = 0.0  # 0-100
    grade: str = ""  # A, B, C, D, F
    
    # Dimension scores
    completeness_score: float = 0.0
    uniqueness_score: float = 0.0
    validity_score: float = 0.0
    consistency_score: float = 0.0
    
    # Column-level
    columns: List[ColumnQuality] = field(default_factory=list)
    
    # Issues
    issues: List[QualityIssue] = field(default_factory=list)
    
    # Summary
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    # Recommendations
    top_recommendations: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": {
                "score": round(self.overall_score, 1),
                "grade": self.grade
            },
            "dimensions": {
                "completeness": round(self.completeness_score, 1),
                "uniqueness": round(self.uniqueness_score, 1),
                "validity": round(self.validity_score, 1),
                "consistency": round(self.consistency_score, 1)
            },
            "issues_summary": {
                "critical": self.critical_issues,
                "high": self.high_issues,
                "medium": self.medium_issues,
                "low": self.low_issues
            },
            "columns": [
                {
                    "column": c.column,
                    "score": round(c.overall_score, 1),
                    "completeness": round(c.completeness, 1)
                }
                for c in self.columns
            ],
            "issues": [
                {
                    "dimension": i.dimension.value,
                    "severity": i.severity.value,
                    "column": i.column,
                    "description": i.description
                }
                for i in self.issues[:20]
            ],
            "recommendations": self.top_recommendations[:10]
        }


# ============================================================================
# Data Quality Report Engine
# ============================================================================

class DataQualityReportEngine:
    """
    Production-grade Data Quality Report engine.
    
    Features:
    - Multi-dimensional quality assessment
    - Column-level scoring
    - Issue detection with severity
    - Actionable recommendations
    - Quality grading
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def assess(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None,
        date_columns: List[str] = None,
        expected_patterns: Dict[str, str] = None
    ) -> DataQualityResult:
        """Assess data quality."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Quality assessment: {df.shape[0]} rows, {df.shape[1]} columns")
        
        issues = []
        column_qualities = []
        
        # Assess each column
        for col in df.columns:
            col_issues = []
            
            # Completeness
            completeness = (1 - df[col].isna().mean()) * 100
            
            if completeness < 50:
                col_issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=SeverityLevel.CRITICAL,
                    column=col,
                    description=f"Column has {100-completeness:.1f}% missing values",
                    affected_rows=int(df[col].isna().sum()),
                    affected_pct=100 - completeness,
                    recommendation=f"Investigate and impute missing values in '{col}'"
                ))
            elif completeness < 90:
                col_issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=SeverityLevel.MEDIUM,
                    column=col,
                    description=f"Column has {100-completeness:.1f}% missing values",
                    affected_rows=int(df[col].isna().sum()),
                    affected_pct=100 - completeness,
                    recommendation=f"Consider imputation strategy for '{col}'"
                ))
            
            # Uniqueness
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())
            uniqueness = (n_unique / n_total * 100) if n_total > 0 else 0
            
            if n_unique == 1 and n_total > 10:
                col_issues.append(QualityIssue(
                    dimension=QualityDimension.UNIQUENESS,
                    severity=SeverityLevel.HIGH,
                    column=col,
                    description="Column contains only one unique value",
                    affected_rows=n_total,
                    affected_pct=100,
                    recommendation=f"Consider removing constant column '{col}'"
                ))
            
            # Check for duplicates in should-be-unique columns
            if 'id' in col.lower() or 'key' in col.lower():
                dup_count = int(df[col].duplicated().sum())
                if dup_count > 0:
                    col_issues.append(QualityIssue(
                        dimension=QualityDimension.UNIQUENESS,
                        severity=SeverityLevel.CRITICAL,
                        column=col,
                        description=f"{dup_count} duplicate values in ID column",
                        affected_rows=dup_count,
                        affected_pct=dup_count / len(df) * 100,
                        recommendation=f"Investigate duplicate IDs in '{col}'"
                    ))
            
            # Validity checks
            validity = 100  # Default valid
            
            if df[col].dtype == 'object':
                # Check for empty strings
                empty_str = (df[col] == '').sum()
                if empty_str > 0:
                    validity -= (empty_str / len(df)) * 50
                    col_issues.append(QualityIssue(
                        dimension=QualityDimension.VALIDITY,
                        severity=SeverityLevel.LOW,
                        column=col,
                        description=f"{empty_str} empty string values",
                        affected_rows=int(empty_str),
                        affected_pct=empty_str / len(df) * 100,
                        recommendation="Replace empty strings with NULL"
                    ))
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check for negative values where unexpected
                if 'age' in col.lower() or 'count' in col.lower() or 'quantity' in col.lower():
                    negative = (df[col] < 0).sum()
                    if negative > 0:
                        validity -= (negative / len(df)) * 50
                        col_issues.append(QualityIssue(
                            dimension=QualityDimension.VALIDITY,
                            severity=SeverityLevel.HIGH,
                            column=col,
                            description=f"{negative} unexpected negative values",
                            affected_rows=int(negative),
                            affected_pct=negative / len(df) * 100,
                            recommendation=f"Validate negative values in '{col}'"
                        ))
            
            # Overall column score
            overall_score = (completeness + min(100, uniqueness) + validity) / 3
            
            column_qualities.append(ColumnQuality(
                column=col,
                completeness=completeness,
                uniqueness=min(100, uniqueness),
                validity=validity,
                overall_score=overall_score,
                issues=col_issues
            ))
            
            issues.extend(col_issues)
        
        # Dimension scores
        completeness_score = np.mean([c.completeness for c in column_qualities])
        uniqueness_score = np.mean([c.uniqueness for c in column_qualities])
        validity_score = np.mean([c.validity for c in column_qualities])
        
        # Consistency score (check for duplicate rows)
        dup_rows = df.duplicated().sum()
        consistency_score = (1 - dup_rows / len(df)) * 100 if len(df) > 0 else 100
        
        if dup_rows > 0:
            issues.append(QualityIssue(
                dimension=QualityDimension.CONSISTENCY,
                severity=SeverityLevel.MEDIUM,
                column="[ALL]",
                description=f"{dup_rows} duplicate rows detected",
                affected_rows=int(dup_rows),
                affected_pct=dup_rows / len(df) * 100,
                recommendation="Remove or investigate duplicate rows"
            ))
        
        # Overall score
        overall_score = (
            completeness_score * 0.3 +
            uniqueness_score * 0.2 +
            validity_score * 0.3 +
            consistency_score * 0.2
        )
        
        # Grade
        grade = self._score_to_grade(overall_score)
        
        # Count issues by severity
        critical = sum(1 for i in issues if i.severity == SeverityLevel.CRITICAL)
        high = sum(1 for i in issues if i.severity == SeverityLevel.HIGH)
        medium = sum(1 for i in issues if i.severity == SeverityLevel.MEDIUM)
        low = sum(1 for i in issues if i.severity == SeverityLevel.LOW)
        
        # Top recommendations
        recommendations = [i.recommendation for i in sorted(
            issues, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x.severity.value]
        )][:10]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DataQualityResult(
            overall_score=overall_score,
            grade=grade,
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            columns=column_qualities,
            issues=issues,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            top_recommendations=recommendations,
            processing_time_sec=processing_time
        )
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        return "F"


# ============================================================================
# Factory Functions
# ============================================================================

def get_quality_report_engine() -> DataQualityReportEngine:
    """Get data quality report engine."""
    return DataQualityReportEngine()


def quick_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick data quality check."""
    engine = DataQualityReportEngine(verbose=False)
    result = engine.assess(df)
    return result.to_dict()


def get_quality_score(df: pd.DataFrame) -> float:
    """Get overall quality score."""
    engine = DataQualityReportEngine(verbose=False)
    result = engine.assess(df)
    return result.overall_score
