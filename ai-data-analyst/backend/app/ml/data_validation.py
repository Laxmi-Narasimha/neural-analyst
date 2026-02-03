# AI Enterprise Data Analyst - Data Validation Engine
# Production-grade data validation with schema enforcement
# Handles: type validation, range checks, pattern matching, referential integrity

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Optional exception import - define locally if not available
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception):
        pass

warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class ValidationLevel(str, Enum):
    """Severity level of validation issues."""
    ERROR = "error"  # Must fix
    WARNING = "warning"  # Should fix
    INFO = "info"  # Nice to know


class ValidationType(str, Enum):
    """Type of validation."""
    TYPE_CHECK = "type_check"
    NULL_CHECK = "null_check"
    RANGE_CHECK = "range_check"
    PATTERN_CHECK = "pattern_check"
    UNIQUE_CHECK = "unique_check"
    REFERENTIAL_CHECK = "referential_check"
    CUSTOM_CHECK = "custom_check"
    FORMAT_CHECK = "format_check"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    column: str
    validation_type: ValidationType
    level: ValidationLevel = ValidationLevel.ERROR
    params: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class ValidationIssue:
    """Single validation issue."""
    rule_name: str
    column: str
    validation_type: ValidationType
    level: ValidationLevel
    message: str
    affected_rows: int
    affected_indices: List[int] = field(default_factory=list)
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class ColumnValidation:
    """Validation status for a column."""
    column: str
    is_valid: bool
    error_count: int
    warning_count: int
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    total_rows: int
    valid_rows: int
    invalid_rows: int
    
    # Summary counts
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    # Column-level results
    column_results: Dict[str, ColumnValidation] = field(default_factory=dict)
    
    # All issues
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Data quality score (0-100)
    quality_score: float = 100.0
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "quality_score": round(self.quality_score, 1),
            "summary": {
                "total_rows": self.total_rows,
                "valid_rows": self.valid_rows,
                "invalid_rows": self.invalid_rows,
                "error_count": self.error_count,
                "warning_count": self.warning_count
            },
            "column_summary": {
                col: {
                    "is_valid": cv.is_valid,
                    "error_count": cv.error_count,
                    "warning_count": cv.warning_count
                }
                for col, cv in self.column_results.items()
            },
            "issues": [
                {
                    "rule": i.rule_name,
                    "column": i.column,
                    "type": i.validation_type.value,
                    "level": i.level.value,
                    "message": i.message,
                    "affected_rows": i.affected_rows,
                    "sample_values": i.sample_values[:5]
                }
                for i in self.issues[:50]
            ]
        }


# ============================================================================
# Validation Engine
# ============================================================================

class DataValidationEngine:
    """
    Production-grade Data Validation engine.
    
    Features:
    - Type validation
    - Null/missing checks
    - Range validation
    - Pattern matching (regex)
    - Uniqueness constraints
    - Custom validation functions
    - Multi-level severity
    """
    
    # Common regex patterns
    PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "phone": r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$',
        "url": r'^https?://[^\s/$.?#].[^\s]*$',
        "date_iso": r'^\d{4}-\d{2}-\d{2}$',
        "date_us": r'^\d{2}/\d{2}/\d{4}$',
        "zipcode_us": r'^\d{5}(-\d{4})?$',
        "ssn": r'^\d{3}-\d{2}-\d{4}$',
        "credit_card": r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
        "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        "ip_address": r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        "alphanumeric": r'^[a-zA-Z0-9]+$',
        "alpha_only": r'^[a-zA-Z]+$',
        "numeric_only": r'^\d+$'
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.rules: List[ValidationRule] = []
        self._compiled_patterns: Dict[str, Pattern] = {}
    
    # ========================================================================
    # Rule Definition Methods
    # ========================================================================
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)
        return self
    
    def require_not_null(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add not-null constraint."""
        self.rules.append(ValidationRule(
            name=f"{column}_not_null",
            column=column,
            validation_type=ValidationType.NULL_CHECK,
            level=level,
            message=f"Column '{column}' contains null values"
        ))
        return self
    
    def require_type(
        self,
        column: str,
        expected_type: str,  # 'numeric', 'string', 'datetime', 'boolean'
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add type constraint."""
        self.rules.append(ValidationRule(
            name=f"{column}_type_{expected_type}",
            column=column,
            validation_type=ValidationType.TYPE_CHECK,
            level=level,
            params={"expected_type": expected_type},
            message=f"Column '{column}' should be {expected_type}"
        ))
        return self
    
    def require_range(
        self,
        column: str,
        min_val: float = None,
        max_val: float = None,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add range constraint."""
        self.rules.append(ValidationRule(
            name=f"{column}_range",
            column=column,
            validation_type=ValidationType.RANGE_CHECK,
            level=level,
            params={"min": min_val, "max": max_val},
            message=f"Column '{column}' values out of range [{min_val}, {max_val}]"
        ))
        return self
    
    def require_pattern(
        self,
        column: str,
        pattern: str,
        pattern_name: str = None,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add regex pattern constraint."""
        # Check if it's a named pattern
        if pattern in self.PATTERNS:
            actual_pattern = self.PATTERNS[pattern]
            pattern_name = pattern_name or pattern
        else:
            actual_pattern = pattern
            pattern_name = pattern_name or "custom"
        
        self.rules.append(ValidationRule(
            name=f"{column}_pattern_{pattern_name}",
            column=column,
            validation_type=ValidationType.PATTERN_CHECK,
            level=level,
            params={"pattern": actual_pattern, "pattern_name": pattern_name},
            message=f"Column '{column}' contains values not matching {pattern_name} pattern"
        ))
        return self
    
    def require_unique(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add uniqueness constraint."""
        self.rules.append(ValidationRule(
            name=f"{column}_unique",
            column=column,
            validation_type=ValidationType.UNIQUE_CHECK,
            level=level,
            message=f"Column '{column}' contains duplicate values"
        ))
        return self
    
    def require_in_set(
        self,
        column: str,
        valid_values: Set[Any],
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add valid values constraint."""
        self.rules.append(ValidationRule(
            name=f"{column}_in_set",
            column=column,
            validation_type=ValidationType.CUSTOM_CHECK,
            level=level,
            params={"valid_values": valid_values, "check_type": "in_set"},
            message=f"Column '{column}' contains invalid values"
        ))
        return self
    
    def require_positive(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add positive values constraint."""
        return self.require_range(column, min_val=0, level=level)
    
    def require_non_negative(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add non-negative constraint."""
        return self.require_range(column, min_val=0, level=level)
    
    def add_custom_rule(
        self,
        column: str,
        validator: Callable[[pd.Series], pd.Series],
        name: str,
        message: str,
        level: ValidationLevel = ValidationLevel.ERROR
    ):
        """Add custom validation function."""
        self.rules.append(ValidationRule(
            name=name,
            column=column,
            validation_type=ValidationType.CUSTOM_CHECK,
            level=level,
            params={"validator": validator, "check_type": "function"},
            message=message
        ))
        return self
    
    # ========================================================================
    # Validation Methods
    # ========================================================================
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame against all rules."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Validating {len(df)} rows against {len(self.rules)} rules")
        
        all_issues: List[ValidationIssue] = []
        column_results: Dict[str, ColumnValidation] = {}
        invalid_row_mask = pd.Series([False] * len(df), index=df.index)
        
        for rule in self.rules:
            if rule.column not in df.columns:
                all_issues.append(ValidationIssue(
                    rule_name=rule.name,
                    column=rule.column,
                    validation_type=rule.validation_type,
                    level=ValidationLevel.ERROR,
                    message=f"Column '{rule.column}' not found in DataFrame",
                    affected_rows=0
                ))
                continue
            
            issue = self._apply_rule(df, rule)
            
            if issue:
                all_issues.append(issue)
                
                # Track invalid rows
                if issue.level == ValidationLevel.ERROR:
                    for idx in issue.affected_indices:
                        if idx in invalid_row_mask.index:
                            invalid_row_mask[idx] = True
        
        # Aggregate column results
        for col in df.columns:
            col_issues = [i for i in all_issues if i.column == col]
            error_count = sum(1 for i in col_issues if i.level == ValidationLevel.ERROR)
            warning_count = sum(1 for i in col_issues if i.level == ValidationLevel.WARNING)
            
            column_results[col] = ColumnValidation(
                column=col,
                is_valid=error_count == 0,
                error_count=error_count,
                warning_count=warning_count,
                issues=col_issues
            )
        
        # Summary
        error_count = sum(1 for i in all_issues if i.level == ValidationLevel.ERROR)
        warning_count = sum(1 for i in all_issues if i.level == ValidationLevel.WARNING)
        info_count = sum(1 for i in all_issues if i.level == ValidationLevel.INFO)
        
        invalid_rows = int(invalid_row_mask.sum())
        valid_rows = len(df) - invalid_rows
        
        # Quality score
        if len(df) > 0:
            quality_score = (valid_rows / len(df)) * 100
            # Deduct for warnings
            quality_score -= warning_count * 0.5
            quality_score = max(0, min(100, quality_score))
        else:
            quality_score = 100.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            is_valid=error_count == 0,
            total_rows=len(df),
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            column_results=column_results,
            issues=all_issues,
            quality_score=quality_score,
            processing_time_sec=processing_time
        )
    
    def _apply_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Apply a single validation rule."""
        series = df[rule.column]
        
        try:
            if rule.validation_type == ValidationType.NULL_CHECK:
                return self._check_nulls(series, rule)
            
            elif rule.validation_type == ValidationType.TYPE_CHECK:
                return self._check_type(series, rule)
            
            elif rule.validation_type == ValidationType.RANGE_CHECK:
                return self._check_range(series, rule)
            
            elif rule.validation_type == ValidationType.PATTERN_CHECK:
                return self._check_pattern(series, rule)
            
            elif rule.validation_type == ValidationType.UNIQUE_CHECK:
                return self._check_unique(series, rule)
            
            elif rule.validation_type == ValidationType.CUSTOM_CHECK:
                return self._check_custom(series, rule)
            
        except Exception as e:
            logger.warning(f"Error applying rule {rule.name}: {e}")
            return None
        
        return None
    
    def _check_nulls(self, series: pd.Series, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check for null values."""
        null_mask = series.isna()
        null_count = int(null_mask.sum())
        
        if null_count == 0:
            return None
        
        return ValidationIssue(
            rule_name=rule.name,
            column=rule.column,
            validation_type=rule.validation_type,
            level=rule.level,
            message=rule.message,
            affected_rows=null_count,
            affected_indices=series[null_mask].index.tolist()[:100]
        )
    
    def _check_type(self, series: pd.Series, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check data type."""
        expected = rule.params.get("expected_type", "string")
        
        if expected == "numeric":
            converted = pd.to_numeric(series, errors='coerce')
            invalid_mask = series.notna() & converted.isna()
        elif expected == "datetime":
            converted = pd.to_datetime(series, errors='coerce')
            invalid_mask = series.notna() & converted.isna()
        elif expected == "boolean":
            valid_bool = series.isin([True, False, 0, 1, '0', '1', 'true', 'false', 'True', 'False'])
            invalid_mask = series.notna() & ~valid_bool
        else:  # string
            invalid_mask = pd.Series([False] * len(series), index=series.index)
        
        invalid_count = int(invalid_mask.sum())
        
        if invalid_count == 0:
            return None
        
        return ValidationIssue(
            rule_name=rule.name,
            column=rule.column,
            validation_type=rule.validation_type,
            level=rule.level,
            message=rule.message,
            affected_rows=invalid_count,
            affected_indices=series[invalid_mask].index.tolist()[:100],
            sample_values=series[invalid_mask].head(5).tolist()
        )
    
    def _check_range(self, series: pd.Series, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check value range."""
        min_val = rule.params.get("min")
        max_val = rule.params.get("max")
        
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        out_of_range = pd.Series([False] * len(series), index=series.index)
        
        if min_val is not None:
            out_of_range = out_of_range | (numeric_series < min_val)
        if max_val is not None:
            out_of_range = out_of_range | (numeric_series > max_val)
        
        # Don't count NaN as out of range
        out_of_range = out_of_range & numeric_series.notna()
        
        invalid_count = int(out_of_range.sum())
        
        if invalid_count == 0:
            return None
        
        return ValidationIssue(
            rule_name=rule.name,
            column=rule.column,
            validation_type=rule.validation_type,
            level=rule.level,
            message=rule.message,
            affected_rows=invalid_count,
            affected_indices=series[out_of_range].index.tolist()[:100],
            sample_values=series[out_of_range].head(5).tolist()
        )
    
    def _check_pattern(self, series: pd.Series, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check regex pattern."""
        pattern = rule.params.get("pattern", ".*")
        
        # Compile and cache pattern
        if pattern not in self._compiled_patterns:
            self._compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
        
        compiled = self._compiled_patterns[pattern]
        
        # Check non-null string values
        str_series = series.dropna().astype(str)
        matches = str_series.apply(lambda x: bool(compiled.match(str(x))))
        
        invalid_mask = ~matches
        invalid_count = int(invalid_mask.sum())
        
        if invalid_count == 0:
            return None
        
        invalid_indices = str_series[invalid_mask].index.tolist()[:100]
        sample_values = str_series[invalid_mask].head(5).tolist()
        
        return ValidationIssue(
            rule_name=rule.name,
            column=rule.column,
            validation_type=rule.validation_type,
            level=rule.level,
            message=rule.message,
            affected_rows=invalid_count,
            affected_indices=invalid_indices,
            sample_values=sample_values
        )
    
    def _check_unique(self, series: pd.Series, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check uniqueness."""
        duplicates = series[series.duplicated(keep=False)]
        dup_count = int(len(duplicates))
        
        if dup_count == 0:
            return None
        
        return ValidationIssue(
            rule_name=rule.name,
            column=rule.column,
            validation_type=rule.validation_type,
            level=rule.level,
            message=rule.message,
            affected_rows=dup_count,
            affected_indices=duplicates.index.tolist()[:100],
            sample_values=duplicates.value_counts().head(5).index.tolist()
        )
    
    def _check_custom(self, series: pd.Series, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Apply custom validation."""
        check_type = rule.params.get("check_type")
        
        if check_type == "in_set":
            valid_values = rule.params.get("valid_values", set())
            invalid_mask = ~series.isin(valid_values) & series.notna()
        elif check_type == "function":
            validator = rule.params.get("validator")
            if validator:
                invalid_mask = ~validator(series)
            else:
                return None
        else:
            return None
        
        invalid_count = int(invalid_mask.sum())
        
        if invalid_count == 0:
            return None
        
        return ValidationIssue(
            rule_name=rule.name,
            column=rule.column,
            validation_type=rule.validation_type,
            level=rule.level,
            message=rule.message,
            affected_rows=invalid_count,
            affected_indices=series[invalid_mask].index.tolist()[:100],
            sample_values=series[invalid_mask].head(5).tolist()
        )
    
    def clear_rules(self):
        """Clear all rules."""
        self.rules = []
        return self


# ============================================================================
# Schema Validation
# ============================================================================

class SchemaValidator:
    """Validate DataFrame against expected schema."""
    
    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        """
        Initialize with schema.
        
        Schema format:
        {
            "column_name": {
                "type": "numeric|string|datetime|boolean",
                "nullable": True/False,
                "min": float,
                "max": float,
                "pattern": "regex or named pattern",
                "values": [list of valid values]
            }
        }
        """
        self.schema = schema
        self.engine = DataValidationEngine(verbose=False)
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame against schema."""
        self.engine.clear_rules()
        
        # Check for missing columns
        schema_cols = set(self.schema.keys())
        df_cols = set(df.columns)
        
        missing_cols = schema_cols - df_cols
        extra_cols = df_cols - schema_cols
        
        # Add rules from schema
        for col, constraints in self.schema.items():
            if col not in df.columns:
                continue
            
            if constraints.get("nullable") is False:
                self.engine.require_not_null(col)
            
            if "type" in constraints:
                self.engine.require_type(col, constraints["type"])
            
            if "min" in constraints or "max" in constraints:
                self.engine.require_range(
                    col,
                    min_val=constraints.get("min"),
                    max_val=constraints.get("max")
                )
            
            if "pattern" in constraints:
                self.engine.require_pattern(col, constraints["pattern"])
            
            if "values" in constraints:
                self.engine.require_in_set(col, set(constraints["values"]))
        
        result = self.engine.validate(df)
        
        # Add missing column issues
        for col in missing_cols:
            result.issues.insert(0, ValidationIssue(
                rule_name=f"{col}_exists",
                column=col,
                validation_type=ValidationType.TYPE_CHECK,
                level=ValidationLevel.ERROR,
                message=f"Required column '{col}' is missing",
                affected_rows=len(df)
            ))
            result.error_count += 1
            result.is_valid = False
        
        return result


# ============================================================================
# Factory Functions
# ============================================================================

def get_validation_engine() -> DataValidationEngine:
    """Get data validation engine."""
    return DataValidationEngine()


def validate_with_schema(
    df: pd.DataFrame,
    schema: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Quick schema validation."""
    validator = SchemaValidator(schema)
    result = validator.validate(df)
    return result.to_dict()


def quick_validate(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick validation with auto-detected rules."""
    engine = DataValidationEngine(verbose=False)
    
    # Auto-add null checks for all columns
    for col in df.columns:
        engine.require_not_null(col, level=ValidationLevel.WARNING)
    
    result = engine.validate(df)
    return result.to_dict()
