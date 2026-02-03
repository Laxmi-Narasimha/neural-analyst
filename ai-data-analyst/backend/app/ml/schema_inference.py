# AI Enterprise Data Analyst - Schema Inference Engine
# Production-grade automatic schema inference
# Handles: type detection, pattern recognition, schema validation

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern

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

class SemanticType(str, Enum):
    """Semantic types for columns."""
    # Basic types
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    
    # Date/Time
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    
    # Identifiers
    ID = "id"
    UUID = "uuid"
    
    # Personal
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    
    # Location
    COUNTRY = "country"
    STATE = "state"
    CITY = "city"
    ZIPCODE = "zipcode"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    
    # Financial
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    
    # Web
    URL = "url"
    IP_ADDRESS = "ip_address"
    
    # Other
    CATEGORICAL = "categorical"
    TEXT = "text"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ColumnInference:
    """Inferred column information."""
    column_name: str
    pandas_dtype: str
    semantic_type: SemanticType
    confidence: float
    
    # Constraints
    nullable: bool = True
    unique: bool = False
    
    # Statistics
    sample_values: List[Any] = field(default_factory=list)
    pattern: str = None  # Detected pattern for strings


@dataclass
class SchemaResult:
    """Complete schema inference result."""
    n_columns: int = 0
    n_rows: int = 0
    
    columns: List[ColumnInference] = field(default_factory=list)
    
    # Quality metrics
    type_coverage: float = 0.0  # % of columns with known types
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_columns": self.n_columns,
            "n_rows": self.n_rows,
            "type_coverage": round(self.type_coverage, 2),
            "columns": [
                {
                    "name": c.column_name,
                    "dtype": c.pandas_dtype,
                    "semantic_type": c.semantic_type.value,
                    "confidence": round(c.confidence, 2),
                    "nullable": c.nullable,
                    "unique": c.unique
                }
                for c in self.columns
            ]
        }


# ============================================================================
# Schema Inference Engine
# ============================================================================

class SchemaInferenceEngine:
    """
    Production-grade Schema Inference engine.
    
    Features:
    - Automatic type detection
    - Semantic type inference
    - Pattern recognition
    - Constraint detection
    """
    
    # Patterns for semantic type detection
    PATTERNS = {
        SemanticType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        SemanticType.URL: re.compile(r'^https?://[^\s]+$'),
        SemanticType.PHONE: re.compile(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$'),
        SemanticType.ZIPCODE: re.compile(r'^\d{5}(-\d{4})?$'),
        SemanticType.UUID: re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'),
        SemanticType.IP_ADDRESS: re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    }
    
    # Column name indicators
    NAME_INDICATORS = {
        SemanticType.EMAIL: ['email', 'mail'],
        SemanticType.PHONE: ['phone', 'mobile', 'cell', 'tel'],
        SemanticType.NAME: ['name', 'first_name', 'last_name', 'fullname'],
        SemanticType.ADDRESS: ['address', 'street', 'addr'],
        SemanticType.CITY: ['city', 'town'],
        SemanticType.STATE: ['state', 'province', 'region'],
        SemanticType.COUNTRY: ['country', 'nation'],
        SemanticType.ZIPCODE: ['zip', 'zipcode', 'postal', 'postcode'],
        SemanticType.LATITUDE: ['lat', 'latitude'],
        SemanticType.LONGITUDE: ['lng', 'lon', 'longitude'],
        SemanticType.ID: ['id', 'key', 'pk'],
        SemanticType.CURRENCY: ['price', 'amount', 'cost', 'revenue', 'salary'],
        SemanticType.PERCENTAGE: ['pct', 'percent', 'rate', 'ratio']
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def infer(
        self,
        df: pd.DataFrame,
        sample_size: int = 1000
    ) -> SchemaResult:
        """Infer schema from DataFrame."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Inferring schema: {df.shape[0]} rows, {df.shape[1]} columns")
        
        columns = []
        
        for col in df.columns:
            inference = self._infer_column(df, col, sample_size)
            columns.append(inference)
        
        # Calculate type coverage
        known_types = sum(1 for c in columns if c.semantic_type != SemanticType.UNKNOWN)
        type_coverage = known_types / len(columns) * 100 if columns else 0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SchemaResult(
            n_columns=len(df.columns),
            n_rows=len(df),
            columns=columns,
            type_coverage=type_coverage,
            processing_time_sec=processing_time
        )
    
    def _infer_column(
        self,
        df: pd.DataFrame,
        col: str,
        sample_size: int
    ) -> ColumnInference:
        """Infer column type."""
        series = df[col]
        dtype = str(series.dtype)
        
        # Sample for efficiency
        sample = series.dropna().head(sample_size)
        
        # Check nullability and uniqueness
        nullable = series.isna().any()
        unique = series.nunique() == len(series.dropna())
        
        # Get sample values
        sample_values = sample.head(5).tolist()
        
        # Infer semantic type
        semantic_type, confidence = self._infer_semantic_type(col, series, sample)
        
        # Detect pattern for strings
        pattern = None
        if series.dtype == 'object' and len(sample) > 0:
            pattern = self._detect_pattern(sample)
        
        return ColumnInference(
            column_name=col,
            pandas_dtype=dtype,
            semantic_type=semantic_type,
            confidence=confidence,
            nullable=nullable,
            unique=unique,
            sample_values=sample_values,
            pattern=pattern
        )
    
    def _infer_semantic_type(
        self,
        col_name: str,
        series: pd.Series,
        sample: pd.Series
    ) -> tuple:
        """Infer semantic type."""
        col_lower = col_name.lower()
        
        # Check numeric types
        if pd.api.types.is_integer_dtype(series):
            # Check for ID
            if any(ind in col_lower for ind in self.NAME_INDICATORS[SemanticType.ID]):
                return SemanticType.ID, 0.9
            return SemanticType.INTEGER, 1.0
        
        elif pd.api.types.is_float_dtype(series):
            # Check for latitude/longitude
            if any(ind in col_lower for ind in self.NAME_INDICATORS[SemanticType.LATITUDE]):
                return SemanticType.LATITUDE, 0.9
            if any(ind in col_lower for ind in self.NAME_INDICATORS[SemanticType.LONGITUDE]):
                return SemanticType.LONGITUDE, 0.9
            # Check for currency/percentage
            if any(ind in col_lower for ind in self.NAME_INDICATORS[SemanticType.CURRENCY]):
                return SemanticType.CURRENCY, 0.8
            if any(ind in col_lower for ind in self.NAME_INDICATORS[SemanticType.PERCENTAGE]):
                return SemanticType.PERCENTAGE, 0.8
            return SemanticType.FLOAT, 1.0
        
        elif pd.api.types.is_bool_dtype(series):
            return SemanticType.BOOLEAN, 1.0
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME, 1.0
        
        elif series.dtype == 'object':
            # Try pattern matching
            if len(sample) > 0:
                for stype, pattern in self.PATTERNS.items():
                    match_count = sample.astype(str).apply(
                        lambda x: bool(pattern.match(str(x)))
                    ).sum()
                    
                    if match_count / len(sample) > 0.8:
                        return stype, match_count / len(sample)
            
            # Check column name
            for stype, indicators in self.NAME_INDICATORS.items():
                if any(ind in col_lower for ind in indicators):
                    return stype, 0.7
            
            # Check for categorical vs text
            if series.nunique() < 20 or series.nunique() / len(series.dropna()) < 0.05:
                return SemanticType.CATEGORICAL, 0.8
            
            avg_len = sample.astype(str).str.len().mean()
            if avg_len > 50:
                return SemanticType.TEXT, 0.7
            
            return SemanticType.STRING, 0.5
        
        return SemanticType.UNKNOWN, 0.0
    
    def _detect_pattern(self, sample: pd.Series) -> Optional[str]:
        """Detect common pattern in string column."""
        if len(sample) < 5:
            return None
        
        sample_strs = sample.astype(str).tolist()
        
        # Check if all same length
        lengths = [len(s) for s in sample_strs]
        if len(set(lengths)) == 1:
            # Create pattern from first value
            first = sample_strs[0]
            pattern_parts = []
            
            for char in first:
                if char.isdigit():
                    pattern_parts.append('\\d')
                elif char.isalpha():
                    pattern_parts.append('[A-Za-z]')
                else:
                    pattern_parts.append(re.escape(char))
            
            return ''.join(pattern_parts)
        
        return None
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        expected: Dict[str, SemanticType]
    ) -> List[str]:
        """Validate DataFrame against expected schema."""
        issues = []
        inferred = self.infer(df)
        
        for col_name, expected_type in expected.items():
            if col_name not in df.columns:
                issues.append(f"Missing column: {col_name}")
                continue
            
            for col in inferred.columns:
                if col.column_name == col_name:
                    if col.semantic_type != expected_type:
                        issues.append(
                            f"Type mismatch for {col_name}: "
                            f"expected {expected_type.value}, got {col.semantic_type.value}"
                        )
                    break
        
        return issues


# ============================================================================
# Factory Functions
# ============================================================================

def get_schema_engine() -> SchemaInferenceEngine:
    """Get schema inference engine."""
    return SchemaInferenceEngine()


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick schema inference."""
    engine = SchemaInferenceEngine(verbose=False)
    result = engine.infer(df)
    return result.to_dict()


def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Get column semantic types."""
    engine = SchemaInferenceEngine(verbose=False)
    result = engine.infer(df)
    return {c.column_name: c.semantic_type.value for c in result.columns}
