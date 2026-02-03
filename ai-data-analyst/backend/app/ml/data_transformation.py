# AI Enterprise Data Analyst - Data Transformation Engine
# Production-grade data transformation operations
# Handles: any dataframe, common transformations

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

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

class TransformationType(str, Enum):
    """Types of transformations."""
    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    BINNING = "binning"
    ENCODING = "encoding"
    CUSTOM = "custom"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TransformationStep:
    """Single transformation step."""
    column: str
    transformation: TransformationType
    params: Dict[str, Any] = field(default_factory=dict)
    output_column: Optional[str] = None


@dataclass
class TransformationResult:
    """Transformation result."""
    transformed_df: pd.DataFrame = None
    n_columns_transformed: int = 0
    n_columns_added: int = 0
    steps_applied: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_columns_transformed": self.n_columns_transformed,
            "n_columns_added": self.n_columns_added,
            "steps_applied": self.steps_applied
        }


# ============================================================================
# Data Transformation Engine
# ============================================================================

class DataTransformationEngine:
    """
    Data Transformation engine.
    
    Features:
    - Mathematical transformations
    - Normalization/standardization
    - Binning and encoding
    - Custom transformations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps: List[TransformationStep] = []
    
    def add_step(
        self,
        column: str,
        transformation: TransformationType,
        params: Dict[str, Any] = None,
        output_column: str = None
    ):
        """Add transformation step."""
        self.steps.append(TransformationStep(
            column=column,
            transformation=transformation,
            params=params or {},
            output_column=output_column or f"{column}_{transformation.value}"
        ))
    
    def transform(self, df: pd.DataFrame) -> TransformationResult:
        """Apply all transformations."""
        start_time = datetime.now()
        
        result_df = df.copy()
        steps_applied = []
        n_added = 0
        
        for step in self.steps:
            if step.column not in result_df.columns:
                continue
            
            col = result_df[step.column]
            
            if step.transformation == TransformationType.LOG:
                transformed = self._log_transform(col)
            elif step.transformation == TransformationType.SQRT:
                transformed = self._sqrt_transform(col)
            elif step.transformation == TransformationType.SQUARE:
                transformed = col ** 2
            elif step.transformation == TransformationType.NORMALIZE:
                transformed = self._normalize(col)
            elif step.transformation == TransformationType.STANDARDIZE:
                transformed = self._standardize(col)
            elif step.transformation == TransformationType.BINNING:
                transformed = self._bin(col, step.params)
            elif step.transformation == TransformationType.ENCODING:
                transformed = self._encode(col)
            else:
                transformed = col
            
            result_df[step.output_column] = transformed
            n_added += 1
            steps_applied.append(f"{step.column} -> {step.transformation.value}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TransformationResult(
            transformed_df=result_df,
            n_columns_transformed=len(self.steps),
            n_columns_added=n_added,
            steps_applied=steps_applied,
            processing_time_sec=processing_time
        )
    
    def _log_transform(self, col: pd.Series) -> pd.Series:
        """Log transformation with handling for zeros/negatives."""
        min_val = col.min()
        if min_val <= 0:
            shifted = col - min_val + 1
        else:
            shifted = col
        return np.log(shifted)
    
    def _sqrt_transform(self, col: pd.Series) -> pd.Series:
        """Square root transformation."""
        return np.sqrt(np.abs(col))
    
    def _normalize(self, col: pd.Series) -> pd.Series:
        """Min-max normalization to [0, 1]."""
        min_val = col.min()
        max_val = col.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(col))
        return (col - min_val) / (max_val - min_val)
    
    def _standardize(self, col: pd.Series) -> pd.Series:
        """Z-score standardization."""
        mean = col.mean()
        std = col.std()
        if std == 0:
            return pd.Series([0] * len(col))
        return (col - mean) / std
    
    def _bin(self, col: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Binning into categories."""
        n_bins = params.get('n_bins', 5)
        labels = params.get('labels', None)
        
        if labels is None:
            labels = [f"bin_{i+1}" for i in range(n_bins)]
        
        return pd.cut(col, bins=n_bins, labels=labels)
    
    def _encode(self, col: pd.Series) -> pd.Series:
        """Label encoding."""
        return col.astype('category').cat.codes
    
    # Quick transformation methods
    def log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Quick log transform."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[f"{col}_log"] = self._log_transform(result[col])
        return result
    
    def normalize_columns(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Quick normalize."""
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        for col in columns:
            result[col] = self._normalize(result[col])
        return result
    
    def standardize_columns(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Quick standardize."""
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        for col in columns:
            result[col] = self._standardize(result[col])
        return result


# ============================================================================
# Factory Functions
# ============================================================================

def get_transformation_engine() -> DataTransformationEngine:
    """Get data transformation engine."""
    return DataTransformationEngine()


def quick_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Quick normalization of all numeric columns."""
    engine = DataTransformationEngine(verbose=False)
    return engine.normalize_columns(df)


def quick_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Quick standardization of all numeric columns."""
    engine = DataTransformationEngine(verbose=False)
    return engine.standardize_columns(df)
