# AI Enterprise Data Analyst - Data Enrichment Engine
# Production-grade data enrichment and augmentation
# Handles: derived features, date extraction, encoding

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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

class EnrichmentType(str, Enum):
    """Types of enrichment."""
    DATE_FEATURES = "date_features"
    TEXT_FEATURES = "text_features"
    NUMERIC_BINNING = "numeric_binning"
    CATEGORICAL_ENCODING = "categorical_encoding"
    INTERACTION = "interaction"
    AGGREGATION = "aggregation"
    CUSTOM = "custom"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EnrichmentStep:
    """Single enrichment step."""
    enrichment_type: EnrichmentType
    source_columns: List[str]
    new_columns: List[str]
    description: str


@dataclass
class EnrichmentResult:
    """Complete enrichment result."""
    n_original_columns: int = 0
    n_new_columns: int = 0
    n_total_columns: int = 0
    
    enriched_df: pd.DataFrame = None
    
    steps_applied: List[EnrichmentStep] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "original_columns": self.n_original_columns,
                "new_columns": self.n_new_columns,
                "total_columns": self.n_total_columns
            },
            "steps": [
                {
                    "type": s.enrichment_type.value,
                    "source": s.source_columns,
                    "new_columns": s.new_columns,
                    "description": s.description
                }
                for s in self.steps_applied
            ]
        }


# ============================================================================
# Data Enrichment Engine
# ============================================================================

class DataEnrichmentEngine:
    """
    Production-grade Data Enrichment engine.
    
    Features:
    - Date feature extraction
    - Text feature extraction
    - Numeric binning
    - Categorical encoding
    - Interaction features
    - Custom transformations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps: List[EnrichmentStep] = []
    
    def enrich(self, df: pd.DataFrame, auto: bool = True) -> EnrichmentResult:
        """Enrich DataFrame with derived features."""
        start_time = datetime.now()
        
        result_df = df.copy()
        original_cols = len(df.columns)
        steps = []
        
        if auto:
            # Auto-detect and enrich
            for col in df.columns:
                # Date columns
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    new_cols = self._extract_date_features(result_df, col)
                    if new_cols:
                        steps.append(EnrichmentStep(
                            enrichment_type=EnrichmentType.DATE_FEATURES,
                            source_columns=[col],
                            new_columns=new_cols,
                            description=f"Extracted date features from {col}"
                        ))
                
                # Text columns
                elif df[col].dtype == 'object':
                    avg_len = df[col].dropna().astype(str).str.len().mean()
                    if avg_len > 50:  # Likely text
                        new_cols = self._extract_text_features(result_df, col)
                        if new_cols:
                            steps.append(EnrichmentStep(
                                enrichment_type=EnrichmentType.TEXT_FEATURES,
                                source_columns=[col],
                                new_columns=new_cols,
                                description=f"Extracted text features from {col}"
                            ))
        
        # Apply manual steps
        for step_config in self.steps:
            new_cols = self._apply_step(result_df, step_config)
            steps.append(step_config)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return EnrichmentResult(
            n_original_columns=original_cols,
            n_new_columns=len(result_df.columns) - original_cols,
            n_total_columns=len(result_df.columns),
            enriched_df=result_df,
            steps_applied=steps,
            processing_time_sec=processing_time
        )
    
    def add_date_features(self, column: str):
        """Add date feature extraction."""
        self.steps.append(EnrichmentStep(
            enrichment_type=EnrichmentType.DATE_FEATURES,
            source_columns=[column],
            new_columns=[],
            description=f"Extract date features from {column}"
        ))
        return self
    
    def add_text_features(self, column: str):
        """Add text feature extraction."""
        self.steps.append(EnrichmentStep(
            enrichment_type=EnrichmentType.TEXT_FEATURES,
            source_columns=[column],
            new_columns=[],
            description=f"Extract text features from {column}"
        ))
        return self
    
    def add_binning(self, column: str, n_bins: int = 5, labels: List[str] = None):
        """Add numeric binning."""
        step = EnrichmentStep(
            enrichment_type=EnrichmentType.NUMERIC_BINNING,
            source_columns=[column],
            new_columns=[f"{column}_binned"],
            description=f"Bin {column} into {n_bins} categories"
        )
        step.params = {'n_bins': n_bins, 'labels': labels}
        self.steps.append(step)
        return self
    
    def add_interaction(self, col1: str, col2: str, operation: str = "multiply"):
        """Add interaction feature."""
        self.steps.append(EnrichmentStep(
            enrichment_type=EnrichmentType.INTERACTION,
            source_columns=[col1, col2],
            new_columns=[f"{col1}_{operation}_{col2}"],
            description=f"{operation} interaction: {col1} and {col2}"
        ))
        return self
    
    def _extract_date_features(self, df: pd.DataFrame, col: str) -> List[str]:
        """Extract features from datetime column."""
        new_cols = []
        
        try:
            dt = pd.to_datetime(df[col], errors='coerce')
            
            df[f'{col}_year'] = dt.dt.year
            new_cols.append(f'{col}_year')
            
            df[f'{col}_month'] = dt.dt.month
            new_cols.append(f'{col}_month')
            
            df[f'{col}_day'] = dt.dt.day
            new_cols.append(f'{col}_day')
            
            df[f'{col}_dayofweek'] = dt.dt.dayofweek
            new_cols.append(f'{col}_dayofweek')
            
            df[f'{col}_quarter'] = dt.dt.quarter
            new_cols.append(f'{col}_quarter')
            
            df[f'{col}_is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            new_cols.append(f'{col}_is_weekend')
            
            df[f'{col}_is_monthend'] = dt.dt.is_month_end.astype(int)
            new_cols.append(f'{col}_is_monthend')
            
            # Has time component
            if dt.dt.hour.nunique() > 1:
                df[f'{col}_hour'] = dt.dt.hour
                new_cols.append(f'{col}_hour')
                
                df[f'{col}_is_business_hour'] = dt.dt.hour.between(9, 17).astype(int)
                new_cols.append(f'{col}_is_business_hour')
                
        except Exception as e:
            logger.warning(f"Error extracting date features from {col}: {e}")
        
        return new_cols
    
    def _extract_text_features(self, df: pd.DataFrame, col: str) -> List[str]:
        """Extract features from text column."""
        new_cols = []
        
        try:
            text = df[col].fillna('').astype(str)
            
            df[f'{col}_length'] = text.str.len()
            new_cols.append(f'{col}_length')
            
            df[f'{col}_word_count'] = text.str.split().str.len()
            new_cols.append(f'{col}_word_count')
            
            df[f'{col}_has_numbers'] = text.str.contains(r'\d', regex=True).astype(int)
            new_cols.append(f'{col}_has_numbers')
            
            df[f'{col}_uppercase_ratio'] = text.apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
            )
            new_cols.append(f'{col}_uppercase_ratio')
            
        except Exception as e:
            logger.warning(f"Error extracting text features from {col}: {e}")
        
        return new_cols
    
    def _apply_step(self, df: pd.DataFrame, step: EnrichmentStep) -> List[str]:
        """Apply enrichment step."""
        new_cols = []
        
        if step.enrichment_type == EnrichmentType.DATE_FEATURES:
            for col in step.source_columns:
                if col in df.columns:
                    new_cols.extend(self._extract_date_features(df, col))
        
        elif step.enrichment_type == EnrichmentType.TEXT_FEATURES:
            for col in step.source_columns:
                if col in df.columns:
                    new_cols.extend(self._extract_text_features(df, col))
        
        elif step.enrichment_type == EnrichmentType.NUMERIC_BINNING:
            col = step.source_columns[0]
            if col in df.columns:
                params = getattr(step, 'params', {})
                n_bins = params.get('n_bins', 5)
                labels = params.get('labels')
                
                new_col = f'{col}_binned'
                df[new_col] = pd.cut(df[col], bins=n_bins, labels=labels)
                new_cols.append(new_col)
        
        elif step.enrichment_type == EnrichmentType.INTERACTION:
            if len(step.source_columns) >= 2:
                col1, col2 = step.source_columns[:2]
                if col1 in df.columns and col2 in df.columns:
                    new_col = step.new_columns[0] if step.new_columns else f'{col1}_x_{col2}'
                    df[new_col] = df[col1] * df[col2]
                    new_cols.append(new_col)
        
        return new_cols


# ============================================================================
# Factory Functions
# ============================================================================

def get_enrichment_engine() -> DataEnrichmentEngine:
    """Get data enrichment engine."""
    return DataEnrichmentEngine()


def auto_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically enrich DataFrame."""
    engine = DataEnrichmentEngine(verbose=False)
    result = engine.enrich(df, auto=True)
    return result.enriched_df


def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extract date features from a column."""
    engine = DataEnrichmentEngine(verbose=False)
    engine.add_date_features(date_col)
    result = engine.enrich(df, auto=False)
    return result.enriched_df
