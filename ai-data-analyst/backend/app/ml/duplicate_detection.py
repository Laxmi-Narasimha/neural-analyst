# AI Enterprise Data Analyst - Duplicate Detection Engine
# Production-grade duplicate and fuzzy matching
# Handles: exact and fuzzy matching, configurable thresholds

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class MatchType(str, Enum):
    """Type of duplicate match."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    PARTIAL = "partial"


class DuplicateStrategy(str, Enum):
    """Strategy for handling duplicates."""
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"
    MERGE = "merge"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DuplicateGroup:
    """Group of duplicate records."""
    group_id: int
    indices: List[int]
    match_type: MatchType
    similarity_score: float
    sample_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateConfig:
    """Configuration for duplicate detection."""
    subset: Optional[List[str]] = None  # Columns to check
    match_type: MatchType = MatchType.EXACT
    fuzzy_threshold: float = 0.8  # For fuzzy matching
    ignore_case: bool = True
    ignore_whitespace: bool = True


@dataclass
class DuplicateResult:
    """Complete duplicate detection result."""
    n_total: int = 0
    n_duplicates: int = 0
    n_unique: int = 0
    duplicate_rate: float = 0.0
    
    # Duplicate groups
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    
    # Indices
    duplicate_indices: List[int] = field(default_factory=list)
    unique_indices: List[int] = field(default_factory=list)
    
    # Cleaned DataFrame
    cleaned_df: pd.DataFrame = None
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_total": self.n_total,
                "n_duplicates": self.n_duplicates,
                "n_unique": self.n_unique,
                "duplicate_rate": round(self.duplicate_rate * 100, 2)
            },
            "duplicate_groups": [
                {
                    "group_id": g.group_id,
                    "n_records": len(g.indices),
                    "match_type": g.match_type.value,
                    "similarity": round(g.similarity_score, 3)
                }
                for g in self.duplicate_groups[:20]
            ],
            "n_groups": len(self.duplicate_groups)
        }


# ============================================================================
# Fuzzy Matching Utilities
# ============================================================================

class FuzzyMatcher:
    """Fuzzy string matching utilities."""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """Calculate similarity ratio (0 to 1)."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = FuzzyMatcher.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        return 1 - (distance / max_len)
    
    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        """Calculate Jaccard similarity for tokens."""
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0


# ============================================================================
# Duplicate Detection Engine
# ============================================================================

class DuplicateDetectionEngine:
    """
    Complete Duplicate Detection engine.
    
    Features:
    - Exact duplicate detection
    - Fuzzy matching
    - Configurable columns
    - Multiple deduplication strategies
    """
    
    def __init__(self, config: DuplicateConfig = None, verbose: bool = True):
        self.config = config or DuplicateConfig()
        self.verbose = verbose
        self.fuzzy = FuzzyMatcher()
    
    def detect(
        self,
        df: pd.DataFrame,
        subset: List[str] = None
    ) -> DuplicateResult:
        """Detect duplicates in DataFrame."""
        start_time = datetime.now()
        
        subset = subset or self.config.subset or df.columns.tolist()
        subset = [c for c in subset if c in df.columns]
        
        if self.verbose:
            logger.info(f"Detecting duplicates in columns: {subset}")
        
        # Preprocess for matching
        processed = self._preprocess(df, subset)
        
        # Detect based on match type
        if self.config.match_type == MatchType.EXACT:
            groups = self._detect_exact(df, processed, subset)
        else:
            groups = self._detect_fuzzy(df, processed, subset)
        
        # Calculate statistics
        all_dup_indices = set()
        for group in groups:
            all_dup_indices.update(group.indices[1:])  # Keep first, mark rest as duplicates
        
        n_duplicates = len(all_dup_indices)
        n_unique = len(df) - n_duplicates
        
        # Create cleaned DataFrame (keep first of each group)
        cleaned_indices = [i for i in range(len(df)) if i not in all_dup_indices]
        cleaned_df = df.iloc[cleaned_indices].reset_index(drop=True)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DuplicateResult(
            n_total=len(df),
            n_duplicates=n_duplicates,
            n_unique=n_unique,
            duplicate_rate=n_duplicates / len(df) if len(df) > 0 else 0,
            duplicate_groups=groups,
            duplicate_indices=list(all_dup_indices),
            unique_indices=cleaned_indices,
            cleaned_df=cleaned_df,
            processing_time_sec=processing_time
        )
    
    def _preprocess(
        self,
        df: pd.DataFrame,
        subset: List[str]
    ) -> pd.DataFrame:
        """Preprocess data for matching."""
        processed = df[subset].copy()
        
        for col in subset:
            if processed[col].dtype == 'object':
                if self.config.ignore_case:
                    processed[col] = processed[col].astype(str).str.lower()
                if self.config.ignore_whitespace:
                    processed[col] = processed[col].astype(str).str.strip()
                    processed[col] = processed[col].str.replace(r'\s+', ' ', regex=True)
        
        return processed
    
    def _detect_exact(
        self,
        df: pd.DataFrame,
        processed: pd.DataFrame,
        subset: List[str]
    ) -> List[DuplicateGroup]:
        """Detect exact duplicates."""
        # Create composite key
        processed['_dup_key'] = processed[subset].astype(str).agg('|'.join, axis=1)
        
        # Find duplicates
        dup_mask = processed.duplicated(subset='_dup_key', keep=False)
        dup_keys = processed.loc[dup_mask, '_dup_key'].unique()
        
        groups = []
        for i, key in enumerate(dup_keys):
            indices = processed[processed['_dup_key'] == key].index.tolist()
            
            if len(indices) > 1:
                groups.append(DuplicateGroup(
                    group_id=i,
                    indices=indices,
                    match_type=MatchType.EXACT,
                    similarity_score=1.0,
                    sample_values=df.iloc[indices[0]][subset].to_dict()
                ))
        
        return groups
    
    def _detect_fuzzy(
        self,
        df: pd.DataFrame,
        processed: pd.DataFrame,
        subset: List[str]
    ) -> List[DuplicateGroup]:
        """Detect fuzzy duplicates."""
        groups = []
        matched = set()
        
        # Create composite strings
        strings = processed[subset].astype(str).agg(' '.join, axis=1).tolist()
        
        for i in range(len(strings)):
            if i in matched:
                continue
            
            group_indices = [i]
            
            for j in range(i + 1, len(strings)):
                if j in matched:
                    continue
                
                sim = self.fuzzy.similarity_ratio(strings[i], strings[j])
                
                if sim >= self.config.fuzzy_threshold:
                    group_indices.append(j)
                    matched.add(j)
            
            if len(group_indices) > 1:
                matched.add(i)
                groups.append(DuplicateGroup(
                    group_id=len(groups),
                    indices=group_indices,
                    match_type=MatchType.FUZZY,
                    similarity_score=self.config.fuzzy_threshold,
                    sample_values=df.iloc[i][subset].to_dict()
                ))
        
        return groups
    
    def deduplicate(
        self,
        df: pd.DataFrame,
        strategy: DuplicateStrategy = DuplicateStrategy.KEEP_FIRST,
        subset: List[str] = None
    ) -> pd.DataFrame:
        """Remove duplicates from DataFrame."""
        result = self.detect(df, subset)
        
        if strategy == DuplicateStrategy.KEEP_FIRST:
            return result.cleaned_df
        elif strategy == DuplicateStrategy.KEEP_LAST:
            # Re-run keeping last
            keep_indices = set(range(len(df)))
            for group in result.duplicate_groups:
                for idx in group.indices[:-1]:
                    keep_indices.discard(idx)
            return df.iloc[list(keep_indices)].reset_index(drop=True)
        else:
            return df


# ============================================================================
# Factory Functions
# ============================================================================

def get_duplicate_engine(config: DuplicateConfig = None) -> DuplicateDetectionEngine:
    """Get duplicate detection engine."""
    return DuplicateDetectionEngine(config=config)


def quick_duplicates(
    df: pd.DataFrame,
    subset: List[str] = None
) -> Dict[str, Any]:
    """Quick duplicate detection."""
    engine = DuplicateDetectionEngine(verbose=False)
    result = engine.detect(df, subset)
    return result.to_dict()
