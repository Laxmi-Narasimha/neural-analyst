# AI Enterprise Data Analyst - Data Sampling Engine
# Production-grade sampling methods for data science
# Handles: random, stratified, cluster, systematic sampling

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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

class SamplingMethod(str, Enum):
    """Sampling methods."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    CLUSTER = "cluster"
    BOOTSTRAP = "bootstrap"
    WEIGHTED = "weighted"
    RESERVOIR = "reservoir"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SamplingResult:
    """Sampling result."""
    method: SamplingMethod
    n_original: int
    n_sampled: int
    sample_fraction: float
    
    # Sampled data
    sampled_df: pd.DataFrame = None
    sampled_indices: List[int] = field(default_factory=list)
    
    # Stratification info
    strata_counts: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    representativeness_score: float = 1.0  # 0-1
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "n_original": self.n_original,
            "n_sampled": self.n_sampled,
            "sample_fraction": round(self.sample_fraction, 4),
            "representativeness": round(self.representativeness_score, 4),
            "strata_counts": self.strata_counts
        }


# ============================================================================
# Data Sampling Engine
# ============================================================================

class DataSamplingEngine:
    """
    Production-grade Data Sampling engine.
    
    Features:
    - Random sampling
    - Stratified sampling
    - Systematic sampling
    - Cluster sampling
    - Bootstrap sampling
    - Weighted sampling
    - Representativeness checking
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        np.random.seed(random_state)
    
    def sample(
        self,
        df: pd.DataFrame,
        n: int = None,
        frac: float = None,
        method: SamplingMethod = SamplingMethod.RANDOM,
        stratify_col: str = None,
        weight_col: str = None,
        cluster_col: str = None,
        replace: bool = False
    ) -> SamplingResult:
        """Sample data using specified method."""
        start_time = datetime.now()
        
        if n is None and frac is None:
            frac = 0.1
        
        if n is None:
            n = int(len(df) * frac)
        
        n = min(n, len(df)) if not replace else n
        
        if self.verbose:
            logger.info(f"Sampling {n}/{len(df)} rows using {method.value}")
        
        if method == SamplingMethod.STRATIFIED:
            sampled, strata_counts = self._stratified_sample(df, n, stratify_col, replace)
        elif method == SamplingMethod.SYSTEMATIC:
            sampled, strata_counts = self._systematic_sample(df, n)
        elif method == SamplingMethod.CLUSTER:
            sampled, strata_counts = self._cluster_sample(df, n, cluster_col)
        elif method == SamplingMethod.BOOTSTRAP:
            sampled, strata_counts = self._bootstrap_sample(df, n)
        elif method == SamplingMethod.WEIGHTED:
            sampled, strata_counts = self._weighted_sample(df, n, weight_col, replace)
        elif method == SamplingMethod.RESERVOIR:
            sampled, strata_counts = self._reservoir_sample(df, n)
        else:
            sampled, strata_counts = self._random_sample(df, n, replace)
        
        # Calculate representativeness
        representativeness = self._calculate_representativeness(df, sampled)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SamplingResult(
            method=method,
            n_original=len(df),
            n_sampled=len(sampled),
            sample_fraction=len(sampled) / len(df),
            sampled_df=sampled,
            sampled_indices=sampled.index.tolist(),
            strata_counts=strata_counts,
            representativeness_score=representativeness,
            processing_time_sec=processing_time
        )
    
    def _random_sample(
        self,
        df: pd.DataFrame,
        n: int,
        replace: bool
    ) -> Tuple[pd.DataFrame, Dict]:
        """Simple random sampling."""
        sampled = df.sample(n=n, replace=replace, random_state=self.random_state)
        return sampled, {}
    
    def _stratified_sample(
        self,
        df: pd.DataFrame,
        n: int,
        stratify_col: str,
        replace: bool
    ) -> Tuple[pd.DataFrame, Dict]:
        """Stratified random sampling."""
        if stratify_col is None or stratify_col not in df.columns:
            # Auto-detect categorical column
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                stratify_col = cat_cols[0]
            else:
                return self._random_sample(df, n, replace)
        
        strata = df[stratify_col].value_counts(normalize=True)
        samples = []
        strata_counts = {}
        
        for stratum, prop in strata.items():
            stratum_n = max(1, int(n * prop))
            stratum_df = df[df[stratify_col] == stratum]
            
            actual_n = min(stratum_n, len(stratum_df)) if not replace else stratum_n
            
            if actual_n > 0:
                stratum_sample = stratum_df.sample(
                    n=actual_n, replace=replace, random_state=self.random_state
                )
                samples.append(stratum_sample)
                strata_counts[str(stratum)] = len(stratum_sample)
        
        if samples:
            sampled = pd.concat(samples)
            # Adjust if we got more or less than n
            if len(sampled) > n:
                sampled = sampled.sample(n=n, random_state=self.random_state)
        else:
            sampled = df.sample(n=min(n, len(df)), random_state=self.random_state)
        
        return sampled, strata_counts
    
    def _systematic_sample(
        self,
        df: pd.DataFrame,
        n: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """Systematic sampling (every k-th element)."""
        k = max(1, len(df) // n)
        start = np.random.randint(0, k)
        
        indices = list(range(start, len(df), k))[:n]
        sampled = df.iloc[indices]
        
        return sampled, {"interval": k, "start": start}
    
    def _cluster_sample(
        self,
        df: pd.DataFrame,
        n: int,
        cluster_col: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Cluster sampling."""
        if cluster_col is None or cluster_col not in df.columns:
            return self._random_sample(df, n, False)
        
        clusters = df[cluster_col].unique()
        avg_cluster_size = len(df) / len(clusters)
        n_clusters = max(1, int(n / avg_cluster_size))
        
        selected_clusters = np.random.choice(
            clusters, size=min(n_clusters, len(clusters)), replace=False
        )
        
        sampled = df[df[cluster_col].isin(selected_clusters)]
        
        # If we got too many, random sample within
        if len(sampled) > n * 1.2:
            sampled = sampled.sample(n=n, random_state=self.random_state)
        
        cluster_counts = {str(c): int((df[cluster_col] == c).sum()) 
                          for c in selected_clusters}
        
        return sampled, cluster_counts
    
    def _bootstrap_sample(
        self,
        df: pd.DataFrame,
        n: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """Bootstrap sampling (with replacement)."""
        sampled = df.sample(n=n, replace=True, random_state=self.random_state)
        unique_count = sampled.index.nunique()
        
        return sampled, {"unique_rows": unique_count, "duplicates": n - unique_count}
    
    def _weighted_sample(
        self,
        df: pd.DataFrame,
        n: int,
        weight_col: str,
        replace: bool
    ) -> Tuple[pd.DataFrame, Dict]:
        """Weighted sampling based on a column."""
        if weight_col is None or weight_col not in df.columns:
            return self._random_sample(df, n, replace)
        
        weights = df[weight_col].fillna(0)
        weights = weights - weights.min() + 1e-10  # Ensure positive
        weights = weights / weights.sum()  # Normalize
        
        sampled = df.sample(
            n=n, replace=replace, weights=weights, random_state=self.random_state
        )
        
        return sampled, {"weight_column": weight_col}
    
    def _reservoir_sample(
        self,
        df: pd.DataFrame,
        n: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """Reservoir sampling (for streaming data)."""
        # Initialize reservoir
        reservoir = list(range(min(n, len(df))))
        
        # Replace elements with decreasing probability
        for i in range(n, len(df)):
            j = np.random.randint(0, i + 1)
            if j < n:
                reservoir[j] = i
        
        sampled = df.iloc[reservoir]
        
        return sampled, {"method": "reservoir"}
    
    def _calculate_representativeness(
        self,
        original: pd.DataFrame,
        sample: pd.DataFrame
    ) -> float:
        """Calculate how representative the sample is."""
        scores = []
        
        # Compare numeric column distributions
        num_cols = original.select_dtypes(include=[np.number]).columns
        
        for col in num_cols[:10]:  # Limit to 10 columns
            orig_mean = original[col].mean()
            sample_mean = sample[col].mean()
            
            if orig_mean != 0:
                diff = abs(sample_mean - orig_mean) / abs(orig_mean)
                scores.append(max(0, 1 - diff))
            else:
                scores.append(1.0 if sample_mean == 0 else 0.5)
        
        # Compare categorical distributions
        cat_cols = original.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols[:5]:  # Limit to 5 columns
            orig_dist = original[col].value_counts(normalize=True)
            sample_dist = sample[col].value_counts(normalize=True)
            
            # Calculate overlap
            common = set(orig_dist.index) & set(sample_dist.index)
            if len(common) > 0:
                diffs = [abs(orig_dist.get(k, 0) - sample_dist.get(k, 0)) 
                        for k in common]
                overlap = 1 - np.mean(diffs)
                scores.append(overlap)
        
        return float(np.mean(scores)) if scores else 1.0
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_col: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        if stratify_col:
            # Stratified split
            test_result = self.sample(
                df, frac=test_size, method=SamplingMethod.STRATIFIED,
                stratify_col=stratify_col
            )
            test = test_result.sampled_df
            train = df.loc[~df.index.isin(test.index)]
        else:
            # Random split
            test = df.sample(frac=test_size, random_state=self.random_state)
            train = df.loc[~df.index.isin(test.index)]
        
        return train, test
    
    def cross_validation_folds(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        stratify_col: str = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate cross-validation folds."""
        folds = []
        
        if stratify_col:
            # Stratified K-fold
            strata = df[stratify_col].unique()
            fold_indices = [[] for _ in range(n_folds)]
            
            for stratum in strata:
                stratum_idx = df[df[stratify_col] == stratum].index.tolist()
                np.random.shuffle(stratum_idx)
                
                for i, idx in enumerate(stratum_idx):
                    fold_indices[i % n_folds].append(idx)
        else:
            # Simple K-fold
            indices = df.index.tolist()
            np.random.shuffle(indices)
            
            fold_size = len(indices) // n_folds
            fold_indices = []
            
            for i in range(n_folds):
                start = i * fold_size
                end = start + fold_size if i < n_folds - 1 else len(indices)
                fold_indices.append(indices[start:end])
        
        # Create train/test pairs
        for i in range(n_folds):
            test_idx = fold_indices[i]
            train_idx = [idx for j, fold in enumerate(fold_indices) 
                        if j != i for idx in fold]
            
            train = df.loc[train_idx]
            test = df.loc[test_idx]
            folds.append((train, test))
        
        return folds


# ============================================================================
# Factory Functions
# ============================================================================

def get_sampling_engine(random_state: int = 42) -> DataSamplingEngine:
    """Get data sampling engine."""
    return DataSamplingEngine(random_state=random_state)


def quick_sample(
    df: pd.DataFrame,
    n: int = None,
    frac: float = 0.1,
    method: str = "random"
) -> pd.DataFrame:
    """Quick sampling."""
    engine = DataSamplingEngine(verbose=False)
    result = engine.sample(df, n=n, frac=frac, method=SamplingMethod(method))
    return result.sampled_df


def stratified_sample(
    df: pd.DataFrame,
    stratify_col: str,
    n: int = None,
    frac: float = 0.1
) -> pd.DataFrame:
    """Quick stratified sampling."""
    engine = DataSamplingEngine(verbose=False)
    result = engine.sample(
        df, n=n, frac=frac, 
        method=SamplingMethod.STRATIFIED,
        stratify_col=stratify_col
    )
    return result.sampled_df
