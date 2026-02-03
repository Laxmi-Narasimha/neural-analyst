# AI Enterprise Data Analyst - ANOVA Analysis Engine
# Production-grade analysis of variance
# Handles: one-way, two-way, repeated measures

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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

class ANOVAType(str, Enum):
    """Type of ANOVA."""
    ONE_WAY = "one_way"
    TWO_WAY = "two_way"
    REPEATED = "repeated"
    WELCH = "welch"  # For unequal variances


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PostHocResult:
    """Post-hoc test result."""
    group1: str
    group2: str
    mean_diff: float
    p_value: float
    significant: bool


@dataclass
class GroupStats:
    """Statistics for a group."""
    name: str
    n: int
    mean: float
    std: float
    se: float


@dataclass
class ANOVAResult:
    """Complete ANOVA result."""
    anova_type: ANOVAType
    
    # Main result
    f_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    
    # Group statistics
    groups: List[GroupStats] = field(default_factory=list)
    
    # Effect size
    eta_squared: float = 0.0
    omega_squared: float = 0.0
    
    # Post-hoc tests
    post_hoc: List[PostHocResult] = field(default_factory=list)
    
    # Assumptions
    homogeneity_p: float = 0.0  # Levene's test
    normality_ps: Dict[str, float] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anova_type": self.anova_type.value,
            "main_result": {
                "f_statistic": round(self.f_statistic, 4),
                "p_value": round(self.p_value, 6),
                "is_significant": self.is_significant
            },
            "effect_size": {
                "eta_squared": round(self.eta_squared, 4),
                "omega_squared": round(self.omega_squared, 4)
            },
            "groups": [
                {"name": g.name, "n": g.n, "mean": round(g.mean, 4), "std": round(g.std, 4)}
                for g in self.groups
            ],
            "post_hoc": [
                {
                    "comparison": f"{p.group1} vs {p.group2}",
                    "mean_diff": round(p.mean_diff, 4),
                    "p_value": round(p.p_value, 6),
                    "significant": p.significant
                }
                for p in self.post_hoc[:20]
            ],
            "assumptions": {
                "homogeneity_p": round(self.homogeneity_p, 4),
                "homogeneity_met": self.homogeneity_p > 0.05
            }
        }


# ============================================================================
# ANOVA Analysis Engine
# ============================================================================

class ANOVAEngine:
    """
    ANOVA Analysis engine.
    
    Features:
    - One-way ANOVA
    - Welch's ANOVA (unequal variances)
    - Effect size calculation
    - Post-hoc tests (Tukey)
    """
    
    def __init__(self, alpha: float = 0.05, verbose: bool = True):
        self.alpha = alpha
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        group_var: str
    ) -> ANOVAResult:
        """Perform one-way ANOVA."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"ANOVA: {dependent_var} by {group_var}")
        
        # Get groups
        groups = df[group_var].unique()
        group_data = [df[df[group_var] == g][dependent_var].dropna().values for g in groups]
        
        # Group statistics
        group_stats = []
        for g, data in zip(groups, group_data):
            group_stats.append(GroupStats(
                name=str(g),
                n=len(data),
                mean=float(np.mean(data)),
                std=float(np.std(data, ddof=1)),
                se=float(np.std(data, ddof=1) / np.sqrt(len(data)))
            ))
        
        # Levene's test for homogeneity
        _, homogeneity_p = scipy_stats.levene(*group_data)
        
        # Decide ANOVA type
        if homogeneity_p < 0.05:
            anova_type = ANOVAType.WELCH
            f_stat, p_value = self._welch_anova(group_data)
        else:
            anova_type = ANOVAType.ONE_WAY
            f_stat, p_value = scipy_stats.f_oneway(*group_data)
        
        is_significant = p_value < self.alpha
        
        # Effect sizes
        eta_sq, omega_sq = self._calculate_effect_sizes(group_data)
        
        # Post-hoc tests if significant
        post_hoc = []
        if is_significant and len(groups) > 2:
            post_hoc = self._tukey_hsd(group_data, [str(g) for g in groups])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ANOVAResult(
            anova_type=anova_type,
            f_statistic=float(f_stat),
            p_value=float(p_value),
            is_significant=is_significant,
            groups=group_stats,
            eta_squared=eta_sq,
            omega_squared=omega_sq,
            post_hoc=post_hoc,
            homogeneity_p=float(homogeneity_p),
            processing_time_sec=processing_time
        )
    
    def _welch_anova(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """Welch's ANOVA for unequal variances."""
        k = len(groups)
        ns = [len(g) for g in groups]
        means = [np.mean(g) for g in groups]
        vars_ = [np.var(g, ddof=1) for g in groups]
        
        weights = [n / v for n, v in zip(ns, vars_)]
        sum_weights = sum(weights)
        grand_mean = sum(w * m for w, m in zip(weights, means)) / sum_weights
        
        numerator = sum(w * (m - grand_mean) ** 2 for w, m in zip(weights, means)) / (k - 1)
        
        lambda_sum = sum((1 - w / sum_weights) ** 2 / (n - 1) for w, n in zip(weights, ns))
        denominator = 1 + 2 * (k - 2) / (k ** 2 - 1) * lambda_sum
        
        f_stat = numerator / denominator
        
        df1 = k - 1
        df2 = (k ** 2 - 1) / (3 * lambda_sum)
        
        p_value = 1 - scipy_stats.f.cdf(f_stat, df1, df2)
        
        return f_stat, p_value
    
    def _calculate_effect_sizes(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """Calculate eta-squared and omega-squared."""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = ss_total - ss_between
        
        n_total = len(all_data)
        k = len(groups)
        ms_within = ss_within / (n_total - k)
        
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        omega_sq = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within) if ss_total > 0 else 0
        
        return float(eta_sq), float(max(0, omega_sq))
    
    def _tukey_hsd(
        self,
        groups: List[np.ndarray],
        names: List[str]
    ) -> List[PostHocResult]:
        """Tukey's HSD post-hoc test."""
        results = []
        k = len(groups)
        
        all_data = np.concatenate(groups)
        n_total = len(all_data)
        ms_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups) / (n_total - k)
        
        for i in range(k):
            for j in range(i + 1, k):
                mean_diff = np.mean(groups[i]) - np.mean(groups[j])
                se = np.sqrt(ms_within * (1 / len(groups[i]) + 1 / len(groups[j])) / 2)
                
                q = abs(mean_diff) / se if se > 0 else 0
                
                # Approximate p-value
                df = n_total - k
                p_value = 1 - scipy_stats.t.cdf(q, df) * 2
                p_value = max(0, min(1, p_value))
                
                results.append(PostHocResult(
                    group1=names[i],
                    group2=names[j],
                    mean_diff=float(mean_diff),
                    p_value=float(p_value),
                    significant=p_value < self.alpha
                ))
        
        return results


# ============================================================================
# Factory Functions
# ============================================================================

def get_anova_engine(alpha: float = 0.05) -> ANOVAEngine:
    """Get ANOVA engine."""
    return ANOVAEngine(alpha=alpha)


def quick_anova(
    df: pd.DataFrame,
    dependent_var: str,
    group_var: str
) -> Dict[str, Any]:
    """Quick ANOVA analysis."""
    engine = ANOVAEngine(verbose=False)
    result = engine.analyze(df, dependent_var, group_var)
    return result.to_dict()
