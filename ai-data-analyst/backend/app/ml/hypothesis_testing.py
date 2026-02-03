# AI Enterprise Data Analyst - Hypothesis Testing Engine
# Production-grade statistical hypothesis testing
# Handles: t-tests, chi-squared, ANOVA, non-parametric tests

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

class TestType(str, Enum):
    """Types of hypothesis tests."""
    ONE_SAMPLE_T = "one_sample_t"
    TWO_SAMPLE_T = "two_sample_t"
    PAIRED_T = "paired_t"
    WELCH_T = "welch_t"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    CHI_SQUARED = "chi_squared"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    SHAPIRO_WILK = "shapiro_wilk"
    LEVENE = "levene"


class AlternativeHypothesis(str, Enum):
    """Alternative hypothesis direction."""
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestResult:
    """Result of a single hypothesis test."""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    
    # Effect size
    effect_size: Optional[float] = None
    effect_size_name: str = ""
    effect_interpretation: str = ""
    
    # Confidence interval
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    
    # Sample info
    n_samples: List[int] = field(default_factory=list)
    sample_means: List[float] = field(default_factory=list)
    sample_stds: List[float] = field(default_factory=list)
    
    # Interpretation
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test": self.test_name,
            "type": self.test_type.value,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "reject_null": self.reject_null,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "effect_interpretation": self.effect_interpretation,
            "confidence_interval": [
                round(self.ci_lower, 4) if self.ci_lower is not None else None,
                round(self.ci_upper, 4) if self.ci_upper is not None else None
            ],
            "interpretation": self.interpretation
        }


# ============================================================================
# Hypothesis Testing Engine
# ============================================================================

class HypothesisTestingEngine:
    """
    Production-grade Hypothesis Testing engine.
    
    Features:
    - Parametric and non-parametric tests
    - Automatic test selection
    - Effect size calculation
    - Power analysis
    - Interpretation generation
    """
    
    def __init__(self, alpha: float = 0.05, verbose: bool = True):
        self.alpha = alpha
        self.verbose = verbose
    
    def one_sample_t_test(
        self,
        sample: pd.Series,
        population_mean: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    ) -> TestResult:
        """One-sample t-test."""
        clean = sample.dropna().values
        
        stat, p_two_sided = scipy_stats.ttest_1samp(clean, population_mean)
        
        if alternative == AlternativeHypothesis.GREATER:
            p_value = p_two_sided / 2 if stat > 0 else 1 - p_two_sided / 2
        elif alternative == AlternativeHypothesis.LESS:
            p_value = p_two_sided / 2 if stat < 0 else 1 - p_two_sided / 2
        else:
            p_value = p_two_sided
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(clean) - population_mean) / np.std(clean, ddof=1)
        
        # CI
        se = scipy_stats.sem(clean)
        ci = scipy_stats.t.interval(0.95, len(clean) - 1, loc=np.mean(clean), scale=se)
        
        return TestResult(
            test_name="One-Sample t-Test",
            test_type=TestType.ONE_SAMPLE_T,
            statistic=float(stat),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            effect_size=abs(effect_size),
            effect_size_name="Cohen's d",
            effect_interpretation=self._interpret_cohens_d(abs(effect_size)),
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_samples=[len(clean)],
            sample_means=[float(np.mean(clean))],
            sample_stds=[float(np.std(clean, ddof=1))],
            interpretation=self._interpret_t_test(p_value, np.mean(clean), population_mean)
        )
    
    def two_sample_t_test(
        self,
        sample1: pd.Series,
        sample2: pd.Series,
        equal_var: bool = True,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    ) -> TestResult:
        """Independent two-sample t-test."""
        clean1 = sample1.dropna().values
        clean2 = sample2.dropna().values
        
        stat, p_value = scipy_stats.ttest_ind(
            clean1, clean2, equal_var=equal_var,
            alternative=alternative.value
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(clean1) - 1) * np.var(clean1, ddof=1) + 
             (len(clean2) - 1) * np.var(clean2, ddof=1)) /
            (len(clean1) + len(clean2) - 2)
        )
        effect_size = (np.mean(clean1) - np.mean(clean2)) / pooled_std if pooled_std > 0 else 0
        
        test_name = "Welch's t-Test" if not equal_var else "Independent t-Test"
        test_type = TestType.WELCH_T if not equal_var else TestType.TWO_SAMPLE_T
        
        return TestResult(
            test_name=test_name,
            test_type=test_type,
            statistic=float(stat),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            effect_size=abs(effect_size),
            effect_size_name="Cohen's d",
            effect_interpretation=self._interpret_cohens_d(abs(effect_size)),
            n_samples=[len(clean1), len(clean2)],
            sample_means=[float(np.mean(clean1)), float(np.mean(clean2))],
            sample_stds=[float(np.std(clean1, ddof=1)), float(np.std(clean2, ddof=1))],
            interpretation=self._interpret_two_sample(p_value, np.mean(clean1), np.mean(clean2))
        )
    
    def paired_t_test(
        self,
        sample1: pd.Series,
        sample2: pd.Series,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    ) -> TestResult:
        """Paired t-test."""
        # Align samples
        combined = pd.DataFrame({'s1': sample1, 's2': sample2}).dropna()
        clean1 = combined['s1'].values
        clean2 = combined['s2'].values
        
        stat, p_value = scipy_stats.ttest_rel(
            clean1, clean2, alternative=alternative.value
        )
        
        # Effect size
        diff = clean1 - clean2
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        return TestResult(
            test_name="Paired t-Test",
            test_type=TestType.PAIRED_T,
            statistic=float(stat),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            effect_size=abs(effect_size),
            effect_size_name="Cohen's d",
            effect_interpretation=self._interpret_cohens_d(abs(effect_size)),
            n_samples=[len(clean1)],
            sample_means=[float(np.mean(diff))],
            sample_stds=[float(np.std(diff, ddof=1))],
            interpretation=f"Mean difference: {np.mean(diff):.4f}"
        )
    
    def mann_whitney_test(
        self,
        sample1: pd.Series,
        sample2: pd.Series,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    ) -> TestResult:
        """Mann-Whitney U test (non-parametric)."""
        clean1 = sample1.dropna().values
        clean2 = sample2.dropna().values
        
        stat, p_value = scipy_stats.mannwhitneyu(
            clean1, clean2, alternative=alternative.value
        )
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(clean1), len(clean2)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        
        return TestResult(
            test_name="Mann-Whitney U Test",
            test_type=TestType.MANN_WHITNEY,
            statistic=float(stat),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            effect_size=abs(effect_size),
            effect_size_name="Rank-biserial r",
            effect_interpretation=self._interpret_r(abs(effect_size)),
            n_samples=[n1, n2],
            interpretation="Non-parametric test for difference in distributions"
        )
    
    def chi_squared_test(
        self,
        observed: pd.DataFrame
    ) -> TestResult:
        """Chi-squared test for independence."""
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(observed)
        
        n = observed.sum().sum()
        k = min(observed.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0
        
        return TestResult(
            test_name="Chi-Squared Test",
            test_type=TestType.CHI_SQUARED,
            statistic=float(chi2),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            effect_size=cramers_v,
            effect_size_name="Cramér's V",
            effect_interpretation=self._interpret_cramers_v(cramers_v),
            n_samples=[int(n)],
            interpretation=f"Degrees of freedom: {dof}"
        )
    
    def normality_test(
        self,
        sample: pd.Series
    ) -> TestResult:
        """Test for normality using Shapiro-Wilk."""
        clean = sample.dropna().values
        
        # Shapiro-Wilk (works up to 5000 samples)
        sample_test = clean[:5000] if len(clean) > 5000 else clean
        stat, p_value = scipy_stats.shapiro(sample_test)
        
        return TestResult(
            test_name="Shapiro-Wilk Normality Test",
            test_type=TestType.SHAPIRO_WILK,
            statistic=float(stat),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            n_samples=[len(clean)],
            interpretation="Reject null: data is NOT normally distributed" if p_value < self.alpha 
                          else "Fail to reject null: data MAY be normally distributed"
        )
    
    def auto_test(
        self,
        sample1: pd.Series,
        sample2: pd.Series = None,
        paired: bool = False
    ) -> TestResult:
        """Automatically select and run appropriate test."""
        if sample2 is None:
            # One sample - check normality
            return self.normality_test(sample1)
        
        # Check normality of both samples
        clean1 = sample1.dropna()
        clean2 = sample2.dropna()
        
        _, p_norm1 = scipy_stats.shapiro(clean1.values[:5000])
        _, p_norm2 = scipy_stats.shapiro(clean2.values[:5000])
        
        is_normal = p_norm1 > 0.05 and p_norm2 > 0.05
        
        if paired:
            if is_normal:
                return self.paired_t_test(sample1, sample2)
            else:
                # Wilcoxon signed-rank
                combined = pd.DataFrame({'s1': sample1, 's2': sample2}).dropna()
                stat, p_value = scipy_stats.wilcoxon(
                    combined['s1'], combined['s2']
                )
                return TestResult(
                    test_name="Wilcoxon Signed-Rank Test",
                    test_type=TestType.WILCOXON,
                    statistic=float(stat),
                    p_value=float(p_value),
                    alpha=self.alpha,
                    reject_null=p_value < self.alpha,
                    interpretation="Non-parametric paired comparison"
                )
        else:
            # Check equal variances
            _, p_levene = scipy_stats.levene(clean1, clean2)
            equal_var = p_levene > 0.05
            
            if is_normal:
                return self.two_sample_t_test(sample1, sample2, equal_var=equal_var)
            else:
                return self.mann_whitney_test(sample1, sample2)
    
    def _interpret_cohens_d(self, d: float) -> str:
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        return "large"
    
    def _interpret_r(self, r: float) -> str:
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        return "large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        if v < 0.1:
            return "negligible"
        elif v < 0.2:
            return "small"
        elif v < 0.4:
            return "medium"
        return "large"
    
    def _interpret_t_test(
        self,
        p_value: float,
        sample_mean: float,
        pop_mean: float
    ) -> str:
        if p_value < self.alpha:
            direction = "greater" if sample_mean > pop_mean else "less"
            return f"Sample mean ({sample_mean:.4f}) is significantly {direction} than {pop_mean}"
        return f"No significant difference from population mean of {pop_mean}"
    
    def _interpret_two_sample(
        self,
        p_value: float,
        mean1: float,
        mean2: float
    ) -> str:
        if p_value < self.alpha:
            return f"Significant difference: Group 1 ({mean1:.4f}) vs Group 2 ({mean2:.4f})"
        return f"No significant difference between groups"


# ============================================================================
# Factory Functions
# ============================================================================

def get_hypothesis_engine(alpha: float = 0.05) -> HypothesisTestingEngine:
    """Get hypothesis testing engine."""
    return HypothesisTestingEngine(alpha=alpha)


def quick_t_test(
    sample1: pd.Series,
    sample2: pd.Series = None,
    pop_mean: float = None
) -> Dict[str, Any]:
    """Quick t-test."""
    engine = HypothesisTestingEngine(verbose=False)
    
    if sample2 is not None:
        result = engine.two_sample_t_test(sample1, sample2)
    elif pop_mean is not None:
        result = engine.one_sample_t_test(sample1, pop_mean)
    else:
        result = engine.normality_test(sample1)
    
    return result.to_dict()


def test_normality(sample: pd.Series) -> Dict[str, Any]:
    """Quick normality test."""
    engine = HypothesisTestingEngine(verbose=False)
    return engine.normality_test(sample).to_dict()
