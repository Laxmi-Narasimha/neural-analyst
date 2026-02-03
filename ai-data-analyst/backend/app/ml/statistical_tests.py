# AI Enterprise Data Analyst - Statistical Testing Engine
# Comprehensive hypothesis testing and effect size calculations

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Test Types
# ============================================================================

class TestType(str, Enum):
    """Statistical test types."""
    T_TEST = "t_test"
    WELCH_T = "welch_t"
    PAIRED_T = "paired_t"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    CORRELATION = "correlation"
    NORMALITY = "normality"
    LEVENE = "levene"


@dataclass
class TestResult:
    """Statistical test result."""
    
    test_type: TestType
    statistic: float
    p_value: float
    
    is_significant: bool = False
    alpha: float = 0.05
    
    effect_size: Optional[float] = None
    effect_interpretation: str = ""
    
    confidence_interval: Optional[tuple[float, float]] = None
    
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "test_type": self.test_type.value,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "effect_interpretation": self.effect_interpretation,
            "confidence_interval": self.confidence_interval,
            "details": self.details
        }


# ============================================================================
# Effect Size Calculators
# ============================================================================

class EffectSizeCalculator:
    """Calculate effect sizes for various tests."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d for two groups."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Hedges' g (corrected Cohen's d)."""
        n = len(group1) + len(group2)
        d = EffectSizeCalculator.cohens_d(group1, group2)
        
        # Correction factor
        correction = 1 - (3 / (4 * n - 9))
        return d * correction
    
    @staticmethod
    def cramers_v(contingency_table: np.ndarray) -> float:
        """Cramér's V for chi-square."""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        
        return np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0
    
    @staticmethod
    def eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
        """Eta-squared for ANOVA."""
        return (f_statistic * df_between) / (f_statistic * df_between + df_within)
    
    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def interpret_correlation(r: float) -> str:
        """Interpret correlation coefficient."""
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"


# ============================================================================
# Statistical Tests
# ============================================================================

class StatisticalTests:
    """Collection of statistical tests."""
    
    @staticmethod
    def t_test(
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
        equal_var: bool = True,
        alpha: float = 0.05
    ) -> TestResult:
        """Perform t-test."""
        if paired:
            stat, p = stats.ttest_rel(group1, group2)
            test_type = TestType.PAIRED_T
        elif equal_var:
            stat, p = stats.ttest_ind(group1, group2, equal_var=True)
            test_type = TestType.T_TEST
        else:
            stat, p = stats.ttest_ind(group1, group2, equal_var=False)
            test_type = TestType.WELCH_T
        
        # Effect size
        d = EffectSizeCalculator.cohens_d(group1, group2)
        
        # Confidence interval for mean difference
        diff = np.mean(group1) - np.mean(group2)
        se = np.sqrt(np.var(group1) / len(group1) + np.var(group2) / len(group2))
        ci = (diff - 1.96 * se, diff + 1.96 * se)
        
        return TestResult(
            test_type=test_type,
            statistic=float(stat),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            effect_size=float(d),
            effect_interpretation=EffectSizeCalculator.interpret_cohens_d(d),
            confidence_interval=ci,
            details={
                "mean1": float(np.mean(group1)),
                "mean2": float(np.mean(group2)),
                "n1": len(group1),
                "n2": len(group2)
            }
        )
    
    @staticmethod
    def anova(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> TestResult:
        """One-way ANOVA."""
        stat, p = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        df_between = len(groups) - 1
        df_within = sum(len(g) - 1 for g in groups)
        eta_sq = EffectSizeCalculator.eta_squared(stat, df_between, df_within)
        
        return TestResult(
            test_type=TestType.ANOVA,
            statistic=float(stat),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            effect_size=float(eta_sq),
            effect_interpretation="small" if eta_sq < 0.06 else "medium" if eta_sq < 0.14 else "large",
            details={
                "df_between": df_between,
                "df_within": df_within,
                "group_means": [float(np.mean(g)) for g in groups]
            }
        )
    
    @staticmethod
    def kruskal_wallis(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> TestResult:
        """Kruskal-Wallis H-test (non-parametric ANOVA)."""
        stat, p = stats.kruskal(*groups)
        
        return TestResult(
            test_type=TestType.KRUSKAL_WALLIS,
            statistic=float(stat),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            details={
                "group_medians": [float(np.median(g)) for g in groups]
            }
        )
    
    @staticmethod
    def mann_whitney(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05
    ) -> TestResult:
        """Mann-Whitney U test."""
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * stat) / (n1 * n2)
        
        return TestResult(
            test_type=TestType.MANN_WHITNEY,
            statistic=float(stat),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            effect_size=float(r),
            effect_interpretation=EffectSizeCalculator.interpret_correlation(r),
            details={
                "median1": float(np.median(group1)),
                "median2": float(np.median(group2))
            }
        )
    
    @staticmethod
    def chi_square(
        observed: np.ndarray,
        expected: np.ndarray = None,
        alpha: float = 0.05
    ) -> TestResult:
        """Chi-square test."""
        if observed.ndim == 2:
            # Contingency table
            chi2, p, dof, expected = stats.chi2_contingency(observed)
            v = EffectSizeCalculator.cramers_v(observed)
        else:
            # Goodness of fit
            if expected is None:
                expected = np.full_like(observed, observed.mean())
            chi2, p = stats.chisquare(observed, expected)
            dof = len(observed) - 1
            v = None
        
        return TestResult(
            test_type=TestType.CHI_SQUARE,
            statistic=float(chi2),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            effect_size=float(v) if v is not None else None,
            effect_interpretation="small" if v and v < 0.1 else "medium" if v and v < 0.3 else "large" if v else "",
            details={"dof": dof}
        )
    
    @staticmethod
    def correlation(
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson",
        alpha: float = 0.05
    ) -> TestResult:
        """Correlation test."""
        if method == "pearson":
            r, p = stats.pearsonr(x, y)
        elif method == "spearman":
            r, p = stats.spearmanr(x, y)
        else:
            r, p = stats.kendalltau(x, y)
        
        return TestResult(
            test_type=TestType.CORRELATION,
            statistic=float(r),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            effect_size=float(r),
            effect_interpretation=EffectSizeCalculator.interpret_correlation(r),
            details={"method": method}
        )
    
    @staticmethod
    def normality_test(
        data: np.ndarray,
        method: str = "shapiro",
        alpha: float = 0.05
    ) -> TestResult:
        """Test for normality."""
        if method == "shapiro":
            stat, p = stats.shapiro(data[:5000])  # Shapiro limited to 5000
        elif method == "dagostino":
            stat, p = stats.normaltest(data)
        else:
            stat, p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        
        return TestResult(
            test_type=TestType.NORMALITY,
            statistic=float(stat),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            details={
                "method": method,
                "is_normal": p >= alpha
            }
        )
    
    @staticmethod
    def levene_test(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> TestResult:
        """Test for equality of variances."""
        stat, p = stats.levene(*groups)
        
        return TestResult(
            test_type=TestType.LEVENE,
            statistic=float(stat),
            p_value=float(p),
            is_significant=p < alpha,
            alpha=alpha,
            details={
                "equal_variance": p >= alpha,
                "variances": [float(np.var(g)) for g in groups]
            }
        )


# ============================================================================
# Statistical Testing Engine
# ============================================================================

class StatisticalTestingEngine:
    """
    Unified statistical testing engine.
    
    Features:
    - Parametric tests (t-test, ANOVA)
    - Non-parametric tests (Mann-Whitney, Kruskal-Wallis)
    - Correlation analysis
    - Chi-square tests
    - Effect size calculations
    - Assumption checking
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.tests = StatisticalTests
        self.effect_size = EffectSizeCalculator
    
    def compare_groups(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        groups: list = None
    ) -> dict[str, Any]:
        """Compare groups with automatic test selection."""
        if groups:
            data = df[df[group_col].isin(groups)]
        else:
            data = df
        
        group_values = [
            data[data[group_col] == g][value_col].dropna().values
            for g in data[group_col].unique()
        ]
        
        n_groups = len(group_values)
        
        # Check assumptions
        normal_tests = [self.tests.normality_test(g) for g in group_values if len(g) > 3]
        all_normal = all(t.details.get("is_normal", False) for t in normal_tests)
        
        variance_test = self.tests.levene_test(*group_values) if len(group_values) > 1 else None
        equal_var = variance_test.details.get("equal_variance", True) if variance_test else True
        
        # Select and run appropriate test
        if n_groups == 2:
            if all_normal:
                main_test = self.tests.t_test(
                    group_values[0], group_values[1], 
                    equal_var=equal_var, alpha=self.alpha
                )
            else:
                main_test = self.tests.mann_whitney(
                    group_values[0], group_values[1], alpha=self.alpha
                )
        else:
            if all_normal:
                main_test = self.tests.anova(*group_values, alpha=self.alpha)
            else:
                main_test = self.tests.kruskal_wallis(*group_values, alpha=self.alpha)
        
        return {
            "main_test": main_test.to_dict(),
            "assumptions": {
                "normality": [t.to_dict() for t in normal_tests],
                "equal_variance": variance_test.to_dict() if variance_test else None
            },
            "test_selection_reason": "parametric" if all_normal else "non-parametric"
        }
    
    def correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
        method: str = "pearson"
    ) -> dict[str, Any]:
        """Calculate correlation matrix with significance."""
        if columns:
            data = df[columns]
        else:
            data = df.select_dtypes(include=[np.number])
        
        n = len(data.columns)
        correlations = {}
        
        for i, col1 in enumerate(data.columns):
            for col2 in data.columns[i + 1:]:
                result = self.tests.correlation(
                    data[col1].dropna().values,
                    data[col2].dropna().values,
                    method=method,
                    alpha=self.alpha
                )
                correlations[f"{col1}_vs_{col2}"] = result.to_dict()
        
        return {
            "correlations": correlations,
            "matrix": data.corr(method=method).to_dict()
        }
    
    def run_test(
        self,
        test_type: TestType,
        data1: np.ndarray,
        data2: np.ndarray = None,
        **kwargs
    ) -> TestResult:
        """Run specific statistical test."""
        alpha = kwargs.get("alpha", self.alpha)
        
        if test_type == TestType.T_TEST:
            return self.tests.t_test(data1, data2, alpha=alpha)
        elif test_type == TestType.WELCH_T:
            return self.tests.t_test(data1, data2, equal_var=False, alpha=alpha)
        elif test_type == TestType.MANN_WHITNEY:
            return self.tests.mann_whitney(data1, data2, alpha=alpha)
        elif test_type == TestType.NORMALITY:
            return self.tests.normality_test(data1, alpha=alpha)
        elif test_type == TestType.CORRELATION:
            return self.tests.correlation(data1, data2, alpha=alpha)
        elif test_type == TestType.CHI_SQUARE:
            return self.tests.chi_square(data1, data2, alpha=alpha)
        else:
            raise ValueError(f"Unknown test type: {test_type}")


# Factory function
def get_statistical_testing_engine(alpha: float = 0.05) -> StatisticalTestingEngine:
    """Get statistical testing engine instance."""
    return StatisticalTestingEngine(alpha)
