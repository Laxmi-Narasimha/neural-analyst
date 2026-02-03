# AI Enterprise Data Analyst - Advanced Statistical Tests Engine
# Auto-selects appropriate tests based on data characteristics
# Handles: any sample size, non-normal data, missing values, outliers

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Test Types and Enums
# ============================================================================

class TestCategory(str, Enum):
    """Categories of statistical tests."""
    NORMALITY = "normality"
    COMPARISON = "comparison"
    CORRELATION = "correlation"
    VARIANCE = "variance"
    INDEPENDENCE = "independence"
    DISTRIBUTION = "distribution"


class TestType(str, Enum):
    """Specific test types."""
    # Normality tests
    SHAPIRO_WILK = "shapiro_wilk"
    DAGOSTINO = "dagostino"
    ANDERSON_DARLING = "anderson_darling"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    JARQUE_BERA = "jarque_bera"
    
    # Comparison tests - Parametric
    TTEST_INDEPENDENT = "ttest_independent"
    TTEST_PAIRED = "ttest_paired"
    TTEST_ONE_SAMPLE = "ttest_one_sample"
    WELCH_TTEST = "welch_ttest"
    ANOVA_ONE_WAY = "anova_one_way"
    ANOVA_TWO_WAY = "anova_two_way"
    
    # Comparison tests - Non-parametric
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    
    # Correlation tests
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    POINT_BISERIAL = "point_biserial"
    
    # Variance tests
    LEVENE = "levene"
    BARTLETT = "bartlett"
    BROWN_FORSYTHE = "brown_forsythe"
    
    # Independence tests
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    CRAMERS_V = "cramers_v"
    
    # Distribution tests
    KS_TWO_SAMPLE = "ks_two_sample"


class EffectSizeType(str, Enum):
    """Effect size measures."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    COHENS_R = "cohens_r"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CRAMERS_V = "cramers_v"


# ============================================================================
# Result Data Classes
# ============================================================================

@dataclass
class TestResult:
    """Result from a statistical test."""
    test_type: TestType
    test_name: str
    
    # Core results
    statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    significance_level: float = 0.05
    
    # Effect size
    effect_size: Optional[float] = None
    effect_size_type: Optional[EffectSizeType] = None
    effect_size_interpretation: str = ""
    
    # Confidence interval
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence_level: float = 0.95
    
    # Power
    power: Optional[float] = None
    
    # Additional info
    degrees_of_freedom: Optional[float] = None
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    descriptive_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Assumptions
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    assumptions_warnings: List[str] = field(default_factory=list)
    
    # Interpretation
    interpretation: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_type": self.test_type.value,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "significance_level": self.significance_level,
            "effect_size": {
                "value": round(self.effect_size, 4) if self.effect_size else None,
                "type": self.effect_size_type.value if self.effect_size_type else None,
                "interpretation": self.effect_size_interpretation
            },
            "confidence_interval": {
                "lower": round(self.ci_lower, 4) if self.ci_lower else None,
                "upper": round(self.ci_upper, 4) if self.ci_upper else None,
                "level": self.confidence_level
            },
            "degrees_of_freedom": self.degrees_of_freedom,
            "sample_sizes": self.sample_sizes,
            "descriptive_stats": self.descriptive_stats,
            "assumptions": {
                "met": self.assumptions_met,
                "warnings": self.assumptions_warnings
            },
            "interpretation": self.interpretation,
            "recommendation": self.recommendation
        }


@dataclass
class MultiTestResult:
    """Result from multiple related tests."""
    primary_test: TestResult
    alternative_tests: List[TestResult] = field(default_factory=list)
    recommendation: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_test": self.primary_test.to_dict(),
            "alternative_tests": [t.to_dict() for t in self.alternative_tests],
            "recommendation": self.recommendation,
            "warnings": self.warnings
        }


# ============================================================================
# Data Analyzer
# ============================================================================

class DataCharacteristicsAnalyzer:
    """Analyze data characteristics to select appropriate tests."""
    
    def analyze_single(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single variable."""
        clean = series.dropna()
        n = len(clean)
        
        result = {
            "n": n,
            "n_missing": len(series) - n,
            "dtype": str(series.dtype),
            "is_numeric": pd.api.types.is_numeric_dtype(series),
            "is_categorical": series.dtype == 'object' or pd.api.types.is_categorical_dtype(series),
            "is_binary": False,
            "is_normal": None,
            "has_outliers": False,
            "n_unique": int(series.nunique())
        }
        
        if result["is_numeric"] and n >= 3:
            # Normality test
            if n >= 8:
                if n <= 5000:
                    _, p_shapiro = scipy_stats.shapiro(clean)
                    result["is_normal"] = p_shapiro > 0.05
                else:
                    # Use D'Agostino for large samples
                    _, p_dagostino = scipy_stats.normaltest(clean)
                    result["is_normal"] = p_dagostino > 0.05
            
            # Outliers
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
            result["has_outliers"] = outlier_count > n * 0.05
            
            # Descriptive stats
            result["mean"] = float(clean.mean())
            result["std"] = float(clean.std())
            result["median"] = float(clean.median())
            result["skewness"] = float(scipy_stats.skew(clean))
            result["kurtosis"] = float(scipy_stats.kurtosis(clean))
        
        # Binary check
        if result["n_unique"] == 2:
            result["is_binary"] = True
        
        return result
    
    def analyze_pair(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Dict[str, Any]:
        """Analyze a pair of variables."""
        char1 = self.analyze_single(series1)
        char2 = self.analyze_single(series2)
        
        # Check if paired (same length and index)
        is_paired = len(series1) == len(series2) and series1.index.equals(series2.index)
        
        # Check homogeneity of variance if both numeric
        homogeneous_variance = None
        if char1["is_numeric"] and char2["is_numeric"]:
            clean1 = series1.dropna()
            clean2 = series2.dropna()
            if len(clean1) >= 3 and len(clean2) >= 3:
                _, p_levene = scipy_stats.levene(clean1, clean2)
                homogeneous_variance = p_levene > 0.05
        
        return {
            "variable1": char1,
            "variable2": char2,
            "is_paired": is_paired,
            "both_numeric": char1["is_numeric"] and char2["is_numeric"],
            "both_normal": char1["is_normal"] and char2["is_normal"] if char1["is_normal"] is not None else None,
            "homogeneous_variance": homogeneous_variance
        }


# ============================================================================
# Test Implementations
# ============================================================================

class StatisticalTests:
    """Collection of statistical test implementations."""
    
    # -------------------------------------------------------------------------
    # Normality Tests
    # -------------------------------------------------------------------------
    
    @staticmethod
    def shapiro_wilk(data: np.ndarray) -> Tuple[float, float]:
        """Shapiro-Wilk normality test."""
        if len(data) < 3:
            return 0.0, 1.0
        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)
        return scipy_stats.shapiro(data)
    
    @staticmethod
    def dagostino(data: np.ndarray) -> Tuple[float, float]:
        """D'Agostino normality test."""
        if len(data) < 8:
            return 0.0, 1.0
        return scipy_stats.normaltest(data)
    
    @staticmethod
    def anderson_darling(data: np.ndarray) -> Tuple[float, List[float], List[float]]:
        """Anderson-Darling normality test."""
        result = scipy_stats.anderson(data, dist='norm')
        return result.statistic, result.critical_values.tolist(), result.significance_level.tolist()
    
    @staticmethod
    def jarque_bera(data: np.ndarray) -> Tuple[float, float]:
        """Jarque-Bera normality test."""
        return scipy_stats.jarque_bera(data)
    
    # -------------------------------------------------------------------------
    # Comparison Tests - Two Groups
    # -------------------------------------------------------------------------
    
    @staticmethod
    def ttest_independent(
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = True
    ) -> Tuple[float, float]:
        """Independent samples t-test."""
        return scipy_stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    @staticmethod
    def ttest_paired(
        before: np.ndarray,
        after: np.ndarray
    ) -> Tuple[float, float]:
        """Paired samples t-test."""
        return scipy_stats.ttest_rel(before, after)
    
    @staticmethod
    def ttest_one_sample(
        data: np.ndarray,
        popmean: float
    ) -> Tuple[float, float]:
        """One-sample t-test."""
        return scipy_stats.ttest_1samp(data, popmean)
    
    @staticmethod
    def mann_whitney(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float]:
        """Mann-Whitney U test (non-parametric)."""
        return scipy_stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    @staticmethod
    def wilcoxon(
        before: np.ndarray,
        after: np.ndarray
    ) -> Tuple[float, float]:
        """Wilcoxon signed-rank test (non-parametric paired)."""
        diff = before - after
        non_zero = diff[diff != 0]
        if len(non_zero) < 6:
            # Too few non-zero differences
            return 0.0, 1.0
        return scipy_stats.wilcoxon(before, after)
    
    # -------------------------------------------------------------------------
    # Comparison Tests - Multiple Groups
    # -------------------------------------------------------------------------
    
    @staticmethod
    def anova_one_way(*groups) -> Tuple[float, float]:
        """One-way ANOVA."""
        return scipy_stats.f_oneway(*groups)
    
    @staticmethod
    def kruskal_wallis(*groups) -> Tuple[float, float]:
        """Kruskal-Wallis H test (non-parametric ANOVA)."""
        return scipy_stats.kruskal(*groups)
    
    # -------------------------------------------------------------------------
    # Correlation Tests
    # -------------------------------------------------------------------------
    
    @staticmethod
    def pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Pearson correlation coefficient."""
        return scipy_stats.pearsonr(x, y)
    
    @staticmethod
    def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Spearman rank correlation."""
        return scipy_stats.spearmanr(x, y)
    
    @staticmethod
    def kendall(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Kendall's tau correlation."""
        return scipy_stats.kendalltau(x, y)
    
    @staticmethod
    def point_biserial(binary: np.ndarray, continuous: np.ndarray) -> Tuple[float, float]:
        """Point-biserial correlation."""
        return scipy_stats.pointbiserialr(binary, continuous)
    
    # -------------------------------------------------------------------------
    # Variance Tests
    # -------------------------------------------------------------------------
    
    @staticmethod
    def levene(*groups, center: str = 'median') -> Tuple[float, float]:
        """Levene's test for equality of variances."""
        return scipy_stats.levene(*groups, center=center)
    
    @staticmethod
    def bartlett(*groups) -> Tuple[float, float]:
        """Bartlett's test for equality of variances."""
        return scipy_stats.bartlett(*groups)
    
    # -------------------------------------------------------------------------
    # Independence Tests
    # -------------------------------------------------------------------------
    
    @staticmethod
    def chi_square(contingency_table: np.ndarray) -> Tuple[float, float, int, np.ndarray]:
        """Chi-square test of independence."""
        return scipy_stats.chi2_contingency(contingency_table)
    
    @staticmethod
    def fisher_exact(contingency_table: np.ndarray) -> Tuple[float, float]:
        """Fisher's exact test (2x2 tables only)."""
        if contingency_table.shape == (2, 2):
            return scipy_stats.fisher_exact(contingency_table)
        return 0.0, 1.0


# ============================================================================
# Effect Size Calculators
# ============================================================================

class EffectSizeCalculator:
    """Calculate effect sizes for various tests."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d for independent samples."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Hedges' g (bias-corrected Cohen's d)."""
        d = EffectSizeCalculator.cohens_d(group1, group2)
        n = len(group1) + len(group2)
        
        # Correction factor
        correction = 1 - (3 / (4 * n - 9))
        
        return d * correction
    
    @staticmethod
    def eta_squared(ss_between: float, ss_total: float) -> float:
        """Eta-squared for ANOVA."""
        if ss_total == 0:
            return 0.0
        return ss_between / ss_total
    
    @staticmethod
    def cramers_v(chi2: float, n: int, min_dim: int) -> float:
        """Cramer's V for chi-square."""
        if n == 0 or min_dim <= 1:
            return 0.0
        return np.sqrt(chi2 / (n * (min_dim - 1)))
    
    @staticmethod
    def cohens_r_from_z(z: float, n: int) -> float:
        """Cohen's r from z-score."""
        return z / np.sqrt(n)
    
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
    def interpret_r(r: float) -> str:
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
# Smart Statistical Test Engine
# ============================================================================

class SmartStatisticalEngine:
    """
    Intelligent statistical testing engine.
    
    Features:
    - Auto-selects appropriate test based on data
    - Checks assumptions automatically
    - Provides non-parametric alternatives when needed
    - Calculates effect sizes
    - Generates interpretations
    """
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.analyzer = DataCharacteristicsAnalyzer()
    
    def compare_two_groups(
        self,
        group1: pd.Series,
        group2: pd.Series,
        paired: bool = False,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> MultiTestResult:
        """
        Compare two groups - automatically selects appropriate test.
        
        Handles:
        - Normal vs non-normal data
        - Equal vs unequal variances
        - Paired vs independent samples
        - Small vs large sample sizes
        """
        # Clean data
        clean1 = group1.dropna().values
        clean2 = group2.dropna().values
        
        if len(clean1) < 2 or len(clean2) < 2:
            raise ValidationException("Each group needs at least 2 observations")
        
        # Analyze characteristics
        char = self.analyzer.analyze_pair(group1, group2)
        
        warnings = []
        alternative_tests = []
        
        # Check assumptions
        is_normal1 = char["variable1"]["is_normal"]
        is_normal2 = char["variable2"]["is_normal"]
        both_normal = is_normal1 and is_normal2 if is_normal1 is not None else None
        homogeneous_var = char["homogeneous_variance"]
        
        # Small sample warning
        if len(clean1) < 30 or len(clean2) < 30:
            warnings.append("Small sample size may affect reliability")
        
        # Select test
        if paired:
            # Paired comparison
            if len(clean1) != len(clean2):
                raise ValidationException("Paired samples must have same length")
            
            if both_normal:
                # Parametric: paired t-test
                primary_test = self._run_paired_ttest(clean1, clean2, group1_name, group2_name)
                alt = self._run_wilcoxon(clean1, clean2, group1_name, group2_name)
                alternative_tests.append(alt)
            else:
                # Non-parametric: Wilcoxon
                primary_test = self._run_wilcoxon(clean1, clean2, group1_name, group2_name)
                if both_normal is not False:
                    alt = self._run_paired_ttest(clean1, clean2, group1_name, group2_name)
                    alternative_tests.append(alt)
        else:
            # Independent comparison
            if both_normal:
                if homogeneous_var:
                    # Standard t-test
                    primary_test = self._run_independent_ttest(
                        clean1, clean2, group1_name, group2_name, equal_var=True
                    )
                else:
                    # Welch's t-test
                    primary_test = self._run_independent_ttest(
                        clean1, clean2, group1_name, group2_name, equal_var=False
                    )
                    primary_test.assumptions_warnings.append("Unequal variances - using Welch's t-test")
                
                alt = self._run_mann_whitney(clean1, clean2, group1_name, group2_name)
                alternative_tests.append(alt)
            else:
                # Non-parametric: Mann-Whitney
                primary_test = self._run_mann_whitney(clean1, clean2, group1_name, group2_name)
                if both_normal is not False:
                    alt = self._run_independent_ttest(
                        clean1, clean2, group1_name, group2_name, equal_var=True
                    )
                    alternative_tests.append(alt)
        
        # Add assumption info
        primary_test.assumptions_met = {
            "normality_group1": is_normal1 if is_normal1 is not None else "not tested",
            "normality_group2": is_normal2 if is_normal2 is not None else "not tested",
            "homogeneous_variance": homogeneous_var if homogeneous_var is not None else "not tested"
        }
        
        # Recommendation
        if primary_test.is_significant:
            rec = f"There is a statistically significant difference between {group1_name} and {group2_name}."
        else:
            rec = f"No statistically significant difference found between {group1_name} and {group2_name}."
        
        return MultiTestResult(
            primary_test=primary_test,
            alternative_tests=alternative_tests,
            recommendation=rec,
            warnings=warnings
        )
    
    def compare_multiple_groups(
        self,
        *groups: pd.Series,
        group_names: List[str] = None
    ) -> MultiTestResult:
        """Compare multiple groups (3+)."""
        if len(groups) < 3:
            raise ValidationException("Need at least 3 groups for multiple comparison")
        
        # Clean data
        clean_groups = [g.dropna().values for g in groups]
        
        if any(len(g) < 2 for g in clean_groups):
            raise ValidationException("Each group needs at least 2 observations")
        
        group_names = group_names or [f"Group {i+1}" for i in range(len(groups))]
        warnings = []
        alternative_tests = []
        
        # Check normality for all groups
        normality_results = []
        for g in clean_groups:
            if len(g) >= 8:
                _, p = scipy_stats.shapiro(g) if len(g) <= 5000 else scipy_stats.normaltest(g)
                normality_results.append(p > 0.05)
            else:
                normality_results.append(None)
        
        all_normal = all(n for n in normality_results if n is not None)
        
        # Check homogeneity of variances
        _, p_levene = StatisticalTests.levene(*clean_groups)
        homogeneous = p_levene > 0.05
        
        if all_normal and homogeneous:
            # ANOVA
            primary_test = self._run_anova(clean_groups, group_names)
            alt = self._run_kruskal_wallis(clean_groups, group_names)
            alternative_tests.append(alt)
        else:
            # Kruskal-Wallis
            primary_test = self._run_kruskal_wallis(clean_groups, group_names)
            alt = self._run_anova(clean_groups, group_names)
            alternative_tests.append(alt)
        
        # Recommendation
        if primary_test.is_significant:
            rec = "Significant difference exists among groups. Consider post-hoc tests."
        else:
            rec = "No significant difference found among groups."
        
        return MultiTestResult(
            primary_test=primary_test,
            alternative_tests=alternative_tests,
            recommendation=rec,
            warnings=warnings
        )
    
    def test_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        x_name: str = "X",
        y_name: str = "Y"
    ) -> MultiTestResult:
        """Test correlation between two variables."""
        # Align and clean
        df = pd.DataFrame({x_name: x, y_name: y}).dropna()
        clean_x = df[x_name].values
        clean_y = df[y_name].values
        
        if len(clean_x) < 3:
            raise ValidationException("Need at least 3 paired observations")
        
        warnings = []
        alternative_tests = []
        
        # Check characteristics
        char_x = self.analyzer.analyze_single(pd.Series(clean_x))
        char_y = self.analyzer.analyze_single(pd.Series(clean_y))
        
        # Check if one is binary
        if char_x["is_binary"] or char_y["is_binary"]:
            if char_x["is_binary"]:
                binary = clean_x
                continuous = clean_y
            else:
                binary = clean_y
                continuous = clean_x
            
            primary_test = self._run_point_biserial(binary, continuous, x_name, y_name)
        elif char_x["is_normal"] and char_y["is_normal"]:
            primary_test = self._run_pearson(clean_x, clean_y, x_name, y_name)
            alt = self._run_spearman(clean_x, clean_y, x_name, y_name)
            alternative_tests.append(alt)
        else:
            primary_test = self._run_spearman(clean_x, clean_y, x_name, y_name)
            alt = self._run_pearson(clean_x, clean_y, x_name, y_name)
            alternative_tests.append(alt)
        
        # Recommendation
        if primary_test.is_significant:
            direction = "positive" if primary_test.statistic > 0 else "negative"
            strength = EffectSizeCalculator.interpret_r(primary_test.statistic)
            rec = f"Significant {strength} {direction} correlation between {x_name} and {y_name}."
        else:
            rec = f"No significant correlation between {x_name} and {y_name}."
        
        return MultiTestResult(
            primary_test=primary_test,
            alternative_tests=alternative_tests,
            recommendation=rec,
            warnings=warnings
        )
    
    def test_independence(
        self,
        var1: pd.Series,
        var2: pd.Series,
        var1_name: str = "Variable 1",
        var2_name: str = "Variable 2"
    ) -> MultiTestResult:
        """Test independence of two categorical variables."""
        # Create contingency table
        df = pd.DataFrame({var1_name: var1, var2_name: var2}).dropna()
        contingency = pd.crosstab(df[var1_name], df[var2_name])
        
        warnings = []
        alternative_tests = []
        
        # Check expected frequencies
        chi2, p_chi, dof, expected = StatisticalTests.chi_square(contingency.values)
        
        # Check for low expected counts
        low_expected = (expected < 5).sum() / expected.size
        if low_expected > 0.2:
            warnings.append(f"{low_expected*100:.1f}% cells have expected count < 5")
        
        # For 2x2 tables, use Fisher's exact if needed
        if contingency.shape == (2, 2) and expected.min() < 5:
            primary_test = self._run_fisher_exact(contingency.values, var1_name, var2_name)
            alt = self._run_chi_square(contingency.values, var1_name, var2_name)
            alternative_tests.append(alt)
        else:
            primary_test = self._run_chi_square(contingency.values, var1_name, var2_name)
            if contingency.shape == (2, 2):
                alt = self._run_fisher_exact(contingency.values, var1_name, var2_name)
                alternative_tests.append(alt)
        
        # Recommendation
        if primary_test.is_significant:
            rec = f"{var1_name} and {var2_name} are not independent (associated)."
        else:
            rec = f"No significant association between {var1_name} and {var2_name}."
        
        return MultiTestResult(
            primary_test=primary_test,
            alternative_tests=alternative_tests,
            recommendation=rec,
            warnings=warnings
        )
    
    def test_normality(self, data: pd.Series, name: str = "Data") -> TestResult:
        """Test normality of a single variable."""
        clean = data.dropna().values
        
        if len(clean) < 3:
            raise ValidationException("Need at least 3 observations")
        
        # Select appropriate test
        if len(clean) <= 5000:
            stat, p = StatisticalTests.shapiro_wilk(clean)
            test_type = TestType.SHAPIRO_WILK
            test_name = "Shapiro-Wilk Test"
        else:
            stat, p = StatisticalTests.dagostino(clean)
            test_type = TestType.DAGOSTINO
            test_name = "D'Agostino-Pearson Test"
        
        is_sig = p < self.alpha
        
        # Descriptive stats
        desc = {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "skewness": float(scipy_stats.skew(clean)),
            "kurtosis": float(scipy_stats.kurtosis(clean))
        }
        
        interp = f"{name} is {'not ' if is_sig else ''}normally distributed (p={p:.4f})."
        
        return TestResult(
            test_type=test_type,
            test_name=test_name,
            statistic=stat,
            p_value=p,
            is_significant=is_sig,
            significance_level=self.alpha,
            sample_sizes={name: len(clean)},
            descriptive_stats={name: desc},
            interpretation=interp,
            recommendation="Use non-parametric tests" if is_sig else "Parametric tests appropriate"
        )
    
    # -------------------------------------------------------------------------
    # Private Test Runners
    # -------------------------------------------------------------------------
    
    def _run_independent_ttest(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        name1: str,
        name2: str,
        equal_var: bool = True
    ) -> TestResult:
        stat, p = StatisticalTests.ttest_independent(g1, g2, equal_var=equal_var)
        d = EffectSizeCalculator.cohens_d(g1, g2)
        
        return TestResult(
            test_type=TestType.TTEST_INDEPENDENT if equal_var else TestType.WELCH_TTEST,
            test_name="Independent t-test" if equal_var else "Welch's t-test",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=d,
            effect_size_type=EffectSizeType.COHENS_D,
            effect_size_interpretation=EffectSizeCalculator.interpret_cohens_d(d),
            degrees_of_freedom=len(g1) + len(g2) - 2,
            sample_sizes={name1: len(g1), name2: len(g2)},
            descriptive_stats={
                name1: {"mean": float(g1.mean()), "std": float(g1.std())},
                name2: {"mean": float(g2.mean()), "std": float(g2.std())}
            },
            interpretation=f"Mean difference: {g1.mean() - g2.mean():.4f}"
        )
    
    def _run_paired_ttest(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        name1: str,
        name2: str
    ) -> TestResult:
        stat, p = StatisticalTests.ttest_paired(g1, g2)
        diff = g1 - g2
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        return TestResult(
            test_type=TestType.TTEST_PAIRED,
            test_name="Paired t-test",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=d,
            effect_size_type=EffectSizeType.COHENS_D,
            effect_size_interpretation=EffectSizeCalculator.interpret_cohens_d(d),
            degrees_of_freedom=len(g1) - 1,
            sample_sizes={"pairs": len(g1)},
            interpretation=f"Mean difference: {diff.mean():.4f}"
        )
    
    def _run_mann_whitney(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        name1: str,
        name2: str
    ) -> TestResult:
        stat, p = StatisticalTests.mann_whitney(g1, g2)
        
        # Effect size: r = Z / sqrt(N)
        n = len(g1) + len(g2)
        z = scipy_stats.norm.ppf(1 - p / 2) if p < 1 else 0
        r = abs(z) / np.sqrt(n)
        
        return TestResult(
            test_type=TestType.MANN_WHITNEY,
            test_name="Mann-Whitney U Test",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=r,
            effect_size_type=EffectSizeType.COHENS_R,
            effect_size_interpretation=EffectSizeCalculator.interpret_r(r),
            sample_sizes={name1: len(g1), name2: len(g2)},
            descriptive_stats={
                name1: {"median": float(np.median(g1))},
                name2: {"median": float(np.median(g2))}
            }
        )
    
    def _run_wilcoxon(
        self,
        g1: np.ndarray,
        g2: np.ndarray,
        name1: str,
        name2: str
    ) -> TestResult:
        stat, p = StatisticalTests.wilcoxon(g1, g2)
        
        return TestResult(
            test_type=TestType.WILCOXON,
            test_name="Wilcoxon Signed-Rank Test",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            sample_sizes={"pairs": len(g1)}
        )
    
    def _run_anova(self, groups: List[np.ndarray], names: List[str]) -> TestResult:
        stat, p = StatisticalTests.anova_one_way(*groups)
        
        # Eta-squared
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
        eta2 = ss_between / ss_total if ss_total > 0 else 0
        
        return TestResult(
            test_type=TestType.ANOVA_ONE_WAY,
            test_name="One-way ANOVA",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=eta2,
            effect_size_type=EffectSizeType.ETA_SQUARED,
            degrees_of_freedom=len(groups) - 1,
            sample_sizes={n: len(g) for n, g in zip(names, groups)}
        )
    
    def _run_kruskal_wallis(self, groups: List[np.ndarray], names: List[str]) -> TestResult:
        stat, p = StatisticalTests.kruskal_wallis(*groups)
        
        return TestResult(
            test_type=TestType.KRUSKAL_WALLIS,
            test_name="Kruskal-Wallis H Test",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            degrees_of_freedom=len(groups) - 1,
            sample_sizes={n: len(g) for n, g in zip(names, groups)}
        )
    
    def _run_pearson(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name_x: str,
        name_y: str
    ) -> TestResult:
        r, p = StatisticalTests.pearson(x, y)
        
        return TestResult(
            test_type=TestType.PEARSON,
            test_name="Pearson Correlation",
            statistic=r,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=abs(r),
            effect_size_type=EffectSizeType.COHENS_R,
            effect_size_interpretation=EffectSizeCalculator.interpret_r(r),
            sample_sizes={"pairs": len(x)}
        )
    
    def _run_spearman(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name_x: str,
        name_y: str
    ) -> TestResult:
        rho, p = StatisticalTests.spearman(x, y)
        
        return TestResult(
            test_type=TestType.SPEARMAN,
            test_name="Spearman Correlation",
            statistic=rho,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=abs(rho),
            effect_size_type=EffectSizeType.COHENS_R,
            effect_size_interpretation=EffectSizeCalculator.interpret_r(rho),
            sample_sizes={"pairs": len(x)}
        )
    
    def _run_point_biserial(
        self,
        binary: np.ndarray,
        continuous: np.ndarray,
        name_x: str,
        name_y: str
    ) -> TestResult:
        r, p = StatisticalTests.point_biserial(binary, continuous)
        
        return TestResult(
            test_type=TestType.POINT_BISERIAL,
            test_name="Point-Biserial Correlation",
            statistic=r,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=abs(r),
            effect_size_type=EffectSizeType.COHENS_R,
            effect_size_interpretation=EffectSizeCalculator.interpret_r(r)
        )
    
    def _run_chi_square(
        self,
        table: np.ndarray,
        name1: str,
        name2: str
    ) -> TestResult:
        chi2, p, dof, expected = StatisticalTests.chi_square(table)
        
        # Cramer's V
        n = table.sum()
        min_dim = min(table.shape) - 1
        v = EffectSizeCalculator.cramers_v(chi2, n, min_dim + 1)
        
        return TestResult(
            test_type=TestType.CHI_SQUARE,
            test_name="Chi-Square Test",
            statistic=chi2,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha,
            effect_size=v,
            effect_size_type=EffectSizeType.CRAMERS_V,
            degrees_of_freedom=dof
        )
    
    def _run_fisher_exact(
        self,
        table: np.ndarray,
        name1: str,
        name2: str
    ) -> TestResult:
        odds_ratio, p = StatisticalTests.fisher_exact(table)
        
        return TestResult(
            test_type=TestType.FISHER_EXACT,
            test_name="Fisher's Exact Test",
            statistic=odds_ratio,
            p_value=p,
            is_significant=p < self.alpha,
            significance_level=self.alpha
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_smart_statistical_engine(alpha: float = 0.05) -> SmartStatisticalEngine:
    """Get a configured statistical engine."""
    return SmartStatisticalEngine(alpha=alpha)


def quick_compare(
    group1: pd.Series,
    group2: pd.Series,
    paired: bool = False
) -> Dict[str, Any]:
    """
    Quick comparison of two groups.
    Auto-selects appropriate test.
    """
    engine = SmartStatisticalEngine()
    result = engine.compare_two_groups(group1, group2, paired=paired)
    return result.to_dict()


def quick_correlate(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """Quick correlation test."""
    engine = SmartStatisticalEngine()
    result = engine.test_correlation(x, y)
    return result.to_dict()
