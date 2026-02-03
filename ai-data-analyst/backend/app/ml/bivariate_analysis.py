# AI Enterprise Data Analyst - Bivariate Analysis Engine
# Production-grade two-variable analysis with complete edge case handling
# Handles: numeric-numeric, numeric-categorical, categorical-categorical

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
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class RelationshipType(str, Enum):
    """Type of bivariate relationship."""
    NUMERIC_NUMERIC = "numeric_numeric"
    NUMERIC_CATEGORICAL = "numeric_categorical"
    CATEGORICAL_CATEGORICAL = "categorical_categorical"


class RelationshipStrength(str, Enum):
    """Strength of relationship."""
    VERY_STRONG = "very_strong"  # > 0.8
    STRONG = "strong"  # 0.6 - 0.8
    MODERATE = "moderate"  # 0.4 - 0.6
    WEAK = "weak"  # 0.2 - 0.4
    NEGLIGIBLE = "negligible"  # < 0.2


class MonotonicDirection(str, Enum):
    """Direction of monotonic relationship."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"
    MIXED = "mixed"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CorrelationMetrics:
    """Correlation metrics for numeric-numeric."""
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    
    # Best correlation
    best_method: str
    best_value: float
    
    # Linear regression
    slope: float
    intercept: float
    r_squared: float
    
    # Relationship characteristics
    is_linear: bool
    is_monotonic: bool
    direction: MonotonicDirection
    strength: RelationshipStrength


@dataclass
class CategoricalAssociation:
    """Association metrics for categorical-categorical."""
    chi_squared: float
    chi_squared_p: float
    degrees_of_freedom: int
    
    cramers_v: float
    contingency_coef: float
    
    is_significant: bool
    strength: RelationshipStrength
    
    # Contingency table
    contingency_table: pd.DataFrame = None


@dataclass
class GroupComparison:
    """Comparison metrics for numeric-categorical."""
    # Test results
    test_name: str  # ANOVA, Kruskal-Wallis, or t-test
    test_statistic: float
    p_value: float
    is_significant: bool
    
    # Effect size
    effect_size: float
    effect_size_name: str  # eta-squared, Cohen's d
    strength: RelationshipStrength
    
    # Group statistics
    group_means: Dict[str, float]
    group_stds: Dict[str, float]
    group_counts: Dict[str, int]
    
    # Post-hoc
    significant_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class BivariateResult:
    """Complete bivariate analysis result."""
    var1_name: str
    var2_name: str
    relationship_type: RelationshipType
    n_complete: int  # Rows with both values present
    n_missing: int  # Rows with at least one missing
    
    # Type-specific results
    correlation: Optional[CorrelationMetrics] = None
    association: Optional[CategoricalAssociation] = None
    group_comparison: Optional[GroupComparison] = None
    
    # Summary
    strength: RelationshipStrength = RelationshipStrength.NEGLIGIBLE
    is_significant: bool = False
    p_value: float = 1.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "variables": [self.var1_name, self.var2_name],
            "relationship_type": self.relationship_type.value,
            "n_complete": self.n_complete,
            "strength": self.strength.value,
            "is_significant": self.is_significant,
            "p_value": round(self.p_value, 6),
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }
        
        if self.correlation:
            c = self.correlation
            result["correlation"] = {
                "pearson_r": self._safe_round(c.pearson_r),
                "spearman_r": self._safe_round(c.spearman_r),
                "r_squared": self._safe_round(c.r_squared),
                "is_linear": c.is_linear,
                "direction": c.direction.value
            }
        
        if self.association:
            a = self.association
            result["association"] = {
                "chi_squared": self._safe_round(a.chi_squared),
                "p_value": self._safe_round(a.chi_squared_p, 6),
                "cramers_v": self._safe_round(a.cramers_v),
                "is_significant": a.is_significant
            }
        
        if self.group_comparison:
            g = self.group_comparison
            result["group_comparison"] = {
                "test": g.test_name,
                "p_value": self._safe_round(g.p_value, 6),
                "effect_size": self._safe_round(g.effect_size),
                "effect_name": g.effect_size_name,
                "group_means": {k: self._safe_round(v) for k, v in g.group_means.items()},
                "significant_pairs": g.significant_pairs[:10]
            }
        
        return result
    
    def _safe_round(self, val: float, decimals: int = 4) -> Optional[float]:
        if val is None or np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)


# ============================================================================
# Bivariate Analysis Engine
# ============================================================================

class BivariateAnalysisEngine:
    """
    Production-grade Bivariate Analysis engine.
    
    Features:
    - Automatic relationship type detection
    - Multiple correlation methods
    - Chi-squared and Cramér's V
    - ANOVA/Kruskal-Wallis comparison
    - Complete edge case handling
    - Actionable recommendations
    """
    
    def __init__(self, alpha: float = 0.05, verbose: bool = True):
        self.alpha = alpha
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str
    ) -> BivariateResult:
        """Analyze relationship between two variables."""
        start_time = datetime.now()
        
        if var1 not in df.columns or var2 not in df.columns:
            raise DataProcessingException(f"Column not found: {var1} or {var2}")
        
        if self.verbose:
            logger.info(f"Bivariate analysis: {var1} vs {var2}")
        
        # Get complete cases
        subset = df[[var1, var2]].dropna()
        n_complete = len(subset)
        n_missing = len(df) - n_complete
        
        # Edge case: insufficient data
        if n_complete < 3:
            return self._create_insufficient_data_result(
                var1, var2, n_complete, n_missing, start_time
            )
        
        # Detect variable types
        type1 = self._detect_type(subset[var1])
        type2 = self._detect_type(subset[var2])
        
        # Determine relationship type
        if type1 == "numeric" and type2 == "numeric":
            rel_type = RelationshipType.NUMERIC_NUMERIC
            result = self._analyze_numeric_numeric(subset, var1, var2)
        elif type1 == "categorical" and type2 == "categorical":
            rel_type = RelationshipType.CATEGORICAL_CATEGORICAL
            result = self._analyze_categorical_categorical(subset, var1, var2)
        else:
            rel_type = RelationshipType.NUMERIC_CATEGORICAL
            # Ensure numeric is first
            if type1 == "categorical":
                var1, var2 = var2, var1
            result = self._analyze_numeric_categorical(subset, var1, var2)
        
        # Unpack result based on type
        if rel_type == RelationshipType.NUMERIC_NUMERIC:
            correlation, recommendations, warnings_list = result
            return BivariateResult(
                var1_name=var1, var2_name=var2,
                relationship_type=rel_type,
                n_complete=n_complete, n_missing=n_missing,
                correlation=correlation,
                strength=correlation.strength,
                is_significant=correlation.pearson_p < self.alpha,
                p_value=correlation.pearson_p,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time_sec=(datetime.now() - start_time).total_seconds()
            )
        
        elif rel_type == RelationshipType.CATEGORICAL_CATEGORICAL:
            association, recommendations, warnings_list = result
            return BivariateResult(
                var1_name=var1, var2_name=var2,
                relationship_type=rel_type,
                n_complete=n_complete, n_missing=n_missing,
                association=association,
                strength=association.strength,
                is_significant=association.is_significant,
                p_value=association.chi_squared_p,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time_sec=(datetime.now() - start_time).total_seconds()
            )
        
        else:  # Numeric-Categorical
            comparison, recommendations, warnings_list = result
            return BivariateResult(
                var1_name=var1, var2_name=var2,
                relationship_type=rel_type,
                n_complete=n_complete, n_missing=n_missing,
                group_comparison=comparison,
                strength=comparison.strength,
                is_significant=comparison.is_significant,
                p_value=comparison.p_value,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time_sec=(datetime.now() - start_time).total_seconds()
            )
    
    def analyze_all_pairs(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> List[BivariateResult]:
        """Analyze all pairs of variables."""
        if columns is None:
            columns = df.columns.tolist()
        
        results = []
        for i, var1 in enumerate(columns):
            for var2 in columns[i+1:]:
                try:
                    result = self.analyze(df, var1, var2)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error analyzing {var1} vs {var2}: {e}")
        
        return results
    
    def _detect_type(self, series: pd.Series) -> str:
        """Detect if variable is numeric or categorical."""
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's really discrete with few values
            if series.nunique() <= 10:
                return "categorical"
            return "numeric"
        return "categorical"
    
    def _analyze_numeric_numeric(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str
    ) -> Tuple[CorrelationMetrics, List[str], List[str]]:
        """Analyze relationship between two numeric variables."""
        x = df[var1].values.astype(float)
        y = df[var2].values.astype(float)
        
        # Handle edge cases
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return self._create_constant_correlation(), [], ["One variable has no variance"]
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
        except:
            pearson_r, pearson_p = 0.0, 1.0
        
        # Spearman correlation (robust to outliers)
        try:
            spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
        except:
            spearman_r, spearman_p = 0.0, 1.0
        
        # Kendall tau
        try:
            kendall_tau, kendall_p = scipy_stats.kendalltau(x, y)
        except:
            kendall_tau, kendall_p = 0.0, 1.0
        
        # Handle NaN results
        pearson_r = 0.0 if np.isnan(pearson_r) else pearson_r
        spearman_r = 0.0 if np.isnan(spearman_r) else spearman_r
        kendall_tau = 0.0 if np.isnan(kendall_tau) else kendall_tau
        
        # Linear regression
        try:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)
            r_squared = r_value ** 2
        except:
            slope, intercept, r_squared = 0.0, 0.0, 0.0
        
        # Determine best correlation
        correlations = {
            "pearson": abs(pearson_r),
            "spearman": abs(spearman_r),
            "kendall": abs(kendall_tau)
        }
        best_method = max(correlations, key=correlations.get)
        best_value = correlations[best_method]
        
        # Determine linearity and monotonicity
        is_linear = abs(pearson_r) > 0.7 and abs(pearson_r - spearman_r) < 0.1
        is_monotonic = abs(spearman_r) > 0.7
        
        # Direction
        if abs(spearman_r) < 0.1:
            direction = MonotonicDirection.NONE
        elif spearman_r > 0:
            direction = MonotonicDirection.POSITIVE
        else:
            direction = MonotonicDirection.NEGATIVE
        
        # Strength
        strength = self._get_strength(best_value)
        
        recommendations = []
        warnings_list = []
        
        if abs(pearson_r - spearman_r) > 0.2:
            warnings_list.append("Large difference between Pearson and Spearman suggests non-linear relationship")
            recommendations.append("Consider polynomial or non-linear regression")
        
        if strength in [RelationshipStrength.STRONG, RelationshipStrength.VERY_STRONG]:
            recommendations.append("Strong correlation - check for multicollinearity if used as features")
        
        if abs(pearson_r) > 0.9:
            warnings_list.append("Very high correlation - possible redundant variable")
        
        correlation = CorrelationMetrics(
            pearson_r=pearson_r, pearson_p=pearson_p,
            spearman_r=spearman_r, spearman_p=spearman_p,
            kendall_tau=kendall_tau, kendall_p=kendall_p,
            best_method=best_method, best_value=best_value,
            slope=slope, intercept=intercept, r_squared=r_squared,
            is_linear=is_linear, is_monotonic=is_monotonic,
            direction=direction, strength=strength
        )
        
        return correlation, recommendations, warnings_list
    
    def _analyze_categorical_categorical(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str
    ) -> Tuple[CategoricalAssociation, List[str], List[str]]:
        """Analyze association between two categorical variables."""
        # Contingency table
        contingency = pd.crosstab(df[var1], df[var2])
        
        # Edge case: single row or column
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return self._create_trivial_association(contingency), [], ["Insufficient categories for chi-squared test"]
        
        # Chi-squared test
        try:
            chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
        except:
            chi2, p_value, dof = 0.0, 1.0, 0
        
        # Check expected frequencies
        if hasattr(expected, 'min') and expected.min() < 5:
            try:
                # Use Fisher's exact for 2x2
                if contingency.shape == (2, 2):
                    _, p_value = scipy_stats.fisher_exact(contingency)
            except:
                pass
        
        # Cramér's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        if min_dim > 0 and n > 0:
            cramers_v = np.sqrt(chi2 / (n * min_dim))
        else:
            cramers_v = 0.0
        
        # Contingency coefficient
        contingency_coef = np.sqrt(chi2 / (chi2 + n)) if (chi2 + n) > 0 else 0.0
        
        is_significant = p_value < self.alpha
        strength = self._get_strength(cramers_v)
        
        recommendations = []
        warnings_list = []
        
        if is_significant and strength != RelationshipStrength.NEGLIGIBLE:
            recommendations.append("Significant association - consider creating interaction features")
        
        if contingency.shape[0] > 10 or contingency.shape[1] > 10:
            warnings_list.append("High cardinality - consider grouping categories")
        
        association = CategoricalAssociation(
            chi_squared=chi2, chi_squared_p=p_value, degrees_of_freedom=dof,
            cramers_v=cramers_v, contingency_coef=contingency_coef,
            is_significant=is_significant, strength=strength,
            contingency_table=contingency
        )
        
        return association, recommendations, warnings_list
    
    def _analyze_numeric_categorical(
        self,
        df: pd.DataFrame,
        numeric_var: str,
        categorical_var: str
    ) -> Tuple[GroupComparison, List[str], List[str]]:
        """Analyze relationship between numeric and categorical variables."""
        groups = df[categorical_var].unique()
        n_groups = len(groups)
        
        # Group statistics
        group_means = {}
        group_stds = {}
        group_counts = {}
        group_data = []
        
        for g in groups:
            data = df[df[categorical_var] == g][numeric_var].values
            if len(data) > 0:
                group_means[str(g)] = float(np.mean(data))
                group_stds[str(g)] = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
                group_counts[str(g)] = len(data)
                group_data.append(data)
        
        # Edge case: not enough groups with data
        valid_groups = [d for d in group_data if len(d) >= 2]
        if len(valid_groups) < 2:
            return self._create_trivial_comparison(group_means, group_stds, group_counts), [], \
                ["Insufficient data in groups for comparison"]
        
        # Test for normality in groups
        all_normal = True
        for data in valid_groups:
            if len(data) >= 8:
                try:
                    _, p = scipy_stats.shapiro(data[:5000])
                    if p < 0.05:
                        all_normal = False
                        break
                except:
                    all_normal = False
        
        # Choose test
        if n_groups == 2:
            # T-test or Mann-Whitney
            if all_normal:
                try:
                    stat, p_value = scipy_stats.ttest_ind(valid_groups[0], valid_groups[1])
                    test_name = "t-test"
                except:
                    stat, p_value = 0.0, 1.0
                    test_name = "t-test"
            else:
                try:
                    stat, p_value = scipy_stats.mannwhitneyu(valid_groups[0], valid_groups[1], alternative='two-sided')
                    test_name = "Mann-Whitney U"
                except:
                    stat, p_value = 0.0, 1.0
                    test_name = "Mann-Whitney U"
            
            # Effect size: Cohen's d
            pooled_std = np.sqrt(
                ((len(valid_groups[0]) - 1) * np.var(valid_groups[0], ddof=1) +
                 (len(valid_groups[1]) - 1) * np.var(valid_groups[1], ddof=1)) /
                (len(valid_groups[0]) + len(valid_groups[1]) - 2)
            )
            if pooled_std > 0:
                effect_size = abs(np.mean(valid_groups[0]) - np.mean(valid_groups[1])) / pooled_std
            else:
                effect_size = 0.0
            effect_name = "Cohen's d"
        
        else:
            # ANOVA or Kruskal-Wallis
            if all_normal:
                try:
                    stat, p_value = scipy_stats.f_oneway(*valid_groups)
                    test_name = "ANOVA"
                except:
                    stat, p_value = 0.0, 1.0
                    test_name = "ANOVA"
            else:
                try:
                    stat, p_value = scipy_stats.kruskal(*valid_groups)
                    test_name = "Kruskal-Wallis"
                except:
                    stat, p_value = 0.0, 1.0
                    test_name = "Kruskal-Wallis"
            
            # Effect size: eta-squared
            all_data = np.concatenate(valid_groups)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in valid_groups)
            ss_total = np.sum((all_data - grand_mean) ** 2)
            effect_size = ss_between / ss_total if ss_total > 0 else 0.0
            effect_name = "eta-squared"
        
        # Handle NaN
        p_value = 1.0 if np.isnan(p_value) else p_value
        stat = 0.0 if np.isnan(stat) else stat
        
        is_significant = p_value < self.alpha
        strength = self._get_strength(np.sqrt(effect_size) if effect_name == "eta-squared" else effect_size / 2)
        
        # Post-hoc for significant ANOVA
        significant_pairs = []
        if is_significant and n_groups > 2:
            # Simple pairwise comparisons with Bonferroni
            adjusted_alpha = self.alpha / (n_groups * (n_groups - 1) / 2)
            group_names = list(group_means.keys())
            
            for i, g1 in enumerate(group_names):
                for g2 in group_names[i+1:]:
                    data1 = df[df[categorical_var].astype(str) == g1][numeric_var].values
                    data2 = df[df[categorical_var].astype(str) == g2][numeric_var].values
                    
                    if len(data1) >= 2 and len(data2) >= 2:
                        try:
                            _, pair_p = scipy_stats.ttest_ind(data1, data2)
                            if pair_p < adjusted_alpha:
                                significant_pairs.append((g1, g2))
                        except:
                            pass
        
        recommendations = []
        warnings_list = []
        
        if is_significant:
            recommendations.append(f"Significant difference between groups (p={p_value:.4f})")
            if len(significant_pairs) > 0:
                recommendations.append(f"Significant pairs: {significant_pairs[:5]}")
        
        if not all_normal:
            warnings_list.append("Non-normal distribution in some groups - used non-parametric test")
        
        comparison = GroupComparison(
            test_name=test_name, test_statistic=stat, p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size, effect_size_name=effect_name, strength=strength,
            group_means=group_means, group_stds=group_stds, group_counts=group_counts,
            significant_pairs=significant_pairs
        )
        
        return comparison, recommendations, warnings_list
    
    def _get_strength(self, value: float) -> RelationshipStrength:
        """Get relationship strength from correlation/effect size."""
        value = abs(value)
        if value >= 0.8:
            return RelationshipStrength.VERY_STRONG
        elif value >= 0.6:
            return RelationshipStrength.STRONG
        elif value >= 0.4:
            return RelationshipStrength.MODERATE
        elif value >= 0.2:
            return RelationshipStrength.WEAK
        return RelationshipStrength.NEGLIGIBLE
    
    def _create_insufficient_data_result(
        self,
        var1: str, var2: str,
        n_complete: int, n_missing: int,
        start_time: datetime
    ) -> BivariateResult:
        """Create result for insufficient data."""
        return BivariateResult(
            var1_name=var1, var2_name=var2,
            relationship_type=RelationshipType.NUMERIC_NUMERIC,
            n_complete=n_complete, n_missing=n_missing,
            warnings=["Insufficient complete cases for analysis"],
            recommendations=["Collect more data or investigate missing values"],
            processing_time_sec=(datetime.now() - start_time).total_seconds()
        )
    
    def _create_constant_correlation(self) -> CorrelationMetrics:
        """Create correlation for constant variable."""
        return CorrelationMetrics(
            pearson_r=0, pearson_p=1, spearman_r=0, spearman_p=1,
            kendall_tau=0, kendall_p=1, best_method="pearson", best_value=0,
            slope=0, intercept=0, r_squared=0,
            is_linear=False, is_monotonic=False,
            direction=MonotonicDirection.NONE, strength=RelationshipStrength.NEGLIGIBLE
        )
    
    def _create_trivial_association(self, contingency: pd.DataFrame) -> CategoricalAssociation:
        """Create association for trivial cases."""
        return CategoricalAssociation(
            chi_squared=0, chi_squared_p=1, degrees_of_freedom=0,
            cramers_v=0, contingency_coef=0,
            is_significant=False, strength=RelationshipStrength.NEGLIGIBLE,
            contingency_table=contingency
        )
    
    def _create_trivial_comparison(
        self, means: Dict, stds: Dict, counts: Dict
    ) -> GroupComparison:
        """Create comparison for trivial cases."""
        return GroupComparison(
            test_name="None", test_statistic=0, p_value=1,
            is_significant=False, effect_size=0, effect_size_name="None",
            strength=RelationshipStrength.NEGLIGIBLE,
            group_means=means, group_stds=stds, group_counts=counts
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_bivariate_engine(alpha: float = 0.05) -> BivariateAnalysisEngine:
    """Get bivariate analysis engine."""
    return BivariateAnalysisEngine(alpha=alpha)


def quick_bivariate(
    df: pd.DataFrame,
    var1: str,
    var2: str
) -> Dict[str, Any]:
    """Quick bivariate analysis."""
    engine = BivariateAnalysisEngine(verbose=False)
    result = engine.analyze(df, var1, var2)
    return result.to_dict()


def correlation_matrix(
    df: pd.DataFrame,
    columns: List[str] = None
) -> pd.DataFrame:
    """Generate correlation matrix for numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].corr()
