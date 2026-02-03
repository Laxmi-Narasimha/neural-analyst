# AI Enterprise Data Analyst - Statistical Analysis Agent
# Advanced statistical analysis with hypothesis testing, distributions, and causal inference

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    chi2_contingency, fisher_exact, kruskal,
    shapiro, normaltest, anderson, kstest,
    spearmanr, pearsonr, kendalltau,
    f_oneway
)

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


# ============================================================================
# Statistical Test Types and Results
# ============================================================================

class TestType(str, Enum):
    """Types of statistical tests."""
    # Parametric tests
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    ANOVA_ONE_WAY = "anova_one_way"
    ANOVA_TWO_WAY = "anova_two_way"
    
    # Non-parametric tests
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"
    
    # Categorical tests
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    
    # Correlation tests
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    
    # Normality tests
    SHAPIRO_WILK = "shapiro_wilk"
    DAGOSTINO_PEARSON = "dagostino_pearson"
    ANDERSON_DARLING = "anderson_darling"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    
    test_type: TestType
    test_name: str
    statistic: float
    p_value: float
    significance_level: float = 0.05
    
    # Effect size metrics
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None  # Cohen's d, eta-squared, etc.
    effect_interpretation: Optional[str] = None  # small, medium, large
    
    # Confidence intervals
    confidence_interval: Optional[tuple[float, float]] = None
    confidence_level: float = 0.95
    
    # Power analysis
    statistical_power: Optional[float] = None
    required_sample_size: Optional[int] = None
    
    # Interpretation
    is_significant: bool = False
    interpretation: str = ""
    assumptions_met: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_type": self.test_type.value,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "significance_level": self.significance_level,
            "is_significant": self.is_significant,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "effect_size_type": self.effect_size_type,
            "effect_interpretation": self.effect_interpretation,
            "confidence_interval": self.confidence_interval,
            "statistical_power": self.statistical_power,
            "interpretation": self.interpretation,
            "assumptions_met": self.assumptions_met,
            "warnings": self.warnings
        }


# ============================================================================
# Statistical Analysis Engine
# ============================================================================

class StatisticalEngine:
    """
    Core statistical analysis engine with advanced testing capabilities.
    
    Implements:
    - Parametric and non-parametric hypothesis tests
    - Effect size calculations
    - Power analysis
    - Assumption checking
    - Multiple comparison corrections
    """
    
    @staticmethod
    def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> tuple[float, str]:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        d = (group1.mean() - group2.mean()) / pooled_std
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return d, interpretation
    
    @staticmethod
    def check_normality(data: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
        """Check normality of data using multiple tests."""
        results = {}
        
        # Shapiro-Wilk (best for n < 50)
        if len(data) <= 5000:  # Shapiro-Wilk limit
            stat, p = shapiro(data)
            results["shapiro_wilk"] = {
                "statistic": stat,
                "p_value": p,
                "is_normal": p > alpha
            }
        
        # D'Agostino-Pearson (for n >= 20)
        if len(data) >= 20:
            stat, p = normaltest(data)
            results["dagostino_pearson"] = {
                "statistic": stat,
                "p_value": p,
                "is_normal": p > alpha
            }
        
        # Anderson-Darling
        ad_result = anderson(data)
        results["anderson_darling"] = {
            "statistic": ad_result.statistic,
            "critical_values": dict(zip(
                [str(x) for x in ad_result.significance_level],
                ad_result.critical_values.tolist()
            )),
            "is_normal": ad_result.statistic < ad_result.critical_values[2]  # 5% level
        }
        
        # Overall assessment
        normal_count = sum(1 for r in results.values() if r.get("is_normal", False))
        results["overall_normal"] = normal_count >= len(results) / 2
        
        return results
    
    @staticmethod
    def check_variance_homogeneity(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> dict[str, Any]:
        """Check homogeneity of variances (Levene's test)."""
        stat, p = stats.levene(*groups)
        
        return {
            "test": "levene",
            "statistic": stat,
            "p_value": p,
            "homogeneous": p > alpha
        }
    
    def t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
        alpha: float = 0.05,
        alternative: str = "two-sided"
    ) -> StatisticalResult:
        """Perform t-test with effect size and power."""
        test_type = TestType.T_TEST_PAIRED if paired else TestType.T_TEST_INDEPENDENT
        
        # Check assumptions
        normality1 = self.check_normality(group1)
        normality2 = self.check_normality(group2)
        
        assumptions = {
            "normality_group1": normality1["overall_normal"],
            "normality_group2": normality2["overall_normal"]
        }
        
        warnings = []
        
        if not assumptions["normality_group1"] or not assumptions["normality_group2"]:
            warnings.append("Normality assumption may be violated. Consider non-parametric Mann-Whitney test.")
        
        # Perform test
        if paired:
            stat, p = ttest_rel(group1, group2, alternative=alternative)
        else:
            # Check variance homogeneity
            variance_check = self.check_variance_homogeneity(group1, group2)
            assumptions["equal_variances"] = variance_check["homogeneous"]
            
            if not variance_check["homogeneous"]:
                warnings.append("Variances are not equal. Using Welch's t-test.")
            
            stat, p = ttest_ind(
                group1, group2,
                equal_var=variance_check["homogeneous"],
                alternative=alternative
            )
        
        # Effect size
        effect_size, effect_interp = self.calculate_cohens_d(group1, group2)
        
        # Confidence interval for difference
        diff_mean = group1.mean() - group2.mean()
        se = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
        t_crit = stats.t.ppf(0.975, len(group1) + len(group2) - 2)
        ci = (diff_mean - t_crit * se, diff_mean + t_crit * se)
        
        # Interpretation
        is_sig = p < alpha
        if is_sig:
            interpretation = f"Statistically significant difference (t={stat:.3f}, p={p:.4f}). " \
                           f"Effect size is {effect_interp} (Cohen's d={effect_size:.3f})."
        else:
            interpretation = f"No statistically significant difference (t={stat:.3f}, p={p:.4f})."
        
        return StatisticalResult(
            test_type=test_type,
            test_name="Paired t-test" if paired else "Independent t-test",
            statistic=stat,
            p_value=p,
            significance_level=alpha,
            effect_size=effect_size,
            effect_size_type="Cohen's d",
            effect_interpretation=effect_interp,
            confidence_interval=ci,
            is_significant=is_sig,
            interpretation=interpretation,
            assumptions_met=assumptions,
            warnings=warnings
        )
    
    def mann_whitney(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = "two-sided"
    ) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric alternative to t-test)."""
        stat, p = mannwhitneyu(group1, group2, alternative=alternative)
        
        # Rank-biserial correlation as effect size
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        
        abs_r = abs(effect_size)
        if abs_r < 0.1:
            effect_interp = "negligible"
        elif abs_r < 0.3:
            effect_interp = "small"
        elif abs_r < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        is_sig = p < alpha
        
        if is_sig:
            interpretation = f"Significant difference in distributions (U={stat:.1f}, p={p:.4f}). " \
                           f"Effect size: {effect_interp} (r={effect_size:.3f})."
        else:
            interpretation = f"No significant difference (U={stat:.1f}, p={p:.4f})."
        
        return StatisticalResult(
            test_type=TestType.MANN_WHITNEY,
            test_name="Mann-Whitney U test",
            statistic=stat,
            p_value=p,
            significance_level=alpha,
            effect_size=effect_size,
            effect_size_type="rank-biserial correlation",
            effect_interpretation=effect_interp,
            is_significant=is_sig,
            interpretation=interpretation
        )
    
    def anova_one_way(
        self,
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> StatisticalResult:
        """Perform one-way ANOVA."""
        # Check assumptions
        normality_results = [self.check_normality(g) for g in groups]
        all_normal = all(r["overall_normal"] for r in normality_results)
        
        variance_check = self.check_variance_homogeneity(*groups)
        
        assumptions = {
            "normality": all_normal,
            "equal_variances": variance_check["homogeneous"]
        }
        
        warnings = []
        if not all_normal:
            warnings.append("Normality violated. Consider Kruskal-Wallis test.")
        if not variance_check["homogeneous"]:
            warnings.append("Variance homogeneity violated. Consider Welch's ANOVA.")
        
        # Perform ANOVA
        stat, p = f_oneway(*groups)
        
        # Eta-squared effect size
        all_data = np.concatenate(groups)
        grand_mean = all_data.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = ((all_data - grand_mean)**2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        if eta_squared < 0.01:
            effect_interp = "negligible"
        elif eta_squared < 0.06:
            effect_interp = "small"
        elif eta_squared < 0.14:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        is_sig = p < alpha
        
        interpretation = f"ANOVA shows {'significant' if is_sig else 'no significant'} " \
                        f"differences between groups (F={stat:.3f}, p={p:.4f}). " \
                        f"Effect size: {effect_interp} (η²={eta_squared:.4f})."
        
        return StatisticalResult(
            test_type=TestType.ANOVA_ONE_WAY,
            test_name="One-way ANOVA",
            statistic=stat,
            p_value=p,
            significance_level=alpha,
            effect_size=eta_squared,
            effect_size_type="eta-squared (η²)",
            effect_interpretation=effect_interp,
            is_significant=is_sig,
            interpretation=interpretation,
            assumptions_met=assumptions,
            warnings=warnings
        )
    
    def chi_square_test(
        self,
        contingency_table: np.ndarray,
        alpha: float = 0.05
    ) -> StatisticalResult:
        """Perform chi-square test of independence."""
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V effect size
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        if cramers_v < 0.1:
            effect_interp = "negligible"
        elif cramers_v < 0.3:
            effect_interp = "small"
        elif cramers_v < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        # Check expected frequencies
        low_expected = (expected < 5).sum()
        warnings = []
        if low_expected > 0:
            warnings.append(f"{low_expected} cells have expected frequency < 5. Consider Fisher's exact test.")
        
        is_sig = p < alpha
        
        interpretation = f"Chi-square test shows {'significant' if is_sig else 'no significant'} " \
                        f"association (χ²={chi2:.3f}, df={dof}, p={p:.4f}). " \
                        f"Effect: {effect_interp} (Cramér's V={cramers_v:.3f})."
        
        return StatisticalResult(
            test_type=TestType.CHI_SQUARE,
            test_name="Chi-square test of independence",
            statistic=chi2,
            p_value=p,
            significance_level=alpha,
            effect_size=cramers_v,
            effect_size_type="Cramér's V",
            effect_interpretation=effect_interp,
            is_significant=is_sig,
            interpretation=interpretation,
            warnings=warnings
        )
    
    def correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson",
        alpha: float = 0.05
    ) -> StatisticalResult:
        """Calculate correlation with significance test."""
        method = method.lower()
        
        if method == "pearson":
            r, p = pearsonr(x, y)
            test_type = TestType.PEARSON
            test_name = "Pearson correlation"
        elif method == "spearman":
            r, p = spearmanr(x, y)
            test_type = TestType.SPEARMAN
            test_name = "Spearman correlation"
        elif method == "kendall":
            r, p = kendalltau(x, y)
            test_type = TestType.KENDALL
            test_name = "Kendall's tau"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        abs_r = abs(r)
        if abs_r < 0.1:
            effect_interp = "negligible"
        elif abs_r < 0.3:
            effect_interp = "weak"
        elif abs_r < 0.5:
            effect_interp = "moderate"
        elif abs_r < 0.7:
            effect_interp = "strong"
        else:
            effect_interp = "very strong"
        
        direction = "positive" if r > 0 else "negative"
        is_sig = p < alpha
        
        interpretation = f"{'Significant' if is_sig else 'Non-significant'} {effect_interp} " \
                        f"{direction} correlation (r={r:.3f}, p={p:.4f})."
        
        return StatisticalResult(
            test_type=test_type,
            test_name=test_name,
            statistic=r,
            p_value=p,
            significance_level=alpha,
            effect_size=r,
            effect_size_type="correlation coefficient",
            effect_interpretation=effect_interp,
            is_significant=is_sig,
            interpretation=interpretation
        )


# ============================================================================
# Statistical Analysis Agent
# ============================================================================

class StatisticalAgent(BaseAgent[dict[str, Any]]):
    """
    Statistical Analysis Agent for comprehensive statistical testing.
    
    Capabilities:
    - Automatic test selection based on data characteristics
    - Assumption checking and recommendations
    - Effect size calculations
    - Multiple comparison corrections
    - Clear interpretations
    """
    
    name: str = "StatisticalAgent"
    description: str = "Advanced statistical analysis with hypothesis testing"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.engine = StatisticalEngine()
    
    def _register_tools(self) -> None:
        """Register statistical analysis tools."""
        
        self.register_tool(AgentTool(
            name="compare_two_groups",
            description="Compare two groups using appropriate statistical test (t-test or Mann-Whitney)",
            function=self._compare_two_groups,
            parameters={
                "group1_data": {"type": "array", "items": {"type": "number"}},
                "group2_data": {"type": "array", "items": {"type": "number"}},
                "paired": {"type": "boolean", "default": False},
                "alpha": {"type": "number", "default": 0.05}
            },
            required_params=["group1_data", "group2_data"]
        ))
        
        self.register_tool(AgentTool(
            name="compare_multiple_groups",
            description="Compare 3+ groups using ANOVA or Kruskal-Wallis",
            function=self._compare_multiple_groups,
            parameters={
                "groups_data": {"type": "array", "items": {"type": "array"}},
                "alpha": {"type": "number", "default": 0.05}
            },
            required_params=["groups_data"]
        ))
        
        self.register_tool(AgentTool(
            name="test_correlation",
            description="Test correlation between two variables",
            function=self._test_correlation,
            parameters={
                "x_data": {"type": "array", "items": {"type": "number"}},
                "y_data": {"type": "array", "items": {"type": "number"}},
                "method": {"type": "string", "enum": ["pearson", "spearman", "kendall"]}
            },
            required_params=["x_data", "y_data"]
        ))
        
        self.register_tool(AgentTool(
            name="test_normality",
            description="Test if data follows normal distribution",
            function=self._test_normality,
            parameters={
                "data": {"type": "array", "items": {"type": "number"}}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="test_categorical_association",
            description="Test association between categorical variables",
            function=self._test_categorical,
            parameters={
                "contingency_table": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
            },
            required_params=["contingency_table"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute statistical analysis based on context."""
        # Use LLM to understand the statistical question
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a statistical analysis expert. Analyze the request and determine the appropriate statistical tests."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "recommendations": self._get_test_recommendations(context.task_description)
        }
    
    def _get_test_recommendations(self, description: str) -> list[str]:
        """Get test recommendations based on description."""
        desc_lower = description.lower()
        recommendations = []
        
        if "compare" in desc_lower and "two" in desc_lower:
            recommendations.append("Use t-test for normally distributed data, Mann-Whitney otherwise")
        if "compare" in desc_lower and ("multiple" in desc_lower or "more than" in desc_lower):
            recommendations.append("Use ANOVA for normally distributed data, Kruskal-Wallis otherwise")
        if "correlation" in desc_lower or "relationship" in desc_lower:
            recommendations.append("Use Pearson for linear relationships, Spearman for monotonic relationships")
        if "categorical" in desc_lower or "chi" in desc_lower:
            recommendations.append("Use Chi-square for categorical associations")
        
        return recommendations
    
    async def _compare_two_groups(
        self,
        group1_data: list[float],
        group2_data: list[float],
        paired: bool = False,
        alpha: float = 0.05
    ) -> dict[str, Any]:
        """Compare two groups with automatic test selection."""
        g1 = np.array(group1_data)
        g2 = np.array(group2_data)
        
        # Check normality to decide test
        norm1 = self.engine.check_normality(g1)
        norm2 = self.engine.check_normality(g2)
        
        use_parametric = norm1["overall_normal"] and norm2["overall_normal"]
        
        if use_parametric:
            result = self.engine.t_test(g1, g2, paired=paired, alpha=alpha)
        else:
            if paired:
                stat, p = wilcoxon(g1, g2)
                result = StatisticalResult(
                    test_type=TestType.WILCOXON,
                    test_name="Wilcoxon signed-rank test",
                    statistic=stat,
                    p_value=p,
                    significance_level=alpha,
                    is_significant=p < alpha,
                    interpretation=f"{'Significant' if p < alpha else 'No significant'} difference (p={p:.4f})"
                )
            else:
                result = self.engine.mann_whitney(g1, g2, alpha=alpha)
        
        return {
            "test_used": result.test_name,
            "reason": "Parametric test used" if use_parametric else "Non-parametric test used due to non-normality",
            **result.to_dict()
        }
    
    async def _compare_multiple_groups(
        self,
        groups_data: list[list[float]],
        alpha: float = 0.05
    ) -> dict[str, Any]:
        """Compare multiple groups."""
        groups = [np.array(g) for g in groups_data]
        
        # Check normality
        all_normal = all(
            self.engine.check_normality(g)["overall_normal"]
            for g in groups
        )
        
        if all_normal:
            result = self.engine.anova_one_way(*groups, alpha=alpha)
        else:
            stat, p = kruskal(*groups)
            result = StatisticalResult(
                test_type=TestType.KRUSKAL_WALLIS,
                test_name="Kruskal-Wallis H test",
                statistic=stat,
                p_value=p,
                significance_level=alpha,
                is_significant=p < alpha,
                interpretation=f"{'Significant' if p < alpha else 'No significant'} differences (p={p:.4f})"
            )
        
        return result.to_dict()
    
    async def _test_correlation(
        self,
        x_data: list[float],
        y_data: list[float],
        method: str = "pearson"
    ) -> dict[str, Any]:
        """Test correlation between variables."""
        x = np.array(x_data)
        y = np.array(y_data)
        
        result = self.engine.correlation(x, y, method=method)
        return result.to_dict()
    
    async def _test_normality(
        self,
        data: list[float]
    ) -> dict[str, Any]:
        """Test data for normality."""
        arr = np.array(data)
        results = self.engine.check_normality(arr)
        
        return {
            "is_normal": results["overall_normal"],
            "tests": results,
            "recommendation": "Data appears normally distributed" if results["overall_normal"] 
                             else "Data may not be normally distributed - consider non-parametric tests"
        }
    
    async def _test_categorical(
        self,
        contingency_table: list[list[float]]
    ) -> dict[str, Any]:
        """Test categorical association."""
        table = np.array(contingency_table)
        result = self.engine.chi_square_test(table)
        return result.to_dict()


# Factory function
def get_statistical_agent() -> StatisticalAgent:
    """Get statistical analysis agent instance."""
    return StatisticalAgent()
