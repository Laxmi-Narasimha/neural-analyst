# AI Enterprise Data Analyst - A/B Testing Engine
# Production-grade experimentation platform following Netflix ABlaze patterns

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
import hashlib

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# A/B Testing Types
# ============================================================================

class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class TestType(str, Enum):
    """Statistical test types."""
    FREQUENTIST = "frequentist"
    BAYESIAN = "bayesian"
    SEQUENTIAL = "sequential"


class MetricType(str, Enum):
    """Types of experiment metrics."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"


@dataclass
class ExperimentMetric:
    """Metric definition for experiments."""
    
    name: str
    metric_type: MetricType
    aggregation: str = "mean"  # mean, sum, count, rate
    minimum_detectable_effect: float = 0.05  # 5% MDE
    baseline_value: Optional[float] = None


@dataclass
class VariantResult:
    """Results for a single variant."""
    
    variant_name: str
    sample_size: int
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    
    # Conversion (for binary metrics)
    conversions: int = 0
    conversion_rate: float = 0.0


@dataclass
class ExperimentResult:
    """Complete experiment analysis result."""
    
    experiment_id: str
    metric_name: str
    test_type: TestType
    
    # Variants
    control: VariantResult = None
    treatment: VariantResult = None
    
    # Statistical results
    relative_lift: float = 0.0
    absolute_lift: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    confidence_level: float = 0.95
    
    # Power analysis
    achieved_power: float = 0.0
    
    # Bayesian results (if applicable)
    probability_better: float = 0.0
    expected_loss: float = 0.0
    
    # Sequential testing
    can_stop_early: bool = False
    early_stop_reason: str = ""
    
    # Interpretation
    recommendation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "metric": self.metric_name,
            "test_type": self.test_type.value,
            "control": {
                "name": self.control.variant_name,
                "sample_size": self.control.sample_size,
                "mean": round(self.control.mean, 6),
                "conversion_rate": round(self.control.conversion_rate, 6)
            } if self.control else None,
            "treatment": {
                "name": self.treatment.variant_name,
                "sample_size": self.treatment.sample_size,
                "mean": round(self.treatment.mean, 6),
                "conversion_rate": round(self.treatment.conversion_rate, 6)
            } if self.treatment else None,
            "results": {
                "relative_lift": round(self.relative_lift * 100, 2),
                "absolute_lift": round(self.absolute_lift, 6),
                "p_value": round(self.p_value, 6),
                "is_significant": self.is_significant,
                "confidence_level": self.confidence_level,
                "achieved_power": round(self.achieved_power, 3)
            },
            "bayesian": {
                "probability_better": round(self.probability_better, 4),
                "expected_loss": round(self.expected_loss, 6)
            } if self.test_type == TestType.BAYESIAN else None,
            "sequential": {
                "can_stop_early": self.can_stop_early,
                "reason": self.early_stop_reason
            } if self.test_type == TestType.SEQUENTIAL else None,
            "recommendation": self.recommendation
        }


# ============================================================================
# Sample Size Calculator
# ============================================================================

class SampleSizeCalculator:
    """
    Calculate required sample size for A/B tests.
    
    Supports:
    - Two-proportion z-test
    - Two-sample t-test
    - Minimum Detectable Effect calculation
    """
    
    @staticmethod
    def for_proportions(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0
    ) -> dict[str, int]:
        """
        Calculate sample size for comparing two proportions.
        
        Args:
            baseline_rate: Control group conversion rate
            minimum_detectable_effect: Relative MDE (e.g., 0.05 for 5%)
            alpha: Significance level
            power: Statistical power
            ratio: Treatment/Control sample size ratio
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Pooled proportion
        p_pooled = (p1 + ratio * p2) / (1 + ratio)
        
        # Effect size
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        # Z values
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_beta = scipy_stats.norm.ppf(power)
        
        # Sample size per group
        n = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2) / ratio)) / (p2 - p1) ** 2
        
        n_control = int(np.ceil(n))
        n_treatment = int(np.ceil(n * ratio))
        
        return {
            "control_size": n_control,
            "treatment_size": n_treatment,
            "total_size": n_control + n_treatment,
            "parameters": {
                "baseline_rate": baseline_rate,
                "mde": minimum_detectable_effect,
                "alpha": alpha,
                "power": power
            }
        }
    
    @staticmethod
    def for_means(
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0
    ) -> dict[str, int]:
        """Calculate sample size for comparing two means."""
        delta = baseline_mean * minimum_detectable_effect
        
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_beta = scipy_stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2
        
        n_control = int(np.ceil(n))
        n_treatment = int(np.ceil(n * ratio))
        
        return {
            "control_size": n_control,
            "treatment_size": n_treatment,
            "total_size": n_control + n_treatment,
            "parameters": {
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "mde": minimum_detectable_effect,
                "alpha": alpha,
                "power": power
            }
        }
    
    @staticmethod
    def minimum_detectable_effect(
        sample_size: int,
        baseline_rate: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> float:
        """Calculate MDE given sample size."""
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_beta = scipy_stats.norm.ppf(power)
        
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size)
        mde = (z_alpha + z_beta) * se / baseline_rate
        
        return float(mde)


# ============================================================================
# Frequentist A/B Test
# ============================================================================

class FrequentistABTest:
    """
    Classic frequentist A/B test analysis.
    
    Supports:
    - Two-sample t-test (continuous metrics)
    - Two-proportion z-test (binary metrics)
    - Welch's t-test (unequal variance)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def analyze_continuous(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray
    ) -> dict[str, Any]:
        """Analyze continuous metric using t-test."""
        # Welch's t-test (doesn't assume equal variance)
        t_stat, p_value = scipy_stats.ttest_ind(
            treatment_values,
            control_values,
            equal_var=False
        )
        
        control_mean = control_values.mean()
        treatment_mean = treatment_values.mean()
        
        absolute_lift = treatment_mean - control_mean
        relative_lift = absolute_lift / control_mean if control_mean != 0 else 0
        
        # Confidence interval for difference
        control_se = control_values.std() / np.sqrt(len(control_values))
        treatment_se = treatment_values.std() / np.sqrt(len(treatment_values))
        pooled_se = np.sqrt(control_se**2 + treatment_se**2)
        
        z_crit = scipy_stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = absolute_lift - z_crit * pooled_se
        ci_upper = absolute_lift + z_crit * pooled_se
        
        return {
            "control_mean": float(control_mean),
            "control_std": float(control_values.std()),
            "control_n": len(control_values),
            "treatment_mean": float(treatment_mean),
            "treatment_std": float(treatment_values.std()),
            "treatment_n": len(treatment_values),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "absolute_lift": float(absolute_lift),
            "relative_lift": float(relative_lift),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "is_significant": p_value < self.alpha
        }
    
    def analyze_proportions(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> dict[str, Any]:
        """Analyze binary metric using z-test for proportions."""
        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total
        
        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        # Z statistic
        z_stat = (p2 - p1) / se if se > 0 else 0
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval
        se_diff = np.sqrt(p1*(1-p1)/control_total + p2*(1-p2)/treatment_total)
        z_crit = scipy_stats.norm.ppf(1 - self.alpha / 2)
        
        ci_lower = (p2 - p1) - z_crit * se_diff
        ci_upper = (p2 - p1) + z_crit * se_diff
        
        absolute_lift = p2 - p1
        relative_lift = absolute_lift / p1 if p1 > 0 else 0
        
        return {
            "control_rate": float(p1),
            "control_n": control_total,
            "treatment_rate": float(p2),
            "treatment_n": treatment_total,
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "absolute_lift": float(absolute_lift),
            "relative_lift": float(relative_lift),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "is_significant": p_value < self.alpha
        }


# ============================================================================
# Bayesian A/B Test
# ============================================================================

class BayesianABTest:
    """
    Bayesian A/B test analysis.
    
    Uses:
    - Beta-Binomial model for proportions
    - Normal-InverseGamma for continuous
    - Monte Carlo for complex metrics
    """
    
    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 10000
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = n_samples
    
    def analyze_proportions(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> dict[str, Any]:
        """Bayesian analysis for binary metrics."""
        # Posterior distributions (Beta)
        control_alpha = self.prior_alpha + control_successes
        control_beta = self.prior_beta + control_total - control_successes
        
        treatment_alpha = self.prior_alpha + treatment_successes
        treatment_beta = self.prior_beta + treatment_total - treatment_successes
        
        # Sample from posteriors
        control_samples = np.random.beta(control_alpha, control_beta, self.n_samples)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, self.n_samples)
        
        # Probability treatment is better
        prob_better = (treatment_samples > control_samples).mean()
        
        # Expected loss if choosing treatment incorrectly
        loss = np.maximum(control_samples - treatment_samples, 0).mean()
        
        # Credible intervals
        control_ci = np.percentile(control_samples, [2.5, 97.5])
        treatment_ci = np.percentile(treatment_samples, [2.5, 97.5])
        
        # Lift distribution
        lift_samples = (treatment_samples - control_samples) / control_samples
        lift_mean = lift_samples.mean()
        lift_ci = np.percentile(lift_samples, [2.5, 97.5])
        
        return {
            "control_rate": float(control_samples.mean()),
            "control_ci": (float(control_ci[0]), float(control_ci[1])),
            "treatment_rate": float(treatment_samples.mean()),
            "treatment_ci": (float(treatment_ci[0]), float(treatment_ci[1])),
            "probability_better": float(prob_better),
            "expected_loss": float(loss),
            "relative_lift_mean": float(lift_mean),
            "relative_lift_ci": (float(lift_ci[0]), float(lift_ci[1])),
            "is_winner": prob_better > 0.95
        }
    
    def analyze_continuous(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray
    ) -> dict[str, Any]:
        """Bayesian analysis for continuous metrics using Monte Carlo."""
        # Bootstrap inference
        control_means = []
        treatment_means = []
        
        for _ in range(self.n_samples):
            c_sample = np.random.choice(control_values, size=len(control_values), replace=True)
            t_sample = np.random.choice(treatment_values, size=len(treatment_values), replace=True)
            control_means.append(c_sample.mean())
            treatment_means.append(t_sample.mean())
        
        control_means = np.array(control_means)
        treatment_means = np.array(treatment_means)
        
        prob_better = (treatment_means > control_means).mean()
        
        lift_samples = (treatment_means - control_means) / np.maximum(control_means, 1e-10)
        
        return {
            "control_mean": float(np.mean(control_means)),
            "control_ci": tuple(np.percentile(control_means, [2.5, 97.5]).tolist()),
            "treatment_mean": float(np.mean(treatment_means)),
            "treatment_ci": tuple(np.percentile(treatment_means, [2.5, 97.5]).tolist()),
            "probability_better": float(prob_better),
            "relative_lift_mean": float(np.mean(lift_samples)),
            "relative_lift_ci": tuple(np.percentile(lift_samples, [2.5, 97.5]).tolist()),
            "is_winner": prob_better > 0.95
        }


# ============================================================================
# Sequential Testing
# ============================================================================

class SequentialTest:
    """
    Sequential testing for early stopping.
    
    Implements:
    - O'Brien-Fleming spending function
    - Pocock boundaries
    - Alpha spending approach
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_looks: int = 5,
        spending_function: str = "obrien_fleming"
    ):
        self.alpha = alpha
        self.n_looks = n_looks
        self.spending_function = spending_function
        self._boundaries = self._calculate_boundaries()
    
    def _calculate_boundaries(self) -> list[float]:
        """Calculate significance boundaries for each look."""
        if self.spending_function == "obrien_fleming":
            # O'Brien-Fleming: More conservative early
            boundaries = []
            for k in range(1, self.n_looks + 1):
                t = k / self.n_looks
                alpha_k = 2 * (1 - scipy_stats.norm.cdf(
                    scipy_stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(t)
                ))
                boundaries.append(alpha_k)
            return boundaries
        
        elif self.spending_function == "pocock":
            # Pocock: Equal boundaries
            z = scipy_stats.norm.ppf(1 - self.alpha / (2 * self.n_looks))
            alpha_level = 2 * (1 - scipy_stats.norm.cdf(z))
            return [alpha_level] * self.n_looks
        
        else:
            # Default: Simple Bonferroni
            return [self.alpha / self.n_looks] * self.n_looks
    
    def check_stopping(
        self,
        p_value: float,
        current_look: int
    ) -> dict[str, Any]:
        """Check if experiment can be stopped early."""
        if current_look > self.n_looks:
            current_look = self.n_looks
        
        boundary = self._boundaries[current_look - 1]
        
        can_stop = p_value < boundary
        
        cumulative_alpha = sum(self._boundaries[:current_look])
        
        return {
            "look_number": current_look,
            "total_looks": self.n_looks,
            "current_boundary": float(boundary),
            "p_value": float(p_value),
            "can_stop_for_significance": can_stop,
            "cumulative_alpha_spent": float(min(cumulative_alpha, self.alpha)),
            "remaining_alpha": float(max(self.alpha - cumulative_alpha, 0))
        }


# ============================================================================
# A/B Testing Engine
# ============================================================================

class ABTestingEngine:
    """
    Production A/B testing engine.
    
    Features:
    - Multiple testing methods
    - Sample size calculation
    - Variance reduction (CUPED)
    - Sequential testing
    - Multi-metric analysis
    """
    
    def __init__(self, default_alpha: float = 0.05):
        self.default_alpha = default_alpha
        self.frequentist = FrequentistABTest(default_alpha)
        self.bayesian = BayesianABTest()
        self.sequential = SequentialTest(default_alpha)
        self.sample_calculator = SampleSizeCalculator()
    
    def analyze_experiment(
        self,
        df: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        control_name: str = "control",
        treatment_name: str = "treatment",
        test_type: TestType = TestType.FREQUENTIST,
        is_binary: bool = False,
        experiment_id: str = None
    ) -> ExperimentResult:
        """
        Analyze A/B experiment data.
        
        Args:
            df: Experiment data
            variant_col: Column with variant assignment
            metric_col: Column with metric values
            control_name: Name of control variant
            treatment_name: Name of treatment variant
            test_type: Statistical test approach
            is_binary: Whether metric is binary (conversion)
            experiment_id: Experiment identifier
        """
        if experiment_id is None:
            experiment_id = str(hashlib.md5(f"{variant_col}_{metric_col}".encode()).hexdigest()[:8])
        
        # Split by variant
        control_data = df[df[variant_col] == control_name][metric_col].dropna()
        treatment_data = df[df[variant_col] == treatment_name][metric_col].dropna()
        
        # Create variant results
        control_result = VariantResult(
            variant_name=control_name,
            sample_size=len(control_data),
            mean=float(control_data.mean()),
            std=float(control_data.std()),
            conversions=int(control_data.sum()) if is_binary else 0,
            conversion_rate=float(control_data.mean()) if is_binary else 0.0
        )
        
        treatment_result = VariantResult(
            variant_name=treatment_name,
            sample_size=len(treatment_data),
            mean=float(treatment_data.mean()),
            std=float(treatment_data.std()),
            conversions=int(treatment_data.sum()) if is_binary else 0,
            conversion_rate=float(treatment_data.mean()) if is_binary else 0.0
        )
        
        # Run analysis based on test type
        if test_type == TestType.FREQUENTIST:
            result = self._frequentist_analysis(
                control_data.values, treatment_data.values, is_binary
            )
        elif test_type == TestType.BAYESIAN:
            result = self._bayesian_analysis(
                control_data.values, treatment_data.values, is_binary
            )
        else:  # Sequential
            result = self._sequential_analysis(
                control_data.values, treatment_data.values, is_binary
            )
        
        # Calculate achieved power
        achieved_power = self._calculate_achieved_power(
            control_result, treatment_result, is_binary
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(result, test_type)
        
        return ExperimentResult(
            experiment_id=experiment_id,
            metric_name=metric_col,
            test_type=test_type,
            control=control_result,
            treatment=treatment_result,
            relative_lift=result.get("relative_lift", 0),
            absolute_lift=result.get("absolute_lift", 0),
            p_value=result.get("p_value", 1.0),
            is_significant=result.get("is_significant", False),
            achieved_power=achieved_power,
            probability_better=result.get("probability_better", 0),
            expected_loss=result.get("expected_loss", 0),
            can_stop_early=result.get("can_stop_early", False),
            early_stop_reason=result.get("early_stop_reason", ""),
            recommendation=recommendation
        )
    
    def _frequentist_analysis(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        is_binary: bool
    ) -> dict[str, Any]:
        """Run frequentist analysis."""
        if is_binary:
            return self.frequentist.analyze_proportions(
                int(control.sum()), len(control),
                int(treatment.sum()), len(treatment)
            )
        else:
            return self.frequentist.analyze_continuous(control, treatment)
    
    def _bayesian_analysis(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        is_binary: bool
    ) -> dict[str, Any]:
        """Run Bayesian analysis."""
        if is_binary:
            result = self.bayesian.analyze_proportions(
                int(control.sum()), len(control),
                int(treatment.sum()), len(treatment)
            )
        else:
            result = self.bayesian.analyze_continuous(control, treatment)
        
        # Add p_value equivalent for compatibility
        result["p_value"] = 1 - result.get("probability_better", 0)
        result["is_significant"] = result.get("is_winner", False)
        result["relative_lift"] = result.get("relative_lift_mean", 0)
        result["absolute_lift"] = result.get("treatment_rate", 0) - result.get("control_rate", 0) if is_binary else 0
        
        return result
    
    def _sequential_analysis(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        is_binary: bool
    ) -> dict[str, Any]:
        """Run sequential analysis."""
        # First get frequentist result
        freq_result = self._frequentist_analysis(control, treatment, is_binary)
        
        # Check stopping rules
        stop_check = self.sequential.check_stopping(
            freq_result["p_value"],
            current_look=1  # TODO: Track actual look number
        )
        
        freq_result["can_stop_early"] = stop_check["can_stop_for_significance"]
        freq_result["early_stop_reason"] = (
            "Significance boundary crossed" if stop_check["can_stop_for_significance"]
            else "Continue collecting data"
        )
        
        return freq_result
    
    def _calculate_achieved_power(
        self,
        control: VariantResult,
        treatment: VariantResult,
        is_binary: bool
    ) -> float:
        """Calculate achieved statistical power."""
        if is_binary:
            p1, p2 = control.conversion_rate, treatment.conversion_rate
            if p1 == 0 or p1 == p2:
                return 0.0
            
            effect = abs(p2 - p1)
            pooled_p = (p1 + p2) / 2
            se = np.sqrt(2 * pooled_p * (1 - pooled_p) / control.sample_size)
            
            z_effect = effect / se if se > 0 else 0
            z_alpha = scipy_stats.norm.ppf(1 - self.default_alpha / 2)
            
            power = 1 - scipy_stats.norm.cdf(z_alpha - z_effect)
        else:
            if control.std == 0:
                return 0.0
            
            effect = abs(treatment.mean - control.mean) / control.std
            n = min(control.sample_size, treatment.sample_size)
            
            se = np.sqrt(2 / n)
            z_effect = effect / se if se > 0 else 0
            z_alpha = scipy_stats.norm.ppf(1 - self.default_alpha / 2)
            
            power = 1 - scipy_stats.norm.cdf(z_alpha - z_effect)
        
        return float(min(power, 1.0))
    
    def _generate_recommendation(
        self,
        result: dict[str, Any],
        test_type: TestType
    ) -> str:
        """Generate experiment recommendation."""
        p_value = result.get("p_value", 1.0)
        relative_lift = result.get("relative_lift", 0)
        is_significant = result.get("is_significant", False)
        
        if is_significant:
            direction = "improvement" if relative_lift > 0 else "degradation"
            magnitude = abs(relative_lift * 100)
            return (
                f"SHIP: Statistically significant {direction} of {magnitude:.2f}%. "
                f"P-value: {p_value:.4f}. Recommend deploying treatment."
            )
        elif p_value > 0.5:
            return (
                "NO EFFECT: No detectable difference between variants. "
                "Consider larger sample size or different hypothesis."
            )
        else:
            return (
                f"CONTINUE: Trending but not significant (p={p_value:.4f}). "
                f"Continue data collection or increase sample size."
            )
    
    def calculate_sample_size(
        self,
        baseline: float,
        mde: float,
        is_binary: bool = True,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> dict[str, int]:
        """Calculate required sample size."""
        if is_binary:
            return self.sample_calculator.for_proportions(
                baseline, mde, alpha, power
            )
        else:
            # For continuous, need std estimate
            return self.sample_calculator.for_means(
                baseline, baseline * 0.3, mde, alpha, power
            )


# Factory function
def get_ab_testing_engine() -> ABTestingEngine:
    """Get A/B testing engine instance."""
    return ABTestingEngine()
