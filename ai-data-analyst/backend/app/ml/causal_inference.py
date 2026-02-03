# AI Enterprise Data Analyst - Causal Inference Engine
# Production-grade causal analysis using DoWhy patterns and CausalML

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import DataProcessingException, ValidationException
except ImportError:
    class DataProcessingException(Exception): pass
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Causal Inference Types
# ============================================================================

class CausalMethod(str, Enum):
    """Causal inference methods."""
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    PROPENSITY_SCORE_WEIGHTING = "propensity_score_weighting"
    STRATIFICATION = "stratification"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    INVERSE_PROBABILITY_WEIGHTING = "inverse_probability_weighting"
    DOUBLY_ROBUST = "doubly_robust"


class EffectType(str, Enum):
    """Types of treatment effects."""
    ATE = "ate"  # Average Treatment Effect
    ATT = "att"  # Average Treatment Effect on Treated
    ATC = "atc"  # Average Treatment Effect on Control
    CATE = "cate"  # Conditional Average Treatment Effect


@dataclass
class CausalEffect:
    """Result of causal effect estimation."""
    
    effect_type: EffectType
    method: CausalMethod
    estimate: float
    std_error: float
    confidence_interval: tuple[float, float]
    p_value: float
    
    # Sample sizes
    n_treated: int = 0
    n_control: int = 0
    n_matched: int = 0
    
    # Diagnostics
    covariate_balance: dict[str, float] = field(default_factory=dict)
    propensity_overlap: float = 0.0
    
    # Interpretation
    is_significant: bool = False
    interpretation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_type": self.effect_type.value,
            "method": self.method.value,
            "estimate": round(self.estimate, 6),
            "std_error": round(self.std_error, 6),
            "confidence_interval": [round(ci, 6) for ci in self.confidence_interval],
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "sample_sizes": {
                "treated": self.n_treated,
                "control": self.n_control,
                "matched": self.n_matched
            },
            "covariate_balance": {k: round(v, 4) for k, v in self.covariate_balance.items()},
            "propensity_overlap": round(self.propensity_overlap, 4),
            "interpretation": self.interpretation
        }


# ============================================================================
# Propensity Score Estimator
# ============================================================================

class PropensityScoreEstimator:
    """
    Estimate propensity scores using logistic regression.
    
    The propensity score is P(treatment=1 | X)
    """
    
    def __init__(self, method: str = "logistic"):
        """
        Initialize propensity score estimator.
        
        Args:
            method: 'logistic', 'gbm', or 'random_forest'
        """
        self.method = method
        self._model = None
        self._fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series
    ) -> "PropensityScoreEstimator":
        """Fit propensity score model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Handle missing values
        X_clean = X.fillna(X.median(numeric_only=True))
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_clean)
        
        if self.method == "logistic":
            self._model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=42
            )
        elif self.method == "gbm":
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                self._model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=42
                )
            except ImportError:
                self._model = LogisticRegression(max_iter=1000)
        else:
            try:
                from sklearn.ensemble import RandomForestClassifier
                self._model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            except ImportError:
                self._model = LogisticRegression(max_iter=1000)
        
        self._model.fit(X_scaled, treatment.values)
        self._fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict propensity scores."""
        if not self._fitted:
            raise ValidationException("Model not fitted")
        
        X_clean = X.fillna(X.median(numeric_only=True))
        X_scaled = self._scaler.transform(X_clean)
        
        return self._model.predict_proba(X_scaled)[:, 1]
    
    def check_overlap(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, Any]:
        """
        Check propensity score overlap between treatment groups.
        
        Good overlap is essential for valid causal inference.
        """
        ps_treated = propensity_scores[treatment == 1]
        ps_control = propensity_scores[treatment == 0]
        
        # Calculate overlap region
        min_treated = ps_treated.min()
        max_treated = ps_treated.max()
        min_control = ps_control.min()
        max_control = ps_control.max()
        
        overlap_min = max(min_treated, min_control)
        overlap_max = min(max_treated, max_control)
        
        # Proportion in overlap region
        in_overlap = (propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)
        overlap_proportion = in_overlap.mean()
        
        # Standardized mean difference
        smd = abs(ps_treated.mean() - ps_control.mean()) / np.sqrt(
            (ps_treated.var() + ps_control.var()) / 2
        )
        
        return {
            "overlap_proportion": float(overlap_proportion),
            "overlap_range": [float(overlap_min), float(overlap_max)],
            "smd": float(smd),
            "treated_range": [float(min_treated), float(max_treated)],
            "control_range": [float(min_control), float(max_control)],
            "is_adequate": overlap_proportion > 0.9 and smd < 0.1
        }


# ============================================================================
# Propensity Score Matching
# ============================================================================

class PropensityScoreMatching:
    """
    Match treated and control units based on propensity scores.
    
    Implements:
    - Nearest neighbor matching
    - Caliper matching
    - Optimal matching
    """
    
    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: Optional[float] = 0.2,
        replacement: bool = False
    ):
        """
        Initialize matcher.
        
        Args:
            n_neighbors: Number of matches per treated unit
            caliper: Maximum distance for matching (in std of propensity)
            replacement: Allow matching with replacement
        """
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replacement = replacement
    
    def match(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Perform propensity score matching.
        
        Returns indices of matched pairs.
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        ps_treated = propensity_scores[treated_idx]
        ps_control = propensity_scores[control_idx]
        
        # Calculate caliper in std units
        caliper_value = None
        if self.caliper:
            caliper_value = self.caliper * propensity_scores.std()
        
        matched_treated = []
        matched_control = []
        used_control = set() if not self.replacement else None
        
        # Match each treated unit to nearest control
        for i, ps in enumerate(ps_treated):
            distances = np.abs(ps_control - ps)
            
            # Exclude already used controls if no replacement
            if not self.replacement and used_control:
                for used_idx in used_control:
                    distances[used_idx] = np.inf
            
            if caliper_value and distances.min() > caliper_value:
                continue  # No match within caliper
            
            # Find best matches
            best_matches = np.argsort(distances)[:self.n_neighbors]
            
            for match_idx in best_matches:
                if caliper_value and distances[match_idx] > caliper_value:
                    break
                
                matched_treated.append(treated_idx[i])
                matched_control.append(control_idx[match_idx])
                
                if not self.replacement:
                    used_control.add(match_idx)
        
        return {
            "treated_idx": np.array(matched_treated),
            "control_idx": np.array(matched_control),
            "n_matched": len(matched_treated),
            "n_unmatched_treated": len(treated_idx) - len(set(matched_treated))
        }


# ============================================================================
# Covariate Balance Checker
# ============================================================================

class CovariateBalanceChecker:
    """
    Check covariate balance after matching or weighting.
    
    Uses standardized mean difference (SMD) as the primary metric.
    """
    
    @staticmethod
    def calculate_smd(
        x_treated: np.ndarray,
        x_control: np.ndarray,
        weights_treated: Optional[np.ndarray] = None,
        weights_control: Optional[np.ndarray] = None
    ) -> float:
        """Calculate standardized mean difference."""
        if weights_treated is not None:
            mean_treated = np.average(x_treated, weights=weights_treated)
            var_treated = np.average(
                (x_treated - mean_treated) ** 2,
                weights=weights_treated
            )
        else:
            mean_treated = x_treated.mean()
            var_treated = x_treated.var()
        
        if weights_control is not None:
            mean_control = np.average(x_control, weights=weights_control)
            var_control = np.average(
                (x_control - mean_control) ** 2,
                weights=weights_control
            )
        else:
            mean_control = x_control.mean()
            var_control = x_control.var()
        
        pooled_std = np.sqrt((var_treated + var_control) / 2)
        
        if pooled_std < 1e-10:
            return 0.0
        
        return abs(mean_treated - mean_control) / pooled_std
    
    @staticmethod
    def check_balance(
        df: pd.DataFrame,
        treatment_col: str,
        covariates: list[str],
        weights: Optional[np.ndarray] = None
    ) -> dict[str, dict[str, float]]:
        """
        Check balance for all covariates.
        
        Returns SMD for each covariate before and after weighting.
        """
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        
        balance = {}
        
        for cov in covariates:
            if cov not in df.columns:
                continue
            
            x_t = treated[cov].values
            x_c = control[cov].values
            
            # Unweighted SMD
            smd_unweighted = CovariateBalanceChecker.calculate_smd(x_t, x_c)
            
            # Weighted SMD if weights provided
            smd_weighted = None
            if weights is not None:
                w_t = weights[df[treatment_col] == 1]
                w_c = weights[df[treatment_col] == 0]
                smd_weighted = CovariateBalanceChecker.calculate_smd(
                    x_t, x_c, w_t, w_c
                )
            
            balance[cov] = {
                "smd_unweighted": float(smd_unweighted),
                "smd_weighted": float(smd_weighted) if smd_weighted else None,
                "is_balanced": smd_unweighted < 0.1
            }
        
        return balance


# ============================================================================
# Causal Inference Engine
# ============================================================================

class CausalInferenceEngine:
    """
    Production-grade causal inference engine.
    
    Implements:
    - Propensity Score Matching (PSM)
    - Inverse Probability Weighting (IPW)
    - Doubly Robust estimation
    - Difference-in-Differences
    - Effect interpretation
    """
    
    def __init__(self):
        self.ps_estimator = PropensityScoreEstimator()
        self.matcher = PropensityScoreMatching()
        self.balance_checker = CovariateBalanceChecker()
    
    def estimate_ate_matching(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str],
        caliper: float = 0.2
    ) -> CausalEffect:
        """
        Estimate Average Treatment Effect using propensity score matching.
        """
        # Prepare data
        X = df[covariates].copy()
        treatment = df[treatment_col].copy()
        outcome = df[outcome_col].copy()
        
        # Estimate propensity scores
        self.ps_estimator.fit(X, treatment)
        propensity_scores = self.ps_estimator.predict_proba(X)
        
        # Check overlap
        overlap = self.ps_estimator.check_overlap(
            propensity_scores,
            treatment.values
        )
        
        # Perform matching
        self.matcher.caliper = caliper
        matches = self.matcher.match(propensity_scores, treatment.values)
        
        # Calculate matched outcome difference
        treated_outcomes = outcome.iloc[matches["treated_idx"]].values
        control_outcomes = outcome.iloc[matches["control_idx"]].values
        
        differences = treated_outcomes - control_outcomes
        
        ate = differences.mean()
        se = differences.std() / np.sqrt(len(differences))
        
        # Confidence interval (95%)
        ci = (ate - 1.96 * se, ate + 1.96 * se)
        
        # P-value (two-sided t-test)
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=len(differences) - 1))
        
        # Check covariate balance after matching
        matched_df = df.iloc[np.concatenate([matches["treated_idx"], matches["control_idx"]])]
        balance = self.balance_checker.check_balance(
            matched_df, treatment_col, covariates
        )
        
        return CausalEffect(
            effect_type=EffectType.ATE,
            method=CausalMethod.PROPENSITY_SCORE_MATCHING,
            estimate=float(ate),
            std_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            n_treated=int((treatment == 1).sum()),
            n_control=int((treatment == 0).sum()),
            n_matched=matches["n_matched"],
            covariate_balance={k: v["smd_unweighted"] for k, v in balance.items()},
            propensity_overlap=overlap["overlap_proportion"],
            is_significant=p_value < 0.05,
            interpretation=self._interpret_effect(ate, ci, p_value, outcome_col)
        )
    
    def estimate_ate_ipw(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str]
    ) -> CausalEffect:
        """
        Estimate ATE using Inverse Probability Weighting.
        """
        X = df[covariates].copy()
        treatment = df[treatment_col].values
        outcome = df[outcome_col].values
        
        # Estimate propensity scores
        self.ps_estimator.fit(X, df[treatment_col])
        ps = self.ps_estimator.predict_proba(X)
        
        # Trim extreme propensity scores for stability
        ps = np.clip(ps, 0.05, 0.95)
        
        # Calculate IPW weights
        weights = treatment / ps + (1 - treatment) / (1 - ps)
        
        # Weighted outcomes
        weighted_treated = np.sum(weights * treatment * outcome) / np.sum(weights * treatment)
        weighted_control = np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment))
        
        ate = weighted_treated - weighted_control
        
        # Bootstrap standard error
        se = self._bootstrap_se_ipw(
            df, treatment_col, outcome_col, covariates, n_bootstrap=100
        )
        
        ci = (ate - 1.96 * se, ate + 1.96 * se)
        z_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        
        # Check balance with weights
        balance = self.balance_checker.check_balance(
            df, treatment_col, covariates, weights
        )
        
        return CausalEffect(
            effect_type=EffectType.ATE,
            method=CausalMethod.INVERSE_PROBABILITY_WEIGHTING,
            estimate=float(ate),
            std_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            n_treated=int((treatment == 1).sum()),
            n_control=int((treatment == 0).sum()),
            covariate_balance={
                k: v.get("smd_weighted", v["smd_unweighted"])
                for k, v in balance.items()
            },
            is_significant=p_value < 0.05,
            interpretation=self._interpret_effect(ate, ci, p_value, outcome_col)
        )
    
    def estimate_ate_doubly_robust(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str]
    ) -> CausalEffect:
        """
        Estimate ATE using Doubly Robust estimation.
        
        Combines propensity scores with outcome regression for robustness.
        """
        from sklearn.linear_model import LinearRegression
        
        X = df[covariates].copy().fillna(0)
        treatment = df[treatment_col].values
        outcome = df[outcome_col].values
        
        # Propensity scores
        self.ps_estimator.fit(X, df[treatment_col])
        ps = np.clip(self.ps_estimator.predict_proba(X), 0.05, 0.95)
        
        # Outcome regression for each treatment group
        model_treated = LinearRegression()
        model_control = LinearRegression()
        
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        model_treated.fit(X[treated_mask], outcome[treated_mask])
        model_control.fit(X[control_mask], outcome[control_mask])
        
        # Predicted outcomes
        mu_1 = model_treated.predict(X)
        mu_0 = model_control.predict(X)
        
        # Doubly robust estimator
        dr_1 = mu_1 + treatment * (outcome - mu_1) / ps
        dr_0 = mu_0 + (1 - treatment) * (outcome - mu_0) / (1 - ps)
        
        ate = (dr_1 - dr_0).mean()
        se = (dr_1 - dr_0).std() / np.sqrt(len(outcome))
        
        ci = (ate - 1.96 * se, ate + 1.96 * se)
        z_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        
        return CausalEffect(
            effect_type=EffectType.ATE,
            method=CausalMethod.DOUBLY_ROBUST,
            estimate=float(ate),
            std_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            n_treated=int(treated_mask.sum()),
            n_control=int(control_mask.sum()),
            is_significant=p_value < 0.05,
            interpretation=self._interpret_effect(ate, ci, p_value, outcome_col)
        )
    
    def estimate_diff_in_diff(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        time_col: str,
        outcome_col: str,
        pre_period_value: Any,
        post_period_value: Any
    ) -> CausalEffect:
        """
        Estimate treatment effect using Difference-in-Differences.
        """
        # Create period indicators
        pre_mask = df[time_col] == pre_period_value
        post_mask = df[time_col] == post_period_value
        
        treated = df[treatment_col] == 1
        control = df[treatment_col] == 0
        
        # Calculate means
        y_treated_pre = df.loc[treated & pre_mask, outcome_col].mean()
        y_treated_post = df.loc[treated & post_mask, outcome_col].mean()
        y_control_pre = df.loc[control & pre_mask, outcome_col].mean()
        y_control_post = df.loc[control & post_mask, outcome_col].mean()
        
        # DiD estimator
        treated_diff = y_treated_post - y_treated_pre
        control_diff = y_control_post - y_control_pre
        did = treated_diff - control_diff
        
        # Standard error via regression
        df_copy = df[pre_mask | post_mask].copy()
        df_copy["post"] = (df_copy[time_col] == post_period_value).astype(int)
        df_copy["treat_post"] = df_copy[treatment_col] * df_copy["post"]
        
        from sklearn.linear_model import LinearRegression
        X = df_copy[[treatment_col, "post", "treat_post"]].values
        y = df_copy[outcome_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Bootstrap SE
        n_bootstrap = 100
        did_estimates = []
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(df_copy), size=len(df_copy), replace=True)
            boot_df = df_copy.iloc[boot_idx]
            
            y_t_pre = boot_df.loc[(boot_df[treatment_col] == 1) & (boot_df["post"] == 0), outcome_col].mean()
            y_t_post = boot_df.loc[(boot_df[treatment_col] == 1) & (boot_df["post"] == 1), outcome_col].mean()
            y_c_pre = boot_df.loc[(boot_df[treatment_col] == 0) & (boot_df["post"] == 0), outcome_col].mean()
            y_c_post = boot_df.loc[(boot_df[treatment_col] == 0) & (boot_df["post"] == 1), outcome_col].mean()
            
            if not any(np.isnan([y_t_pre, y_t_post, y_c_pre, y_c_post])):
                did_estimates.append((y_t_post - y_t_pre) - (y_c_post - y_c_pre))
        
        se = np.std(did_estimates) if did_estimates else 0
        ci = (did - 1.96 * se, did + 1.96 * se)
        z_stat = did / se if se > 0 else 0
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        
        return CausalEffect(
            effect_type=EffectType.ATT,
            method=CausalMethod.DIFFERENCE_IN_DIFFERENCES,
            estimate=float(did),
            std_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            n_treated=int(treated.sum()),
            n_control=int(control.sum()),
            is_significant=p_value < 0.05,
            interpretation=self._interpret_effect(did, ci, p_value, outcome_col)
        )
    
    def _bootstrap_se_ipw(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str],
        n_bootstrap: int = 100
    ) -> float:
        """Calculate bootstrap standard error for IPW."""
        estimates = []
        
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[boot_idx]
            
            X = boot_df[covariates].fillna(0)
            treatment = boot_df[treatment_col].values
            outcome = boot_df[outcome_col].values
            
            ps_estimator = PropensityScoreEstimator()
            ps_estimator.fit(X, boot_df[treatment_col])
            ps = np.clip(ps_estimator.predict_proba(X), 0.05, 0.95)
            
            weights = treatment / ps + (1 - treatment) / (1 - ps)
            
            w_t = np.sum(weights * treatment * outcome) / np.sum(weights * treatment)
            w_c = np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment))
            
            estimates.append(w_t - w_c)
        
        return np.std(estimates)
    
    def _interpret_effect(
        self,
        estimate: float,
        ci: tuple[float, float],
        p_value: float,
        outcome_name: str
    ) -> str:
        """Generate interpretation of causal effect."""
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
        direction = "increases" if estimate > 0 else "decreases"
        
        interpretation = (
            f"The treatment {direction} {outcome_name} by {abs(estimate):.4f} units "
            f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]). "
            f"This effect is {significance} (p = {p_value:.4f})."
        )
        
        if p_value < 0.001:
            interpretation += " Strong evidence of causal effect."
        elif p_value < 0.05:
            interpretation += " Moderate evidence of causal effect."
        else:
            interpretation += " Insufficient evidence of causal effect."
        
        return interpretation


# Factory function
def get_causal_inference_engine() -> CausalInferenceEngine:
    """Get causal inference engine instance."""
    return CausalInferenceEngine()
