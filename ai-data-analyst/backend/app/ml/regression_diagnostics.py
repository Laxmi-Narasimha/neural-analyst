# AI Enterprise Data Analyst - Regression Diagnostics Engine
# Production-grade regression model diagnostics
# Handles: any regression model, comprehensive diagnostics

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

class DiagnosticStatus(str, Enum):
    """Diagnostic test status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DiagnosticTest:
    """Single diagnostic test result."""
    name: str
    description: str
    statistic: float
    p_value: Optional[float]
    status: DiagnosticStatus
    interpretation: str


@dataclass
class RegressionDiagnosticsResult:
    """Complete regression diagnostics result."""
    # Model fit metrics
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    
    # Diagnostic tests
    tests: List[DiagnosticTest] = field(default_factory=list)
    
    # Residual analysis
    residual_mean: float = 0.0
    residual_std: float = 0.0
    
    # Influence analysis
    high_leverage_points: List[int] = field(default_factory=list)
    high_influence_points: List[int] = field(default_factory=list)
    
    # Overall assessment
    model_valid: bool = True
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_fit": {
                "r_squared": round(self.r_squared, 4),
                "adj_r_squared": round(self.adj_r_squared, 4),
                "rmse": round(self.rmse, 4),
                "mae": round(self.mae, 4)
            },
            "diagnostics": [
                {
                    "name": t.name,
                    "statistic": round(t.statistic, 4),
                    "p_value": round(t.p_value, 4) if t.p_value else None,
                    "status": t.status.value,
                    "interpretation": t.interpretation
                }
                for t in self.tests
            ],
            "model_valid": self.model_valid,
            "issues": self.issues,
            "recommendations": self.recommendations
        }


# ============================================================================
# Regression Diagnostics Engine
# ============================================================================

class RegressionDiagnosticsEngine:
    """
    Regression Diagnostics engine.
    
    Features:
    - Residual analysis
    - Normality testing
    - Heteroscedasticity testing
    - Autocorrelation testing
    - Influence analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def diagnose(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: np.ndarray = None,
        alpha: float = 0.05
    ) -> RegressionDiagnosticsResult:
        """Perform comprehensive regression diagnostics."""
        start_time = datetime.now()
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if self.verbose:
            logger.info(f"Running regression diagnostics on {len(y_true)} observations")
        
        residuals = y_true - y_pred
        n = len(residuals)
        k = X.shape[1] if X is not None else 1
        
        # Model fit metrics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else r_squared
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        
        tests = []
        issues = []
        recommendations = []
        
        # 1. Normality of residuals
        normality_test = self._test_normality(residuals, alpha)
        tests.append(normality_test)
        if normality_test.status == DiagnosticStatus.FAILED:
            issues.append("Residuals are not normally distributed")
            recommendations.append("Consider non-linear transformation or robust regression")
        
        # 2. Homoscedasticity (Breusch-Pagan-like)
        hetero_test = self._test_heteroscedasticity(residuals, y_pred, alpha)
        tests.append(hetero_test)
        if hetero_test.status == DiagnosticStatus.FAILED:
            issues.append("Heteroscedasticity detected")
            recommendations.append("Consider weighted least squares or log transformation")
        
        # 3. Autocorrelation (Durbin-Watson)
        autocorr_test = self._test_autocorrelation(residuals)
        tests.append(autocorr_test)
        if autocorr_test.status == DiagnosticStatus.FAILED:
            issues.append("Autocorrelation in residuals")
            recommendations.append("Consider time series model or add lagged variables")
        
        # 4. Mean of residuals
        mean_test = self._test_residual_mean(residuals, alpha)
        tests.append(mean_test)
        
        # 5. Influential points
        if X is not None:
            leverage_points, influence_points = self._analyze_influence(X, residuals)
        else:
            leverage_points, influence_points = [], []
        
        if len(influence_points) > 0:
            issues.append(f"Found {len(influence_points)} highly influential points")
            recommendations.append("Investigate and consider removing influential outliers")
        
        # Overall assessment
        model_valid = len([t for t in tests if t.status == DiagnosticStatus.FAILED]) <= 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RegressionDiagnosticsResult(
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            rmse=rmse,
            mae=mae,
            tests=tests,
            residual_mean=float(np.mean(residuals)),
            residual_std=float(np.std(residuals)),
            high_leverage_points=leverage_points,
            high_influence_points=influence_points,
            model_valid=model_valid,
            issues=issues,
            recommendations=recommendations,
            processing_time_sec=processing_time
        )
    
    def _test_normality(self, residuals: np.ndarray, alpha: float) -> DiagnosticTest:
        """Test normality of residuals."""
        if len(residuals) < 20:
            stat, p = scipy_stats.shapiro(residuals)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = scipy_stats.normaltest(residuals)
            test_name = "D'Agostino-Pearson"
        
        if p > alpha:
            status = DiagnosticStatus.PASSED
            interp = "Residuals are normally distributed"
        elif p > 0.01:
            status = DiagnosticStatus.WARNING
            interp = "Weak evidence of non-normality"
        else:
            status = DiagnosticStatus.FAILED
            interp = "Residuals are not normally distributed"
        
        return DiagnosticTest(
            name=f"Normality ({test_name})",
            description="Tests if residuals follow normal distribution",
            statistic=stat,
            p_value=p,
            status=status,
            interpretation=interp
        )
    
    def _test_heteroscedasticity(
        self,
        residuals: np.ndarray,
        y_pred: np.ndarray,
        alpha: float
    ) -> DiagnosticTest:
        """Test for heteroscedasticity."""
        # Simple test: correlation between |residuals| and predicted values
        abs_residuals = np.abs(residuals)
        
        corr, p = scipy_stats.spearmanr(abs_residuals, y_pred)
        
        if p > alpha:
            status = DiagnosticStatus.PASSED
            interp = "Homoscedasticity assumption holds"
        elif p > 0.01:
            status = DiagnosticStatus.WARNING
            interp = "Mild heteroscedasticity detected"
        else:
            status = DiagnosticStatus.FAILED
            interp = "Significant heteroscedasticity detected"
        
        return DiagnosticTest(
            name="Heteroscedasticity",
            description="Tests if variance of residuals is constant",
            statistic=abs(corr),
            p_value=p,
            status=status,
            interpretation=interp
        )
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> DiagnosticTest:
        """Test for autocorrelation using Durbin-Watson."""
        n = len(residuals)
        diff_sum = np.sum(np.diff(residuals) ** 2)
        res_sum = np.sum(residuals ** 2)
        
        dw = diff_sum / res_sum if res_sum > 0 else 2.0
        
        if 1.5 <= dw <= 2.5:
            status = DiagnosticStatus.PASSED
            interp = "No significant autocorrelation"
        elif 1.0 <= dw <= 3.0:
            status = DiagnosticStatus.WARNING
            interp = "Possible mild autocorrelation"
        else:
            status = DiagnosticStatus.FAILED
            interp = "Significant autocorrelation detected"
        
        return DiagnosticTest(
            name="Durbin-Watson",
            description="Tests for autocorrelation in residuals",
            statistic=dw,
            p_value=None,
            status=status,
            interpretation=interp
        )
    
    def _test_residual_mean(self, residuals: np.ndarray, alpha: float) -> DiagnosticTest:
        """Test if mean of residuals is zero."""
        stat, p = scipy_stats.ttest_1samp(residuals, 0)
        
        if p > alpha:
            status = DiagnosticStatus.PASSED
            interp = "Mean of residuals is not significantly different from zero"
        else:
            status = DiagnosticStatus.FAILED
            interp = "Mean of residuals differs from zero (bias detected)"
        
        return DiagnosticTest(
            name="Residual Mean",
            description="Tests if mean of residuals equals zero",
            statistic=np.mean(residuals),
            p_value=p,
            status=status,
            interpretation=interp
        )
    
    def _analyze_influence(
        self,
        X: np.ndarray,
        residuals: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Analyze influential observations."""
        n, k = X.shape
        
        # Hat matrix diagonal (leverage)
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            H = X @ XtX_inv @ X.T
            leverage = np.diag(H)
        except:
            leverage = np.zeros(n)
        
        # Cook's distance
        mse = np.sum(residuals ** 2) / (n - k)
        cooks_d = (residuals ** 2 / (k * mse)) * (leverage / (1 - leverage + 1e-10) ** 2)
        
        # Thresholds
        leverage_threshold = 2 * k / n
        cooks_threshold = 4 / n
        
        high_leverage = np.where(leverage > leverage_threshold)[0].tolist()
        high_influence = np.where(cooks_d > cooks_threshold)[0].tolist()
        
        return high_leverage, high_influence


# ============================================================================
# Factory Functions
# ============================================================================

def get_regression_diagnostics_engine() -> RegressionDiagnosticsEngine:
    """Get regression diagnostics engine."""
    return RegressionDiagnosticsEngine()


def quick_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """Quick regression diagnostics."""
    engine = RegressionDiagnosticsEngine(verbose=False)
    result = engine.diagnose(y_true, y_pred)
    return result.to_dict()
