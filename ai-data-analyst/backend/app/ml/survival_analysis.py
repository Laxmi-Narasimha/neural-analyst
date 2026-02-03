# AI Enterprise Data Analyst - Survival Analysis Engine
# Kaplan-Meier, Cox PH, Churn Survival Curves

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np
import pandas as pd
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SurvivalMethod(str, Enum):
    KAPLAN_MEIER = "kaplan_meier"
    COX_PH = "cox_proportional_hazards"


@dataclass
class SurvivalPoint:
    time: float
    survival_prob: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_at_risk: int = 0
    n_events: int = 0


@dataclass  
class SurvivalCurve:
    group: str = "all"
    points: list[SurvivalPoint] = field(default_factory=list)
    median_survival: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "median_survival": self.median_survival,
            "curve": [{"time": p.time, "survival": round(p.survival_prob, 4)} for p in self.points]
        }


class KaplanMeierEstimator:
    """Non-parametric survival estimator with CI."""
    
    def fit(self, durations: np.ndarray, events: np.ndarray) -> SurvivalCurve:
        sorted_idx = np.argsort(durations)
        times, events = durations[sorted_idx], events[sorted_idx]
        unique_times = np.unique(times[events == 1])
        
        survival_prob = 1.0
        points = [SurvivalPoint(0, 1.0, 1.0, 1.0, len(times), 0)]
        variance_sum = 0.0
        
        for t in unique_times:
            n_at_risk = int((times >= t).sum())
            n_events = int(((times == t) & (events == 1)).sum())
            
            if n_at_risk > 0 and n_events > 0:
                survival_prob *= (n_at_risk - n_events) / n_at_risk
                if n_at_risk > n_events:
                    variance_sum += n_events / (n_at_risk * (n_at_risk - n_events))
                
                std_err = survival_prob * np.sqrt(variance_sum)
                points.append(SurvivalPoint(
                    float(t), float(survival_prob),
                    max(0, survival_prob - 1.96 * std_err),
                    min(1, survival_prob + 1.96 * std_err),
                    n_at_risk, n_events
                ))
        
        median = self._find_median(points)
        return SurvivalCurve(points=points, median_survival=median)
    
    def _find_median(self, points: list[SurvivalPoint]) -> Optional[float]:
        for p in points:
            if p.survival_prob <= 0.5:
                return p.time
        return None


class LogRankTest:
    """Compare survival between two groups."""
    
    def compare(self, g1_dur, g1_ev, g2_dur, g2_ev, alpha=0.05) -> dict:
        from scipy import stats
        all_times = np.unique(np.concatenate([g1_dur[g1_ev==1], g2_dur[g2_ev==1]]))
        
        O1, E1, V = 0, 0, 0
        for t in all_times:
            n1, n2 = (g1_dur >= t).sum(), (g2_dur >= t).sum()
            d1 = ((g1_dur == t) & (g1_ev == 1)).sum()
            d = d1 + ((g2_dur == t) & (g2_ev == 1)).sum()
            n = n1 + n2
            
            if n > 1:
                O1 += d1
                E1 += n1 * d / n
                V += n1 * n2 * d * (n - d) / (n * n * (n - 1))
        
        chi2 = (O1 - E1) ** 2 / V if V > 0 else 0
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        return {"chi2": round(chi2, 4), "p_value": round(p_value, 6), "significant": p_value < alpha}


class SurvivalAnalysisEngine:
    """Production survival analysis."""
    
    def __init__(self):
        self.km = KaplanMeierEstimator()
        self.logrank = LogRankTest()
    
    def kaplan_meier(self, df: pd.DataFrame, duration_col: str, event_col: str, 
                     group_col: str = None) -> dict:
        results = {"curves": []}
        
        if group_col is None:
            curve = self.km.fit(df[duration_col].values, df[event_col].values)
            results["curves"].append(curve.to_dict())
        else:
            for grp in df[group_col].unique():
                g = df[df[group_col] == grp]
                curve = self.km.fit(g[duration_col].values, g[event_col].values)
                curve.group = str(grp)
                results["curves"].append(curve.to_dict())
            
            if len(df[group_col].unique()) == 2:
                groups = list(df[group_col].unique())
                g1, g2 = df[df[group_col]==groups[0]], df[df[group_col]==groups[1]]
                results["logrank"] = self.logrank.compare(
                    g1[duration_col].values, g1[event_col].values,
                    g2[duration_col].values, g2[event_col].values
                )
        return results
    
    def churn_survival(self, df: pd.DataFrame, signup_col: str, 
                       churn_col: str = None) -> dict:
        df = df.copy()
        df[signup_col] = pd.to_datetime(df[signup_col])
        ref_date = pd.Timestamp.now()
        
        if churn_col and churn_col in df.columns:
            df[churn_col] = pd.to_datetime(df[churn_col])
            df["duration"] = (df[churn_col].fillna(ref_date) - df[signup_col]).dt.days
            df["churned"] = df[churn_col].notna().astype(int)
        else:
            df["duration"] = (ref_date - df[signup_col]).dt.days
            df["churned"] = 0
        
        df = df[df["duration"] > 0]
        curve = self.km.fit(df["duration"].values, df["churned"].values)
        
        return {
            "curve": curve.to_dict(),
            "n_customers": len(df),
            "churn_rate": round(df["churned"].mean() * 100, 2),
            "median_lifetime_days": curve.median_survival
        }


def get_survival_analysis_engine() -> SurvivalAnalysisEngine:
    return SurvivalAnalysisEngine()
