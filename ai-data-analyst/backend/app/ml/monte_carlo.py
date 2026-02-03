# AI Enterprise Data Analyst - Monte Carlo Simulation Engine
# Scenario simulation, risk analysis, revenue forecasting

from __future__ import annotations
from dataclasses import dataclass
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


class DistributionType(str, Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BETA = "beta"
    POISSON = "poisson"


@dataclass
class SimulationResult:
    mean: float
    std: float
    median: float
    percentile_5: float
    percentile_95: float
    min_value: float
    max_value: float
    simulations: np.ndarray
    
    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "median": round(self.median, 2),
            "ci_90": [round(self.percentile_5, 2), round(self.percentile_95, 2)],
            "range": [round(self.min_value, 2), round(self.max_value, 2)],
            "probability_positive": round((self.simulations > 0).mean() * 100, 1)
        }


class MonteCarloEngine:
    """Production Monte Carlo simulation engine."""
    
    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_simulations = n_simulations
        np.random.seed(seed)
    
    def simulate_distribution(
        self, dist_type: DistributionType, params: dict, n: int = None
    ) -> np.ndarray:
        """Generate random samples from distribution."""
        n = n or self.n_simulations
        
        if dist_type == DistributionType.NORMAL:
            return np.random.normal(params.get("mean", 0), params.get("std", 1), n)
        elif dist_type == DistributionType.LOGNORMAL:
            return np.random.lognormal(params.get("mean", 0), params.get("sigma", 1), n)
        elif dist_type == DistributionType.UNIFORM:
            return np.random.uniform(params.get("low", 0), params.get("high", 1), n)
        elif dist_type == DistributionType.TRIANGULAR:
            return np.random.triangular(params.get("left", 0), params.get("mode", 0.5), 
                                        params.get("right", 1), n)
        elif dist_type == DistributionType.POISSON:
            return np.random.poisson(params.get("lam", 1), n)
        else:
            return np.random.normal(0, 1, n)
    
    def revenue_forecast(
        self, base_revenue: float, growth_mean: float, growth_std: float, 
        periods: int = 12
    ) -> dict:
        """Simulate revenue with growth uncertainty."""
        results = []
        
        for _ in range(self.n_simulations):
            revenue = base_revenue
            path = [revenue]
            for _ in range(periods):
                growth = np.random.normal(growth_mean, growth_std)
                revenue *= (1 + growth)
                path.append(revenue)
            results.append(path)
        
        results = np.array(results)
        final_revenues = results[:, -1]
        
        return {
            "base_revenue": base_revenue,
            "periods": periods,
            "final_revenue": SimulationResult(
                mean=final_revenues.mean(), std=final_revenues.std(),
                median=np.median(final_revenues),
                percentile_5=np.percentile(final_revenues, 5),
                percentile_95=np.percentile(final_revenues, 95),
                min_value=final_revenues.min(), max_value=final_revenues.max(),
                simulations=final_revenues
            ).to_dict(),
            "paths_summary": {
                "mean_path": results.mean(axis=0).tolist(),
                "upper_bound": np.percentile(results, 95, axis=0).tolist(),
                "lower_bound": np.percentile(results, 5, axis=0).tolist()
            }
        }
    
    def var_analysis(self, returns: np.ndarray, confidence: float = 0.95) -> dict:
        """Value at Risk analysis."""
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        
        return {
            "VaR": round(float(var), 4),
            "CVaR": round(float(cvar), 4) if not np.isnan(cvar) else None,
            "confidence_level": confidence,
            "interpretation": f"{confidence*100:.0f}% confidence that losses will not exceed {abs(var):.2%}"
        }
    
    def scenario_analysis(self, scenarios: list[dict]) -> dict:
        """Run multiple scenario simulations."""
        results = []
        
        for scenario in scenarios:
            name = scenario.get("name", "Scenario")
            prob = scenario.get("probability", 1.0 / len(scenarios))
            outcome_mean = scenario.get("outcome_mean", 0)
            outcome_std = scenario.get("outcome_std", 1)
            
            samples = np.random.normal(outcome_mean, outcome_std, 
                                       int(self.n_simulations * prob))
            
            results.append({
                "scenario": name,
                "probability": prob,
                "expected_outcome": round(outcome_mean, 2),
                "simulated_mean": round(samples.mean(), 2),
                "simulated_std": round(samples.std(), 2)
            })
        
        all_outcomes = np.concatenate([
            np.random.normal(s["outcome_mean"], s["outcome_std"], 
                           int(self.n_simulations * s["probability"]))
            for s in scenarios
        ])
        
        return {
            "scenarios": results,
            "combined": {
                "expected_value": round(all_outcomes.mean(), 2),
                "std": round(all_outcomes.std(), 2),
                "p5": round(np.percentile(all_outcomes, 5), 2),
                "p95": round(np.percentile(all_outcomes, 95), 2)
            }
        }
    
    def sensitivity_analysis(
        self, model_func: callable, base_params: dict, 
        param_ranges: dict[str, tuple]
    ) -> dict:
        """Analyze sensitivity of output to input parameters."""
        base_result = model_func(**base_params)
        sensitivities = {}
        
        for param, (low, high) in param_ranges.items():
            results = []
            test_values = np.linspace(low, high, 20)
            
            for val in test_values:
                test_params = base_params.copy()
                test_params[param] = val
                results.append(model_func(**test_params))
            
            sensitivities[param] = {
                "values": test_values.tolist(),
                "outputs": results,
                "elasticity": round((max(results) - min(results)) / base_result, 4) 
                              if base_result != 0 else 0
            }
        
        return {
            "base_result": base_result,
            "sensitivities": sensitivities,
            "most_sensitive": max(sensitivities.keys(), 
                                  key=lambda k: abs(sensitivities[k]["elasticity"]))
        }


def get_monte_carlo_engine(n_simulations: int = 10000) -> MonteCarloEngine:
    return MonteCarloEngine(n_simulations)
