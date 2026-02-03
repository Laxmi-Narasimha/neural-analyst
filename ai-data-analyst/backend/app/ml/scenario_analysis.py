# AI Enterprise Data Analyst - Scenario Analysis Engine
# Production-grade what-if and scenario analysis
# Handles: any numeric data, sensitivity analysis

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

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

class ScenarioType(str, Enum):
    """Types of scenarios."""
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"
    CUSTOM = "custom"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Scenario:
    """Single scenario definition."""
    name: str
    scenario_type: ScenarioType
    adjustments: Dict[str, float]  # Variable: adjustment factor
    output_value: Optional[float] = None
    change_from_base: Optional[float] = None


@dataclass
class SensitivityPoint:
    """Single sensitivity data point."""
    variable: str
    adjustment: float
    output_value: float
    change_pct: float


@dataclass
class ScenarioResult:
    """Complete scenario analysis result."""
    base_value: float = 0.0
    
    # Scenarios
    scenarios: List[Scenario] = field(default_factory=list)
    
    # Sensitivity analysis
    sensitivity: Dict[str, List[SensitivityPoint]] = field(default_factory=dict)
    
    # Key insights
    most_sensitive_variable: str = ""
    sensitivity_ranking: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_value": round(self.base_value, 2),
            "scenarios": [
                {
                    "name": s.name,
                    "type": s.scenario_type.value,
                    "output": round(s.output_value, 2) if s.output_value else None,
                    "change_pct": round(s.change_from_base * 100, 2) if s.change_from_base else None
                }
                for s in self.scenarios
            ],
            "most_sensitive": self.most_sensitive_variable,
            "sensitivity_ranking": self.sensitivity_ranking[:10]
        }


# ============================================================================
# Scenario Analysis Engine
# ============================================================================

class ScenarioAnalysisEngine:
    """
    Scenario Analysis engine.
    
    Features:
    - What-if analysis
    - Sensitivity analysis
    - Multiple scenario comparison
    - Variable impact ranking
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        base_values: Dict[str, float],
        formula: Callable[[Dict[str, float]], float],
        scenarios: List[Dict[str, Any]] = None,
        sensitivity_range: float = 0.2
    ) -> ScenarioResult:
        """Perform scenario analysis."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Scenario analysis with {len(base_values)} variables")
        
        # Calculate base value
        base_output = formula(base_values)
        
        # Generate default scenarios if not provided
        if scenarios is None:
            scenarios = self._generate_default_scenarios(base_values)
        
        # Calculate scenario outputs
        scenario_results = []
        for scenario_def in scenarios:
            adjusted = base_values.copy()
            for var, adj in scenario_def.get('adjustments', {}).items():
                if var in adjusted:
                    adjusted[var] = adjusted[var] * (1 + adj)
            
            output = formula(adjusted)
            change = (output - base_output) / base_output if base_output != 0 else 0
            
            scenario_results.append(Scenario(
                name=scenario_def.get('name', 'Custom'),
                scenario_type=ScenarioType(scenario_def.get('type', 'custom')),
                adjustments=scenario_def.get('adjustments', {}),
                output_value=output,
                change_from_base=change
            ))
        
        # Sensitivity analysis
        sensitivity = {}
        sensitivity_impacts = {}
        
        for var in base_values:
            sensitivity[var] = []
            
            for adj in np.linspace(-sensitivity_range, sensitivity_range, 11):
                adjusted = base_values.copy()
                adjusted[var] = adjusted[var] * (1 + adj)
                
                output = formula(adjusted)
                change = (output - base_output) / base_output if base_output != 0 else 0
                
                sensitivity[var].append(SensitivityPoint(
                    variable=var,
                    adjustment=adj,
                    output_value=output,
                    change_pct=change * 100
                ))
            
            # Calculate sensitivity impact (slope)
            changes = [p.change_pct for p in sensitivity[var]]
            sensitivity_impacts[var] = max(changes) - min(changes)
        
        # Rank by sensitivity
        ranking = sorted(sensitivity_impacts.keys(), key=lambda x: -sensitivity_impacts[x])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ScenarioResult(
            base_value=base_output,
            scenarios=scenario_results,
            sensitivity=sensitivity,
            most_sensitive_variable=ranking[0] if ranking else "",
            sensitivity_ranking=ranking,
            processing_time_sec=processing_time
        )
    
    def _generate_default_scenarios(
        self,
        base_values: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate default optimistic/pessimistic scenarios."""
        return [
            {
                "name": "Base Case",
                "type": "base",
                "adjustments": {}
            },
            {
                "name": "Optimistic (+10%)",
                "type": "optimistic",
                "adjustments": {k: 0.10 for k in base_values}
            },
            {
                "name": "Pessimistic (-10%)",
                "type": "pessimistic",
                "adjustments": {k: -0.10 for k in base_values}
            },
            {
                "name": "Best Case (+20%)",
                "type": "optimistic",
                "adjustments": {k: 0.20 for k in base_values}
            },
            {
                "name": "Worst Case (-20%)",
                "type": "pessimistic",
                "adjustments": {k: -0.20 for k in base_values}
            }
        ]
    
    def monte_carlo(
        self,
        base_values: Dict[str, float],
        formula: Callable[[Dict[str, float]], float],
        std_devs: Dict[str, float],
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Monte Carlo simulation."""
        results = []
        
        for _ in range(n_simulations):
            simulated = {}
            for var, base in base_values.items():
                std = std_devs.get(var, base * 0.1)
                simulated[var] = np.random.normal(base, std)
            
            results.append(formula(simulated))
        
        results = np.array(results)
        
        return {
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "median": float(np.median(results)),
            "percentile_5": float(np.percentile(results, 5)),
            "percentile_95": float(np.percentile(results, 95)),
            "min": float(np.min(results)),
            "max": float(np.max(results))
        }


# ============================================================================
# Factory Functions
# ============================================================================

def get_scenario_engine() -> ScenarioAnalysisEngine:
    """Get scenario analysis engine."""
    return ScenarioAnalysisEngine()


def quick_scenario(
    base_values: Dict[str, float],
    formula: Callable[[Dict[str, float]], float]
) -> Dict[str, Any]:
    """Quick scenario analysis."""
    engine = ScenarioAnalysisEngine(verbose=False)
    result = engine.analyze(base_values, formula)
    return result.to_dict()
