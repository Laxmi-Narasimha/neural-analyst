# AI Enterprise Data Analyst - Investment Analysis Engine
# Production-grade investment metrics and analysis
# Handles: NPV, IRR, ROI, payback period

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

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

class InvestmentDecision(str, Enum):
    """Investment decision recommendation."""
    STRONGLY_ACCEPT = "strongly_accept"
    ACCEPT = "accept"
    MARGINAL = "marginal"
    REJECT = "reject"
    STRONGLY_REJECT = "strongly_reject"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InvestmentMetrics:
    """Investment analysis metrics."""
    npv: float  # Net Present Value
    irr: Optional[float]  # Internal Rate of Return
    roi: float  # Return on Investment
    payback_period: Optional[float]  # Years to recoup investment
    profitability_index: float
    breakeven_point: Optional[float]


@dataclass
class InvestmentResult:
    """Complete investment analysis result."""
    initial_investment: float = 0.0
    total_cashflows: float = 0.0
    discount_rate: float = 0.0
    
    # Metrics
    metrics: InvestmentMetrics = None
    
    # Decision
    decision: InvestmentDecision = InvestmentDecision.MARGINAL
    decision_rationale: str = ""
    
    # Sensitivity
    npv_sensitivity: Dict[float, float] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "initial_investment": round(self.initial_investment, 2),
                "total_cashflows": round(self.total_cashflows, 2),
                "discount_rate": f"{self.discount_rate * 100:.1f}%"
            },
            "metrics": {
                "npv": round(self.metrics.npv, 2),
                "irr": f"{self.metrics.irr * 100:.2f}%" if self.metrics.irr else None,
                "roi": f"{self.metrics.roi * 100:.1f}%",
                "payback_period": f"{self.metrics.payback_period:.1f} years" if self.metrics.payback_period else "Never",
                "profitability_index": round(self.metrics.profitability_index, 2)
            },
            "decision": {
                "recommendation": self.decision.value,
                "rationale": self.decision_rationale
            },
            "npv_sensitivity": {f"{k*100:.0f}%": round(v, 2) for k, v in self.npv_sensitivity.items()}
        }


# ============================================================================
# Investment Analysis Engine
# ============================================================================

class InvestmentAnalysisEngine:
    """
    Investment Analysis engine.
    
    Features:
    - NPV calculation
    - IRR calculation
    - Payback period
    - ROI and profitability index
    - Sensitivity analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        initial_investment: float,
        cashflows: List[float],
        discount_rate: float = 0.10,
        hurdle_rate: float = None
    ) -> InvestmentResult:
        """Perform investment analysis."""
        start_time = datetime.now()
        
        hurdle_rate = hurdle_rate or discount_rate
        
        if self.verbose:
            logger.info(f"Analyzing investment: ${initial_investment:,.0f}, {len(cashflows)} periods")
        
        # NPV
        npv = self._calculate_npv(initial_investment, cashflows, discount_rate)
        
        # IRR
        irr = self._calculate_irr(initial_investment, cashflows)
        
        # ROI
        total_cashflows = sum(cashflows)
        roi = (total_cashflows - initial_investment) / initial_investment if initial_investment > 0 else 0
        
        # Payback period
        payback = self._calculate_payback(initial_investment, cashflows)
        
        # Profitability Index
        pv_cashflows = sum(cf / (1 + discount_rate) ** (t + 1) for t, cf in enumerate(cashflows))
        pi = pv_cashflows / initial_investment if initial_investment > 0 else 0
        
        metrics = InvestmentMetrics(
            npv=npv,
            irr=irr,
            roi=roi,
            payback_period=payback,
            profitability_index=pi,
            breakeven_point=None
        )
        
        # Decision
        decision, rationale = self._make_decision(metrics, hurdle_rate)
        
        # Sensitivity analysis
        sensitivity = {}
        for rate in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            sensitivity[rate] = self._calculate_npv(initial_investment, cashflows, rate)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return InvestmentResult(
            initial_investment=initial_investment,
            total_cashflows=total_cashflows,
            discount_rate=discount_rate,
            metrics=metrics,
            decision=decision,
            decision_rationale=rationale,
            npv_sensitivity=sensitivity,
            processing_time_sec=processing_time
        )
    
    def _calculate_npv(
        self,
        initial: float,
        cashflows: List[float],
        rate: float
    ) -> float:
        """Calculate Net Present Value."""
        pv = sum(cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
        return pv - initial
    
    def _calculate_irr(
        self,
        initial: float,
        cashflows: List[float],
        max_iter: int = 100
    ) -> Optional[float]:
        """Calculate Internal Rate of Return using Newton's method."""
        all_cfs = [-initial] + list(cashflows)
        
        # Check if IRR exists
        if sum(cashflows) <= initial:
            return None
        
        # Newton-Raphson
        irr = 0.10  # Initial guess
        
        for _ in range(max_iter):
            npv = sum(cf / (1 + irr) ** t for t, cf in enumerate(all_cfs))
            npv_deriv = sum(-t * cf / (1 + irr) ** (t + 1) for t, cf in enumerate(all_cfs))
            
            if abs(npv_deriv) < 1e-10:
                break
            
            new_irr = irr - npv / npv_deriv
            
            if abs(new_irr - irr) < 1e-7:
                return new_irr
            
            irr = new_irr
            
            # Bounds check
            if irr < -0.99 or irr > 10:
                return None
        
        return irr if -0.99 < irr < 10 else None
    
    def _calculate_payback(
        self,
        initial: float,
        cashflows: List[float]
    ) -> Optional[float]:
        """Calculate payback period."""
        cumulative = 0
        
        for t, cf in enumerate(cashflows):
            cumulative += cf
            
            if cumulative >= initial:
                # Interpolate
                if t == 0:
                    return initial / cf if cf > 0 else None
                
                prev_cumulative = cumulative - cf
                fraction = (initial - prev_cumulative) / cf if cf > 0 else 0
                return t + fraction
        
        return None  # Never pays back
    
    def _make_decision(
        self,
        metrics: InvestmentMetrics,
        hurdle_rate: float
    ) -> tuple:
        """Make investment decision."""
        score = 0
        reasons = []
        
        # NPV criterion
        if metrics.npv > 0:
            score += 2
            reasons.append("Positive NPV")
        else:
            score -= 2
            reasons.append("Negative NPV")
        
        # IRR criterion
        if metrics.irr is not None:
            if metrics.irr > hurdle_rate * 1.5:
                score += 2
                reasons.append(f"IRR exceeds hurdle by 50%+")
            elif metrics.irr > hurdle_rate:
                score += 1
                reasons.append("IRR exceeds hurdle rate")
            else:
                score -= 1
                reasons.append("IRR below hurdle rate")
        
        # PI criterion
        if metrics.profitability_index > 1.2:
            score += 1
            reasons.append("Strong profitability index")
        elif metrics.profitability_index < 1.0:
            score -= 1
            reasons.append("PI below 1.0")
        
        # Payback criterion
        if metrics.payback_period is not None:
            if metrics.payback_period <= 3:
                score += 1
                reasons.append("Quick payback")
            elif metrics.payback_period > 7:
                score -= 1
                reasons.append("Slow payback")
        
        # Determine decision
        if score >= 4:
            decision = InvestmentDecision.STRONGLY_ACCEPT
        elif score >= 2:
            decision = InvestmentDecision.ACCEPT
        elif score >= 0:
            decision = InvestmentDecision.MARGINAL
        elif score >= -2:
            decision = InvestmentDecision.REJECT
        else:
            decision = InvestmentDecision.STRONGLY_REJECT
        
        return decision, "; ".join(reasons)


# ============================================================================
# Factory Functions
# ============================================================================

def get_investment_engine() -> InvestmentAnalysisEngine:
    """Get investment analysis engine."""
    return InvestmentAnalysisEngine()


def quick_investment(
    initial: float,
    cashflows: List[float],
    discount_rate: float = 0.10
) -> Dict[str, Any]:
    """Quick investment analysis."""
    engine = InvestmentAnalysisEngine(verbose=False)
    result = engine.analyze(initial, cashflows, discount_rate)
    return result.to_dict()
