# AI Enterprise Data Analyst - Profit Analysis Engine
# Production-grade profitability and margin analysis
# Handles: any financial data with revenue and costs

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
# Data Classes
# ============================================================================

@dataclass
class MarginMetrics:
    """Margin metrics."""
    gross_margin: float
    gross_margin_pct: float
    operating_margin: float
    operating_margin_pct: float
    net_margin: float
    net_margin_pct: float


@dataclass
class SegmentProfit:
    """Profitability by segment."""
    segment: str
    revenue: float
    cost: float
    profit: float
    margin_pct: float
    contribution_pct: float


@dataclass
class ProfitResult:
    """Complete profit analysis result."""
    # Totals
    total_revenue: float = 0.0
    total_cost: float = 0.0
    gross_profit: float = 0.0
    
    # Margins
    margins: MarginMetrics = None
    
    # By segment
    by_segment: List[SegmentProfit] = field(default_factory=list)
    by_product: List[SegmentProfit] = field(default_factory=list)
    by_period: List[SegmentProfit] = field(default_factory=list)
    
    # Insights
    most_profitable_segment: str = ""
    least_profitable_segment: str = ""
    
    # Trends
    margin_trend: str = ""  # improving, declining, stable
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_revenue": round(self.total_revenue, 2),
                "total_cost": round(self.total_cost, 2),
                "gross_profit": round(self.gross_profit, 2)
            },
            "margins": {
                "gross_margin_pct": round(self.margins.gross_margin_pct, 2) if self.margins else 0,
                "operating_margin_pct": round(self.margins.operating_margin_pct, 2) if self.margins else 0,
                "net_margin_pct": round(self.margins.net_margin_pct, 2) if self.margins else 0
            },
            "by_segment": [
                {
                    "segment": s.segment,
                    "revenue": round(s.revenue, 2),
                    "profit": round(s.profit, 2),
                    "margin_pct": round(s.margin_pct, 2),
                    "contribution_pct": round(s.contribution_pct, 2)
                }
                for s in self.by_segment[:10]
            ],
            "insights": {
                "most_profitable": self.most_profitable_segment,
                "least_profitable": self.least_profitable_segment,
                "margin_trend": self.margin_trend
            }
        }


# ============================================================================
# Profit Analysis Engine
# ============================================================================

class ProfitAnalysisEngine:
    """
    Production-grade Profit Analysis engine.
    
    Features:
    - Gross, operating, and net margins
    - Segment-level profitability
    - Product profitability
    - Period-over-period trends
    - Contribution analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        revenue_col: str = None,
        cost_col: str = None,
        segment_col: str = None,
        product_col: str = None,
        date_col: str = None,
        operating_cost_col: str = None,
        other_income_col: str = None,
        other_expense_col: str = None
    ) -> ProfitResult:
        """Perform profit analysis."""
        start_time = datetime.now()
        
        # Auto-detect columns
        if revenue_col is None:
            revenue_col = self._detect_column(df, ['revenue', 'sales', 'income', 'amount'])
        if cost_col is None:
            cost_col = self._detect_column(df, ['cost', 'cogs', 'expense', 'spend'])
        
        if revenue_col is None or cost_col is None:
            raise ValueError("Could not detect revenue or cost columns")
        
        if self.verbose:
            logger.info(f"Profit analysis: revenue={revenue_col}, cost={cost_col}")
        
        # Total calculations
        total_revenue = df[revenue_col].sum()
        total_cost = df[cost_col].sum()
        gross_profit = total_revenue - total_cost
        
        # Calculate margins
        operating_cost = df[operating_cost_col].sum() if operating_cost_col else 0
        other_income = df[other_income_col].sum() if other_income_col else 0
        other_expense = df[other_expense_col].sum() if other_expense_col else 0
        
        operating_profit = gross_profit - operating_cost
        net_profit = operating_profit + other_income - other_expense
        
        margins = MarginMetrics(
            gross_margin=gross_profit,
            gross_margin_pct=(gross_profit / total_revenue * 100) if total_revenue > 0 else 0,
            operating_margin=operating_profit,
            operating_margin_pct=(operating_profit / total_revenue * 100) if total_revenue > 0 else 0,
            net_margin=net_profit,
            net_margin_pct=(net_profit / total_revenue * 100) if total_revenue > 0 else 0
        )
        
        # Segment analysis
        by_segment = []
        if segment_col and segment_col in df.columns:
            by_segment = self._analyze_by_group(df, segment_col, revenue_col, cost_col, gross_profit)
        
        # Product analysis
        by_product = []
        if product_col and product_col in df.columns:
            by_product = self._analyze_by_group(df, product_col, revenue_col, cost_col, gross_profit)
        
        # Period analysis
        by_period = []
        if date_col and date_col in df.columns:
            by_period = self._analyze_by_period(df, date_col, revenue_col, cost_col, gross_profit)
        
        # Find best/worst segments
        most_profitable = ""
        least_profitable = ""
        if by_segment:
            sorted_segments = sorted(by_segment, key=lambda x: -x.margin_pct)
            most_profitable = sorted_segments[0].segment
            profitable_segments = [s for s in sorted_segments if s.margin_pct >= 0]
            if profitable_segments:
                least_profitable = profitable_segments[-1].segment
        
        # Trend
        margin_trend = "stable"
        if len(by_period) >= 3:
            recent_margins = [p.margin_pct for p in by_period[-3:]]
            if all(recent_margins[i] < recent_margins[i+1] for i in range(len(recent_margins)-1)):
                margin_trend = "improving"
            elif all(recent_margins[i] > recent_margins[i+1] for i in range(len(recent_margins)-1)):
                margin_trend = "declining"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProfitResult(
            total_revenue=total_revenue,
            total_cost=total_cost,
            gross_profit=gross_profit,
            margins=margins,
            by_segment=by_segment,
            by_product=by_product,
            by_period=by_period,
            most_profitable_segment=most_profitable,
            least_profitable_segment=least_profitable,
            margin_trend=margin_trend,
            processing_time_sec=processing_time
        )
    
    def _analyze_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        revenue_col: str,
        cost_col: str,
        total_profit: float
    ) -> List[SegmentProfit]:
        """Analyze profitability by group."""
        grouped = df.groupby(group_col).agg({
            revenue_col: 'sum',
            cost_col: 'sum'
        }).reset_index()
        
        results = []
        for _, row in grouped.iterrows():
            revenue = row[revenue_col]
            cost = row[cost_col]
            profit = revenue - cost
            margin_pct = (profit / revenue * 100) if revenue > 0 else 0
            contribution = (profit / total_profit * 100) if total_profit > 0 else 0
            
            results.append(SegmentProfit(
                segment=str(row[group_col]),
                revenue=float(revenue),
                cost=float(cost),
                profit=float(profit),
                margin_pct=float(margin_pct),
                contribution_pct=float(contribution)
            ))
        
        # Sort by profit descending
        results.sort(key=lambda x: -x.profit)
        return results
    
    def _analyze_by_period(
        self,
        df: pd.DataFrame,
        date_col: str,
        revenue_col: str,
        cost_col: str,
        total_profit: float
    ) -> List[SegmentProfit]:
        """Analyze profitability by time period."""
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_work = df_work.dropna(subset=[date_col])
        
        df_work['period'] = df_work[date_col].dt.to_period('M').astype(str)
        
        grouped = df_work.groupby('period').agg({
            revenue_col: 'sum',
            cost_col: 'sum'
        }).reset_index()
        
        results = []
        for _, row in grouped.iterrows():
            revenue = row[revenue_col]
            cost = row[cost_col]
            profit = revenue - cost
            margin_pct = (profit / revenue * 100) if revenue > 0 else 0
            
            results.append(SegmentProfit(
                segment=row['period'],
                revenue=float(revenue),
                cost=float(cost),
                profit=float(profit),
                margin_pct=float(margin_pct),
                contribution_pct=0
            ))
        
        # Sort by period
        results.sort(key=lambda x: x.segment)
        return results
    
    def _detect_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Detect column by name patterns."""
        for pattern in patterns:
            for col in df.columns:
                if pattern in col.lower():
                    if df[col].dtype in [np.float64, np.int64]:
                        return col
        return None
    
    def break_even_analysis(
        self,
        fixed_costs: float,
        variable_cost_per_unit: float,
        price_per_unit: float
    ) -> Dict[str, Any]:
        """Calculate break-even point."""
        contribution_margin = price_per_unit - variable_cost_per_unit
        
        if contribution_margin <= 0:
            return {
                "break_even_units": None,
                "break_even_revenue": None,
                "error": "Contribution margin is non-positive"
            }
        
        break_even_units = fixed_costs / contribution_margin
        break_even_revenue = break_even_units * price_per_unit
        
        return {
            "fixed_costs": fixed_costs,
            "variable_cost_per_unit": variable_cost_per_unit,
            "price_per_unit": price_per_unit,
            "contribution_margin": round(contribution_margin, 2),
            "contribution_margin_pct": round(contribution_margin / price_per_unit * 100, 2),
            "break_even_units": round(break_even_units, 0),
            "break_even_revenue": round(break_even_revenue, 2)
        }


# ============================================================================
# Factory Functions
# ============================================================================

def get_profit_engine() -> ProfitAnalysisEngine:
    """Get profit analysis engine."""
    return ProfitAnalysisEngine()


def quick_profit(
    df: pd.DataFrame,
    revenue_col: str = None,
    cost_col: str = None
) -> Dict[str, Any]:
    """Quick profit analysis."""
    engine = ProfitAnalysisEngine(verbose=False)
    result = engine.analyze(df, revenue_col, cost_col)
    return result.to_dict()


def calculate_margins(
    revenue: float,
    cogs: float,
    operating_expenses: float = 0,
    taxes: float = 0
) -> Dict[str, float]:
    """Quick margin calculation."""
    gross_profit = revenue - cogs
    operating_profit = gross_profit - operating_expenses
    net_profit = operating_profit - taxes
    
    return {
        "gross_margin_pct": round(gross_profit / revenue * 100, 2) if revenue > 0 else 0,
        "operating_margin_pct": round(operating_profit / revenue * 100, 2) if revenue > 0 else 0,
        "net_margin_pct": round(net_profit / revenue * 100, 2) if revenue > 0 else 0
    }
