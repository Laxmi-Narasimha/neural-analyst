# AI Enterprise Data Analyst - Financial Ratios Engine
# Production-grade financial ratio calculations
# Handles: any financial data, comprehensive ratio analysis

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums and Types
# ============================================================================

class RatioCategory(str, Enum):
    """Categories of financial ratios."""
    LIQUIDITY = "liquidity"
    PROFITABILITY = "profitability"
    EFFICIENCY = "efficiency"
    LEVERAGE = "leverage"
    VALUATION = "valuation"
    GROWTH = "growth"


class RatioInterpretation(str, Enum):
    """Ratio interpretation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FinancialRatio:
    """Single financial ratio calculation."""
    name: str
    value: float
    category: RatioCategory
    formula: str
    interpretation: RatioInterpretation
    benchmark: Optional[float] = None
    industry_avg: Optional[float] = None
    trend: Optional[str] = None


@dataclass
class FinancialData:
    """Financial data inputs."""
    # Balance Sheet
    current_assets: float = 0.0
    current_liabilities: float = 0.0
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    shareholders_equity: float = 0.0
    inventory: float = 0.0
    accounts_receivable: float = 0.0
    accounts_payable: float = 0.0
    cash: float = 0.0
    short_term_debt: float = 0.0
    long_term_debt: float = 0.0
    
    # Income Statement
    revenue: float = 0.0
    cost_of_goods_sold: float = 0.0
    gross_profit: float = 0.0
    operating_income: float = 0.0
    net_income: float = 0.0
    ebitda: float = 0.0
    interest_expense: float = 0.0
    
    # Cash Flow
    operating_cash_flow: float = 0.0
    capital_expenditures: float = 0.0
    free_cash_flow: float = 0.0
    dividends_paid: float = 0.0
    
    # Market Data
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    stock_price: float = 0.0
    
    # Previous Period (for growth)
    prev_revenue: float = 0.0
    prev_net_income: float = 0.0
    prev_total_assets: float = 0.0


@dataclass
class FinancialRatioResult:
    """Complete financial ratio analysis result."""
    ratios: List[FinancialRatio] = field(default_factory=list)
    by_category: Dict[str, List[FinancialRatio]] = field(default_factory=dict)
    overall_health: str = ""
    key_insights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_health": self.overall_health,
            "ratios": [
                {
                    "name": r.name,
                    "value": round(r.value, 4) if not np.isnan(r.value) and not np.isinf(r.value) else None,
                    "category": r.category.value,
                    "interpretation": r.interpretation.value
                }
                for r in self.ratios
            ],
            "by_category": {
                cat: [{"name": r.name, "value": round(r.value, 4) if not np.isnan(r.value) else None} for r in ratios]
                for cat, ratios in self.by_category.items()
            },
            "key_insights": self.key_insights[:10],
            "warnings": self.warnings[:5]
        }


# ============================================================================
# Financial Ratio Calculators
# ============================================================================

class LiquidityRatios:
    """Liquidity ratio calculations."""
    
    @staticmethod
    def current_ratio(data: FinancialData) -> float:
        if data.current_liabilities == 0:
            return np.nan
        return data.current_assets / data.current_liabilities
    
    @staticmethod
    def quick_ratio(data: FinancialData) -> float:
        if data.current_liabilities == 0:
            return np.nan
        return (data.current_assets - data.inventory) / data.current_liabilities
    
    @staticmethod
    def cash_ratio(data: FinancialData) -> float:
        if data.current_liabilities == 0:
            return np.nan
        return data.cash / data.current_liabilities


class ProfitabilityRatios:
    """Profitability ratio calculations."""
    
    @staticmethod
    def gross_margin(data: FinancialData) -> float:
        if data.revenue == 0:
            return np.nan
        return data.gross_profit / data.revenue
    
    @staticmethod
    def operating_margin(data: FinancialData) -> float:
        if data.revenue == 0:
            return np.nan
        return data.operating_income / data.revenue
    
    @staticmethod
    def net_margin(data: FinancialData) -> float:
        if data.revenue == 0:
            return np.nan
        return data.net_income / data.revenue
    
    @staticmethod
    def return_on_assets(data: FinancialData) -> float:
        if data.total_assets == 0:
            return np.nan
        return data.net_income / data.total_assets
    
    @staticmethod
    def return_on_equity(data: FinancialData) -> float:
        if data.shareholders_equity == 0:
            return np.nan
        return data.net_income / data.shareholders_equity
    
    @staticmethod
    def return_on_capital(data: FinancialData) -> float:
        invested_capital = data.shareholders_equity + data.long_term_debt
        if invested_capital == 0:
            return np.nan
        return data.operating_income / invested_capital


class EfficiencyRatios:
    """Efficiency ratio calculations."""
    
    @staticmethod
    def asset_turnover(data: FinancialData) -> float:
        if data.total_assets == 0:
            return np.nan
        return data.revenue / data.total_assets
    
    @staticmethod
    def inventory_turnover(data: FinancialData) -> float:
        if data.inventory == 0:
            return np.nan
        return data.cost_of_goods_sold / data.inventory
    
    @staticmethod
    def receivables_turnover(data: FinancialData) -> float:
        if data.accounts_receivable == 0:
            return np.nan
        return data.revenue / data.accounts_receivable
    
    @staticmethod
    def days_sales_outstanding(data: FinancialData) -> float:
        turnover = EfficiencyRatios.receivables_turnover(data)
        if np.isnan(turnover) or turnover == 0:
            return np.nan
        return 365 / turnover
    
    @staticmethod
    def days_inventory_outstanding(data: FinancialData) -> float:
        turnover = EfficiencyRatios.inventory_turnover(data)
        if np.isnan(turnover) or turnover == 0:
            return np.nan
        return 365 / turnover


class LeverageRatios:
    """Leverage ratio calculations."""
    
    @staticmethod
    def debt_to_equity(data: FinancialData) -> float:
        if data.shareholders_equity == 0:
            return np.nan
        return data.total_liabilities / data.shareholders_equity
    
    @staticmethod
    def debt_ratio(data: FinancialData) -> float:
        if data.total_assets == 0:
            return np.nan
        return data.total_liabilities / data.total_assets
    
    @staticmethod
    def interest_coverage(data: FinancialData) -> float:
        if data.interest_expense == 0:
            return np.nan
        return data.operating_income / data.interest_expense
    
    @staticmethod
    def debt_to_ebitda(data: FinancialData) -> float:
        if data.ebitda == 0:
            return np.nan
        return (data.short_term_debt + data.long_term_debt) / data.ebitda


class ValuationRatios:
    """Valuation ratio calculations."""
    
    @staticmethod
    def price_to_earnings(data: FinancialData) -> float:
        if data.shares_outstanding == 0 or data.net_income == 0:
            return np.nan
        eps = data.net_income / data.shares_outstanding
        return data.stock_price / eps
    
    @staticmethod
    def price_to_book(data: FinancialData) -> float:
        if data.shares_outstanding == 0 or data.shareholders_equity == 0:
            return np.nan
        book_value_per_share = data.shareholders_equity / data.shares_outstanding
        return data.stock_price / book_value_per_share
    
    @staticmethod
    def ev_to_ebitda(data: FinancialData) -> float:
        if data.ebitda == 0:
            return np.nan
        ev = data.market_cap + data.total_liabilities - data.cash
        return ev / data.ebitda


class GrowthRatios:
    """Growth ratio calculations."""
    
    @staticmethod
    def revenue_growth(data: FinancialData) -> float:
        if data.prev_revenue == 0:
            return np.nan
        return (data.revenue - data.prev_revenue) / data.prev_revenue
    
    @staticmethod
    def earnings_growth(data: FinancialData) -> float:
        if data.prev_net_income == 0:
            return np.nan
        return (data.net_income - data.prev_net_income) / abs(data.prev_net_income)
    
    @staticmethod
    def asset_growth(data: FinancialData) -> float:
        if data.prev_total_assets == 0:
            return np.nan
        return (data.total_assets - data.prev_total_assets) / data.prev_total_assets


# ============================================================================
# Financial Ratio Engine
# ============================================================================

class FinancialRatioEngine:
    """
    Complete Financial Ratio Analysis engine.
    
    Features:
    - All major financial ratio categories
    - Automatic interpretation
    - Benchmarking
    - Health assessment
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        data: Union[FinancialData, Dict[str, float], pd.DataFrame]
    ) -> FinancialRatioResult:
        """Perform comprehensive financial ratio analysis."""
        start_time = datetime.now()
        
        # Convert input to FinancialData
        if isinstance(data, dict):
            fin_data = FinancialData(**{k: v for k, v in data.items() if hasattr(FinancialData, k)})
        elif isinstance(data, pd.DataFrame):
            fin_data = self._dataframe_to_financial_data(data)
        else:
            fin_data = data
        
        if self.verbose:
            logger.info("Calculating financial ratios...")
        
        ratios = []
        
        # Liquidity Ratios
        ratios.extend(self._calculate_liquidity_ratios(fin_data))
        
        # Profitability Ratios
        ratios.extend(self._calculate_profitability_ratios(fin_data))
        
        # Efficiency Ratios
        ratios.extend(self._calculate_efficiency_ratios(fin_data))
        
        # Leverage Ratios
        ratios.extend(self._calculate_leverage_ratios(fin_data))
        
        # Valuation Ratios
        ratios.extend(self._calculate_valuation_ratios(fin_data))
        
        # Growth Ratios
        ratios.extend(self._calculate_growth_ratios(fin_data))
        
        # Organize by category
        by_category = {}
        for ratio in ratios:
            cat = ratio.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ratio)
        
        # Overall health assessment
        overall_health = self._assess_overall_health(ratios)
        
        # Key insights
        insights = self._generate_insights(ratios)
        
        # Warnings
        warnings = self._generate_warnings(ratios)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FinancialRatioResult(
            ratios=ratios,
            by_category=by_category,
            overall_health=overall_health,
            key_insights=insights,
            warnings=warnings,
            processing_time_sec=processing_time
        )
    
    def _calculate_liquidity_ratios(self, data: FinancialData) -> List[FinancialRatio]:
        ratios = []
        
        current = LiquidityRatios.current_ratio(data)
        ratios.append(FinancialRatio(
            name="Current Ratio",
            value=current,
            category=RatioCategory.LIQUIDITY,
            formula="Current Assets / Current Liabilities",
            interpretation=self._interpret_current_ratio(current),
            benchmark=2.0
        ))
        
        quick = LiquidityRatios.quick_ratio(data)
        ratios.append(FinancialRatio(
            name="Quick Ratio",
            value=quick,
            category=RatioCategory.LIQUIDITY,
            formula="(Current Assets - Inventory) / Current Liabilities",
            interpretation=self._interpret_quick_ratio(quick),
            benchmark=1.0
        ))
        
        cash = LiquidityRatios.cash_ratio(data)
        ratios.append(FinancialRatio(
            name="Cash Ratio",
            value=cash,
            category=RatioCategory.LIQUIDITY,
            formula="Cash / Current Liabilities",
            interpretation=self._interpret_generic(cash, 0.5, 0.2, 1.0),
            benchmark=0.5
        ))
        
        return ratios
    
    def _calculate_profitability_ratios(self, data: FinancialData) -> List[FinancialRatio]:
        ratios = []
        
        gross = ProfitabilityRatios.gross_margin(data)
        ratios.append(FinancialRatio(
            name="Gross Margin",
            value=gross,
            category=RatioCategory.PROFITABILITY,
            formula="Gross Profit / Revenue",
            interpretation=self._interpret_margin(gross),
            benchmark=0.40
        ))
        
        operating = ProfitabilityRatios.operating_margin(data)
        ratios.append(FinancialRatio(
            name="Operating Margin",
            value=operating,
            category=RatioCategory.PROFITABILITY,
            formula="Operating Income / Revenue",
            interpretation=self._interpret_margin(operating),
            benchmark=0.15
        ))
        
        net = ProfitabilityRatios.net_margin(data)
        ratios.append(FinancialRatio(
            name="Net Margin",
            value=net,
            category=RatioCategory.PROFITABILITY,
            formula="Net Income / Revenue",
            interpretation=self._interpret_margin(net),
            benchmark=0.10
        ))
        
        roa = ProfitabilityRatios.return_on_assets(data)
        ratios.append(FinancialRatio(
            name="Return on Assets (ROA)",
            value=roa,
            category=RatioCategory.PROFITABILITY,
            formula="Net Income / Total Assets",
            interpretation=self._interpret_generic(roa, 0.05, 0.02, 0.10),
            benchmark=0.05
        ))
        
        roe = ProfitabilityRatios.return_on_equity(data)
        ratios.append(FinancialRatio(
            name="Return on Equity (ROE)",
            value=roe,
            category=RatioCategory.PROFITABILITY,
            formula="Net Income / Shareholders' Equity",
            interpretation=self._interpret_generic(roe, 0.15, 0.05, 0.25),
            benchmark=0.15
        ))
        
        return ratios
    
    def _calculate_efficiency_ratios(self, data: FinancialData) -> List[FinancialRatio]:
        ratios = []
        
        asset_turn = EfficiencyRatios.asset_turnover(data)
        ratios.append(FinancialRatio(
            name="Asset Turnover",
            value=asset_turn,
            category=RatioCategory.EFFICIENCY,
            formula="Revenue / Total Assets",
            interpretation=self._interpret_generic(asset_turn, 1.0, 0.5, 2.0),
            benchmark=1.0
        ))
        
        inv_turn = EfficiencyRatios.inventory_turnover(data)
        ratios.append(FinancialRatio(
            name="Inventory Turnover",
            value=inv_turn,
            category=RatioCategory.EFFICIENCY,
            formula="COGS / Inventory",
            interpretation=self._interpret_generic(inv_turn, 6.0, 2.0, 12.0),
            benchmark=6.0
        ))
        
        dso = EfficiencyRatios.days_sales_outstanding(data)
        ratios.append(FinancialRatio(
            name="Days Sales Outstanding",
            value=dso,
            category=RatioCategory.EFFICIENCY,
            formula="365 / Receivables Turnover",
            interpretation=self._interpret_dso(dso),
            benchmark=45.0
        ))
        
        return ratios
    
    def _calculate_leverage_ratios(self, data: FinancialData) -> List[FinancialRatio]:
        ratios = []
        
        dte = LeverageRatios.debt_to_equity(data)
        ratios.append(FinancialRatio(
            name="Debt to Equity",
            value=dte,
            category=RatioCategory.LEVERAGE,
            formula="Total Liabilities / Shareholders' Equity",
            interpretation=self._interpret_debt_equity(dte),
            benchmark=1.0
        ))
        
        debt_ratio = LeverageRatios.debt_ratio(data)
        ratios.append(FinancialRatio(
            name="Debt Ratio",
            value=debt_ratio,
            category=RatioCategory.LEVERAGE,
            formula="Total Liabilities / Total Assets",
            interpretation=self._interpret_debt_ratio(debt_ratio),
            benchmark=0.50
        ))
        
        interest_cov = LeverageRatios.interest_coverage(data)
        ratios.append(FinancialRatio(
            name="Interest Coverage",
            value=interest_cov,
            category=RatioCategory.LEVERAGE,
            formula="Operating Income / Interest Expense",
            interpretation=self._interpret_interest_coverage(interest_cov),
            benchmark=3.0
        ))
        
        return ratios
    
    def _calculate_valuation_ratios(self, data: FinancialData) -> List[FinancialRatio]:
        ratios = []
        
        pe = ValuationRatios.price_to_earnings(data)
        ratios.append(FinancialRatio(
            name="Price to Earnings (P/E)",
            value=pe,
            category=RatioCategory.VALUATION,
            formula="Stock Price / EPS",
            interpretation=self._interpret_pe(pe),
            benchmark=20.0
        ))
        
        pb = ValuationRatios.price_to_book(data)
        ratios.append(FinancialRatio(
            name="Price to Book (P/B)",
            value=pb,
            category=RatioCategory.VALUATION,
            formula="Stock Price / Book Value per Share",
            interpretation=self._interpret_generic(pb, 3.0, 1.0, 5.0),
            benchmark=3.0
        ))
        
        return ratios
    
    def _calculate_growth_ratios(self, data: FinancialData) -> List[FinancialRatio]:
        ratios = []
        
        rev_growth = GrowthRatios.revenue_growth(data)
        ratios.append(FinancialRatio(
            name="Revenue Growth",
            value=rev_growth,
            category=RatioCategory.GROWTH,
            formula="(Current Revenue - Prior Revenue) / Prior Revenue",
            interpretation=self._interpret_growth(rev_growth),
            benchmark=0.10
        ))
        
        earnings_growth = GrowthRatios.earnings_growth(data)
        ratios.append(FinancialRatio(
            name="Earnings Growth",
            value=earnings_growth,
            category=RatioCategory.GROWTH,
            formula="(Current Net Income - Prior) / |Prior|",
            interpretation=self._interpret_growth(earnings_growth),
            benchmark=0.10
        ))
        
        return ratios
    
    def _interpret_current_ratio(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value >= 2.0:
            return RatioInterpretation.EXCELLENT
        elif value >= 1.5:
            return RatioInterpretation.GOOD
        elif value >= 1.0:
            return RatioInterpretation.AVERAGE
        elif value >= 0.5:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_quick_ratio(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value >= 1.5:
            return RatioInterpretation.EXCELLENT
        elif value >= 1.0:
            return RatioInterpretation.GOOD
        elif value >= 0.7:
            return RatioInterpretation.AVERAGE
        elif value >= 0.4:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_margin(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value >= 0.20:
            return RatioInterpretation.EXCELLENT
        elif value >= 0.10:
            return RatioInterpretation.GOOD
        elif value >= 0.05:
            return RatioInterpretation.AVERAGE
        elif value >= 0:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_generic(self, value: float, mid: float, low: float, high: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value >= high:
            return RatioInterpretation.EXCELLENT
        elif value >= mid:
            return RatioInterpretation.GOOD
        elif value >= low:
            return RatioInterpretation.AVERAGE
        elif value >= 0:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_debt_equity(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value <= 0.5:
            return RatioInterpretation.EXCELLENT
        elif value <= 1.0:
            return RatioInterpretation.GOOD
        elif value <= 2.0:
            return RatioInterpretation.AVERAGE
        elif value <= 3.0:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_debt_ratio(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value <= 0.3:
            return RatioInterpretation.EXCELLENT
        elif value <= 0.5:
            return RatioInterpretation.GOOD
        elif value <= 0.7:
            return RatioInterpretation.AVERAGE
        elif value <= 0.9:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_interest_coverage(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value >= 5.0:
            return RatioInterpretation.EXCELLENT
        elif value >= 3.0:
            return RatioInterpretation.GOOD
        elif value >= 1.5:
            return RatioInterpretation.AVERAGE
        elif value >= 1.0:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_pe(self, value: float) -> RatioInterpretation:
        if np.isnan(value) or value < 0:
            return RatioInterpretation.CRITICAL
        if value <= 15:
            return RatioInterpretation.EXCELLENT  # Potentially undervalued
        elif value <= 25:
            return RatioInterpretation.GOOD
        elif value <= 35:
            return RatioInterpretation.AVERAGE
        elif value <= 50:
            return RatioInterpretation.POOR  # Expensive
        return RatioInterpretation.CRITICAL
    
    def _interpret_dso(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value <= 30:
            return RatioInterpretation.EXCELLENT
        elif value <= 45:
            return RatioInterpretation.GOOD
        elif value <= 60:
            return RatioInterpretation.AVERAGE
        elif value <= 90:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _interpret_growth(self, value: float) -> RatioInterpretation:
        if np.isnan(value):
            return RatioInterpretation.AVERAGE
        if value >= 0.20:
            return RatioInterpretation.EXCELLENT
        elif value >= 0.10:
            return RatioInterpretation.GOOD
        elif value >= 0:
            return RatioInterpretation.AVERAGE
        elif value >= -0.10:
            return RatioInterpretation.POOR
        return RatioInterpretation.CRITICAL
    
    def _assess_overall_health(self, ratios: List[FinancialRatio]) -> str:
        scores = {
            RatioInterpretation.EXCELLENT: 5,
            RatioInterpretation.GOOD: 4,
            RatioInterpretation.AVERAGE: 3,
            RatioInterpretation.POOR: 2,
            RatioInterpretation.CRITICAL: 1
        }
        
        valid_ratios = [r for r in ratios if not np.isnan(r.value)]
        if not valid_ratios:
            return "Insufficient data"
        
        avg_score = np.mean([scores[r.interpretation] for r in valid_ratios])
        
        if avg_score >= 4.5:
            return "Excellent"
        elif avg_score >= 3.5:
            return "Good"
        elif avg_score >= 2.5:
            return "Average"
        elif avg_score >= 1.5:
            return "Poor"
        return "Critical"
    
    def _generate_insights(self, ratios: List[FinancialRatio]) -> List[str]:
        insights = []
        
        for ratio in ratios:
            if np.isnan(ratio.value):
                continue
            
            if ratio.interpretation == RatioInterpretation.EXCELLENT:
                insights.append(f"{ratio.name} is excellent at {ratio.value:.2f}")
            elif ratio.interpretation == RatioInterpretation.CRITICAL:
                insights.append(f"{ratio.name} is critical at {ratio.value:.2f} - needs attention")
        
        return insights
    
    def _generate_warnings(self, ratios: List[FinancialRatio]) -> List[str]:
        warnings = []
        
        for ratio in ratios:
            if np.isnan(ratio.value):
                continue
            
            if ratio.interpretation in [RatioInterpretation.POOR, RatioInterpretation.CRITICAL]:
                warnings.append(f"Warning: {ratio.name} = {ratio.value:.2f}")
        
        return warnings
    
    def _dataframe_to_financial_data(self, df: pd.DataFrame) -> FinancialData:
        """Convert DataFrame row to FinancialData."""
        row = df.iloc[0] if len(df) > 0 else {}
        
        mapping = {
            'current_assets': ['current_assets', 'currentassets', 'ca'],
            'current_liabilities': ['current_liabilities', 'currentliabilities', 'cl'],
            'total_assets': ['total_assets', 'totalassets', 'assets'],
            'total_liabilities': ['total_liabilities', 'totalliabilities', 'liabilities'],
            'revenue': ['revenue', 'sales', 'net_sales'],
            'net_income': ['net_income', 'netincome', 'profit']
        }
        
        data_dict = {}
        for field, patterns in mapping.items():
            for col in df.columns:
                if col.lower().replace(' ', '_') in patterns:
                    data_dict[field] = float(row[col]) if col in row else 0.0
                    break
        
        return FinancialData(**data_dict)


# ============================================================================
# Factory Functions
# ============================================================================

def get_financial_ratio_engine() -> FinancialRatioEngine:
    """Get financial ratio engine."""
    return FinancialRatioEngine()


def quick_financial_ratios(data: Dict[str, float]) -> Dict[str, Any]:
    """Quick financial ratio analysis."""
    engine = FinancialRatioEngine(verbose=False)
    result = engine.analyze(data)
    return result.to_dict()
