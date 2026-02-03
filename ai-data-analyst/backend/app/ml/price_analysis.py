# AI Enterprise Data Analyst - Price Analysis Engine
# Production-grade price and pricing analysis
# Handles: price elasticity, competitive pricing, optimization

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
# Data Classes
# ============================================================================

@dataclass
class PriceElasticity:
    """Price elasticity result."""
    elasticity: float
    interpretation: str
    is_elastic: bool
    r_squared: float


@dataclass
class PriceTier:
    """Price tier definition."""
    tier_name: str
    min_price: float
    max_price: float
    count: int
    percentage: float
    avg_quantity: float


@dataclass
class CompetitorPrice:
    """Competitor pricing comparison."""
    competitor: str
    avg_price: float
    price_diff: float
    price_diff_pct: float


@dataclass
class PriceResult:
    """Complete price analysis result."""
    n_products: int = 0
    
    # Price statistics
    avg_price: float = 0.0
    median_price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    price_std: float = 0.0
    
    # Elasticity
    elasticity: PriceElasticity = None
    
    # Price distribution
    price_tiers: List[PriceTier] = field(default_factory=list)
    
    # Competitive analysis
    competitor_comparison: List[CompetitorPrice] = field(default_factory=list)
    
    # Optimal price (if quantity data available)
    optimal_price: Optional[float] = None
    revenue_at_optimal: Optional[float] = None
    
    # Trends
    price_trend: str = ""  # increasing, decreasing, stable
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_products": self.n_products,
                "avg_price": round(self.avg_price, 2),
                "median_price": round(self.median_price, 2),
                "price_range": [round(self.min_price, 2), round(self.max_price, 2)],
                "price_std": round(self.price_std, 2)
            },
            "elasticity": {
                "value": round(self.elasticity.elasticity, 4) if self.elasticity else None,
                "interpretation": self.elasticity.interpretation if self.elasticity else None,
                "is_elastic": self.elasticity.is_elastic if self.elasticity else None
            },
            "price_tiers": [
                {
                    "tier": t.tier_name,
                    "range": [round(t.min_price, 2), round(t.max_price, 2)],
                    "count": t.count,
                    "pct": round(t.percentage, 1)
                }
                for t in self.price_tiers
            ],
            "optimization": {
                "optimal_price": round(self.optimal_price, 2) if self.optimal_price else None,
                "revenue_at_optimal": round(self.revenue_at_optimal, 2) if self.revenue_at_optimal else None
            },
            "trend": self.price_trend
        }


# ============================================================================
# Price Analysis Engine
# ============================================================================

class PriceAnalysisEngine:
    """
    Production-grade Price Analysis engine.
    
    Features:
    - Price elasticity calculation
    - Price tier analysis
    - Optimal price finding
    - Competitive analysis
    - Trend detection
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        price_col: str = None,
        quantity_col: str = None,
        product_col: str = None,
        competitor_col: str = None,
        date_col: str = None,
        n_tiers: int = 5
    ) -> PriceResult:
        """Analyze pricing data."""
        start_time = datetime.now()
        
        # Auto-detect price column
        if price_col is None:
            price_col = self._detect_column(df, ['price', 'amount', 'cost', 'value'])
        
        if price_col is None or price_col not in df.columns:
            raise ValueError("Could not detect price column")
        
        prices = df[price_col].dropna()
        
        if self.verbose:
            logger.info(f"Price analysis: {len(prices)} price points")
        
        # Basic statistics
        avg_price = float(prices.mean())
        median_price = float(prices.median())
        min_price = float(prices.min())
        max_price = float(prices.max())
        price_std = float(prices.std())
        
        # Price elasticity
        elasticity = None
        if quantity_col and quantity_col in df.columns:
            elasticity = self._calculate_elasticity(df, price_col, quantity_col)
        
        # Price tiers
        price_tiers = self._create_price_tiers(prices, n_tiers)
        
        # Competitive analysis
        competitor_comparison = []
        if competitor_col and competitor_col in df.columns:
            competitor_comparison = self._analyze_competitors(df, price_col, competitor_col)
        
        # Optimal price
        optimal_price = None
        revenue_at_optimal = None
        if quantity_col and quantity_col in df.columns:
            optimal_price, revenue_at_optimal = self._find_optimal_price(
                df, price_col, quantity_col
            )
        
        # Trend
        price_trend = "stable"
        if date_col and date_col in df.columns:
            price_trend = self._detect_trend(df, price_col, date_col)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PriceResult(
            n_products=len(prices),
            avg_price=avg_price,
            median_price=median_price,
            min_price=min_price,
            max_price=max_price,
            price_std=price_std,
            elasticity=elasticity,
            price_tiers=price_tiers,
            competitor_comparison=competitor_comparison,
            optimal_price=optimal_price,
            revenue_at_optimal=revenue_at_optimal,
            price_trend=price_trend,
            processing_time_sec=processing_time
        )
    
    def _calculate_elasticity(
        self,
        df: pd.DataFrame,
        price_col: str,
        quantity_col: str
    ) -> PriceElasticity:
        """Calculate price elasticity of demand."""
        clean = df[[price_col, quantity_col]].dropna()
        
        if len(clean) < 5:
            return PriceElasticity(
                elasticity=0, interpretation="Insufficient data",
                is_elastic=False, r_squared=0
            )
        
        # Log-log regression for elasticity
        log_price = np.log(clean[price_col] + 1)
        log_qty = np.log(clean[quantity_col] + 1)
        
        slope, intercept, r_value, _, _ = scipy_stats.linregress(log_price, log_qty)
        elasticity = slope
        
        # Interpretation
        if abs(elasticity) > 1:
            interpretation = "Elastic demand - sensitive to price changes"
            is_elastic = True
        elif abs(elasticity) < 1:
            interpretation = "Inelastic demand - less sensitive to price"
            is_elastic = False
        else:
            interpretation = "Unit elastic"
            is_elastic = False
        
        return PriceElasticity(
            elasticity=float(elasticity),
            interpretation=interpretation,
            is_elastic=is_elastic,
            r_squared=float(r_value ** 2)
        )
    
    def _create_price_tiers(
        self,
        prices: pd.Series,
        n_tiers: int
    ) -> List[PriceTier]:
        """Create price tiers."""
        tier_labels = ['Budget', 'Economy', 'Standard', 'Premium', 'Luxury']
        if n_tiers > 5:
            tier_labels = [f'Tier_{i+1}' for i in range(n_tiers)]
        
        # Use quantiles
        quantiles = [i / n_tiers for i in range(n_tiers + 1)]
        bin_edges = prices.quantile(quantiles).values
        
        # Handle duplicate edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return []
        
        tiers = []
        for i in range(len(bin_edges) - 1):
            mask = (prices >= bin_edges[i]) & (prices < bin_edges[i + 1])
            if i == len(bin_edges) - 2:  # Include max in last tier
                mask = (prices >= bin_edges[i]) & (prices <= bin_edges[i + 1])
            
            count = int(mask.sum())
            
            tier_name = tier_labels[min(i, len(tier_labels) - 1)]
            
            tiers.append(PriceTier(
                tier_name=tier_name,
                min_price=float(bin_edges[i]),
                max_price=float(bin_edges[i + 1]),
                count=count,
                percentage=count / len(prices) * 100,
                avg_quantity=0  # Would need quantity data
            ))
        
        return tiers
    
    def _analyze_competitors(
        self,
        df: pd.DataFrame,
        price_col: str,
        competitor_col: str
    ) -> List[CompetitorPrice]:
        """Analyze competitor pricing."""
        grouped = df.groupby(competitor_col)[price_col].mean()
        overall_avg = df[price_col].mean()
        
        comparisons = []
        for competitor, avg in grouped.items():
            diff = avg - overall_avg
            diff_pct = (diff / overall_avg * 100) if overall_avg > 0 else 0
            
            comparisons.append(CompetitorPrice(
                competitor=str(competitor),
                avg_price=float(avg),
                price_diff=float(diff),
                price_diff_pct=float(diff_pct)
            ))
        
        comparisons.sort(key=lambda x: x.avg_price)
        return comparisons
    
    def _find_optimal_price(
        self,
        df: pd.DataFrame,
        price_col: str,
        quantity_col: str
    ) -> tuple:
        """Find revenue-maximizing price."""
        clean = df[[price_col, quantity_col]].dropna()
        
        if len(clean) < 10:
            return None, None
        
        # Simple approach: find price point with max revenue
        clean['revenue'] = clean[price_col] * clean[quantity_col]
        
        # Group by price ranges
        price_range = clean[price_col].max() - clean[price_col].min()
        n_bins = min(20, len(clean) // 10)
        
        if n_bins < 3:
            return None, None
        
        clean['price_bin'] = pd.cut(clean[price_col], bins=n_bins)
        
        revenue_by_price = clean.groupby('price_bin')['revenue'].sum()
        
        if len(revenue_by_price) == 0:
            return None, None
        
        best_bin = revenue_by_price.idxmax()
        optimal_price = (best_bin.left + best_bin.right) / 2
        revenue_at_optimal = float(revenue_by_price.max())
        
        return float(optimal_price), revenue_at_optimal
    
    def _detect_trend(
        self,
        df: pd.DataFrame,
        price_col: str,
        date_col: str
    ) -> str:
        """Detect price trend over time."""
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_sorted = df_work.dropna(subset=[date_col]).sort_values(date_col)
        
        if len(df_sorted) < 5:
            return "insufficient_data"
        
        # Linear regression on price over time
        x = np.arange(len(df_sorted))
        y = df_sorted[price_col].values
        
        slope, _, _, _, _ = scipy_stats.linregress(x, y)
        
        # Normalize by mean price
        normalized_slope = slope / np.mean(y) * 100
        
        if normalized_slope > 1:
            return "increasing"
        elif normalized_slope < -1:
            return "decreasing"
        return "stable"
    
    def _detect_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Detect column by name patterns."""
        for pattern in patterns:
            for col in df.columns:
                if pattern in col.lower():
                    if df[col].dtype in [np.float64, np.int64]:
                        return col
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_price_engine() -> PriceAnalysisEngine:
    """Get price analysis engine."""
    return PriceAnalysisEngine()


def quick_price_analysis(
    df: pd.DataFrame,
    price_col: str = None
) -> Dict[str, Any]:
    """Quick price analysis."""
    engine = PriceAnalysisEngine(verbose=False)
    result = engine.analyze(df, price_col=price_col)
    return result.to_dict()


def calculate_elasticity(
    prices: List[float],
    quantities: List[float]
) -> float:
    """Calculate price elasticity from lists."""
    df = pd.DataFrame({'price': prices, 'quantity': quantities})
    engine = PriceAnalysisEngine(verbose=False)
    result = engine.analyze(df, 'price', 'quantity')
    return result.elasticity.elasticity if result.elasticity else 0
