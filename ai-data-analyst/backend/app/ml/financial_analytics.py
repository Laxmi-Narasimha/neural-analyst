# AI Enterprise Data Analyst - Financial Analytics Engine
# Sharpe ratio, VaR, Beta, drawdown, portfolio analysis

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    var_95: float
    cvar_95: float
    
    def to_dict(self) -> dict:
        return {
            "total_return": f"{self.total_return:.2%}",
            "annualized_return": f"{self.annualized_return:.2%}",
            "volatility": f"{self.volatility:.2%}",
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "beta": round(self.beta, 3),
            "alpha": f"{self.alpha:.2%}",
            "VaR_95": f"{self.var_95:.2%}",
            "CVaR_95": f"{self.cvar_95:.2%}"
        }


class FinancialAnalyticsEngine:
    """Production financial analytics engine."""
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / self.trading_days
        if returns.std() == 0:
            return 0.0
        return float(np.sqrt(self.trading_days) * excess_returns.mean() / returns.std())
    
    def sortino_ratio(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        excess = returns - target
        downside = returns[returns < target]
        
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        
        return float(np.sqrt(self.trading_days) * excess.mean() / downside.std())
    
    def max_drawdown(self, prices: np.ndarray) -> tuple[float, int, int]:
        """Calculate maximum drawdown and peak/trough indices."""
        cumulative = np.maximum.accumulate(prices)
        drawdowns = (prices - cumulative) / cumulative
        
        trough_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(prices[:trough_idx+1])) if trough_idx > 0 else 0
        max_dd = float(drawdowns[trough_idx])
        
        return max_dd, peak_idx, trough_idx
    
    def beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate beta relative to market."""
        if len(asset_returns) != len(market_returns):
            min_len = min(len(asset_returns), len(market_returns))
            asset_returns = asset_returns[:min_len]
            market_returns = market_returns[:min_len]
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return float(covariance / market_variance) if market_variance > 0 else 0.0
    
    def alpha(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate Jensen's alpha."""
        beta_val = self.beta(asset_returns, market_returns)
        
        asset_ann = np.mean(asset_returns) * self.trading_days
        market_ann = np.mean(market_returns) * self.trading_days
        
        expected = self.risk_free_rate + beta_val * (market_ann - self.risk_free_rate)
        return float(asset_ann - expected)
    
    def value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    def conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.value_at_risk(returns, confidence)
        cvar = returns[returns <= var].mean()
        return float(cvar) if not np.isnan(cvar) else var
    
    def analyze_portfolio(
        self, prices: pd.Series, benchmark_prices: pd.Series = None
    ) -> PortfolioMetrics:
        """Complete portfolio analysis."""
        returns = prices.pct_change().dropna().values
        
        total_return = float((prices.iloc[-1] / prices.iloc[0]) - 1)
        n_years = len(prices) / self.trading_days
        ann_return = float((1 + total_return) ** (1/n_years) - 1) if n_years > 0 else 0
        volatility = float(returns.std() * np.sqrt(self.trading_days))
        
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        max_dd, _, _ = self.max_drawdown(prices.values)
        var = self.value_at_risk(returns)
        cvar = self.conditional_var(returns)
        
        if benchmark_prices is not None:
            bench_returns = benchmark_prices.pct_change().dropna().values
            beta_val = self.beta(returns, bench_returns)
            alpha_val = self.alpha(returns, bench_returns)
        else:
            beta_val = 1.0
            alpha_val = 0.0
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            beta=beta_val,
            alpha=alpha_val,
            var_95=var,
            cvar_95=cvar
        )
    
    def rolling_metrics(
        self, prices: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling financial metrics."""
        returns = prices.pct_change()
        
        result = pd.DataFrame(index=prices.index)
        result["rolling_return"] = prices.pct_change(window)
        result["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(self.trading_days)
        result["rolling_sharpe"] = (
            returns.rolling(window).mean() * self.trading_days - self.risk_free_rate
        ) / result["rolling_volatility"]
        
        # Rolling drawdown
        rolling_max = prices.rolling(window, min_periods=1).max()
        result["rolling_drawdown"] = (prices - rolling_max) / rolling_max
        
        return result.dropna()
    
    def correlation_analysis(self, prices_df: pd.DataFrame) -> dict:
        """Analyze correlation between assets."""
        returns = prices_df.pct_change().dropna()
        corr = returns.corr()
        
        # Find most/least correlated pairs
        pairs = []
        for i, col1 in enumerate(corr.columns):
            for col2 in corr.columns[i+1:]:
                pairs.append({
                    "asset1": col1,
                    "asset2": col2,
                    "correlation": round(corr.loc[col1, col2], 4)
                })
        
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "correlation_matrix": corr.round(4).to_dict(),
            "most_correlated": pairs[:5],
            "least_correlated": pairs[-5:]
        }


def get_financial_analytics_engine(
    risk_free_rate: float = 0.02
) -> FinancialAnalyticsEngine:
    return FinancialAnalyticsEngine(risk_free_rate)
