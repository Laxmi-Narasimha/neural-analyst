# AI Enterprise Data Analyst - Benchmark Analysis Engine
# Production-grade benchmarking and performance comparison
# Handles: any metrics, internal/external benchmarks

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
# Enums
# ============================================================================

class BenchmarkStatus(str, Enum):
    """Performance relative to benchmark."""
    ABOVE = "above"
    AT = "at"
    BELOW = "below"
    FAR_BELOW = "far_below"


class BenchmarkType(str, Enum):
    """Type of benchmark."""
    INTERNAL = "internal"  # Historical performance
    EXTERNAL = "external"  # Industry/competitor
    TARGET = "target"  # Goals


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkMetric:
    """Single benchmark comparison."""
    metric: str
    current_value: float
    benchmark_value: float
    variance: float
    variance_pct: float
    status: BenchmarkStatus
    percentile: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark analysis result."""
    n_metrics: int = 0
    
    # Metrics
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    
    # Summary
    above_benchmark: int = 0
    at_benchmark: int = 0
    below_benchmark: int = 0
    
    # Overall score
    overall_score: float = 0.0  # 0-100
    
    # Top performers and laggards
    top_performers: List[str] = field(default_factory=list)
    underperformers: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_metrics": self.n_metrics,
                "above_benchmark": self.above_benchmark,
                "at_benchmark": self.at_benchmark,
                "below_benchmark": self.below_benchmark,
                "overall_score": round(self.overall_score, 1)
            },
            "metrics": [
                {
                    "metric": m.metric,
                    "current": round(m.current_value, 2),
                    "benchmark": round(m.benchmark_value, 2),
                    "variance_pct": round(m.variance_pct, 1),
                    "status": m.status.value
                }
                for m in self.metrics[:20]
            ],
            "top_performers": self.top_performers[:5],
            "underperformers": self.underperformers[:5]
        }


# ============================================================================
# Benchmark Analysis Engine
# ============================================================================

class BenchmarkAnalysisEngine:
    """
    Benchmark Analysis engine.
    
    Features:
    - Multi-metric benchmarking
    - Percentile ranking
    - Gap analysis
    - Performance scoring
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        current: Dict[str, float],
        benchmarks: Dict[str, float],
        higher_is_better: Dict[str, bool] = None
    ) -> BenchmarkResult:
        """Perform benchmark analysis."""
        start_time = datetime.now()
        
        if higher_is_better is None:
            higher_is_better = {m: True for m in current}
        
        if self.verbose:
            logger.info(f"Benchmarking {len(current)} metrics")
        
        metrics = []
        
        for metric, value in current.items():
            if metric not in benchmarks:
                continue
            
            benchmark = benchmarks[metric]
            variance = value - benchmark
            variance_pct = (variance / abs(benchmark) * 100) if benchmark != 0 else 0
            
            is_better = higher_is_better.get(metric, True)
            
            if is_better:
                if variance_pct >= 5:
                    status = BenchmarkStatus.ABOVE
                elif variance_pct >= -5:
                    status = BenchmarkStatus.AT
                elif variance_pct >= -20:
                    status = BenchmarkStatus.BELOW
                else:
                    status = BenchmarkStatus.FAR_BELOW
            else:
                if variance_pct <= -5:
                    status = BenchmarkStatus.ABOVE
                elif variance_pct <= 5:
                    status = BenchmarkStatus.AT
                elif variance_pct <= 20:
                    status = BenchmarkStatus.BELOW
                else:
                    status = BenchmarkStatus.FAR_BELOW
            
            metrics.append(BenchmarkMetric(
                metric=metric,
                current_value=value,
                benchmark_value=benchmark,
                variance=variance,
                variance_pct=variance_pct,
                status=status
            ))
        
        # Summary
        above = sum(1 for m in metrics if m.status == BenchmarkStatus.ABOVE)
        at = sum(1 for m in metrics if m.status == BenchmarkStatus.AT)
        below = sum(1 for m in metrics if m.status in [BenchmarkStatus.BELOW, BenchmarkStatus.FAR_BELOW])
        
        # Overall score
        scores = []
        for m in metrics:
            is_better = higher_is_better.get(m.metric, True)
            if is_better:
                score = min(100, max(0, 50 + m.variance_pct))
            else:
                score = min(100, max(0, 50 - m.variance_pct))
            scores.append(score)
        
        overall_score = np.mean(scores) if scores else 50
        
        # Top/bottom performers
        sorted_metrics = sorted(metrics, key=lambda x: x.variance_pct, reverse=True)
        top_performers = [m.metric for m in sorted_metrics[:5] if m.status == BenchmarkStatus.ABOVE]
        underperformers = [m.metric for m in sorted_metrics[-5:] if m.status in [BenchmarkStatus.BELOW, BenchmarkStatus.FAR_BELOW]]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BenchmarkResult(
            n_metrics=len(metrics),
            metrics=metrics,
            above_benchmark=above,
            at_benchmark=at,
            below_benchmark=below,
            overall_score=overall_score,
            top_performers=top_performers,
            underperformers=underperformers,
            processing_time_sec=processing_time
        )
    
    def compute_percentiles(
        self,
        current: Dict[str, float],
        distributions: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute percentile rankings against distributions."""
        percentiles = {}
        
        for metric, value in current.items():
            if metric in distributions:
                dist = distributions[metric]
                percentile = sum(1 for v in dist if v <= value) / len(dist) * 100
                percentiles[metric] = percentile
        
        return percentiles


# ============================================================================
# Factory Functions
# ============================================================================

def get_benchmark_engine() -> BenchmarkAnalysisEngine:
    """Get benchmark analysis engine."""
    return BenchmarkAnalysisEngine()


def quick_benchmark(
    current: Dict[str, float],
    benchmarks: Dict[str, float]
) -> Dict[str, Any]:
    """Quick benchmark analysis."""
    engine = BenchmarkAnalysisEngine(verbose=False)
    result = engine.analyze(current, benchmarks)
    return result.to_dict()
