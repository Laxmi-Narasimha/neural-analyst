# AI Enterprise Data Analyst - Benchmarking Engine
# Production-grade performance benchmarking
# Handles: timing, memory, throughput measurements

from __future__ import annotations

import gc
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
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
# Data Classes
# ============================================================================

@dataclass
class BenchmarkRun:
    """Single benchmark run."""
    name: str
    duration_sec: float
    iterations: int
    avg_time_sec: float
    min_time_sec: float
    max_time_sec: float
    std_time_sec: float
    
    # Throughput
    throughput: float = 0.0  # items/sec
    
    # Memory (approximate)
    memory_mb: float = 0.0
    
    result: Any = None


@dataclass
class BenchmarkComparison:
    """Comparison of benchmarks."""
    baseline: str
    competitor: str
    speedup: float  # >1 means competitor is faster
    baseline_avg: float
    competitor_avg: float


@dataclass
class BenchmarkResult:
    """Complete benchmarking result."""
    n_benchmarks: int = 0
    total_time_sec: float = 0.0
    
    runs: List[BenchmarkRun] = field(default_factory=list)
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    
    # Best performer
    fastest: str = ""
    slowest: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_benchmarks": self.n_benchmarks,
            "total_time_sec": round(self.total_time_sec, 3),
            "runs": [
                {
                    "name": r.name,
                    "avg_time_sec": round(r.avg_time_sec, 6),
                    "min_time_sec": round(r.min_time_sec, 6),
                    "throughput": round(r.throughput, 2) if r.throughput else None
                }
                for r in self.runs
            ],
            "fastest": self.fastest,
            "slowest": self.slowest,
            "comparisons": [
                {
                    "baseline": c.baseline,
                    "competitor": c.competitor,
                    "speedup": round(c.speedup, 2)
                }
                for c in self.comparisons
            ]
        }


# ============================================================================
# Benchmarking Engine
# ============================================================================

class BenchmarkingEngine:
    """
    Production-grade Benchmarking engine.
    
    Features:
    - Accurate timing
    - Multiple iterations
    - Memory tracking
    - Throughput calculation
    - Comparison analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.runs: List[BenchmarkRun] = []
    
    def benchmark(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 10,
        warmup: int = 2,
        n_items: int = None
    ) -> BenchmarkRun:
        """Benchmark a function."""
        kwargs = kwargs or {}
        
        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Force garbage collection
        gc.collect()
        
        # Timed runs
        times = []
        result = None
        
        for i in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        
        # Calculate throughput
        throughput = 0.0
        if n_items:
            throughput = n_items / np.mean(times)
        
        run = BenchmarkRun(
            name=name,
            duration_sec=float(np.sum(times)),
            iterations=iterations,
            avg_time_sec=float(np.mean(times)),
            min_time_sec=float(np.min(times)),
            max_time_sec=float(np.max(times)),
            std_time_sec=float(np.std(times)),
            throughput=throughput,
            result=result
        )
        
        self.runs.append(run)
        
        if self.verbose:
            logger.info(f"Benchmark '{name}': {run.avg_time_sec:.6f}s avg ({iterations} iterations)")
        
        return run
    
    def compare(
        self,
        baseline_name: str,
        competitor_name: str
    ) -> Optional[BenchmarkComparison]:
        """Compare two benchmarks."""
        baseline = None
        competitor = None
        
        for run in self.runs:
            if run.name == baseline_name:
                baseline = run
            elif run.name == competitor_name:
                competitor = run
        
        if not baseline or not competitor:
            return None
        
        speedup = baseline.avg_time_sec / competitor.avg_time_sec
        
        return BenchmarkComparison(
            baseline=baseline_name,
            competitor=competitor_name,
            speedup=speedup,
            baseline_avg=baseline.avg_time_sec,
            competitor_avg=competitor.avg_time_sec
        )
    
    def get_results(self) -> BenchmarkResult:
        """Get all benchmark results."""
        if not self.runs:
            return BenchmarkResult()
        
        # Find fastest/slowest
        sorted_runs = sorted(self.runs, key=lambda x: x.avg_time_sec)
        fastest = sorted_runs[0].name
        slowest = sorted_runs[-1].name
        
        # Pairwise comparisons
        comparisons = []
        if len(self.runs) > 1:
            baseline = sorted_runs[-1]  # Slowest as baseline
            for run in sorted_runs[:-1]:
                comp = BenchmarkComparison(
                    baseline=baseline.name,
                    competitor=run.name,
                    speedup=baseline.avg_time_sec / run.avg_time_sec,
                    baseline_avg=baseline.avg_time_sec,
                    competitor_avg=run.avg_time_sec
                )
                comparisons.append(comp)
        
        return BenchmarkResult(
            n_benchmarks=len(self.runs),
            total_time_sec=sum(r.duration_sec for r in self.runs),
            runs=self.runs,
            comparisons=comparisons,
            fastest=fastest,
            slowest=slowest
        )
    
    def time_it(self, name: str, iterations: int = 10):
        """Decorator for timing functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                return self.benchmark(name, func, args, kwargs, iterations).result
            return wrapper
        return decorator
    
    def reset(self):
        """Reset all benchmarks."""
        self.runs = []


def time_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    iterations: int = 10
) -> Dict[str, float]:
    """Quick function timing."""
    engine = BenchmarkingEngine(verbose=False)
    run = engine.benchmark("function", func, args, kwargs, iterations)
    
    return {
        "avg_time_sec": run.avg_time_sec,
        "min_time_sec": run.min_time_sec,
        "max_time_sec": run.max_time_sec,
        "iterations": run.iterations
    }


def compare_functions(
    functions: Dict[str, Callable],
    args: tuple = (),
    kwargs: dict = None,
    iterations: int = 10
) -> Dict[str, Any]:
    """Compare multiple functions."""
    engine = BenchmarkingEngine(verbose=False)
    
    for name, func in functions.items():
        engine.benchmark(name, func, args, kwargs, iterations)
    
    return engine.get_results().to_dict()


# ============================================================================
# Factory Functions
# ============================================================================

def get_benchmark_engine() -> BenchmarkingEngine:
    """Get benchmarking engine."""
    return BenchmarkingEngine()
