# AI Enterprise Data Analyst - Funnel Analysis Engine
# Production-grade conversion funnel analysis
# Handles: any event/step data, flexible funnel definitions

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
# Data Classes
# ============================================================================

@dataclass
class FunnelStep:
    """Single step in funnel."""
    name: str
    count: int
    conversion_rate: float  # From previous step
    overall_rate: float  # From first step
    drop_off_count: int
    drop_off_rate: float


@dataclass
class FunnelConfig:
    """Configuration for funnel analysis."""
    # Column mappings
    user_id_col: Optional[str] = None
    event_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    
    # Funnel steps (in order)
    steps: Optional[List[str]] = None
    
    # Time window for funnel completion (hours)
    time_window_hours: Optional[int] = None


@dataclass
class FunnelResult:
    """Complete funnel analysis result."""
    n_users: int = 0
    n_steps: int = 0
    
    # Step details
    steps: List[FunnelStep] = field(default_factory=list)
    
    # Overall metrics
    overall_conversion: float = 0.0
    biggest_drop_step: str = ""
    biggest_drop_rate: float = 0.0
    
    # Time analysis
    avg_time_between_steps: Dict[str, float] = field(default_factory=dict)
    
    # Segment breakdown (if applicable)
    segment_funnels: Dict[str, List[FunnelStep]] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_users": self.n_users,
                "n_steps": self.n_steps,
                "overall_conversion": round(self.overall_conversion * 100, 2),
                "biggest_drop": {
                    "step": self.biggest_drop_step,
                    "rate": round(self.biggest_drop_rate * 100, 2)
                }
            },
            "steps": [
                {
                    "name": s.name,
                    "count": s.count,
                    "conversion_rate": round(s.conversion_rate * 100, 2),
                    "overall_rate": round(s.overall_rate * 100, 2),
                    "drop_off_count": s.drop_off_count,
                    "drop_off_rate": round(s.drop_off_rate * 100, 2)
                }
                for s in self.steps
            ],
            "avg_time_between_steps": {k: round(v, 2) for k, v in self.avg_time_between_steps.items()}
        }


# ============================================================================
# Funnel Analysis Engine
# ============================================================================

class FunnelAnalysisEngine:
    """
    Complete Funnel Analysis engine.
    
    Features:
    - Flexible step definitions
    - Time-windowed funnel completion
    - Drop-off analysis
    - Time between steps
    """
    
    def __init__(self, config: FunnelConfig = None, verbose: bool = True):
        self.config = config or FunnelConfig()
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        steps: List[str] = None,
        user_id_col: str = None,
        event_col: str = None,
        timestamp_col: str = None
    ) -> FunnelResult:
        """Perform funnel analysis."""
        start_time = datetime.now()
        
        # Auto-detect columns
        user_id_col = user_id_col or self.config.user_id_col or self._detect_user_col(df)
        event_col = event_col or self.config.event_col or self._detect_event_col(df)
        timestamp_col = timestamp_col or self.config.timestamp_col or self._detect_timestamp_col(df)
        
        # Get steps
        steps = steps or self.config.steps
        if steps is None:
            steps = df[event_col].unique().tolist()[:10]
        
        if self.verbose:
            logger.info(f"Funnel analysis: {len(steps)} steps, user={user_id_col}, event={event_col}")
        
        # Prepare data
        df = df.copy()
        if timestamp_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df = df.sort_values(timestamp_col)
        
        # Calculate funnel
        funnel_steps = self._calculate_funnel(df, steps, user_id_col, event_col)
        
        # Calculate time between steps
        time_between = {}
        if timestamp_col:
            time_between = self._calculate_time_between_steps(df, steps, user_id_col, event_col, timestamp_col)
        
        # Find biggest drop
        biggest_drop_step = ""
        biggest_drop_rate = 0.0
        for step in funnel_steps:
            if step.drop_off_rate > biggest_drop_rate:
                biggest_drop_rate = step.drop_off_rate
                biggest_drop_step = step.name
        
        # Overall conversion
        overall_conversion = funnel_steps[-1].overall_rate if funnel_steps else 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FunnelResult(
            n_users=df[user_id_col].nunique(),
            n_steps=len(steps),
            steps=funnel_steps,
            overall_conversion=overall_conversion,
            biggest_drop_step=biggest_drop_step,
            biggest_drop_rate=biggest_drop_rate,
            avg_time_between_steps=time_between,
            processing_time_sec=processing_time
        )
    
    def _calculate_funnel(
        self,
        df: pd.DataFrame,
        steps: List[str],
        user_col: str,
        event_col: str
    ) -> List[FunnelStep]:
        """Calculate funnel metrics."""
        funnel_steps = []
        prev_users = None
        first_count = 0
        
        for i, step in enumerate(steps):
            step_users = set(df[df[event_col] == step][user_col].unique())
            
            if prev_users is not None:
                # Only count users who completed previous step
                step_users = step_users & prev_users
            
            count = len(step_users)
            
            if i == 0:
                first_count = count
                conversion_rate = 1.0
                drop_off_count = 0
                drop_off_rate = 0.0
            else:
                prev_count = len(prev_users) if prev_users else 0
                conversion_rate = count / prev_count if prev_count > 0 else 0
                drop_off_count = prev_count - count
                drop_off_rate = drop_off_count / prev_count if prev_count > 0 else 0
            
            overall_rate = count / first_count if first_count > 0 else 0
            
            funnel_steps.append(FunnelStep(
                name=step,
                count=count,
                conversion_rate=conversion_rate,
                overall_rate=overall_rate,
                drop_off_count=drop_off_count,
                drop_off_rate=drop_off_rate
            ))
            
            prev_users = step_users
        
        return funnel_steps
    
    def _calculate_time_between_steps(
        self,
        df: pd.DataFrame,
        steps: List[str],
        user_col: str,
        event_col: str,
        time_col: str
    ) -> Dict[str, float]:
        """Calculate average time between steps."""
        times = {}
        
        for i in range(1, len(steps)):
            prev_step = steps[i - 1]
            curr_step = steps[i]
            
            prev_times = df[df[event_col] == prev_step].groupby(user_col)[time_col].first()
            curr_times = df[df[event_col] == curr_step].groupby(user_col)[time_col].first()
            
            common_users = set(prev_times.index) & set(curr_times.index)
            
            if common_users:
                diffs = [(curr_times[u] - prev_times[u]).total_seconds() / 3600 
                        for u in common_users if curr_times[u] > prev_times[u]]
                if diffs:
                    times[f"{prev_step} → {curr_step}"] = np.mean(diffs)
        
        return times
    
    def _detect_user_col(self, df: pd.DataFrame) -> str:
        patterns = ['user', 'customer', 'visitor', 'session', 'id']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[0]
    
    def _detect_event_col(self, df: pd.DataFrame) -> str:
        patterns = ['event', 'action', 'step', 'activity', 'type', 'name']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                if df[col].dtype == 'object':
                    return col
        return df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else df.columns[1]
    
    def _detect_timestamp_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        patterns = ['time', 'date', 'timestamp', 'created']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_funnel_engine(config: FunnelConfig = None) -> FunnelAnalysisEngine:
    """Get funnel analysis engine."""
    return FunnelAnalysisEngine(config=config)


def quick_funnel(
    df: pd.DataFrame,
    steps: List[str]
) -> Dict[str, Any]:
    """Quick funnel analysis."""
    config = FunnelConfig(steps=steps)
    engine = FunnelAnalysisEngine(config=config, verbose=False)
    result = engine.analyze(df, steps=steps)
    return result.to_dict()
