# AI Enterprise Data Analyst - Session Analysis Engine
# Production-grade user session and behavior analysis
# Handles: session metrics, user journeys, engagement

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
class SessionMetrics:
    """Session-level metrics."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    duration_sec: float
    page_views: int
    actions: int
    is_bounce: bool
    converted: bool


@dataclass
class SessionSummary:
    """Session analysis summary."""
    total_sessions: int
    total_users: int
    avg_session_duration: float
    median_session_duration: float
    avg_pages_per_session: float
    bounce_rate: float
    conversion_rate: float
    
    # Engagement
    engaged_sessions: int  # >30 seconds
    deeply_engaged: int  # >5 pages or >2 minutes


@dataclass
class SessionResult:
    """Complete session analysis result."""
    summary: SessionSummary = None
    sessions: List[SessionMetrics] = field(default_factory=list)
    
    # By hour distribution
    hourly_distribution: Dict[int, int] = field(default_factory=dict)
    
    # By day of week
    daily_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Top paths
    top_entry_pages: Dict[str, int] = field(default_factory=dict)
    top_exit_pages: Dict[str, int] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_sessions": self.summary.total_sessions if self.summary else 0,
                "total_users": self.summary.total_users if self.summary else 0,
                "avg_duration_sec": round(self.summary.avg_session_duration, 1) if self.summary else 0,
                "avg_pages": round(self.summary.avg_pages_per_session, 2) if self.summary else 0,
                "bounce_rate": round(self.summary.bounce_rate, 2) if self.summary else 0,
                "conversion_rate": round(self.summary.conversion_rate, 2) if self.summary else 0
            },
            "hourly_distribution": self.hourly_distribution,
            "daily_distribution": self.daily_distribution,
            "top_entry_pages": dict(list(self.top_entry_pages.items())[:5]),
            "top_exit_pages": dict(list(self.top_exit_pages.items())[:5])
        }


# ============================================================================
# Session Analysis Engine
# ============================================================================

class SessionAnalysisEngine:
    """
    Production-grade Session Analysis engine.
    
    Features:
    - Session reconstruction
    - Duration analysis
    - Bounce rate calculation
    - Conversion tracking
    - Engagement metrics
    """
    
    def __init__(
        self,
        session_timeout_min: int = 30,
        verbose: bool = True
    ):
        self.session_timeout = timedelta(minutes=session_timeout_min)
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        user_col: str = None,
        timestamp_col: str = None,
        page_col: str = None,
        action_col: str = None,
        conversion_col: str = None
    ) -> SessionResult:
        """Analyze sessions."""
        start_time = datetime.now()
        
        # Auto-detect columns
        if user_col is None:
            user_col = self._detect_column(df, ['user', 'visitor', 'customer', 'id'])
        if timestamp_col is None:
            timestamp_col = self._detect_column(df, ['timestamp', 'time', 'date', 'datetime'])
        if page_col is None:
            page_col = self._detect_column(df, ['page', 'url', 'path', 'screen'])
        
        if user_col is None or timestamp_col is None:
            raise ValueError("Could not detect user or timestamp columns")
        
        df_work = df.copy()
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col], errors='coerce')
        df_work = df_work.dropna(subset=[timestamp_col])
        df_work = df_work.sort_values([user_col, timestamp_col])
        
        if self.verbose:
            logger.info(f"Session analysis: {len(df_work)} events, {df_work[user_col].nunique()} users")
        
        # Reconstruct sessions
        sessions = self._reconstruct_sessions(
            df_work, user_col, timestamp_col, page_col, action_col, conversion_col
        )
        
        # Calculate summary
        if sessions:
            summary = SessionSummary(
                total_sessions=len(sessions),
                total_users=len(set(s.user_id for s in sessions)),
                avg_session_duration=np.mean([s.duration_sec for s in sessions]),
                median_session_duration=np.median([s.duration_sec for s in sessions]),
                avg_pages_per_session=np.mean([s.page_views for s in sessions]),
                bounce_rate=sum(1 for s in sessions if s.is_bounce) / len(sessions) * 100,
                conversion_rate=sum(1 for s in sessions if s.converted) / len(sessions) * 100,
                engaged_sessions=sum(1 for s in sessions if s.duration_sec > 30),
                deeply_engaged=sum(1 for s in sessions if s.page_views > 5 or s.duration_sec > 120)
            )
        else:
            summary = SessionSummary(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Distributions
        hourly = {}
        daily = {}
        
        for s in sessions:
            hour = s.start_time.hour
            hourly[hour] = hourly.get(hour, 0) + 1
            
            day = s.start_time.strftime('%A')
            daily[day] = daily.get(day, 0) + 1
        
        # Entry/exit pages
        entry_pages = {}
        exit_pages = {}
        
        if page_col:
            for user_id in df_work[user_col].unique():
                user_data = df_work[df_work[user_col] == user_id]
                if len(user_data) > 0:
                    entry = str(user_data.iloc[0][page_col])
                    exit_page = str(user_data.iloc[-1][page_col])
                    
                    entry_pages[entry] = entry_pages.get(entry, 0) + 1
                    exit_pages[exit_page] = exit_pages.get(exit_page, 0) + 1
        
        # Sort
        entry_pages = dict(sorted(entry_pages.items(), key=lambda x: -x[1]))
        exit_pages = dict(sorted(exit_pages.items(), key=lambda x: -x[1]))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SessionResult(
            summary=summary,
            sessions=sessions[:1000],  # Limit for memory
            hourly_distribution=hourly,
            daily_distribution=daily,
            top_entry_pages=entry_pages,
            top_exit_pages=exit_pages,
            processing_time_sec=processing_time
        )
    
    def _reconstruct_sessions(
        self,
        df: pd.DataFrame,
        user_col: str,
        timestamp_col: str,
        page_col: str,
        action_col: str,
        conversion_col: str
    ) -> List[SessionMetrics]:
        """Reconstruct sessions from events."""
        sessions = []
        session_counter = 0
        
        for user_id in df[user_col].unique():
            user_data = df[df[user_col] == user_id].sort_values(timestamp_col)
            
            if len(user_data) == 0:
                continue
            
            # Split into sessions based on timeout
            session_start = None
            session_events = []
            
            for idx, row in user_data.iterrows():
                timestamp = row[timestamp_col]
                
                if session_start is None:
                    session_start = timestamp
                    session_events = [row]
                elif timestamp - session_events[-1][timestamp_col] > self.session_timeout:
                    # End current session, start new one
                    sessions.append(self._create_session(
                        session_counter, user_id, session_events,
                        timestamp_col, page_col, action_col, conversion_col
                    ))
                    session_counter += 1
                    session_start = timestamp
                    session_events = [row]
                else:
                    session_events.append(row)
            
            # Final session
            if session_events:
                sessions.append(self._create_session(
                    session_counter, user_id, session_events,
                    timestamp_col, page_col, action_col, conversion_col
                ))
                session_counter += 1
        
        return sessions
    
    def _create_session(
        self,
        session_id: int,
        user_id: str,
        events: List[pd.Series],
        timestamp_col: str,
        page_col: str,
        action_col: str,
        conversion_col: str
    ) -> SessionMetrics:
        """Create session from events."""
        start_time = events[0][timestamp_col]
        end_time = events[-1][timestamp_col]
        duration = (end_time - start_time).total_seconds()
        
        page_views = len(set(e[page_col] for e in events)) if page_col else len(events)
        actions = len(events)
        is_bounce = len(events) == 1
        
        converted = False
        if conversion_col:
            converted = any(e.get(conversion_col) for e in events)
        
        return SessionMetrics(
            session_id=f"session_{session_id}",
            user_id=str(user_id),
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration,
            page_views=page_views,
            actions=actions,
            is_bounce=is_bounce,
            converted=converted
        )
    
    def _detect_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Detect column by patterns."""
        for pattern in patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_session_engine() -> SessionAnalysisEngine:
    """Get session analysis engine."""
    return SessionAnalysisEngine()


def quick_session_analysis(
    df: pd.DataFrame,
    user_col: str = None,
    timestamp_col: str = None
) -> Dict[str, Any]:
    """Quick session analysis."""
    engine = SessionAnalysisEngine(verbose=False)
    result = engine.analyze(df, user_col, timestamp_col)
    return result.to_dict()


def calculate_bounce_rate(
    df: pd.DataFrame,
    user_col: str,
    timestamp_col: str
) -> float:
    """Calculate bounce rate."""
    engine = SessionAnalysisEngine(verbose=False)
    result = engine.analyze(df, user_col, timestamp_col)
    return result.summary.bounce_rate if result.summary else 0
