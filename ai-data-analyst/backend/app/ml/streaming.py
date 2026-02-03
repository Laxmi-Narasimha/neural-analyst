# AI Enterprise Data Analyst - Real-Time Streaming Analytics
# Event processing, windowing, and streaming patterns (Kafka/Flink inspired)

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Callable, Iterator
from uuid import uuid4
import time
import threading
import asyncio

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Streaming Types
# ============================================================================

class WindowType(str, Enum):
    """Types of time windows."""
    TUMBLING = "tumbling"  # Fixed, non-overlapping
    SLIDING = "sliding"    # Overlapping
    SESSION = "session"    # Gap-based
    HOPPING = "hopping"    # Fixed with hop


class AggregationType(str, Enum):
    """Aggregation functions for windowed data."""
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    PERCENTILE = "percentile"


@dataclass
class StreamEvent:
    """Single event in a data stream."""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    key: Optional[str] = None
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class WindowResult:
    """Result of windowed aggregation."""
    
    window_start: datetime
    window_end: datetime
    key: Optional[str]
    aggregation: AggregationType
    value: float
    count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "key": self.key,
            "aggregation": self.aggregation.value,
            "value": round(self.value, 6),
            "count": self.count
        }


# ============================================================================
# Time Windows
# ============================================================================

class TimeWindow(ABC):
    """Base class for time windows."""
    
    def __init__(self, duration: timedelta):
        self.duration = duration
        self._buffer: deque[StreamEvent] = deque()
    
    @abstractmethod
    def add(self, event: StreamEvent) -> list[StreamEvent]:
        """Add event and return any expired events."""
        pass
    
    @abstractmethod
    def get_windows(self) -> Iterator[tuple[datetime, datetime, list[StreamEvent]]]:
        """Get all current windows with their events."""
        pass
    
    def clear(self) -> None:
        """Clear the window buffer."""
        self._buffer.clear()


class TumblingWindow(TimeWindow):
    """Fixed-size, non-overlapping time window."""
    
    def __init__(self, duration: timedelta):
        super().__init__(duration)
        self._window_start: Optional[datetime] = None
    
    def add(self, event: StreamEvent) -> list[StreamEvent]:
        """Add event, emit window if complete."""
        if self._window_start is None:
            self._window_start = self._align_to_window(event.timestamp)
        
        window_end = self._window_start + self.duration
        
        expired = []
        if event.timestamp >= window_end:
            # Window complete, emit all events
            expired = list(self._buffer)
            self._buffer.clear()
            self._window_start = self._align_to_window(event.timestamp)
        
        self._buffer.append(event)
        return expired
    
    def _align_to_window(self, ts: datetime) -> datetime:
        """Align timestamp to window boundary."""
        epoch = datetime(1970, 1, 1)
        seconds = (ts - epoch).total_seconds()
        window_seconds = self.duration.total_seconds()
        aligned = int(seconds // window_seconds) * window_seconds
        return epoch + timedelta(seconds=aligned)
    
    def get_windows(self) -> Iterator[tuple[datetime, datetime, list[StreamEvent]]]:
        """Get current window."""
        if self._window_start and self._buffer:
            yield (
                self._window_start,
                self._window_start + self.duration,
                list(self._buffer)
            )


class SlidingWindow(TimeWindow):
    """Overlapping time window with slide interval."""
    
    def __init__(self, duration: timedelta, slide: timedelta):
        super().__init__(duration)
        self.slide = slide
    
    def add(self, event: StreamEvent) -> list[StreamEvent]:
        """Add event and remove expired events."""
        self._buffer.append(event)
        
        # Remove events outside window
        cutoff = event.timestamp - self.duration
        expired = []
        while self._buffer and self._buffer[0].timestamp < cutoff:
            expired.append(self._buffer.popleft())
        
        return expired
    
    def get_windows(self) -> Iterator[tuple[datetime, datetime, list[StreamEvent]]]:
        """Get overlapping windows."""
        if not self._buffer:
            return
        
        earliest = self._buffer[0].timestamp
        latest = self._buffer[-1].timestamp
        
        current = earliest
        while current <= latest:
            window_end = current + self.duration
            events = [e for e in self._buffer if current <= e.timestamp < window_end]
            if events:
                yield (current, window_end, events)
            current += self.slide


class SessionWindow(TimeWindow):
    """Gap-based session window."""
    
    def __init__(self, gap: timedelta):
        super().__init__(gap)
        self._sessions: list[list[StreamEvent]] = []
    
    def add(self, event: StreamEvent) -> list[StreamEvent]:
        """Add event, detect session boundaries."""
        if not self._sessions:
            self._sessions.append([event])
            return []
        
        last_session = self._sessions[-1]
        last_event = last_session[-1]
        
        if event.timestamp - last_event.timestamp > self.duration:
            # Gap exceeded, start new session
            completed = last_session
            self._sessions.append([event])
            return completed
        else:
            last_session.append(event)
            return []
    
    def get_windows(self) -> Iterator[tuple[datetime, datetime, list[StreamEvent]]]:
        """Get all sessions."""
        for session in self._sessions:
            if session:
                yield (
                    session[0].timestamp,
                    session[-1].timestamp,
                    session
                )


# ============================================================================
# Stream Aggregator
# ============================================================================

class StreamAggregator:
    """Aggregate streaming data over windows."""
    
    def __init__(
        self,
        window: TimeWindow,
        aggregation: AggregationType = AggregationType.MEAN,
        key_fn: Optional[Callable[[StreamEvent], str]] = None
    ):
        self.window = window
        self.aggregation = aggregation
        self.key_fn = key_fn
        self._results: list[WindowResult] = []
    
    def process(self, event: StreamEvent) -> list[WindowResult]:
        """Process event and return any completed aggregations."""
        expired = self.window.add(event)
        
        if not expired:
            return []
        
        # Aggregate expired events
        if self.key_fn:
            # Group by key
            groups: dict[str, list[StreamEvent]] = {}
            for e in expired:
                key = self.key_fn(e)
                groups.setdefault(key, []).append(e)
            
            results = []
            for key, events in groups.items():
                result = self._aggregate(events, key)
                results.append(result)
                self._results.append(result)
            return results
        else:
            result = self._aggregate(expired, None)
            self._results.append(result)
            return [result]
    
    def _aggregate(self, events: list[StreamEvent], key: Optional[str]) -> WindowResult:
        """Compute aggregation over events."""
        values = [e.value for e in events if isinstance(e.value, (int, float))]
        
        if not values:
            agg_value = 0.0
        elif self.aggregation == AggregationType.COUNT:
            agg_value = float(len(values))
        elif self.aggregation == AggregationType.SUM:
            agg_value = sum(values)
        elif self.aggregation == AggregationType.MEAN:
            agg_value = np.mean(values)
        elif self.aggregation == AggregationType.MIN:
            agg_value = min(values)
        elif self.aggregation == AggregationType.MAX:
            agg_value = max(values)
        elif self.aggregation == AggregationType.STDDEV:
            agg_value = np.std(values)
        else:
            agg_value = np.mean(values)
        
        return WindowResult(
            window_start=events[0].timestamp if events else datetime.utcnow(),
            window_end=events[-1].timestamp if events else datetime.utcnow(),
            key=key,
            aggregation=self.aggregation,
            value=float(agg_value),
            count=len(events)
        )
    
    def get_results(self) -> list[WindowResult]:
        """Get all aggregation results."""
        return self._results


# ============================================================================
# Stream Processor
# ============================================================================

class StreamProcessor:
    """
    Process streaming data with transformations.
    
    Operations:
    - Filter
    - Map
    - FlatMap
    - Reduce
    - Window aggregations
    """
    
    def __init__(self, name: str = "processor"):
        self.name = name
        self._operations: list[tuple[str, Callable]] = []
        self._output: deque[StreamEvent] = deque(maxlen=10000)
    
    def filter(self, predicate: Callable[[StreamEvent], bool]) -> "StreamProcessor":
        """Filter events based on predicate."""
        self._operations.append(("filter", predicate))
        return self
    
    def map(self, fn: Callable[[StreamEvent], StreamEvent]) -> "StreamProcessor":
        """Transform each event."""
        self._operations.append(("map", fn))
        return self
    
    def flat_map(self, fn: Callable[[StreamEvent], list[StreamEvent]]) -> "StreamProcessor":
        """Transform each event to multiple events."""
        self._operations.append(("flat_map", fn))
        return self
    
    def key_by(self, key_fn: Callable[[StreamEvent], str]) -> "StreamProcessor":
        """Set key for downstream operations."""
        self._operations.append(("key_by", key_fn))
        return self
    
    def process(self, event: StreamEvent) -> list[StreamEvent]:
        """Process a single event through the pipeline."""
        events = [event]
        
        for op_type, op_fn in self._operations:
            new_events = []
            
            for e in events:
                if op_type == "filter":
                    if op_fn(e):
                        new_events.append(e)
                elif op_type == "map":
                    new_events.append(op_fn(e))
                elif op_type == "flat_map":
                    new_events.extend(op_fn(e))
                elif op_type == "key_by":
                    e.key = op_fn(e)
                    new_events.append(e)
                else:
                    new_events.append(e)
            
            events = new_events
        
        for e in events:
            self._output.append(e)
        
        return events
    
    def process_batch(self, events: list[StreamEvent]) -> list[StreamEvent]:
        """Process batch of events."""
        results = []
        for event in events:
            results.extend(self.process(event))
        return results
    
    def get_output(self) -> list[StreamEvent]:
        """Get processed output."""
        return list(self._output)


# ============================================================================
# Streaming Analytics Engine
# ============================================================================

class StreamingAnalyticsEngine:
    """
    Real-time streaming analytics engine.
    
    Features:
    - Event ingestion
    - Windowed aggregations
    - Stream processing pipelines
    - Real-time metrics
    """
    
    def __init__(self):
        self._processors: dict[str, StreamProcessor] = {}
        self._aggregators: dict[str, StreamAggregator] = {}
        self._metrics: dict[str, float] = {}
        self._event_count = 0
        self._start_time = datetime.utcnow()
    
    def create_processor(self, name: str) -> StreamProcessor:
        """Create a named stream processor."""
        processor = StreamProcessor(name)
        self._processors[name] = processor
        return processor
    
    def create_aggregator(
        self,
        name: str,
        window_type: WindowType,
        duration: timedelta,
        aggregation: AggregationType = AggregationType.MEAN,
        slide: Optional[timedelta] = None
    ) -> StreamAggregator:
        """Create a named windowed aggregator."""
        if window_type == WindowType.TUMBLING:
            window = TumblingWindow(duration)
        elif window_type == WindowType.SLIDING:
            window = SlidingWindow(duration, slide or duration / 2)
        elif window_type == WindowType.SESSION:
            window = SessionWindow(duration)
        else:
            window = TumblingWindow(duration)
        
        aggregator = StreamAggregator(window, aggregation)
        self._aggregators[name] = aggregator
        return aggregator
    
    def ingest(self, event: StreamEvent) -> dict[str, Any]:
        """Ingest a single event."""
        self._event_count += 1
        
        results = {
            "event_id": event.event_id,
            "processor_results": {},
            "aggregator_results": {}
        }
        
        # Process through all processors
        for name, processor in self._processors.items():
            output = processor.process(event)
            results["processor_results"][name] = len(output)
        
        # Process through all aggregators
        for name, aggregator in self._aggregators.items():
            window_results = aggregator.process(event)
            if window_results:
                results["aggregator_results"][name] = [r.to_dict() for r in window_results]
        
        return results
    
    def ingest_batch(self, events: list[dict]) -> dict[str, Any]:
        """Ingest batch of events from dictionaries."""
        stream_events = []
        for e in events:
            stream_events.append(StreamEvent(
                timestamp=pd.to_datetime(e.get("timestamp", datetime.utcnow())),
                key=e.get("key"),
                value=e.get("value"),
                metadata=e.get("metadata", {})
            ))
        
        results = {"events_processed": len(stream_events), "aggregations": []}
        
        for event in stream_events:
            result = self.ingest(event)
            for agg_results in result["aggregator_results"].values():
                results["aggregations"].extend(agg_results)
        
        return results
    
    def get_metrics(self) -> dict[str, Any]:
        """Get streaming metrics."""
        elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            "total_events": self._event_count,
            "events_per_second": self._event_count / elapsed if elapsed > 0 else 0,
            "uptime_seconds": elapsed,
            "processors": list(self._processors.keys()),
            "aggregators": list(self._aggregators.keys())
        }
    
    def get_aggregator_results(self, name: str) -> list[dict]:
        """Get results from a specific aggregator."""
        if name in self._aggregators:
            return [r.to_dict() for r in self._aggregators[name].get_results()]
        return []


# Factory function
def get_streaming_engine() -> StreamingAnalyticsEngine:
    """Get streaming analytics engine instance."""
    return StreamingAnalyticsEngine()
