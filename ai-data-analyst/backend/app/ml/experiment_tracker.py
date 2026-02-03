# AI Enterprise Data Analyst - Experiment Tracker Engine
# Production-grade ML experiment tracking
# Handles: runs, parameters, metrics, artifacts

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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

class RunStatus(str, Enum):
    """Experiment run status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExperimentRun:
    """Single experiment run."""
    run_id: str
    experiment_name: str
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics (can be logged multiple times)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Tags
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Status
    status: RunStatus = RunStatus.RUNNING
    
    # Timestamps
    start_time: str = ""
    end_time: str = ""
    duration_sec: float = 0.0
    
    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()


@dataclass
class Experiment:
    """Experiment with multiple runs."""
    name: str
    description: str = ""
    
    runs: Dict[str, ExperimentRun] = field(default_factory=dict)
    
    # Best run tracking
    best_run_id: str = None
    best_metric_name: str = None
    best_metric_value: float = None
    
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ExperimentResult:
    """Experiment tracker status."""
    n_experiments: int = 0
    n_runs: int = 0
    
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_experiments": self.n_experiments,
            "n_runs": self.n_runs,
            "experiments": self.experiments
        }


# ============================================================================
# Experiment Tracker Engine
# ============================================================================

class ExperimentTrackerEngine:
    """
    Production-grade Experiment Tracker engine.
    
    Features:
    - Experiment management
    - Run tracking
    - Parameter logging
    - Metric history
    - Best run tracking
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.experiments: Dict[str, Experiment] = {}
        self._active_run: ExperimentRun = None
        self._run_counter = 0
    
    def create_experiment(
        self,
        name: str,
        description: str = ""
    ) -> str:
        """Create an experiment."""
        if name not in self.experiments:
            self.experiments[name] = Experiment(
                name=name,
                description=description
            )
        
        if self.verbose:
            logger.info(f"Created experiment: {name}")
        
        return name
    
    def start_run(
        self,
        experiment_name: str,
        run_name: str = None,
        tags: Dict[str, str] = None
    ) -> ExperimentRun:
        """Start a new run."""
        # Auto-create experiment
        if experiment_name not in self.experiments:
            self.create_experiment(experiment_name)
        
        self._run_counter += 1
        run_id = run_name or f"run_{self._run_counter:04d}"
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            tags=tags or {}
        )
        
        self.experiments[experiment_name].runs[run_id] = run
        self._active_run = run
        
        if self.verbose:
            logger.info(f"Started run: {run_id}")
        
        return run
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self._active_run:
            self._active_run.parameters[key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self._active_run:
            self._active_run.parameters.update(params)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric."""
        if self._active_run:
            if key not in self._active_run.metrics:
                self._active_run.metrics[key] = []
            self._active_run.metrics[key].append(value)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value)
    
    def log_artifact(self, name: str, path: str):
        """Log an artifact."""
        if self._active_run:
            self._active_run.artifacts[name] = path
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        if self._active_run:
            self._active_run.tags[key] = value
    
    def end_run(self, status: RunStatus = RunStatus.COMPLETED):
        """End the current run."""
        if self._active_run:
            self._active_run.end_time = datetime.now().isoformat()
            self._active_run.status = status
            
            start = datetime.fromisoformat(self._active_run.start_time)
            end = datetime.fromisoformat(self._active_run.end_time)
            self._active_run.duration_sec = (end - start).total_seconds()
            
            if self.verbose:
                logger.info(f"Ended run: {self._active_run.run_id}")
            
            self._active_run = None
    
    def get_run(self, experiment_name: str, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run."""
        if experiment_name in self.experiments:
            return self.experiments[experiment_name].runs.get(run_id)
        return None
    
    def search_runs(
        self,
        experiment_name: str,
        filter_string: str = None,
        order_by: str = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """Search runs."""
        if experiment_name not in self.experiments:
            return pd.DataFrame()
        
        runs = self.experiments[experiment_name].runs.values()
        
        data = []
        for run in runs:
            row = {
                'run_id': run.run_id,
                'status': run.status.value,
                'duration_sec': run.duration_sec
            }
            row.update({f"param_{k}": v for k, v in run.parameters.items()})
            row.update({
                f"metric_{k}": v[-1] if v else None 
                for k, v in run.metrics.items()
            })
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if order_by and order_by in df.columns:
            df = df.sort_values(order_by, ascending=False)
        
        return df.head(max_results)
    
    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        maximize: bool = True
    ) -> Optional[ExperimentRun]:
        """Get the best run by a metric."""
        if experiment_name not in self.experiments:
            return None
        
        runs = self.experiments[experiment_name].runs.values()
        
        best_run = None
        best_value = float('-inf') if maximize else float('inf')
        
        for run in runs:
            if metric in run.metrics and run.metrics[metric]:
                value = run.metrics[metric][-1]
                
                if maximize and value > best_value:
                    best_value = value
                    best_run = run
                elif not maximize and value < best_value:
                    best_value = value
                    best_run = run
        
        # Update experiment
        if best_run:
            exp = self.experiments[experiment_name]
            exp.best_run_id = best_run.run_id
            exp.best_metric_name = metric
            exp.best_metric_value = best_value
        
        return best_run
    
    def compare_runs(
        self,
        experiment_name: str,
        run_ids: List[str]
    ) -> pd.DataFrame:
        """Compare specific runs."""
        if experiment_name not in self.experiments:
            return pd.DataFrame()
        
        data = []
        for run_id in run_ids:
            run = self.experiments[experiment_name].runs.get(run_id)
            if run:
                row = {'run_id': run_id}
                row.update(run.parameters)
                row.update({k: v[-1] if v else None for k, v in run.metrics.items()})
                data.append(row)
        
        return pd.DataFrame(data)
    
    def get_status(self) -> ExperimentResult:
        """Get tracker status."""
        total_runs = sum(len(e.runs) for e in self.experiments.values())
        
        return ExperimentResult(
            n_experiments=len(self.experiments),
            n_runs=total_runs,
            experiments=[
                {
                    'name': e.name,
                    'n_runs': len(e.runs),
                    'best_run': e.best_run_id,
                    'best_metric': f"{e.best_metric_name}={e.best_metric_value}"
                    if e.best_metric_name else None
                }
                for e in self.experiments.values()
            ]
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_experiment_tracker() -> ExperimentTrackerEngine:
    """Get experiment tracker engine."""
    return ExperimentTrackerEngine()


def track_experiment(
    experiment_name: str,
    params: Dict[str, Any],
    train_func,
    **kwargs
) -> ExperimentRun:
    """Track a training experiment."""
    tracker = ExperimentTrackerEngine(verbose=False)
    run = tracker.start_run(experiment_name)
    
    tracker.log_params(params)
    
    try:
        result = train_func(**kwargs)
        
        if isinstance(result, dict):
            tracker.log_metrics(result)
        
        tracker.end_run(RunStatus.COMPLETED)
    except Exception as e:
        tracker.set_tag('error', str(e))
        tracker.end_run(RunStatus.FAILED)
        raise
    
    return run
