# AI Enterprise Data Analyst - Alert Manager Engine
# Production-grade data monitoring and alerting
# Handles: thresholds, anomaly alerts, notifications

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
# Enums
# ============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    MISSING_DATA = "missing_data"
    DATA_DRIFT = "data_drift"
    CUSTOM = "custom"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Condition
    metric: str = None
    condition: str = None  # >, <, ==, !=, >=, <=
    threshold: float = None
    
    # Custom check function
    check_func: Callable = None
    
    # Status
    enabled: bool = True
    last_triggered: str = None


@dataclass
class Alert:
    """Single alert instance."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    alert_type: AlertType
    
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Timestamps
    triggered_at: str = ""
    acknowledged_at: str = None
    resolved_at: str = None
    
    def __post_init__(self):
        if not self.triggered_at:
            self.triggered_at = datetime.now().isoformat()


@dataclass
class AlertResult:
    """Alert check result."""
    n_rules: int = 0
    n_active_alerts: int = 0
    
    alerts: List[Alert] = field(default_factory=list)
    
    summary: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rules": self.n_rules,
            "n_active_alerts": self.n_active_alerts,
            "by_severity": self.summary,
            "alerts": [
                {
                    "id": a.alert_id,
                    "severity": a.severity.value,
                    "message": a.message,
                    "status": a.status.value
                }
                for a in self.alerts
            ]
        }


# ============================================================================
# Alert Manager Engine
# ============================================================================

class AlertManagerEngine:
    """
    Production-grade Alert Manager engine.
    
    Features:
    - Rule-based alerting
    - Threshold monitoring
    - Anomaly detection
    - Alert history
    - Notification channels
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self._alert_counter = 0
        self._rule_counter = 0
    
    def add_threshold_rule(
        self,
        name: str,
        metric: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> str:
        """Add a threshold-based alert rule."""
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"
        
        self.rules[rule_id] = AlertRule(
            rule_id=rule_id,
            name=name,
            alert_type=AlertType.THRESHOLD,
            severity=severity,
            metric=metric,
            condition=condition,
            threshold=threshold
        )
        
        if self.verbose:
            logger.info(f"Added threshold rule: {name}")
        
        return rule_id
    
    def add_missing_data_rule(
        self,
        name: str,
        column: str,
        max_missing_pct: float = 10.0,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> str:
        """Add a missing data alert rule."""
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"
        
        def check_missing(df: pd.DataFrame) -> Optional[str]:
            if column not in df.columns:
                return f"Column {column} not found"
            
            missing_pct = df[column].isna().mean() * 100
            if missing_pct > max_missing_pct:
                return f"{column} has {missing_pct:.1f}% missing (threshold: {max_missing_pct}%)"
            return None
        
        self.rules[rule_id] = AlertRule(
            rule_id=rule_id,
            name=name,
            alert_type=AlertType.MISSING_DATA,
            severity=severity,
            metric=column,
            check_func=check_missing
        )
        
        return rule_id
    
    def add_anomaly_rule(
        self,
        name: str,
        column: str,
        n_std: float = 3.0,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> str:
        """Add an anomaly detection rule."""
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"
        
        def check_anomaly(df: pd.DataFrame) -> Optional[str]:
            if column not in df.columns:
                return None
            
            values = df[column].dropna()
            if len(values) < 10:
                return None
            
            mean = values.mean()
            std = values.std()
            
            anomalies = (values < mean - n_std * std) | (values > mean + n_std * std)
            n_anomalies = anomalies.sum()
            
            if n_anomalies > 0:
                return f"{n_anomalies} anomalies detected in {column} (>{n_std} std from mean)"
            return None
        
        self.rules[rule_id] = AlertRule(
            rule_id=rule_id,
            name=name,
            alert_type=AlertType.ANOMALY,
            severity=severity,
            metric=column,
            check_func=check_anomaly
        )
        
        return rule_id
    
    def add_custom_rule(
        self,
        name: str,
        check_func: Callable,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> str:
        """Add a custom alert rule."""
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"
        
        self.rules[rule_id] = AlertRule(
            rule_id=rule_id,
            name=name,
            alert_type=AlertType.CUSTOM,
            severity=severity,
            check_func=check_func
        )
        
        return rule_id
    
    def check_rules(
        self,
        df: pd.DataFrame = None,
        metrics: Dict[str, float] = None
    ) -> AlertResult:
        """Check all rules against data."""
        new_alerts = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            alert_message = None
            
            # Threshold rules with metrics
            if rule.alert_type == AlertType.THRESHOLD and metrics:
                if rule.metric in metrics:
                    value = metrics[rule.metric]
                    triggered = self._check_condition(value, rule.condition, rule.threshold)
                    if triggered:
                        alert_message = f"{rule.metric} = {value} {rule.condition} {rule.threshold}"
            
            # Custom/function-based rules
            elif rule.check_func and df is not None:
                alert_message = rule.check_func(df)
            
            # Create alert if triggered
            if alert_message:
                self._alert_counter += 1
                alert = Alert(
                    alert_id=f"alert_{self._alert_counter:06d}",
                    rule_id=rule_id,
                    severity=rule.severity,
                    alert_type=rule.alert_type,
                    message=f"[{rule.name}] {alert_message}"
                )
                
                new_alerts.append(alert)
                self.alerts.append(alert)
                rule.last_triggered = datetime.now().isoformat()
                
                if self.verbose:
                    logger.warning(f"Alert triggered: {alert.message}")
        
        # Count by severity
        active = [a for a in self.alerts if a.status == AlertStatus.ACTIVE]
        summary = {
            'critical': sum(1 for a in active if a.severity == AlertSeverity.CRITICAL),
            'warning': sum(1 for a in active if a.severity == AlertSeverity.WARNING),
            'info': sum(1 for a in active if a.severity == AlertSeverity.INFO)
        }
        
        return AlertResult(
            n_rules=len(self.rules),
            n_active_alerts=len(active),
            alerts=new_alerts,
            summary=summary
        )
    
    def acknowledge(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now().isoformat()
                break
    
    def resolve(self, alert_id: str):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now().isoformat()
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [a for a in self.alerts if a.status == AlertStatus.ACTIVE]
    
    def disable_rule(self, rule_id: str):
        """Disable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    def enable_rule(self, rule_id: str):
        """Enable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def _check_condition(
        self,
        value: float,
        condition: str,
        threshold: float
    ) -> bool:
        """Check if condition is met."""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return value == threshold
        elif condition == '!=':
            return value != threshold
        return False


# ============================================================================
# Factory Functions
# ============================================================================

def get_alert_manager() -> AlertManagerEngine:
    """Get alert manager engine."""
    return AlertManagerEngine()


def quick_data_check(
    df: pd.DataFrame,
    max_missing_pct: float = 20.0
) -> List[str]:
    """Quick data quality check with alerts."""
    manager = AlertManagerEngine(verbose=False)
    
    for col in df.columns:
        manager.add_missing_data_rule(
            f"Missing data: {col}",
            col,
            max_missing_pct
        )
    
    result = manager.check_rules(df)
    return [a.message for a in result.alerts]
