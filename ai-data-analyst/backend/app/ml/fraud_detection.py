# AI Enterprise Data Analyst - Fraud Detection Engine
# Stripe Radar and PayPal-inspired fraud detection patterns

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Callable
import hashlib

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Fraud Detection Types
# ============================================================================

class RiskLevel(str, Enum):
    """Transaction risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudType(str, Enum):
    """Types of fraud."""
    PAYMENT_FRAUD = "payment_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    IDENTITY_THEFT = "identity_theft"
    CHARGEBACK = "chargeback"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    FRIENDLY_FRAUD = "friendly_fraud"


@dataclass
class RiskSignal:
    """Individual risk signal."""
    
    name: str
    score: float  # 0-1
    weight: float = 1.0
    category: str = "general"
    description: str = ""
    
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class FraudScore:
    """Complete fraud risk assessment."""
    
    transaction_id: str
    overall_score: float  # 0-100
    risk_level: RiskLevel
    
    signals: list[RiskSignal] = field(default_factory=list)
    model_scores: dict[str, float] = field(default_factory=dict)
    
    recommended_action: str = "allow"
    review_reason: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "overall_score": round(self.overall_score, 2),
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action,
            "review_reason": self.review_reason,
            "signals": [
                {"name": s.name, "score": round(s.score, 3), "category": s.category}
                for s in self.signals
            ],
            "model_scores": {k: round(v, 4) for k, v in self.model_scores.items()}
        }


# ============================================================================
# Rule Engine
# ============================================================================

@dataclass
class FraudRule:
    """Single fraud detection rule."""
    
    name: str
    condition: Callable[[dict], bool]
    risk_score: float  # Impact on overall score
    category: str = "general"
    description: str = ""
    enabled: bool = True


class RuleEngine:
    """Rule-based fraud detection engine."""
    
    def __init__(self):
        self._rules: list[FraudRule] = []
        self._init_default_rules()
    
    def _init_default_rules(self) -> None:
        """Initialize default fraud rules."""
        self.add_rule(FraudRule(
            name="high_amount",
            condition=lambda t: t.get("amount", 0) > 10000,
            risk_score=25,
            category="amount",
            description="Transaction amount exceeds $10,000"
        ))
        
        self.add_rule(FraudRule(
            name="night_transaction",
            condition=lambda t: self._is_night_transaction(t),
            risk_score=10,
            category="timing",
            description="Transaction during late night hours"
        ))
        
        self.add_rule(FraudRule(
            name="new_device",
            condition=lambda t: t.get("device_age_days", 999) < 1,
            risk_score=20,
            category="device",
            description="Transaction from new device"
        ))
        
        self.add_rule(FraudRule(
            name="velocity_check",
            condition=lambda t: t.get("transactions_last_hour", 0) > 5,
            risk_score=30,
            category="velocity",
            description="High transaction velocity"
        ))
        
        self.add_rule(FraudRule(
            name="geo_mismatch",
            condition=lambda t: t.get("ip_country") != t.get("card_country"),
            risk_score=15,
            category="location",
            description="IP country doesn't match card country"
        ))
        
        self.add_rule(FraudRule(
            name="suspicious_email",
            condition=lambda t: self._is_suspicious_email(t.get("email", "")),
            risk_score=20,
            category="identity",
            description="Email pattern looks suspicious"
        ))
    
    def _is_night_transaction(self, t: dict) -> bool:
        """Check if transaction is during night hours."""
        hour = t.get("hour", 12)
        return hour >= 0 and hour < 5
    
    def _is_suspicious_email(self, email: str) -> bool:
        """Check for suspicious email patterns."""
        if not email:
            return False
        
        suspicious_patterns = [
            email.count('.') > 3,
            email.count('+') > 0,
            len(email.split('@')[0]) > 25,
            any(c.isdigit() for c in email.split('@')[0][-5:])
        ]
        
        return sum(suspicious_patterns) >= 2
    
    def add_rule(self, rule: FraudRule) -> None:
        """Add a fraud detection rule."""
        self._rules.append(rule)
    
    def evaluate(self, transaction: dict) -> list[RiskSignal]:
        """Evaluate all rules against a transaction."""
        signals = []
        
        for rule in self._rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.condition(transaction):
                    signals.append(RiskSignal(
                        name=rule.name,
                        score=rule.risk_score / 100,
                        category=rule.category,
                        description=rule.description
                    ))
            except Exception as e:
                logger.warning(f"Rule {rule.name} failed: {e}")
        
        return signals


# ============================================================================
# Velocity Checker
# ============================================================================

class VelocityChecker:
    """
    Check transaction velocity patterns.
    
    Stripe Radar pattern for detecting rapid transactions.
    """
    
    def __init__(self):
        self._history: dict[str, list[datetime]] = {}
        self._amount_history: dict[str, list[float]] = {}
    
    def check(
        self,
        entity_id: str,
        timestamp: datetime = None,
        amount: float = 0
    ) -> dict[str, Any]:
        """Check velocity for an entity."""
        timestamp = timestamp or datetime.utcnow()
        
        # Get history
        history = self._history.get(entity_id, [])
        amount_hist = self._amount_history.get(entity_id, [])
        
        # Count transactions in windows
        now = timestamp
        last_minute = sum(1 for t in history if now - t < timedelta(minutes=1))
        last_hour = sum(1 for t in history if now - t < timedelta(hours=1))
        last_day = sum(1 for t in history if now - t < timedelta(days=1))
        
        # Amount in windows
        recent_amounts = [a for a, t in zip(amount_hist, history) if now - t < timedelta(hours=1)]
        amount_last_hour = sum(recent_amounts)
        
        # Update history
        history.append(timestamp)
        amount_hist.append(amount)
        
        # Keep only last 7 days
        cutoff = now - timedelta(days=7)
        self._history[entity_id] = [t for t in history if t > cutoff]
        self._amount_history[entity_id] = [
            a for a, t in zip(amount_hist, history) if t > cutoff
        ]
        
        return {
            "transactions_last_minute": last_minute,
            "transactions_last_hour": last_hour,
            "transactions_last_day": last_day,
            "amount_last_hour": amount_last_hour,
            "velocity_score": min(last_hour / 10, 1.0)  # Normalized 0-1
        }


# ============================================================================
# Device Fingerprinting
# ============================================================================

class DeviceFingerprint:
    """Device fingerprinting for fraud detection."""
    
    def __init__(self):
        self._known_devices: dict[str, dict] = {}
    
    def generate_fingerprint(self, device_info: dict) -> str:
        """Generate device fingerprint from available signals."""
        components = [
            device_info.get("user_agent", ""),
            device_info.get("screen_resolution", ""),
            device_info.get("timezone", ""),
            device_info.get("language", ""),
            device_info.get("platform", ""),
            str(device_info.get("plugins", [])),
        ]
        
        fingerprint_str = "|".join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]
    
    def check_device(
        self,
        user_id: str,
        device_info: dict
    ) -> dict[str, Any]:
        """Check if device is known and trusted."""
        fingerprint = self.generate_fingerprint(device_info)
        
        user_devices = self._known_devices.get(user_id, {})
        
        if fingerprint in user_devices:
            device_data = user_devices[fingerprint]
            return {
                "is_known": True,
                "first_seen": device_data["first_seen"],
                "last_seen": device_data["last_seen"],
                "transaction_count": device_data["count"],
                "risk_score": 0.0
            }
        else:
            # New device
            now = datetime.utcnow()
            if user_id not in self._known_devices:
                self._known_devices[user_id] = {}
            
            self._known_devices[user_id][fingerprint] = {
                "first_seen": now,
                "last_seen": now,
                "count": 1
            }
            
            return {
                "is_known": False,
                "first_seen": now,
                "risk_score": 0.3  # New device penalty
            }
    
    def update_device(self, user_id: str, device_info: dict) -> None:
        """Update device history after successful transaction."""
        fingerprint = self.generate_fingerprint(device_info)
        
        if user_id in self._known_devices and fingerprint in self._known_devices[user_id]:
            self._known_devices[user_id][fingerprint]["last_seen"] = datetime.utcnow()
            self._known_devices[user_id][fingerprint]["count"] += 1


# ============================================================================
# ML Fraud Model
# ============================================================================

class MLFraudModel:
    """Machine learning fraud detection model."""
    
    def __init__(self):
        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] = None
    ) -> dict[str, float]:
        """Train fraud detection model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            self._feature_names = feature_names or X.columns.tolist()
            
            # Scale features
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            
            # Train model
            self._model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                random_state=42
            )
            
            # Cross-validation
            scores = cross_val_score(self._model, X_scaled, y, cv=5, scoring='roc_auc')
            
            # Fit on full data
            self._model.fit(X_scaled, y)
            self._fitted = True
            
            return {
                "auc_mean": float(scores.mean()),
                "auc_std": float(scores.std()),
                "feature_importance": dict(zip(
                    self._feature_names,
                    self._model.feature_importances_.tolist()
                ))
            }
            
        except ImportError:
            logger.warning("sklearn not available for ML model")
            return {"error": "sklearn not installed"}
    
    def predict_proba(self, features: dict) -> float:
        """Predict fraud probability."""
        if not self._fitted:
            return 0.5  # Default if not trained
        
        # Create feature vector
        X = np.array([[features.get(f, 0) for f in self._feature_names]])
        X_scaled = self._scaler.transform(X)
        
        proba = self._model.predict_proba(X_scaled)[0][1]
        return float(proba)


# ============================================================================
# Fraud Detection Engine
# ============================================================================

class FraudDetectionEngine:
    """
    Production fraud detection engine.
    
    Inspired by:
    - Stripe Radar (rule + ML hybrid)
    - PayPal risk engine
    - Square fraud detection
    
    Features:
    - Rule-based detection
    - ML model scoring
    - Velocity checks
    - Device fingerprinting
    - Risk scoring integration
    """
    
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.velocity = VelocityChecker()
        self.device_fp = DeviceFingerprint()
        self.ml_model = MLFraudModel()
        
        # Thresholds
        self.low_threshold = 30
        self.medium_threshold = 60
        self.high_threshold = 80
    
    def evaluate(
        self,
        transaction: dict,
        user_id: str = None,
        device_info: dict = None
    ) -> FraudScore:
        """
        Evaluate transaction for fraud risk.
        
        Args:
            transaction: Transaction data
            user_id: User identifier
            device_info: Device fingerprint data
        """
        transaction_id = transaction.get("id", str(hashlib.md5(str(transaction).encode()).hexdigest()[:8]))
        signals: list[RiskSignal] = []
        model_scores: dict[str, float] = {}
        
        # 1. Rule-based signals
        rule_signals = self.rule_engine.evaluate(transaction)
        signals.extend(rule_signals)
        
        # 2. Velocity check
        if user_id:
            velocity = self.velocity.check(
                user_id,
                amount=transaction.get("amount", 0)
            )
            if velocity["velocity_score"] > 0.3:
                signals.append(RiskSignal(
                    name="high_velocity",
                    score=velocity["velocity_score"],
                    category="velocity",
                    description=f"{velocity['transactions_last_hour']} transactions in last hour"
                ))
            model_scores["velocity"] = velocity["velocity_score"]
        
        # 3. Device check
        if device_info and user_id:
            device_risk = self.device_fp.check_device(user_id, device_info)
            if not device_risk["is_known"]:
                signals.append(RiskSignal(
                    name="unknown_device",
                    score=device_risk["risk_score"],
                    category="device",
                    description="Transaction from unknown device"
                ))
            model_scores["device"] = device_risk["risk_score"]
        
        # 4. ML model score
        if self.ml_model._fitted:
            ml_score = self.ml_model.predict_proba(transaction)
            model_scores["ml_model"] = ml_score
            if ml_score > 0.5:
                signals.append(RiskSignal(
                    name="ml_high_risk",
                    score=ml_score,
                    category="ml",
                    description=f"ML model predicts {ml_score:.1%} fraud probability"
                ))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(signals, model_scores)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        # Determine action
        action, reason = self._determine_action(overall_score, signals)
        
        return FraudScore(
            transaction_id=transaction_id,
            overall_score=overall_score,
            risk_level=risk_level,
            signals=signals,
            model_scores=model_scores,
            recommended_action=action,
            review_reason=reason
        )
    
    def _calculate_overall_score(
        self,
        signals: list[RiskSignal],
        model_scores: dict[str, float]
    ) -> float:
        """Calculate overall fraud score (0-100)."""
        # Rule-based component (40% weight)
        rule_score = sum(s.score for s in signals if s.category != "ml") * 100
        rule_component = min(rule_score, 100) * 0.4
        
        # ML component (40% weight)
        ml_score = model_scores.get("ml_model", 0.5)
        ml_component = ml_score * 100 * 0.4
        
        # Velocity component (10% weight)
        velocity_score = model_scores.get("velocity", 0)
        velocity_component = velocity_score * 100 * 0.1
        
        # Device component (10% weight)
        device_score = model_scores.get("device", 0)
        device_component = device_score * 100 * 0.1
        
        return min(rule_component + ml_component + velocity_component + device_component, 100)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score."""
        if score < self.low_threshold:
            return RiskLevel.LOW
        elif score < self.medium_threshold:
            return RiskLevel.MEDIUM
        elif score < self.high_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _determine_action(
        self,
        score: float,
        signals: list[RiskSignal]
    ) -> tuple[str, Optional[str]]:
        """Determine recommended action."""
        if score < self.low_threshold:
            return "allow", None
        elif score < self.medium_threshold:
            return "allow_with_monitoring", "Elevated risk signals detected"
        elif score < self.high_threshold:
            top_signals = sorted(signals, key=lambda s: s.score, reverse=True)[:3]
            reason = ", ".join(s.name for s in top_signals)
            return "manual_review", f"High risk: {reason}"
        else:
            return "block", "Critical risk level exceeded"
    
    def add_rule(self, rule: FraudRule) -> None:
        """Add custom fraud rule."""
        self.rule_engine.add_rule(rule)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train ML fraud model."""
        return self.ml_model.fit(X, y)


# Factory function
def get_fraud_detection_engine() -> FraudDetectionEngine:
    """Get fraud detection engine instance."""
    return FraudDetectionEngine()
