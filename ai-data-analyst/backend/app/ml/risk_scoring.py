# AI Enterprise Data Analyst - Risk Scoring Engine
# Production-grade risk assessment and scoring
# Handles: any entity data, multiple risk factors

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

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

class RiskLevel(str, Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RiskFactor:
    """Single risk factor definition."""
    name: str
    weight: float
    direction: str = "higher"  # higher or lower is riskier
    thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class EntityRisk:
    """Risk assessment for single entity."""
    entity_id: Any
    risk_score: float  # 0-100
    risk_level: RiskLevel
    factor_scores: Dict[str, float] = field(default_factory=dict)
    top_risk_factors: List[str] = field(default_factory=list)


@dataclass
class RiskResult:
    """Complete risk scoring result."""
    n_entities: int = 0
    
    # Entity scores
    entities: List[EntityRisk] = field(default_factory=list)
    
    # Distribution
    distribution: Dict[str, int] = field(default_factory=dict)
    
    # Summary
    avg_risk_score: float = 0.0
    high_risk_count: int = 0
    critical_count: int = 0
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_entities": self.n_entities,
                "avg_risk_score": round(self.avg_risk_score, 1),
                "high_risk_count": self.high_risk_count,
                "critical_count": self.critical_count
            },
            "distribution": self.distribution,
            "high_risk_entities": [
                {
                    "id": e.entity_id,
                    "score": round(e.risk_score, 1),
                    "level": e.risk_level.value,
                    "top_factors": e.top_risk_factors[:3]
                }
                for e in self.entities if e.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ][:20]
        }


# ============================================================================
# Risk Scoring Engine
# ============================================================================

class RiskScoringEngine:
    """
    Risk Scoring engine.
    
    Features:
    - Weighted risk factors
    - Automatic normalization
    - Risk level classification
    - Factor contribution analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.factors: List[RiskFactor] = []
    
    def add_factor(
        self,
        name: str,
        weight: float = 1.0,
        direction: str = "higher"
    ):
        """Add a risk factor."""
        self.factors.append(RiskFactor(
            name=name,
            weight=weight,
            direction=direction
        ))
    
    def score(
        self,
        df: pd.DataFrame,
        entity_col: str,
        factor_cols: List[str] = None,
        weights: Dict[str, float] = None
    ) -> RiskResult:
        """Calculate risk scores."""
        start_time = datetime.now()
        
        # Use all numeric columns if not specified
        if factor_cols is None:
            factor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if entity_col in factor_cols:
                factor_cols.remove(entity_col)
        
        if weights is None:
            weights = {col: 1.0 for col in factor_cols}
        
        if self.verbose:
            logger.info(f"Scoring {len(df)} entities with {len(factor_cols)} factors")
        
        # Normalize factors
        normalized = pd.DataFrame()
        normalized[entity_col] = df[entity_col]
        
        for col in factor_cols:
            values = df[col].fillna(df[col].median())
            min_val = values.min()
            max_val = values.max()
            
            if max_val > min_val:
                normalized[col] = (values - min_val) / (max_val - min_val) * 100
            else:
                normalized[col] = 50
        
        # Calculate weighted scores
        entities = []
        total_weight = sum(weights.get(c, 1.0) for c in factor_cols)
        
        for _, row in normalized.iterrows():
            factor_scores = {}
            weighted_sum = 0
            
            for col in factor_cols:
                w = weights.get(col, 1.0)
                score = row[col]
                factor_scores[col] = float(score)
                weighted_sum += score * w
            
            risk_score = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Determine risk level
            risk_level = self._classify_risk(risk_score)
            
            # Top contributing factors
            sorted_factors = sorted(factor_scores.items(), key=lambda x: -x[1])
            top_factors = [f[0] for f in sorted_factors[:3]]
            
            entities.append(EntityRisk(
                entity_id=row[entity_col],
                risk_score=float(risk_score),
                risk_level=risk_level,
                factor_scores=factor_scores,
                top_risk_factors=top_factors
            ))
        
        # Distribution
        distribution = {
            RiskLevel.VERY_LOW.value: sum(1 for e in entities if e.risk_level == RiskLevel.VERY_LOW),
            RiskLevel.LOW.value: sum(1 for e in entities if e.risk_level == RiskLevel.LOW),
            RiskLevel.MEDIUM.value: sum(1 for e in entities if e.risk_level == RiskLevel.MEDIUM),
            RiskLevel.HIGH.value: sum(1 for e in entities if e.risk_level == RiskLevel.HIGH),
            RiskLevel.CRITICAL.value: sum(1 for e in entities if e.risk_level == RiskLevel.CRITICAL)
        }
        
        # Sort by risk score
        entities.sort(key=lambda x: -x.risk_score)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RiskResult(
            n_entities=len(entities),
            entities=entities,
            distribution=distribution,
            avg_risk_score=np.mean([e.risk_score for e in entities]),
            high_risk_count=distribution[RiskLevel.HIGH.value],
            critical_count=distribution[RiskLevel.CRITICAL.value],
            processing_time_sec=processing_time
        )
    
    def _classify_risk(self, score: float) -> RiskLevel:
        """Classify risk level from score."""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MEDIUM
        elif score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW


# ============================================================================
# Factory Functions
# ============================================================================

def get_risk_engine() -> RiskScoringEngine:
    """Get risk scoring engine."""
    return RiskScoringEngine()


def quick_risk_score(
    df: pd.DataFrame,
    entity_col: str
) -> Dict[str, Any]:
    """Quick risk scoring."""
    engine = RiskScoringEngine(verbose=False)
    result = engine.score(df, entity_col)
    return result.to_dict()
