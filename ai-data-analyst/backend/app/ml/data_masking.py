# AI Enterprise Data Analyst - Data Masking Engine
# Production-grade PII detection and data masking
# Handles: email, phone, SSN, credit cards, names, addresses

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern

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

class PIIType(str, Enum):
    """Types of PII."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    CUSTOM = "custom"


class MaskingStrategy(str, Enum):
    """Masking strategies."""
    REDACT = "redact"  # Replace with fixed string
    HASH = "hash"  # SHA256 hash
    PARTIAL = "partial"  # Show first/last chars
    TOKENIZE = "tokenize"  # Replace with token
    GENERALIZE = "generalize"  # Generalize value
    NULL = "null"  # Replace with null


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PIIDetection:
    """Detected PII in a column."""
    column: str
    pii_type: PIIType
    n_detected: int
    detection_rate: float
    sample_matches: List[str]
    confidence: float


@dataclass
class MaskingResult:
    """Complete masking result."""
    n_columns_masked: int = 0
    n_values_masked: int = 0
    
    # PII detected
    detections: List[PIIDetection] = field(default_factory=list)
    
    # Masked DataFrame
    masked_df: pd.DataFrame = None
    
    # Mapping (for reversible masks)
    token_mapping: Dict[str, str] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "columns_masked": self.n_columns_masked,
                "values_masked": self.n_values_masked
            },
            "detections": [
                {
                    "column": d.column,
                    "pii_type": d.pii_type.value,
                    "n_detected": d.n_detected,
                    "detection_rate": round(d.detection_rate, 2),
                    "confidence": round(d.confidence, 2)
                }
                for d in self.detections
            ]
        }


# ============================================================================
# PII Patterns
# ============================================================================

class PIIPatterns:
    """Regex patterns for PII detection."""
    
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            re.IGNORECASE
        ),
        PIIType.PHONE: re.compile(
            r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$'
        ),
        PIIType.SSN: re.compile(
            r'^\d{3}-\d{2}-\d{4}$'
        ),
        PIIType.CREDIT_CARD: re.compile(
            r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'
        ),
        PIIType.IP_ADDRESS: re.compile(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        )
    }
    
    # Name indicators
    NAME_PREFIXES = {'mr', 'mrs', 'ms', 'dr', 'prof'}
    
    # Common name patterns (simplified)
    NAME_PATTERN = re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+')


# ============================================================================
# Data Masking Engine
# ============================================================================

class DataMaskingEngine:
    """
    Production-grade Data Masking engine.
    
    Features:
    - Automatic PII detection
    - Multiple masking strategies
    - Reversible tokenization
    - Custom pattern support
    - Partial masking
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.patterns = PIIPatterns()
        self._token_counter = 0
    
    def detect_pii(
        self,
        df: pd.DataFrame,
        sample_size: int = 1000
    ) -> List[PIIDetection]:
        """Detect PII in DataFrame."""
        detections = []
        
        for col in df.columns:
            if not df[col].dtype == 'object':
                continue
            
            # Sample for efficiency
            sample = df[col].dropna().astype(str).head(sample_size)
            
            for pii_type, pattern in self.patterns.PATTERNS.items():
                matches = sample.apply(lambda x: bool(pattern.match(str(x))))
                n_matches = matches.sum()
                
                if n_matches > 0:
                    detection_rate = n_matches / len(sample) * 100
                    
                    # Confidence based on detection rate
                    confidence = min(1.0, detection_rate / 50)
                    
                    detections.append(PIIDetection(
                        column=col,
                        pii_type=pii_type,
                        n_detected=int(n_matches),
                        detection_rate=detection_rate,
                        sample_matches=sample[matches].head(3).tolist(),
                        confidence=confidence
                    ))
            
            # Check for name pattern
            if self._looks_like_name_column(col, sample):
                detections.append(PIIDetection(
                    column=col,
                    pii_type=PIIType.NAME,
                    n_detected=len(sample),
                    detection_rate=100,
                    sample_matches=sample.head(3).tolist(),
                    confidence=0.7
                ))
        
        return detections
    
    def mask(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        strategy: MaskingStrategy = MaskingStrategy.PARTIAL,
        pii_types: List[PIIType] = None
    ) -> MaskingResult:
        """Mask PII in DataFrame."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Masking PII with strategy: {strategy.value}")
        
        # Detect PII if columns not specified
        if columns is None:
            detections = self.detect_pii(df)
            columns = list(set(d.column for d in detections))
        else:
            detections = [d for d in self.detect_pii(df) if d.column in columns]
        
        # Filter by PII type if specified
        if pii_types:
            detections = [d for d in detections if d.pii_type in pii_types]
            columns = list(set(d.column for d in detections))
        
        masked_df = df.copy()
        n_values_masked = 0
        token_mapping = {}
        
        for col in columns:
            if col not in masked_df.columns:
                continue
            
            # Get PII type for this column
            col_detections = [d for d in detections if d.column == col]
            pii_type = col_detections[0].pii_type if col_detections else PIIType.CUSTOM
            
            original = masked_df[col].copy()
            
            if strategy == MaskingStrategy.REDACT:
                masked_df[col] = self._redact(original, pii_type)
            elif strategy == MaskingStrategy.HASH:
                masked_df[col] = self._hash_values(original)
            elif strategy == MaskingStrategy.PARTIAL:
                masked_df[col] = self._partial_mask(original, pii_type)
            elif strategy == MaskingStrategy.TOKENIZE:
                masked_df[col], mapping = self._tokenize(original)
                token_mapping.update(mapping)
            elif strategy == MaskingStrategy.GENERALIZE:
                masked_df[col] = self._generalize(original, pii_type)
            elif strategy == MaskingStrategy.NULL:
                masked_df[col] = None
            
            n_values_masked += original.notna().sum()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MaskingResult(
            n_columns_masked=len(columns),
            n_values_masked=n_values_masked,
            detections=detections,
            masked_df=masked_df,
            token_mapping=token_mapping,
            processing_time_sec=processing_time
        )
    
    def _redact(self, series: pd.Series, pii_type: PIIType) -> pd.Series:
        """Replace with redaction string."""
        redaction_strings = {
            PIIType.EMAIL: "[EMAIL REDACTED]",
            PIIType.PHONE: "[PHONE REDACTED]",
            PIIType.SSN: "[SSN REDACTED]",
            PIIType.CREDIT_CARD: "[CC REDACTED]",
            PIIType.IP_ADDRESS: "[IP REDACTED]",
            PIIType.NAME: "[NAME REDACTED]",
            PIIType.ADDRESS: "[ADDRESS REDACTED]",
            PIIType.CUSTOM: "[REDACTED]"
        }
        
        redact_str = redaction_strings.get(pii_type, "[REDACTED]")
        return series.fillna('').apply(lambda x: redact_str if x else None)
    
    def _hash_values(self, series: pd.Series) -> pd.Series:
        """Hash values with SHA256."""
        def hash_val(x):
            if pd.isna(x) or x == '':
                return None
            return hashlib.sha256(str(x).encode()).hexdigest()[:16]
        
        return series.apply(hash_val)
    
    def _partial_mask(self, series: pd.Series, pii_type: PIIType) -> pd.Series:
        """Partially mask values."""
        def mask_value(x):
            if pd.isna(x) or str(x) == '':
                return None
            
            s = str(x)
            
            if pii_type == PIIType.EMAIL:
                parts = s.split('@')
                if len(parts) == 2:
                    local = parts[0]
                    if len(local) > 2:
                        return local[0] + '*' * (len(local) - 2) + local[-1] + '@' + parts[1]
                    return '*' * len(local) + '@' + parts[1]
            
            elif pii_type == PIIType.PHONE:
                digits = re.sub(r'\D', '', s)
                if len(digits) >= 4:
                    return '*' * (len(digits) - 4) + digits[-4:]
            
            elif pii_type == PIIType.SSN:
                return '***-**-' + s[-4:] if len(s) >= 4 else '***-**-****'
            
            elif pii_type == PIIType.CREDIT_CARD:
                digits = re.sub(r'\D', '', s)
                if len(digits) >= 4:
                    return '**** **** **** ' + digits[-4:]
            
            elif pii_type == PIIType.NAME:
                parts = s.split()
                if len(parts) >= 2:
                    return parts[0][0] + '.' + ' ' + parts[-1][0] + '.'
                return s[0] + '*' * (len(s) - 1)
            
            # Default: show first and last char
            if len(s) > 2:
                return s[0] + '*' * (len(s) - 2) + s[-1]
            return '*' * len(s)
        
        return series.apply(mask_value)
    
    def _tokenize(self, series: pd.Series) -> tuple:
        """Replace with reversible tokens."""
        mapping = {}
        
        def tokenize_val(x):
            if pd.isna(x) or str(x) == '':
                return None
            
            s = str(x)
            if s not in mapping:
                self._token_counter += 1
                token = f"TOKEN_{self._token_counter:06d}"
                mapping[token] = s
            else:
                token = [k for k, v in mapping.items() if v == s][0]
            
            return token
        
        tokenized = series.apply(tokenize_val)
        return tokenized, mapping
    
    def _generalize(self, series: pd.Series, pii_type: PIIType) -> pd.Series:
        """Generalize values."""
        def generalize_val(x):
            if pd.isna(x) or str(x) == '':
                return None
            
            s = str(x)
            
            if pii_type == PIIType.EMAIL:
                parts = s.split('@')
                if len(parts) == 2:
                    return '*@' + parts[1].split('.')[-1]
            
            elif pii_type == PIIType.IP_ADDRESS:
                parts = s.split('.')
                if len(parts) == 4:
                    return parts[0] + '.xxx.xxx.xxx'
            
            return '[GENERALIZED]'
        
        return series.apply(generalize_val)
    
    def _looks_like_name_column(self, col: str, sample: pd.Series) -> bool:
        """Check if column looks like names."""
        col_lower = col.lower()
        name_indicators = ['name', 'customer', 'user', 'person', 'contact']
        
        if any(ind in col_lower for ind in name_indicators):
            # Check values
            name_pattern_matches = sample.apply(
                lambda x: bool(self.patterns.NAME_PATTERN.match(str(x)))
            ).mean()
            
            return name_pattern_matches > 0.5
        
        return False


# ============================================================================
# Factory Functions
# ============================================================================

def get_masking_engine() -> DataMaskingEngine:
    """Get data masking engine."""
    return DataMaskingEngine()


def detect_pii(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Quick PII detection."""
    engine = DataMaskingEngine(verbose=False)
    detections = engine.detect_pii(df)
    return [
        {
            "column": d.column,
            "pii_type": d.pii_type.value,
            "n_detected": d.n_detected,
            "confidence": round(d.confidence, 2)
        }
        for d in detections
    ]


def mask_pii(
    df: pd.DataFrame,
    strategy: str = "partial"
) -> pd.DataFrame:
    """Quick PII masking."""
    engine = DataMaskingEngine(verbose=False)
    result = engine.mask(df, strategy=MaskingStrategy(strategy))
    return result.masked_df
