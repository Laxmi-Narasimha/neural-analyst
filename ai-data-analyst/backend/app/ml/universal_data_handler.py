# AI Enterprise Data Analyst - Universal Data Handler
# Production-grade automatic data handling for ANY dataset
# Handles ALL edge cases: nulls, outliers, mixed types, encoding, scaling, etc.

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, 
        LabelEncoder, OneHotEncoder, OrdinalEncoder,
        PowerTransformer, QuantileTransformer
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Data Type Detection
# ============================================================================

class SemanticType(str, Enum):
    """Semantic data types for intelligent handling."""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    NUMERIC_RATIO = "numeric_ratio"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    CATEGORICAL_BINARY = "categorical_binary"
    DATETIME = "datetime"
    TEXT_SHORT = "text_short"
    TEXT_LONG = "text_long"
    IDENTIFIER = "identifier"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    COORDINATE_LAT = "coordinate_lat"
    COORDINATE_LON = "coordinate_lon"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass
class ColumnMetadata:
    """Comprehensive metadata for a column."""
    name: str
    original_dtype: str
    semantic_type: SemanticType
    
    # Statistics
    total_count: int = 0
    null_count: int = 0
    unique_count: int = 0
    
    # Quality metrics
    completeness: float = 1.0
    validity: float = 1.0
    
    # Value ranges
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    
    # Distribution info
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    is_normal: bool = False
    
    # Outlier info
    outlier_count: int = 0
    outlier_pct: float = 0.0
    
    # Recommendations
    recommended_imputation: str = "none"
    recommended_scaling: str = "none"
    recommended_encoding: str = "none"
    
    # For categorical
    top_categories: List[Tuple[Any, int]] = field(default_factory=list)
    
    # Flags
    is_constant: bool = False
    is_potential_id: bool = False
    is_datetime_parseable: bool = False
    has_negative: bool = False
    has_zero: bool = False
    requires_log_transform: bool = False


class IntelligentTypeDetector:
    """
    Advanced type detection that goes beyond pandas dtypes.
    Detects semantic meaning and optimal handling strategies.
    """
    
    # Regex patterns for type detection
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'url': r'^https?://[^\s]+$',
        'phone': r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,}$',
        'currency': r'^[$€£¥₹₽]?\s*-?[\d,]+\.?\d*$',
        'percentage': r'^-?\d+\.?\d*\s*%$',
        'date_iso': r'^\d{4}-\d{2}-\d{2}',
        'date_us': r'^\d{1,2}/\d{1,2}/\d{2,4}',
        'date_eu': r'^\d{1,2}\.\d{1,2}\.\d{2,4}',
        'coordinate': r'^-?\d{1,3}\.\d+$',
        'json': r'^\s*[\[{].*[\]}]\s*$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    }
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self._compiled_patterns = {
            k: re.compile(v, re.IGNORECASE) for k, v in self.PATTERNS.items()
        }
    
    def analyze_column(self, series: pd.Series) -> ColumnMetadata:
        """Perform comprehensive column analysis."""
        meta = ColumnMetadata(
            name=str(series.name),
            original_dtype=str(series.dtype),
            semantic_type=SemanticType.UNKNOWN,
            total_count=len(series),
            null_count=int(series.isna().sum()),
            unique_count=int(series.nunique())
        )
        
        # Basic completeness
        meta.completeness = 1 - (meta.null_count / meta.total_count) if meta.total_count > 0 else 0
        
        # Handle empty or all-null columns
        if meta.null_count == meta.total_count:
            meta.semantic_type = SemanticType.UNKNOWN
            meta.recommended_imputation = "drop_column"
            return meta
        
        # Get non-null data for analysis
        non_null = series.dropna()
        
        # Check for constant
        if meta.unique_count <= 1:
            meta.is_constant = True
            meta.recommended_imputation = "drop_column"
            return meta
        
        # Detect semantic type
        meta.semantic_type = self._detect_semantic_type(non_null, meta)
        
        # Add statistics for numeric types
        if meta.semantic_type in [
            SemanticType.NUMERIC_CONTINUOUS,
            SemanticType.NUMERIC_DISCRETE,
            SemanticType.NUMERIC_RATIO,
            SemanticType.CURRENCY,
            SemanticType.PERCENTAGE
        ]:
            self._add_numeric_stats(non_null, meta)
        
        # Add categorical stats
        if meta.semantic_type in [
            SemanticType.CATEGORICAL_NOMINAL,
            SemanticType.CATEGORICAL_ORDINAL,
            SemanticType.CATEGORICAL_BINARY
        ]:
            self._add_categorical_stats(non_null, meta)
        
        # Generate recommendations
        self._generate_recommendations(meta)
        
        return meta
    
    def _detect_semantic_type(self, series: pd.Series, meta: ColumnMetadata) -> SemanticType:
        """Detect the semantic type of the column."""
        sample = series.head(self.sample_size)
        
        # Check pandas dtype first
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME
        
        if pd.api.types.is_bool_dtype(series):
            return SemanticType.CATEGORICAL_BINARY
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            return self._classify_numeric(series, meta)
        
        # String analysis
        if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
            return self._classify_string(sample, meta)
        
        # For category dtype
        if pd.api.types.is_categorical_dtype(series):
            if meta.unique_count == 2:
                return SemanticType.CATEGORICAL_BINARY
            return SemanticType.CATEGORICAL_NOMINAL
        
        return SemanticType.UNKNOWN
    
    def _classify_numeric(self, series: pd.Series, meta: ColumnMetadata) -> SemanticType:
        """Classify numeric column type."""
        non_null = series.dropna()
        
        meta.has_negative = bool((non_null < 0).any())
        meta.has_zero = bool((non_null == 0).any())
        
        # Check for identifier (unique int sequence)
        if pd.api.types.is_integer_dtype(series):
            if meta.unique_count / meta.total_count > 0.95:
                meta.is_potential_id = True
                return SemanticType.IDENTIFIER
        
        # Check for binary
        unique_vals = set(non_null.unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return SemanticType.CATEGORICAL_BINARY
        
        # Check for discrete vs continuous
        if pd.api.types.is_integer_dtype(series):
            if meta.unique_count < 20:
                return SemanticType.NUMERIC_DISCRETE
            return SemanticType.NUMERIC_CONTINUOUS
        
        # Float - check if actually discrete
        if all(float(x).is_integer() for x in non_null.head(100) if pd.notna(x)):
            if meta.unique_count < 20:
                return SemanticType.NUMERIC_DISCRETE
        
        # Check for ratio (always positive)
        if not meta.has_negative and not meta.has_zero:
            return SemanticType.NUMERIC_RATIO
        
        return SemanticType.NUMERIC_CONTINUOUS
    
    def _classify_string(self, sample: pd.Series, meta: ColumnMetadata) -> SemanticType:
        """Classify string column type."""
        str_sample = sample.astype(str)
        
        # Check patterns
        pattern_matches = {}
        for pattern_name, pattern in self._compiled_patterns.items():
            matches = str_sample.apply(lambda x: bool(pattern.match(str(x))) if pd.notna(x) else False)
            match_rate = matches.mean()
            pattern_matches[pattern_name] = match_rate
        
        # Email
        if pattern_matches.get('email', 0) > 0.8:
            return SemanticType.EMAIL
        
        # URL
        if pattern_matches.get('url', 0) > 0.8:
            return SemanticType.URL
        
        # Phone
        if pattern_matches.get('phone', 0) > 0.8:
            return SemanticType.PHONE
        
        # Currency
        if pattern_matches.get('currency', 0) > 0.8:
            return SemanticType.CURRENCY
        
        # Percentage
        if pattern_matches.get('percentage', 0) > 0.8:
            return SemanticType.PERCENTAGE
        
        # Datetime
        date_match = max(
            pattern_matches.get('date_iso', 0),
            pattern_matches.get('date_us', 0),
            pattern_matches.get('date_eu', 0)
        )
        if date_match > 0.8:
            meta.is_datetime_parseable = True
            return SemanticType.DATETIME
        
        # Coordinates
        if pattern_matches.get('coordinate', 0) > 0.9:
            # Check lat vs lon based on value range
            try:
                numeric_vals = pd.to_numeric(sample, errors='coerce').dropna()
                if numeric_vals.between(-90, 90).all():
                    return SemanticType.COORDINATE_LAT
                elif numeric_vals.between(-180, 180).all():
                    return SemanticType.COORDINATE_LON
            except:
                pass
        
        # JSON
        if pattern_matches.get('json', 0) > 0.5:
            return SemanticType.JSON
        
        # UUID/Identifier
        if pattern_matches.get('uuid', 0) > 0.8 or meta.unique_count / meta.total_count > 0.95:
            meta.is_potential_id = True
            return SemanticType.IDENTIFIER
        
        # Text length analysis
        avg_length = str_sample.str.len().mean()
        
        # Binary categorical
        if meta.unique_count == 2:
            return SemanticType.CATEGORICAL_BINARY
        
        # Long text
        if avg_length > 100:
            return SemanticType.TEXT_LONG
        
        # Short text vs categorical
        if avg_length > 50 or meta.unique_count / meta.total_count > 0.5:
            return SemanticType.TEXT_SHORT
        
        return SemanticType.CATEGORICAL_NOMINAL
    
    def _add_numeric_stats(self, series: pd.Series, meta: ColumnMetadata) -> None:
        """Add comprehensive numeric statistics."""
        try:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(series):
                # Try to extract numeric values
                series = pd.to_numeric(
                    series.astype(str).str.replace(r'[^\d.\-]', '', regex=True),
                    errors='coerce'
                ).dropna()
            
            if len(series) == 0:
                return
            
            meta.min_val = float(series.min())
            meta.max_val = float(series.max())
            meta.mean_val = float(series.mean())
            meta.median_val = float(series.median())
            meta.std_val = float(series.std())
            
            # Distribution analysis
            if len(series) >= 20:
                meta.skewness = float(scipy_stats.skew(series))
                meta.kurtosis = float(scipy_stats.kurtosis(series))
                
                # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
                try:
                    if len(series) < 5000:
                        sample = series.sample(min(len(series), 5000))
                        _, p_value = scipy_stats.shapiro(sample)
                    else:
                        _, p_value = scipy_stats.normaltest(series)
                    meta.is_normal = p_value > 0.05
                except:
                    pass
                
                # Check if log transform would help
                if meta.skewness and abs(meta.skewness) > 1:
                    if not meta.has_negative and not meta.has_zero:
                        meta.requires_log_transform = True
            
            # Outlier detection using IQR
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((series < lower) | (series > upper)).sum()
            meta.outlier_count = int(outliers)
            meta.outlier_pct = outliers / len(series) * 100
            
        except Exception as e:
            logger.warning(f"Error computing numeric stats for {meta.name}: {e}")
    
    def _add_categorical_stats(self, series: pd.Series, meta: ColumnMetadata) -> None:
        """Add categorical statistics."""
        try:
            value_counts = series.value_counts().head(20)
            meta.top_categories = list(value_counts.items())
        except Exception as e:
            logger.warning(f"Error computing categorical stats for {meta.name}: {e}")
    
    def _generate_recommendations(self, meta: ColumnMetadata) -> None:
        """Generate processing recommendations based on analysis."""
        
        # Imputation recommendations
        if meta.null_count > 0:
            null_pct = meta.null_count / meta.total_count * 100
            
            if null_pct > 70:
                meta.recommended_imputation = "drop_column"
            elif meta.semantic_type in [
                SemanticType.NUMERIC_CONTINUOUS,
                SemanticType.NUMERIC_RATIO
            ]:
                if meta.is_normal:
                    meta.recommended_imputation = "mean"
                else:
                    meta.recommended_imputation = "median"
            elif meta.semantic_type == SemanticType.NUMERIC_DISCRETE:
                meta.recommended_imputation = "mode"
            elif meta.semantic_type in [
                SemanticType.CATEGORICAL_NOMINAL,
                SemanticType.CATEGORICAL_ORDINAL,
                SemanticType.CATEGORICAL_BINARY
            ]:
                meta.recommended_imputation = "mode"
            elif null_pct < 5:
                meta.recommended_imputation = "knn"
            else:
                meta.recommended_imputation = "iterative"
        
        # Scaling recommendations
        if meta.semantic_type in [
            SemanticType.NUMERIC_CONTINUOUS,
            SemanticType.NUMERIC_RATIO
        ]:
            if meta.outlier_pct > 5:
                meta.recommended_scaling = "robust"
            elif meta.requires_log_transform:
                meta.recommended_scaling = "log_standard"
            elif meta.is_normal:
                meta.recommended_scaling = "standard"
            else:
                meta.recommended_scaling = "quantile"
        elif meta.semantic_type == SemanticType.NUMERIC_DISCRETE:
            meta.recommended_scaling = "standard"
        
        # Encoding recommendations
        if meta.semantic_type == SemanticType.CATEGORICAL_BINARY:
            meta.recommended_encoding = "label"
        elif meta.semantic_type == SemanticType.CATEGORICAL_NOMINAL:
            if meta.unique_count <= 10:
                meta.recommended_encoding = "onehot"
            else:
                meta.recommended_encoding = "target"
        elif meta.semantic_type == SemanticType.CATEGORICAL_ORDINAL:
            meta.recommended_encoding = "ordinal"


# ============================================================================
# Universal Data Preprocessor
# ============================================================================

@dataclass
class PreprocessingResult:
    """Result of data preprocessing."""
    X: np.ndarray
    y: Optional[np.ndarray]
    feature_names: List[str]
    column_metadata: Dict[str, ColumnMetadata]
    transformations_applied: List[str]
    warnings: List[str]
    dropped_columns: List[str]
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]


class UniversalDataPreprocessor:
    """
    Universal data preprocessor that handles ANY data automatically.
    
    Features:
    - Automatic type detection and conversion
    - Smart null handling with optimal imputation strategy
    - Automatic outlier handling
    - Intelligent scaling and encoding
    - Handles edge cases: empty columns, constant columns, mixed types
    - Memory efficient for large datasets
    - Provides detailed transformation log
    """
    
    def __init__(
        self,
        max_categories_onehot: int = 10,
        outlier_handling: str = 'clip',  # 'clip', 'remove', 'cap', 'none'
        null_threshold_drop: float = 0.7,
        constant_threshold: float = 0.999,
        high_cardinality_threshold: float = 0.95,
        date_features: bool = True,
        text_features: bool = True,
        verbose: bool = True
    ):
        self.max_categories_onehot = max_categories_onehot
        self.outlier_handling = outlier_handling
        self.null_threshold_drop = null_threshold_drop
        self.constant_threshold = constant_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.date_features = date_features
        self.text_features = text_features
        self.verbose = verbose
        
        self.type_detector = IntelligentTypeDetector()
        self._metadata: Dict[str, ColumnMetadata] = {}
        self._transformers: Dict[str, Any] = {}
        self._fitted = False
        
        self._warnings: List[str] = []
        self._transformations: List[str] = []
        self._dropped_columns: List[str] = []
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> PreprocessingResult:
        """
        Fit the preprocessor and transform data.
        
        Handles ANY data - nulls, outliers, mixed types, etc.
        """
        original_shape = df.shape
        self._warnings = []
        self._transformations = []
        self._dropped_columns = []
        
        # Validate input
        if df.empty:
            raise DataProcessingException("Empty dataframe provided")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Setup columns to process
        exclude = set(exclude_columns or [])
        if target_column:
            exclude.add(target_column)
        
        feature_columns = [c for c in df.columns if c not in exclude]
        
        # Step 1: Analyze all columns
        if self.verbose:
            logger.info("Step 1: Analyzing column types...")
        
        for col in feature_columns:
            self._metadata[col] = self.type_detector.analyze_column(df[col])
        
        # Step 2: Drop problematic columns
        columns_to_drop = self._identify_columns_to_drop(feature_columns)
        for col in columns_to_drop:
            feature_columns.remove(col)
            self._dropped_columns.append(col)
            self._transformations.append(f"Dropped column: {col}")
        
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Step 3: Parse and convert types
        if self.verbose:
            logger.info("Step 2: Converting data types...")
        
        df = self._convert_types(df, feature_columns)
        
        # Step 4: Handle missing values
        if self.verbose:
            logger.info("Step 3: Handling missing values...")
        
        df = self._handle_missing_values(df, feature_columns)
        
        # Step 5: Handle outliers
        if self.verbose:
            logger.info("Step 4: Handling outliers...")
        
        df = self._handle_outliers(df, feature_columns)
        
        # Step 6: Extract datetime features
        if self.date_features:
            df, feature_columns = self._extract_datetime_features(df, feature_columns)
        
        # Step 7: Encode categoricals
        if self.verbose:
            logger.info("Step 5: Encoding categorical variables...")
        
        df, feature_columns = self._encode_categoricals(df, feature_columns)
        
        # Step 8: Scale numeric features
        if self.verbose:
            logger.info("Step 6: Scaling numeric features...")
        
        df = self._scale_numerics(df, feature_columns)
        
        # Step 9: Final cleanup
        df = self._final_cleanup(df, feature_columns)
        
        # Extract X and y
        X = df[feature_columns].values if feature_columns else np.array([])
        y = None
        
        if target_column and target_column in df.columns:
            y = self._process_target(df[target_column])
        
        self._fitted = True
        
        return PreprocessingResult(
            X=X,
            y=y,
            feature_names=feature_columns,
            column_metadata=self._metadata,
            transformations_applied=self._transformations,
            warnings=self._warnings,
            dropped_columns=self._dropped_columns,
            original_shape=original_shape,
            processed_shape=(X.shape[0], X.shape[1]) if X.size > 0 else (0, 0)
        )
    
    def _identify_columns_to_drop(self, columns: List[str]) -> List[str]:
        """Identify columns that should be dropped."""
        to_drop = []
        
        for col in columns:
            meta = self._metadata.get(col)
            if not meta:
                continue
            
            # Drop if too many nulls
            if meta.completeness < (1 - self.null_threshold_drop):
                to_drop.append(col)
                self._warnings.append(f"Column '{col}' dropped: {100 - meta.completeness*100:.1f}% missing")
                continue
            
            # Drop constant columns
            if meta.is_constant:
                to_drop.append(col)
                self._warnings.append(f"Column '{col}' dropped: constant value")
                continue
            
            # Drop identifier columns (unless explicitly needed)
            if meta.is_potential_id and meta.semantic_type == SemanticType.IDENTIFIER:
                to_drop.append(col)
                self._warnings.append(f"Column '{col}' dropped: potential identifier")
                continue
        
        return to_drop
    
    def _convert_types(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convert columns to appropriate types."""
        for col in columns:
            meta = self._metadata.get(col)
            if not meta:
                continue
            
            try:
                # Parse dates
                if meta.is_datetime_parseable or meta.semantic_type == SemanticType.DATETIME:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                        self._transformations.append(f"Converted '{col}' to datetime")
                
                # Parse numerics from strings (currency, percentage)
                elif meta.semantic_type in [SemanticType.CURRENCY, SemanticType.PERCENTAGE]:
                    # Remove currency symbols and percentage signs
                    df[col] = df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    self._transformations.append(f"Converted '{col}' to numeric")
                
            except Exception as e:
                self._warnings.append(f"Type conversion failed for '{col}': {e}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle missing values with optimal strategy per column."""
        for col in columns:
            if df[col].isna().sum() == 0:
                continue
            
            meta = self._metadata.get(col)
            strategy = meta.recommended_imputation if meta else "median"
            
            try:
                if strategy == "drop_column":
                    continue  # Already handled
                
                elif strategy == "mean":
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    self._transformations.append(f"Imputed '{col}' with mean: {mean_val:.4f}")
                
                elif strategy == "median":
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self._transformations.append(f"Imputed '{col}' with median: {median_val:.4f}")
                
                elif strategy == "mode":
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else df[col].iloc[0]
                    df[col] = df[col].fillna(mode_val)
                    self._transformations.append(f"Imputed '{col}' with mode: {mode_val}")
                
                elif strategy == "knn" and HAS_SKLEARN:
                    # KNN imputation for small missing percentages
                    if pd.api.types.is_numeric_dtype(df[col]):
                        imputer = KNNImputer(n_neighbors=5)
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                        self._transformers[f"{col}_imputer"] = imputer
                        self._transformations.append(f"Imputed '{col}' with KNN")
                    else:
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else '')
                
                elif strategy == "iterative" and HAS_SKLEARN:
                    # Iterative imputation for complex missing patterns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        imputer = IterativeImputer(random_state=42, max_iter=10)
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                        self._transformers[f"{col}_imputer"] = imputer
                        self._transformations.append(f"Imputed '{col}' with iterative imputer")
                    else:
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else '')
                
                else:
                    # Fallback: forward fill then backward fill
                    df[col] = df[col].ffill().bfill()
                    if df[col].isna().any():
                        # Ultimate fallback for still-missing values
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna('unknown')
                    self._transformations.append(f"Imputed '{col}' with ffill/bfill")
                    
            except Exception as e:
                self._warnings.append(f"Imputation failed for '{col}': {e}")
                # Ultimate fallback
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median() if df[col].median() == df[col].median() else 0)
                else:
                    df[col] = df[col].fillna('unknown')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        if self.outlier_handling == 'none':
            return df
        
        for col in columns:
            meta = self._metadata.get(col)
            if not meta or meta.semantic_type not in [
                SemanticType.NUMERIC_CONTINUOUS,
                SemanticType.NUMERIC_RATIO,
                SemanticType.NUMERIC_DISCRETE
            ]:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if meta.outlier_pct < 0.1:  # Skip if negligible outliers
                continue
            
            try:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                outlier_count = outlier_mask.sum()
                
                if outlier_count == 0:
                    continue
                
                if self.outlier_handling == 'clip':
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    self._transformations.append(f"Clipped outliers in '{col}' to [{lower:.4f}, {upper:.4f}]")
                
                elif self.outlier_handling == 'cap':
                    # Cap at 1st and 99th percentiles
                    p1 = df[col].quantile(0.01)
                    p99 = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower=p1, upper=p99)
                    self._transformations.append(f"Capped outliers in '{col}' to [{p1:.4f}, {p99:.4f}]")
                
                elif self.outlier_handling == 'remove':
                    # Replace with NaN (will be imputed)
                    df.loc[outlier_mask, col] = np.nan
                    df[col] = df[col].fillna(df[col].median())
                    self._transformations.append(f"Replaced {outlier_count} outliers in '{col}' with median")
                    
            except Exception as e:
                self._warnings.append(f"Outlier handling failed for '{col}': {e}")
        
        return df
    
    def _extract_datetime_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Extract features from datetime columns."""
        new_columns = []
        cols_to_drop = []
        
        for col in columns[:]:  # Copy to allow modification
            meta = self._metadata.get(col)
            if not meta or meta.semantic_type != SemanticType.DATETIME:
                continue
            
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            try:
                # Extract useful features
                prefix = col
                
                # Year
                df[f'{prefix}_year'] = df[col].dt.year
                new_columns.append(f'{prefix}_year')
                
                # Month
                df[f'{prefix}_month'] = df[col].dt.month
                new_columns.append(f'{prefix}_month')
                
                # Day
                df[f'{prefix}_day'] = df[col].dt.day
                new_columns.append(f'{prefix}_day')
                
                # Day of week
                df[f'{prefix}_dayofweek'] = df[col].dt.dayofweek
                new_columns.append(f'{prefix}_dayofweek')
                
                # Is weekend
                df[f'{prefix}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
                new_columns.append(f'{prefix}_is_weekend')
                
                # Quarter
                df[f'{prefix}_quarter'] = df[col].dt.quarter
                new_columns.append(f'{prefix}_quarter')
                
                # Fill any NaT values with 0
                for new_col in new_columns[-6:]:
                    df[new_col] = df[new_col].fillna(0).astype(int)
                
                cols_to_drop.append(col)
                self._transformations.append(f"Extracted datetime features from '{col}'")
                
            except Exception as e:
                self._warnings.append(f"Datetime extraction failed for '{col}': {e}")
        
        # Update columns list
        for col in cols_to_drop:
            if col in columns:
                columns.remove(col)
        columns.extend(new_columns)
        
        return df, columns
    
    def _encode_categoricals(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables."""
        new_columns = []
        cols_to_remove = []
        
        for col in columns[:]:  # Copy to allow modification
            meta = self._metadata.get(col)
            if not meta:
                continue
            
            if meta.semantic_type not in [
                SemanticType.CATEGORICAL_NOMINAL,
                SemanticType.CATEGORICAL_ORDINAL,
                SemanticType.CATEGORICAL_BINARY,
                SemanticType.TEXT_SHORT
            ]:
                continue
            
            n_unique = df[col].nunique()
            
            try:
                if meta.semantic_type == SemanticType.CATEGORICAL_BINARY or n_unique == 2:
                    # Label encode binary
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._transformers[f"{col}_encoder"] = le
                    self._transformations.append(f"Label encoded '{col}'")
                
                elif n_unique <= self.max_categories_onehot:
                    # One-hot encode low cardinality
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                    df = pd.concat([df, dummies], axis=1)
                    new_columns.extend(dummies.columns.tolist())
                    cols_to_remove.append(col)
                    self._transformations.append(f"One-hot encoded '{col}' ({n_unique} categories)")
                
                else:
                    # Frequency encode high cardinality
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0)
                    self._transformers[f"{col}_freq_map"] = freq_map
                    self._transformations.append(f"Frequency encoded '{col}' ({n_unique} categories)")
                    
            except Exception as e:
                self._warnings.append(f"Encoding failed for '{col}': {e}")
                # Fallback: frequency encoding
                try:
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0)
                except:
                    df[col] = 0
        
        # Update columns
        for col in cols_to_remove:
            if col in columns:
                columns.remove(col)
        columns.extend(new_columns)
        
        return df, columns
    
    def _scale_numerics(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Scale numeric columns with optimal strategy."""
        for col in columns:
            meta = self._metadata.get(col)
            if not meta:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            strategy = meta.recommended_scaling if meta else "standard"
            
            if strategy == "none":
                continue
            
            try:
                values = df[col].values.reshape(-1, 1)
                
                if strategy == "robust":
                    scaler = RobustScaler()
                    df[col] = scaler.fit_transform(values).ravel()
                    self._transformers[f"{col}_scaler"] = scaler
                    self._transformations.append(f"Robust scaled '{col}'")
                
                elif strategy == "log_standard":
                    # Log transform first, then standardize
                    # Shift to ensure positive values
                    min_val = df[col].min()
                    if min_val <= 0:
                        df[col] = df[col] - min_val + 1
                    df[col] = np.log1p(df[col])
                    
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]]).ravel()
                    self._transformers[f"{col}_scaler"] = scaler
                    self._transformations.append(f"Log + standard scaled '{col}'")
                
                elif strategy == "quantile":
                    if HAS_SKLEARN:
                        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
                        df[col] = scaler.fit_transform(values).ravel()
                        self._transformers[f"{col}_scaler"] = scaler
                        self._transformations.append(f"Quantile transformed '{col}'")
                    else:
                        scaler = StandardScaler()
                        df[col] = scaler.fit_transform(values).ravel()
                        self._transformers[f"{col}_scaler"] = scaler
                
                else:  # standard
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(values).ravel()
                    self._transformers[f"{col}_scaler"] = scaler
                    self._transformations.append(f"Standard scaled '{col}'")
                    
            except Exception as e:
                self._warnings.append(f"Scaling failed for '{col}': {e}")
        
        return df
    
    def _final_cleanup(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Final cleanup to ensure no NaN, Inf values."""
        for col in columns:
            if col not in df.columns:
                continue
            
            # Replace infinities
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Final NaN fill
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('unknown')
        
        return df
    
    def _process_target(self, series: pd.Series) -> np.ndarray:
        """Process target variable."""
        # Handle nulls in target
        if series.isna().any():
            self._warnings.append(f"Target column has {series.isna().sum()} null values - using mode for imputation")
            mode_val = series.mode().iloc[0] if len(series.mode()) > 0 else series.dropna().iloc[0]
            series = series.fillna(mode_val)
        
        # Encode if categorical
        if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            le = LabelEncoder()
            series = pd.Series(le.fit_transform(series.astype(str)))
            self._transformers['target_encoder'] = le
            self._transformations.append("Label encoded target variable")
        
        return series.values
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self._fitted:
            raise DataProcessingException("Preprocessor not fitted. Call fit_transform first.")
        
        # Apply same transformations (implementation would go here)
        # For now, return warning
        raise NotImplementedError("Transform method for new data not yet implemented")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing."""
        return {
            "columns_analyzed": len(self._metadata),
            "columns_dropped": self._dropped_columns,
            "transformations_applied": self._transformations,
            "warnings": self._warnings,
            "transformers_fitted": list(self._transformers.keys())
        }


# ============================================================================
# Factory Functions
# ============================================================================

def get_universal_preprocessor(**kwargs) -> UniversalDataPreprocessor:
    """Get a configured universal preprocessor."""
    return UniversalDataPreprocessor(**kwargs)


def analyze_dataset(df: pd.DataFrame) -> Dict[str, ColumnMetadata]:
    """Quick analysis of all columns in a dataframe."""
    detector = IntelligentTypeDetector()
    return {col: detector.analyze_column(df[col]) for col in df.columns}


def preprocess_for_ml(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    **kwargs
) -> PreprocessingResult:
    """
    One-shot preprocessing for machine learning.
    Handles ANY data automatically.
    """
    preprocessor = UniversalDataPreprocessor(**kwargs)
    return preprocessor.fit_transform(df, target_column=target_column)
