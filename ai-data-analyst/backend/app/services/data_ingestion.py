# AI Enterprise Data Analyst - Data Ingestion Service
# Production-grade data loading with format detection, validation, and profiling

from __future__ import annotations

import hashlib
import io
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, IO, Optional, Type

import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    FileFormatException,
    FileParseException,
    FileUploadException,
    DataQualityException
)
from app.core.logging import get_logger, LogContext, log_execution_time

logger = get_logger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class FileFormat(str, Enum):
    """Supported file formats."""
    CSV = "csv"
    EXCEL_XLS = "xls"
    EXCEL_XLSX = "xlsx"
    JSON = "json"
    PARQUET = "parquet"
    FEATHER = "feather"
    TSV = "tsv"
    XML = "xml"
    SQL = "sql"


class DataType(str, Enum):
    """Inferred column data types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UUID = "uuid"
    JSON = "json"
    BINARY = "binary"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    
    name: str
    original_name: str
    position: int
    inferred_type: DataType
    pandas_dtype: str
    
    # Basic stats
    total_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    
    # Sample values
    sample_values: list[Any] = field(default_factory=list)
    
    # Numeric stats (if applicable)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_value: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    
    # String stats (if applicable)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Flags
    has_outliers: bool = False
    is_constant: bool = False
    is_unique_identifier: bool = False
    is_potential_pii: bool = False
    
    # Distribution
    value_distribution: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "original_name": self.original_name,
            "position": self.position,
            "inferred_type": self.inferred_type.value,
            "pandas_dtype": self.pandas_dtype,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "null_percentage": round(self.null_percentage, 4),
            "unique_count": self.unique_count,
            "unique_percentage": round(self.unique_percentage, 4),
            "sample_values": self.sample_values[:5],
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": round(self.mean_value, 6) if self.mean_value else None,
            "median_value": self.median_value,
            "std_value": round(self.std_value, 6) if self.std_value else None,
            "has_outliers": self.has_outliers,
            "is_constant": self.is_constant,
            "is_unique_identifier": self.is_unique_identifier,
            "is_potential_pii": self.is_potential_pii
        }


@dataclass
class DataProfile:
    """Complete profile for a dataset."""
    
    filename: str
    file_format: FileFormat
    file_size_bytes: int
    file_hash: str
    
    row_count: int = 0
    column_count: int = 0
    memory_size_bytes: int = 0
    
    columns: list[ColumnProfile] = field(default_factory=list)
    
    # Data quality scores (0-1)
    completeness_score: float = 0.0
    uniqueness_score: float = 0.0
    consistency_score: float = 0.0
    overall_quality_score: float = 0.0
    
    # Warnings and issues
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    # Timing
    processing_time_ms: float = 0.0
    profiled_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "file_format": self.file_format.value,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_size_bytes": self.memory_size_bytes,
            "columns": [c.to_dict() for c in self.columns],
            "completeness_score": round(self.completeness_score, 4),
            "uniqueness_score": round(self.uniqueness_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "overall_quality_score": round(self.overall_quality_score, 4),
            "warnings": self.warnings,
            "errors": self.errors,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "profiled_at": self.profiled_at.isoformat()
        }


# ============================================================================
# File Parser Interface and Implementations
# ============================================================================

class BaseFileParser(ABC):
    """Abstract base class for file parsers (Strategy Pattern)."""
    
    supported_formats: list[FileFormat] = []
    
    @abstractmethod
    def parse(
        self,
        file_data: BinaryIO,
        filename: str,
        **options: Any
    ) -> pd.DataFrame:
        """Parse file into DataFrame."""
        pass
    
    @classmethod
    def supports(cls, file_format: FileFormat) -> bool:
        """Check if parser supports the format."""
        return file_format in cls.supported_formats


class CSVParser(BaseFileParser):
    """CSV file parser with intelligent encoding detection."""
    
    supported_formats = [FileFormat.CSV, FileFormat.TSV]
    
    def parse(
        self,
        file_data: BinaryIO,
        filename: str,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        **options: Any
    ) -> pd.DataFrame:
        """Parse CSV/TSV file."""
        # Detect encoding if not specified
        if encoding is None:
            encoding = self._detect_encoding(file_data)
            file_data.seek(0)
        
        # Detect delimiter if not specified
        if delimiter is None:
            if filename.lower().endswith('.tsv'):
                delimiter = '\t'
            else:
                delimiter = self._detect_delimiter(file_data, encoding)
                file_data.seek(0)
        
        try:
            df = pd.read_csv(
                file_data,
                encoding=encoding,
                delimiter=delimiter,
                low_memory=False,
                on_bad_lines='warn',
                **options
            )
            return df
        except Exception as e:
            raise FileParseException(
                filename=filename,
                parse_errors=[str(e)]
            )
    
    def _detect_encoding(self, file_data: BinaryIO) -> str:
        """Detect file encoding."""
        # Read sample
        sample = file_data.read(10000)
        file_data.seek(0)
        
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                sample.decode(encoding)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        return 'utf-8'  # Default
    
    def _detect_delimiter(self, file_data: BinaryIO, encoding: str) -> str:
        """Detect CSV delimiter."""
        sample = file_data.read(5000).decode(encoding)
        file_data.seek(0)
        
        delimiters = [',', ';', '\t', '|']
        counts = {d: sample.count(d) for d in delimiters}
        
        return max(counts, key=counts.get)


class ExcelParser(BaseFileParser):
    """Excel file parser."""
    
    supported_formats = [FileFormat.EXCEL_XLS, FileFormat.EXCEL_XLSX]
    
    def parse(
        self,
        file_data: BinaryIO,
        filename: str,
        sheet_name: Optional[str | int] = 0,
        **options: Any
    ) -> pd.DataFrame:
        """Parse Excel file."""
        try:
            engine = 'openpyxl' if filename.lower().endswith('.xlsx') else 'xlrd'
            df = pd.read_excel(
                file_data,
                sheet_name=sheet_name,
                engine=engine,
                **options
            )
            return df
        except Exception as e:
            raise FileParseException(
                filename=filename,
                parse_errors=[str(e)]
            )


class JSONParser(BaseFileParser):
    """JSON file parser."""
    
    supported_formats = [FileFormat.JSON]
    
    def parse(
        self,
        file_data: BinaryIO,
        filename: str,
        **options: Any
    ) -> pd.DataFrame:
        """Parse JSON file."""
        try:
            df = pd.read_json(file_data, **options)
            return df
        except ValueError:
            # Try reading as JSON lines
            file_data.seek(0)
            try:
                df = pd.read_json(file_data, lines=True, **options)
                return df
            except Exception as e:
                raise FileParseException(
                    filename=filename,
                    parse_errors=[str(e)]
                )


class ParquetParser(BaseFileParser):
    """Parquet file parser."""
    
    supported_formats = [FileFormat.PARQUET]
    
    def parse(
        self,
        file_data: BinaryIO,
        filename: str,
        **options: Any
    ) -> pd.DataFrame:
        """Parse Parquet file."""
        try:
            df = pd.read_parquet(file_data, **options)
            return df
        except Exception as e:
            raise FileParseException(
                filename=filename,
                parse_errors=[str(e)]
            )


# ============================================================================
# Data Type Inference
# ============================================================================

class TypeInferencer:
    """Intelligent data type inference."""
    
    # Regex patterns for semantic types
    import re
    
    PATTERNS = {
        DataType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        DataType.URL: re.compile(r'^https?://[^\s]+$'),
        DataType.PHONE: re.compile(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$'),
        DataType.UUID: re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I),
    }
    
    # PII-related column name patterns
    PII_PATTERNS = [
        'email', 'phone', 'ssn', 'social_security', 'passport',
        'credit_card', 'password', 'address', 'birthdate', 'dob',
        'name', 'first_name', 'last_name', 'full_name'
    ]
    
    @classmethod
    def infer_column_type(cls, series: pd.Series) -> DataType:
        """Infer semantic data type for a column."""
        # Get pandas dtype
        dtype = series.dtype
        
        # Handle numeric types
        if pd.api.types.is_integer_dtype(dtype):
            return DataType.INTEGER
        
        if pd.api.types.is_float_dtype(dtype):
            return DataType.FLOAT
        
        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN
        
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return DataType.DATETIME
        
        # Handle string/object types - need semantic analysis
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            return cls._infer_string_type(series)
        
        return DataType.UNKNOWN
    
    @classmethod
    def _infer_string_type(cls, series: pd.Series) -> DataType:
        """Infer type for string/object columns."""
        # Get non-null sample
        sample = series.dropna()
        if len(sample) == 0:
            return DataType.STRING
        
        # Convert to string
        sample = sample.astype(str)
        
        # Check sample size (max 1000 for performance)
        if len(sample) > 1000:
            sample = sample.sample(1000, random_state=42)
        
        # Try datetime parsing
        try:
            pd.to_datetime(sample, errors='raise')
            return DataType.DATETIME
        except:
            pass
        
        # Check semantic patterns
        for dtype, pattern in cls.PATTERNS.items():
            matches = sample.str.match(pattern).sum()
            if matches / len(sample) > 0.8:  # 80% threshold
                return dtype
        
        # Check if categorical (low cardinality)
        unique_ratio = sample.nunique() / len(sample)
        if unique_ratio < 0.05 and sample.nunique() < 100:
            return DataType.CATEGORICAL
        
        # Check if text (long strings)
        avg_length = sample.str.len().mean()
        if avg_length > 100:
            return DataType.TEXT
        
        return DataType.STRING
    
    @classmethod
    def is_potential_pii(cls, column_name: str, series: pd.Series) -> bool:
        """Check if column might contain PII."""
        # Check column name
        name_lower = column_name.lower()
        for pattern in cls.PII_PATTERNS:
            if pattern in name_lower:
                return True
        
        # Check data patterns
        dtype = cls.infer_column_type(series)
        return dtype in [DataType.EMAIL, DataType.PHONE, DataType.UUID]


# ============================================================================
# Main Data Ingestion Service
# ============================================================================

class DataIngestionService:
    """
    Production-grade data ingestion service.
    
    Handles:
    - Multi-format file parsing
    - Automatic encoding detection
    - Schema inference
    - Data profiling
    - Quality assessment
    """
    
    # Parser registry (Factory Pattern)
    PARSERS: list[Type[BaseFileParser]] = [
        CSVParser,
        ExcelParser,
        JSONParser,
        ParquetParser,
    ]
    
    def __init__(self) -> None:
        self._parser_instances: dict[FileFormat, BaseFileParser] = {}
    
    def get_parser(self, file_format: FileFormat) -> BaseFileParser:
        """Get parser for file format."""
        # Check cache
        if file_format in self._parser_instances:
            return self._parser_instances[file_format]
        
        # Find suitable parser
        for parser_class in self.PARSERS:
            if parser_class.supports(file_format):
                parser = parser_class()
                self._parser_instances[file_format] = parser
                return parser
        
        raise FileFormatException(
            filename="",
            actual_format=file_format.value,
            supported_formats=[f.value for f in FileFormat]
        )
    
    def detect_format(self, filename: str, file_data: Optional[BinaryIO] = None) -> FileFormat:
        """Detect file format from filename and content."""
        # Extension-based detection
        ext = Path(filename).suffix.lower().lstrip('.')
        
        format_map = {
            'csv': FileFormat.CSV,
            'tsv': FileFormat.TSV,
            'xls': FileFormat.EXCEL_XLS,
            'xlsx': FileFormat.EXCEL_XLSX,
            'json': FileFormat.JSON,
            'parquet': FileFormat.PARQUET,
            'feather': FileFormat.FEATHER,
        }
        
        if ext in format_map:
            return format_map[ext]
        
        raise FileFormatException(
            filename=filename,
            actual_format=ext,
            supported_formats=list(format_map.keys())
        )
    
    @log_execution_time(operation_name="ingest_file")
    def ingest_file(
        self,
        file_data: BinaryIO,
        filename: str,
        file_format: Optional[FileFormat] = None,
        **parse_options: Any
    ) -> tuple[pd.DataFrame, DataProfile]:
        """
        Ingest file and generate profile.
        
        Args:
            file_data: File-like object with binary data
            filename: Original filename
            file_format: File format (auto-detected if None)
            **parse_options: Additional parsing options
        
        Returns:
            Tuple of (DataFrame, DataProfile)
        """
        context = LogContext(
            component="DataIngestionService",
            operation="ingest_file"
        )
        
        start_time = datetime.utcnow()
        
        # Detect format
        if file_format is None:
            file_format = self.detect_format(filename)
        
        logger.info(
            f"Ingesting file: {filename}",
            context=context,
            format=file_format.value
        )
        
        # Calculate file hash and size without reading the entire file into memory.
        file_data.seek(0)
        hasher = hashlib.sha256()
        file_size = 0
        while True:
            chunk = file_data.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            file_size += len(chunk)
        file_hash = hasher.hexdigest()
        file_data.seek(0)
        
        # Parse file
        parser = self.get_parser(file_format)
        df = parser.parse(file_data, filename, **parse_options)
        
        # Generate profile
        profile = self._generate_profile(
            df=df,
            filename=filename,
            file_format=file_format,
            file_size=file_size,
            file_hash=file_hash
        )
        
        profile.processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(
            f"Ingestion complete: {df.shape[0]} rows, {df.shape[1]} columns",
            context=context,
            rows=df.shape[0],
            columns=df.shape[1],
            quality_score=profile.overall_quality_score
        )
        
        return df, profile

    def parse_file(
        self,
        file_data: IO[bytes],
        filename: str,
        file_format: Optional[FileFormat] = None,
        **parse_options: Any,
    ) -> pd.DataFrame:
        """
        Parse a dataset file into a DataFrame without profiling.

        This is used by compute paths that only need the DataFrame and want to
        avoid recomputing profiles on every load.
        """
        if file_format is None:
            file_format = self.detect_format(filename)
        parser = self.get_parser(file_format)
        return parser.parse(file_data, filename, **parse_options)
    
    def _generate_profile(
        self,
        df: pd.DataFrame,
        filename: str,
        file_format: FileFormat,
        file_size: int,
        file_hash: str
    ) -> DataProfile:
        """Generate comprehensive data profile."""
        profile = DataProfile(
            filename=filename,
            file_format=file_format,
            file_size_bytes=file_size,
            file_hash=file_hash,
            row_count=len(df),
            column_count=len(df.columns),
            memory_size_bytes=df.memory_usage(deep=True).sum()
        )
        
        # Profile each column
        for i, col in enumerate(df.columns):
            col_profile = self._profile_column(df[col], col, i)
            profile.columns.append(col_profile)
        
        # Calculate overall quality scores
        profile.completeness_score = self._calculate_completeness(profile)
        profile.uniqueness_score = self._calculate_uniqueness(profile, df)
        profile.consistency_score = self._calculate_consistency(profile)
        profile.overall_quality_score = (
            profile.completeness_score * 0.4 +
            profile.uniqueness_score * 0.3 +
            profile.consistency_score * 0.3
        )
        
        # Generate warnings
        profile.warnings = self._generate_warnings(profile)
        
        return profile
    
    def _profile_column(
        self,
        series: pd.Series,
        name: str,
        position: int
    ) -> ColumnProfile:
        """Profile a single column."""
        total = len(series)
        null_count = series.isna().sum()
        non_null = series.dropna()
        unique_count = series.nunique()
        
        profile = ColumnProfile(
            name=str(name),
            original_name=str(name),
            position=position,
            inferred_type=TypeInferencer.infer_column_type(series),
            pandas_dtype=str(series.dtype),
            total_count=total,
            null_count=int(null_count),
            null_percentage=null_count / total if total > 0 else 0.0,
            unique_count=unique_count,
            unique_percentage=unique_count / total if total > 0 else 0.0
        )
        
        # Sample values
        if len(non_null) > 0:
            profile.sample_values = non_null.head(5).tolist()
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            try:
                profile.min_value = float(series.min())
                profile.max_value = float(series.max())
                profile.mean_value = float(series.mean())
                profile.median_value = float(series.median())
                profile.std_value = float(series.std())
                profile.q25 = float(series.quantile(0.25))
                profile.q75 = float(series.quantile(0.75))
                
                # Outlier detection (IQR method)
                iqr = profile.q75 - profile.q25
                lower_bound = profile.q25 - 1.5 * iqr
                upper_bound = profile.q75 + 1.5 * iqr
                outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                profile.has_outliers = outliers > 0.01 * total
            except:
                pass
        
        # String statistics
        if profile.inferred_type in [DataType.STRING, DataType.TEXT, DataType.CATEGORICAL]:
            try:
                lengths = non_null.astype(str).str.len()
                profile.min_length = int(lengths.min()) if len(lengths) > 0 else None
                profile.max_length = int(lengths.max()) if len(lengths) > 0 else None
                profile.avg_length = float(lengths.mean()) if len(lengths) > 0 else None
            except:
                pass
        
        # Value distribution (for categorical)
        if unique_count <= 20:
            value_counts = series.value_counts().head(20)
            profile.value_distribution = value_counts.to_dict()
        
        # Flags
        profile.is_constant = unique_count <= 1
        profile.is_unique_identifier = unique_count == total and null_count == 0
        profile.is_potential_pii = TypeInferencer.is_potential_pii(name, series)
        
        return profile
    
    def _calculate_completeness(self, profile: DataProfile) -> float:
        """Calculate data completeness score."""
        if profile.row_count == 0:
            return 0.0
        
        total_cells = profile.row_count * profile.column_count
        null_cells = sum(c.null_count for c in profile.columns)
        
        return 1.0 - (null_cells / total_cells) if total_cells > 0 else 0.0
    
    def _calculate_uniqueness(self, profile: DataProfile, df: pd.DataFrame) -> float:
        """Calculate data uniqueness score."""
        if profile.row_count == 0:
            return 0.0
        
        duplicate_rows = df.duplicated().sum()
        return 1.0 - (duplicate_rows / profile.row_count)
    
    def _calculate_consistency(self, profile: DataProfile) -> float:
        """Calculate data consistency score."""
        if not profile.columns:
            return 0.0
        
        # Check for consistent types within columns
        type_consistent = sum(
            1 for c in profile.columns
            if c.inferred_type != DataType.UNKNOWN
        )
        
        return type_consistent / len(profile.columns)
    
    def _generate_warnings(self, profile: DataProfile) -> list[str]:
        """Generate warnings based on profile."""
        warnings = []
        
        # High null percentage warnings
        for col in profile.columns:
            if col.null_percentage > 0.5:
                warnings.append(
                    f"Column '{col.name}' has {col.null_percentage:.1%} missing values"
                )
        
        # Constant column warnings
        constant_cols = [c.name for c in profile.columns if c.is_constant]
        if constant_cols:
            warnings.append(
                f"Constant columns (consider removing): {', '.join(constant_cols)}"
            )
        
        # PII warnings
        pii_cols = [c.name for c in profile.columns if c.is_potential_pii]
        if pii_cols:
            warnings.append(
                f"Potential PII detected in columns: {', '.join(pii_cols)}"
            )
        
        # Low quality warning
        if profile.overall_quality_score < 0.5:
            warnings.append(
                f"Overall data quality is low ({profile.overall_quality_score:.1%})"
            )
        
        return warnings


# Factory function
def get_ingestion_service() -> DataIngestionService:
    """Get data ingestion service instance."""
    return DataIngestionService()
