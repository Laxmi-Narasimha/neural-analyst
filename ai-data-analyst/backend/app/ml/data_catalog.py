# AI Enterprise Data Analyst - Data Catalog Engine
# Production-grade data catalog and discovery
# Handles: dataset registry, schema, tags, search

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
# Enums
# ============================================================================

class DatasetType(str, Enum):
    """Types of datasets."""
    TABLE = "table"
    FILE = "file"
    STREAM = "stream"
    VIEW = "view"
    EXTERNAL = "external"


class ColumnDataType(str, Enum):
    """Column data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ColumnSchema:
    """Schema for a column."""
    name: str
    data_type: ColumnDataType
    nullable: bool = True
    description: str = ""
    
    # Statistics
    unique_count: int = 0
    null_count: int = 0
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class DatasetSchema:
    """Schema for a dataset."""
    columns: List[ColumnSchema] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    partition_columns: List[str] = field(default_factory=list)


@dataclass
class DatasetEntry:
    """Entry in the data catalog."""
    dataset_id: str
    name: str
    description: str = ""
    dataset_type: DatasetType = DatasetType.TABLE
    
    # Schema
    schema: DatasetSchema = None
    
    # Location
    location: str = ""
    format: str = ""
    
    # Statistics
    row_count: int = 0
    size_bytes: int = 0
    
    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Ownership
    owner: str = ""
    
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    last_accessed: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
            self.updated_at = self.created_at


@dataclass
class CatalogResult:
    """Data catalog status."""
    n_datasets: int = 0
    n_columns: int = 0
    
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_datasets": self.n_datasets,
            "n_columns": self.n_columns,
            "datasets": self.datasets
        }


# ============================================================================
# Data Catalog Engine
# ============================================================================

class DataCatalogEngine:
    """
    Production-grade Data Catalog engine.
    
    Features:
    - Dataset registration
    - Schema management
    - Tag-based discovery
    - Full-text search
    - Data lineage integration
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.datasets: Dict[str, DatasetEntry] = {}
    
    def register_dataset(
        self,
        name: str,
        df: pd.DataFrame = None,
        description: str = "",
        dataset_type: DatasetType = DatasetType.TABLE,
        tags: List[str] = None,
        owner: str = "",
        location: str = ""
    ) -> str:
        """Register a dataset."""
        dataset_id = self._generate_id(name)
        
        schema = None
        row_count = 0
        
        if df is not None:
            schema = self._infer_schema(df)
            row_count = len(df)
        
        entry = DatasetEntry(
            dataset_id=dataset_id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            schema=schema,
            location=location,
            row_count=row_count,
            tags=tags or [],
            owner=owner
        )
        
        self.datasets[dataset_id] = entry
        
        if self.verbose:
            logger.info(f"Registered dataset: {name}")
        
        return dataset_id
    
    def _infer_schema(self, df: pd.DataFrame) -> DatasetSchema:
        """Infer schema from DataFrame."""
        columns = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                col_type = ColumnDataType.INTEGER
            elif pd.api.types.is_float_dtype(dtype):
                col_type = ColumnDataType.FLOAT
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = ColumnDataType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = ColumnDataType.DATETIME
            else:
                col_type = ColumnDataType.STRING
            
            columns.append(ColumnSchema(
                name=col,
                data_type=col_type,
                nullable=df[col].isna().any(),
                unique_count=int(df[col].nunique()),
                null_count=int(df[col].isna().sum()),
                sample_values=df[col].dropna().head(3).tolist()
            ))
        
        return DatasetSchema(columns=columns)
    
    def update_schema(
        self,
        dataset_id: str,
        df: pd.DataFrame
    ):
        """Update schema from DataFrame."""
        if dataset_id not in self.datasets:
            return
        
        self.datasets[dataset_id].schema = self._infer_schema(df)
        self.datasets[dataset_id].row_count = len(df)
        self.datasets[dataset_id].updated_at = datetime.now().isoformat()
    
    def add_tags(self, dataset_id: str, tags: List[str]):
        """Add tags to a dataset."""
        if dataset_id in self.datasets:
            self.datasets[dataset_id].tags.extend(tags)
            self.datasets[dataset_id].tags = list(set(self.datasets[dataset_id].tags))
    
    def search(
        self,
        query: str = None,
        tags: List[str] = None,
        owner: str = None
    ) -> List[DatasetEntry]:
        """Search datasets."""
        results = []
        
        for entry in self.datasets.values():
            match = True
            
            if query:
                query_lower = query.lower()
                if (query_lower not in entry.name.lower() and 
                    query_lower not in entry.description.lower()):
                    match = False
            
            if tags:
                if not any(t in entry.tags for t in tags):
                    match = False
            
            if owner:
                if entry.owner != owner:
                    match = False
            
            if match:
                results.append(entry)
        
        return results
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetEntry]:
        """Get a dataset by ID."""
        entry = self.datasets.get(dataset_id)
        if entry:
            entry.last_accessed = datetime.now().isoformat()
        return entry
    
    def get_dataset_by_name(self, name: str) -> Optional[DatasetEntry]:
        """Get a dataset by name."""
        for entry in self.datasets.values():
            if entry.name == name:
                entry.last_accessed = datetime.now().isoformat()
                return entry
        return None
    
    def list_datasets(
        self,
        tags: List[str] = None,
        sort_by: str = "name"
    ) -> List[Dict[str, Any]]:
        """List all datasets."""
        datasets = list(self.datasets.values())
        
        if tags:
            datasets = [d for d in datasets if any(t in d.tags for t in tags)]
        
        if sort_by == "name":
            datasets.sort(key=lambda x: x.name)
        elif sort_by == "updated":
            datasets.sort(key=lambda x: x.updated_at, reverse=True)
        elif sort_by == "rows":
            datasets.sort(key=lambda x: x.row_count, reverse=True)
        
        return [
            {
                "id": d.dataset_id,
                "name": d.name,
                "type": d.dataset_type.value,
                "rows": d.row_count,
                "columns": len(d.schema.columns) if d.schema else 0,
                "tags": d.tags
            }
            for d in datasets
        ]
    
    def get_column_info(
        self,
        dataset_id: str,
        column_name: str
    ) -> Optional[ColumnSchema]:
        """Get column information."""
        entry = self.datasets.get(dataset_id)
        if entry and entry.schema:
            for col in entry.schema.columns:
                if col.name == column_name:
                    return col
        return None
    
    def get_status(self) -> CatalogResult:
        """Get catalog status."""
        total_columns = sum(
            len(d.schema.columns) if d.schema else 0 
            for d in self.datasets.values()
        )
        
        return CatalogResult(
            n_datasets=len(self.datasets),
            n_columns=total_columns,
            datasets=self.list_datasets()
        )
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID."""
        ts = datetime.now().isoformat()
        return hashlib.md5(f"{name}_{ts}".encode()).hexdigest()[:12]


# ============================================================================
# Factory Functions
# ============================================================================

def get_data_catalog() -> DataCatalogEngine:
    """Get data catalog engine."""
    return DataCatalogEngine()


def catalog_dataframe(
    df: pd.DataFrame,
    name: str,
    description: str = ""
) -> DataCatalogEngine:
    """Create catalog with a DataFrame."""
    catalog = DataCatalogEngine(verbose=False)
    catalog.register_dataset(name, df, description)
    return catalog
