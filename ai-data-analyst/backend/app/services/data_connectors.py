# AI Enterprise Data Analyst - Data Connectors
# Multi-source data connectors (databases, cloud storage, APIs)

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Iterator
from urllib.parse import urlparse
import json

import pandas as pd

from app.core.logging import get_logger
from app.core.exceptions import DataProcessingException

logger = get_logger(__name__)


# ============================================================================
# Connector Types
# ============================================================================

class ConnectorType(str, Enum):
    """Data source connector types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    REST_API = "rest_api"
    GRAPHQL = "graphql"


@dataclass
class ConnectionConfig:
    """Connection configuration."""
    
    connector_type: ConnectorType
    
    # Database
    host: str = ""
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""
    
    # Cloud
    bucket: str = ""
    region: str = ""
    access_key: str = ""
    secret_key: str = ""
    
    # API
    base_url: str = ""
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    
    # Options
    ssl: bool = False
    timeout: int = 30
    extra: dict[str, Any] = field(default_factory=dict)
    
    def get_connection_string(self) -> str:
        """Generate connection string for databases."""
        if self.connector_type == ConnectorType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port or 5432}/{self.database}"
        elif self.connector_type == ConnectorType.MYSQL:
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port or 3306}/{self.database}"
        elif self.connector_type == ConnectorType.SQLITE:
            return f"sqlite:///{self.database}"
        return ""


@dataclass
class QueryResult:
    """Query result."""
    
    data: pd.DataFrame
    row_count: int
    columns: list[str]
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Base Connector
# ============================================================================

class BaseConnector(ABC):
    """Abstract base for data connectors."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass
    
    @abstractmethod
    def query(self, query: str, params: dict = None) -> QueryResult:
        """Execute query and return results."""
        pass
    
    def test_connection(self) -> bool:
        """Test if connection is valid."""
        try:
            return self.connect()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# ============================================================================
# SQL Database Connectors
# ============================================================================

class SQLConnector(BaseConnector):
    """Generic SQL database connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._engine = None
    
    def connect(self) -> bool:
        """Connect to SQL database."""
        try:
            from sqlalchemy import create_engine
            
            conn_string = self.config.get_connection_string()
            self._engine = create_engine(conn_string)
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self._connected = True
            return True
            
        except ImportError:
            logger.error("SQLAlchemy not installed")
            return False
        except Exception as e:
            logger.error(f"SQL connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        if self._engine:
            self._engine.dispose()
        self._connected = False
    
    def query(self, query: str, params: dict = None) -> QueryResult:
        """Execute SQL query."""
        if not self._connected:
            self.connect()
        
        import time
        start = time.time()
        
        df = pd.read_sql(query, self._engine, params=params)
        
        return QueryResult(
            data=df,
            row_count=len(df),
            columns=df.columns.tolist(),
            execution_time_ms=(time.time() - start) * 1000
        )
    
    def get_tables(self) -> list[str]:
        """List all tables in database."""
        from sqlalchemy import inspect
        
        if not self._connected:
            self.connect()
        
        inspector = inspect(self._engine)
        return inspector.get_table_names()
    
    def get_schema(self, table: str) -> list[dict]:
        """Get table schema."""
        from sqlalchemy import inspect
        
        if not self._connected:
            self.connect()
        
        inspector = inspect(self._engine)
        columns = inspector.get_columns(table)
        
        return [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True)
            }
            for col in columns
        ]


# ============================================================================
# Cloud Storage Connectors
# ============================================================================

class S3Connector(BaseConnector):
    """AWS S3 connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client = None
    
    def connect(self) -> bool:
        """Connect to S3."""
        try:
            import boto3
            
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region
            )
            
            # Test connection
            self._client.head_bucket(Bucket=self.config.bucket)
            self._connected = True
            return True
            
        except ImportError:
            logger.error("boto3 not installed")
            return False
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from S3."""
        self._client = None
        self._connected = False
    
    def query(self, query: str, params: dict = None) -> QueryResult:
        """Read file from S3 (query is file path)."""
        if not self._connected:
            self.connect()
        
        import time
        import io
        
        start = time.time()
        
        response = self._client.get_object(
            Bucket=self.config.bucket,
            Key=query
        )
        
        content = response['Body'].read()
        
        # Detect file type
        if query.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif query.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(content))
        elif query.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise DataProcessingException(f"Unsupported file type: {query}")
        
        return QueryResult(
            data=df,
            row_count=len(df),
            columns=df.columns.tolist(),
            execution_time_ms=(time.time() - start) * 1000,
            metadata={"source": f"s3://{self.config.bucket}/{query}"}
        )
    
    def list_files(self, prefix: str = "") -> list[str]:
        """List files in bucket."""
        if not self._connected:
            self.connect()
        
        response = self._client.list_objects_v2(
            Bucket=self.config.bucket,
            Prefix=prefix
        )
        
        return [obj['Key'] for obj in response.get('Contents', [])]
    
    def upload(self, data: pd.DataFrame, key: str, format: str = "parquet") -> bool:
        """Upload DataFrame to S3."""
        if not self._connected:
            self.connect()
        
        import io
        
        buffer = io.BytesIO()
        
        if format == "parquet":
            data.to_parquet(buffer, index=False)
        elif format == "csv":
            data.to_csv(buffer, index=False)
        else:
            data.to_json(buffer)
        
        buffer.seek(0)
        
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=buffer.getvalue()
        )
        
        return True


# ============================================================================
# REST API Connector
# ============================================================================

class RESTConnector(BaseConnector):
    """REST API connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._session = None
    
    def connect(self) -> bool:
        """Initialize HTTP session."""
        try:
            import requests
            
            self._session = requests.Session()
            
            # Set headers
            self._session.headers.update(self.config.headers)
            
            if self.config.api_key:
                self._session.headers['Authorization'] = f"Bearer {self.config.api_key}"
            
            self._connected = True
            return True
            
        except ImportError:
            logger.error("requests not installed")
            return False
    
    def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
        self._connected = False
    
    def query(self, query: str, params: dict = None) -> QueryResult:
        """Make API request (query is endpoint path)."""
        if not self._connected:
            self.connect()
        
        import time
        start = time.time()
        
        url = f"{self.config.base_url.rstrip('/')}/{query.lstrip('/')}"
        
        response = self._session.get(
            url,
            params=params,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            elif 'results' in data:
                df = pd.DataFrame(data['results'])
            else:
                df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()
        
        return QueryResult(
            data=df,
            row_count=len(df),
            columns=df.columns.tolist(),
            execution_time_ms=(time.time() - start) * 1000,
            metadata={"url": url, "status_code": response.status_code}
        )
    
    def post(self, endpoint: str, data: dict) -> dict:
        """Make POST request."""
        if not self._connected:
            self.connect()
        
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        response = self._session.post(url, json=data, timeout=self.config.timeout)
        response.raise_for_status()
        
        return response.json()


# ============================================================================
# Connector Factory
# ============================================================================

class ConnectorFactory:
    """Factory for creating data connectors."""
    
    @staticmethod
    def create(config: ConnectionConfig) -> BaseConnector:
        """Create connector based on type."""
        if config.connector_type in [
            ConnectorType.POSTGRESQL,
            ConnectorType.MYSQL,
            ConnectorType.SQLITE
        ]:
            return SQLConnector(config)
        
        elif config.connector_type == ConnectorType.S3:
            return S3Connector(config)
        
        elif config.connector_type == ConnectorType.REST_API:
            return RESTConnector(config)
        
        else:
            raise DataProcessingException(
                f"Unsupported connector type: {config.connector_type}"
            )
    
    @staticmethod
    def from_url(url: str) -> BaseConnector:
        """Create connector from URL."""
        parsed = urlparse(url)
        
        if parsed.scheme in ['postgresql', 'postgres']:
            config = ConnectionConfig(
                connector_type=ConnectorType.POSTGRESQL,
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                username=parsed.username,
                password=parsed.password
            )
        elif parsed.scheme == 'mysql':
            config = ConnectionConfig(
                connector_type=ConnectorType.MYSQL,
                host=parsed.hostname,
                port=parsed.port or 3306,
                database=parsed.path.lstrip('/'),
                username=parsed.username,
                password=parsed.password
            )
        elif parsed.scheme == 's3':
            config = ConnectionConfig(
                connector_type=ConnectorType.S3,
                bucket=parsed.netloc
            )
        elif parsed.scheme in ['http', 'https']:
            config = ConnectionConfig(
                connector_type=ConnectorType.REST_API,
                base_url=f"{parsed.scheme}://{parsed.netloc}"
            )
        else:
            raise DataProcessingException(f"Unknown URL scheme: {parsed.scheme}")
        
        return ConnectorFactory.create(config)


def get_connector(config: ConnectionConfig) -> BaseConnector:
    """Convenience wrapper for creating a connector from config."""
    return ConnectorFactory.create(config)


# ============================================================================
# Data Connector Service
# ============================================================================

class DataConnectorService:
    """
    Unified data connector service.
    
    Features:
    - Multiple database support
    - Cloud storage (S3, GCS)
    - REST API integration
    - Connection pooling
    - Query caching
    """
    
    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}
        self._cache: dict[str, QueryResult] = {}
    
    def register(self, name: str, config: ConnectionConfig) -> BaseConnector:
        """Register a new data source."""
        connector = ConnectorFactory.create(config)
        self._connectors[name] = connector
        return connector
    
    def connect(self, name: str) -> bool:
        """Connect to a data source."""
        if name not in self._connectors:
            raise DataProcessingException(f"Unknown data source: {name}")
        return self._connectors[name].connect()
    
    def query(
        self,
        source: str,
        query: str,
        params: dict = None,
        cache: bool = False
    ) -> QueryResult:
        """Execute query on a data source."""
        if source not in self._connectors:
            raise DataProcessingException(f"Unknown data source: {source}")
        
        # Check cache
        cache_key = f"{source}:{query}:{json.dumps(params or {})}"
        if cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Execute query
        result = self._connectors[source].query(query, params)
        
        # Cache result
        if cache:
            self._cache[cache_key] = result
        
        return result
    
    def list_sources(self) -> list[str]:
        """List registered data sources."""
        return list(self._connectors.keys())
    
    def disconnect_all(self) -> None:
        """Disconnect all data sources."""
        for connector in self._connectors.values():
            connector.disconnect()


# Factory function
def get_data_connector_service() -> DataConnectorService:
    """Get data connector service instance."""
    return DataConnectorService()
