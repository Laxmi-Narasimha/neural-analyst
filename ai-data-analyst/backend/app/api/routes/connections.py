# AI Enterprise Data Analyst - Database Connections API
# REST API for managing external database connections

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.database import get_db_session
from app.core.logging import get_logger
from app.services.data_connectors import (
    ConnectionConfig,
    ConnectorType,
    SQLConnector,
    get_connector,
)

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class ConnectionCreate(BaseModel):
    """Request to create a new connection."""
    name: str = Field(..., min_length=1, max_length=100)
    connector_type: str = Field(..., description="postgresql, mysql, sqlite, mongodb, bigquery, snowflake")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(..., min_length=1)
    username: str = Field(default="")
    password: str = Field(default="")
    ssl: bool = Field(default=False)
    extra: dict[str, Any] = Field(default_factory=dict)


class ConnectionResponse(BaseModel):
    """Connection response."""
    id: str
    name: str
    connector_type: str
    host: str
    port: int
    database: str
    username: str
    ssl: bool
    status: str
    created_at: datetime
    last_tested: Optional[datetime] = None


class ConnectionListResponse(BaseModel):
    """List of connections."""
    items: list[ConnectionResponse]
    total: int


class TestConnectionResponse(BaseModel):
    """Test connection result."""
    success: bool
    message: str
    latency_ms: Optional[float] = None


class QueryRequest(BaseModel):
    """SQL query request."""
    query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    """Query result."""
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float


class ImportRequest(BaseModel):
    """Import table request."""
    table_name: str
    dataset_name: str


# ============================================================================
# In-Memory Storage (Replace with DB in production)
# ============================================================================

_connections: dict[str, dict] = {}


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
async def create_connection(
    request: ConnectionCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new database connection.
    
    Supports: PostgreSQL, MySQL, SQLite, MongoDB, BigQuery, Snowflake
    """
    connection_id = str(uuid4())
    
    # Store connection (encrypt password in production!)
    connection = {
        "id": connection_id,
        "name": request.name,
        "connector_type": request.connector_type,
        "host": request.host,
        "port": request.port,
        "database": request.database,
        "username": request.username,
        "password": request.password,  # Encrypt in production!
        "ssl": request.ssl,
        "extra": request.extra,
        "status": "created",
        "created_at": datetime.utcnow(),
        "last_tested": None,
    }
    
    _connections[connection_id] = connection
    
    logger.info(f"Created connection: {request.name} ({request.connector_type})")
    
    return ConnectionResponse(
        id=connection_id,
        name=request.name,
        connector_type=request.connector_type,
        host=request.host,
        port=request.port,
        database=request.database,
        username=request.username,
        ssl=request.ssl,
        status="created",
        created_at=connection["created_at"],
    )


@router.get("", response_model=ConnectionListResponse)
async def list_connections(
    db: AsyncSession = Depends(get_db_session),
):
    """List all database connections."""
    items = [
        ConnectionResponse(
            id=conn["id"],
            name=conn["name"],
            connector_type=conn["connector_type"],
            host=conn["host"],
            port=conn["port"],
            database=conn["database"],
            username=conn["username"],
            ssl=conn["ssl"],
            status=conn["status"],
            created_at=conn["created_at"],
            last_tested=conn.get("last_tested"),
        )
        for conn in _connections.values()
    ]
    
    return ConnectionListResponse(items=items, total=len(items))


@router.get("/{connection_id}", response_model=ConnectionResponse)
async def get_connection(
    connection_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get connection details."""
    if connection_id not in _connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = _connections[connection_id]
    return ConnectionResponse(
        id=conn["id"],
        name=conn["name"],
        connector_type=conn["connector_type"],
        host=conn["host"],
        port=conn["port"],
        database=conn["database"],
        username=conn["username"],
        ssl=conn["ssl"],
        status=conn["status"],
        created_at=conn["created_at"],
        last_tested=conn.get("last_tested"),
    )


@router.post("/{connection_id}/test", response_model=TestConnectionResponse)
async def test_connection(
    connection_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Test database connection."""
    if connection_id not in _connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = _connections[connection_id]
    start_time = datetime.utcnow()
    
    try:
        # Create connector config
        config = ConnectionConfig(
            connector_type=ConnectorType(conn["connector_type"]),
            host=conn["host"],
            port=conn["port"],
            database=conn["database"],
            username=conn["username"],
            password=conn["password"],
            ssl=conn["ssl"],
        )
        
        # Get connector and test
        connector = get_connector(config)
        connector.connect()
        result = connector.test_connection()
        connector.disconnect()
        
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Update status
        _connections[connection_id]["status"] = "active" if result else "failed"
        _connections[connection_id]["last_tested"] = datetime.utcnow()
        
        return TestConnectionResponse(
            success=result,
            message="Connection successful" if result else "Connection failed",
            latency_ms=elapsed,
        )
        
    except Exception as e:
        _connections[connection_id]["status"] = "failed"
        _connections[connection_id]["last_tested"] = datetime.utcnow()
        
        return TestConnectionResponse(
            success=False,
            message=str(e),
        )


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    connection_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a connection."""
    if connection_id not in _connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    del _connections[connection_id]
    logger.info(f"Deleted connection: {connection_id}")


@router.post("/{connection_id}/query", response_model=QueryResponse)
async def query_database(
    connection_id: str,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Execute SQL query on connection."""
    if connection_id not in _connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = _connections[connection_id]
    
    try:
        config = ConnectionConfig(
            connector_type=ConnectorType(conn["connector_type"]),
            host=conn["host"],
            port=conn["port"],
            database=conn["database"],
            username=conn["username"],
            password=conn["password"],
            ssl=conn["ssl"],
        )
        
        connector = get_connector(config)
        connector.connect()
        
        start = datetime.utcnow()
        result = connector.query(request.query)
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        
        connector.disconnect()
        
        return QueryResponse(
            columns=result.columns,
            rows=result.data.to_dict(orient="records"),
            row_count=result.row_count,
            execution_time_ms=elapsed,
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{connection_id}/import")
async def import_table_as_dataset(
    connection_id: str,
    request: ImportRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Import a table from the database as a new dataset."""
    if connection_id not in _connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = _connections[connection_id]
    
    try:
        config = ConnectionConfig(
            connector_type=ConnectorType(conn["connector_type"]),
            host=conn["host"],
            port=conn["port"],
            database=conn["database"],
            username=conn["username"],
            password=conn["password"],
            ssl=conn["ssl"],
        )
        
        connector = get_connector(config)
        connector.connect()
        
        # Query the table
        result = connector.query(f"SELECT * FROM {request.table_name}")
        connector.disconnect()
        
        # TODO: Save as dataset using DatasetRepository
        # For now, return the data info
        
        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "table_name": request.table_name,
            "row_count": result.row_count,
            "columns": result.columns,
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{connection_id}/tables")
async def list_tables(
    connection_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """List tables in the connected database."""
    if connection_id not in _connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = _connections[connection_id]
    
    try:
        config = ConnectionConfig(
            connector_type=ConnectorType(conn["connector_type"]),
            host=conn["host"],
            port=conn["port"],
            database=conn["database"],
            username=conn["username"],
            password=conn["password"],
            ssl=conn["ssl"],
        )
        
        connector = get_connector(config)
        connector.connect()
        
        if hasattr(connector, 'get_tables'):
            tables = connector.get_tables()
        else:
            tables = []
        
        connector.disconnect()
        
        return {"tables": tables}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
