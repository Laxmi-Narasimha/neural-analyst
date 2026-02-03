# AI Enterprise Data Analyst - Services Package
"""Production-grade services - Complete Module Exports."""

# Core Services
from app.services.database import (
    DatabaseManager,
    db_manager,
    get_db_session,
    init_database,
    close_database,
)

from app.services.base_repository import (
    BaseRepository,
)

from app.services.llm_service import (
    LLMService,
    Message,
    LLMResponse,
    get_llm_service,
)

from app.services.data_ingestion import (
    DataIngestionService,
    DataProfile,
    FileFormat,
    DataType,
    get_ingestion_service as get_data_ingestion_service,
)

# Authentication
from app.services.auth_service import (
    AuthenticationService,
    UserRole,
    Permission,
    TokenPayload,
    AuthUser,
    PasswordHasher,
    JWTService,
    APIKeyService,
    get_auth_service,
)

# Data Connectors
from app.services.data_connectors import (
    DataConnectorService,
    ConnectorType,
    ConnectionConfig,
    QueryResult,
    BaseConnector,
    SQLConnector,
    S3Connector,
    RESTConnector,
    ConnectorFactory,
    get_data_connector_service,
)

# Caching
from app.services.cache_service import (
    CacheService,
    CacheTier,
    CacheStrategy,
    CacheEntry,
    CacheStats,
    LRUCache,
    RedisCache,
    MultiTierCache,
    cached,
    async_cached,
    get_cache_service,
)

# Query Optimization
from app.services.query_optimizer import (
    QueryOptimizerService,
    QueryType,
    JoinType,
    QueryPlan,
    SQLParser,
    QueryOptimizer,
    QueryBuilder,
    get_query_optimizer,
)

# Report Generation
from app.services.report_generator import (
    ReportEngine,
    ReportFormat,
    ReportType,
    Report,
    ReportSection,
    HTMLFormatter,
    MarkdownFormatter,
    JSONFormatter,
    EDAReportGenerator,
    MLModelReportGenerator,
    get_report_engine,
)


__all__ = [
    # Database
    "DatabaseManager", "db_manager", "get_db_session", "init_database", "close_database",
    
    # Repository
    "BaseRepository",
    
    # LLM
    "LLMService", "Message", "LLMResponse", "get_llm_service",
    
    # Data Ingestion
    "DataIngestionService", "DataProfile", "FileFormat", "DataType",
    "get_data_ingestion_service",
    
    # Authentication
    "AuthenticationService", "UserRole", "Permission", "TokenPayload",
    "AuthUser", "PasswordHasher", "JWTService", "APIKeyService",
    "get_auth_service",
    
    # Data Connectors
    "DataConnectorService", "ConnectorType", "ConnectionConfig",
    "QueryResult", "BaseConnector", "SQLConnector", "S3Connector",
    "RESTConnector", "ConnectorFactory", "get_data_connector_service",
    
    # Caching
    "CacheService", "CacheTier", "CacheStrategy", "CacheEntry",
    "CacheStats", "LRUCache", "RedisCache", "MultiTierCache",
    "cached", "async_cached", "get_cache_service",
    
    # Query Optimization
    "QueryOptimizerService", "QueryType", "JoinType", "QueryPlan",
    "SQLParser", "QueryOptimizer", "QueryBuilder", "get_query_optimizer",
    
    # Report Generation
    "ReportEngine", "ReportFormat", "ReportType", "Report", "ReportSection",
    "HTMLFormatter", "MarkdownFormatter", "JSONFormatter",
    "EDAReportGenerator", "MLModelReportGenerator", "get_report_engine",
]
