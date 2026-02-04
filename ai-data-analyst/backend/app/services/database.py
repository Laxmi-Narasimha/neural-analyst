# AI Enterprise Data Analyst - Database Session Management
# Production-grade async database connection with pooling and context managers

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy import text
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import Base

logger = get_logger(__name__)


class DatabaseManager:
    """
    Database connection manager implementing the Singleton pattern.
    
    Manages async database connections with proper pooling and
    lifecycle management for enterprise applications.
    """
    
    _instance: DatabaseManager | None = None
    _engine: AsyncEngine | None = None
    _session_factory: async_sessionmaker[AsyncSession] | None = None
    
    def __new__(cls) -> DatabaseManager:
        """Ensure only one instance exists (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def engine(self) -> AsyncEngine:
        """Get or create the async engine."""
        if self._engine is None:
            url = settings.database.async_url

            # Support a lightweight local dev DB without Postgres.
            if url.startswith("sqlite"):
                self._engine = create_async_engine(
                    url,
                    echo=settings.database.echo,
                    future=True,
                    poolclass=NullPool,
                    connect_args={"check_same_thread": False},
                )
            else:
                self._engine = create_async_engine(
                    url,
                    echo=settings.database.echo,
                    pool_size=settings.database.pool_size,
                    max_overflow=settings.database.max_overflow,
                    pool_timeout=settings.database.pool_timeout,
                    pool_recycle=settings.database.pool_recycle,
                    poolclass=AsyncAdaptedQueuePool,
                    # Production optimizations
                    pool_pre_ping=True,  # Validate connections before use
                    future=True,
                )
            logger.info(
                "Database engine created",
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow
            )
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    
    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped")
    
    async def close(self) -> None:
        """Close the database engine and clean up connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database engine closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions.
        
        Implements the Unit of Work pattern with automatic
        commit/rollback handling.
        
        Usage:
            async with db_manager.session() as session:
                # Perform database operations
                session.add(entity)
                await session.commit()
        """
        session = self.session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global instance
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage in routes:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with db_manager.session() as session:
        yield session


# Application lifecycle hooks
async def init_database() -> None:
    """Initialize database on application startup."""
    logger.info("Initializing database connection...")
    
    # Test connection
    db_healthy = await db_manager.health_check()
    if db_healthy:
        logger.info("Database connection established successfully")
    else:
        # In production, fail fast. In development, allow the app to boot so /health works.
        if settings.is_production:
            raise ConnectionError("Failed to connect to database")

        logger.warning(
            "Database not available; continuing startup without DB. "
            "Most API endpoints will fail until DB is configured."
        )
        return
    
    # Create tables if configured (useful for first deploys) or in development
    if settings.is_development or settings.database.auto_create_tables:
        await db_manager.create_tables()


async def close_database() -> None:
    """Close database on application shutdown."""
    await db_manager.close()
