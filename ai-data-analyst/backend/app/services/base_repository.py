# AI Enterprise Data Analyst - Base Repository Pattern
# Production-grade generic repository with CRUD operations and query building

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, Type, TypeVar
from uuid import UUID

from sqlalchemy import Select, and_, asc, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import DataNotFoundException, DataAlreadyExistsException
from app.core.logging import get_logger, LogContext
from app.models.database import Base

logger = get_logger(__name__)

# Type variables for generic repository
ModelT = TypeVar("ModelT", bound=Base)
CreateSchemaT = TypeVar("CreateSchemaT")
UpdateSchemaT = TypeVar("UpdateSchemaT")


class IRepository(ABC, Generic[ModelT]):
    """
    Abstract repository interface defining standard CRUD operations.
    
    Implements the Repository Pattern for data access abstraction,
    following SOLID principles (Interface Segregation).
    """
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[ModelT]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        **filters: Any
    ) -> Sequence[ModelT]:
        """Get all entities with pagination and filtering."""
        pass
    
    @abstractmethod
    async def create(self, data: dict[str, Any]) -> ModelT:
        """Create new entity."""
        pass
    
    @abstractmethod
    async def update(self, id: UUID, data: dict[str, Any]) -> ModelT:
        """Update existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete entity."""
        pass
    
    @abstractmethod
    async def count(self, **filters: Any) -> int:
        """Count entities matching filters."""
        pass


class BaseRepository(IRepository[ModelT]):
    """
    Generic base repository implementation.
    
    Provides reusable CRUD operations for all database models,
    implementing the Template Method pattern for customization.
    
    Type Parameters:
        ModelT: SQLAlchemy model type
    
    Usage:
        class UserRepository(BaseRepository[User]):
            model = User
    """
    
    model: Type[ModelT]
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self._session = session
        self._model_name = self.model.__name__
    
    @property
    def session(self) -> AsyncSession:
        """Get the database session."""
        return self._session
    
    def _build_base_query(self) -> Select:
        """Build base query with soft delete filter if applicable."""
        query = select(self.model)
        
        # Apply soft delete filter if model supports it
        if hasattr(self.model, "is_deleted"):
            query = query.where(self.model.is_deleted == False)
        
        return query
    
    def _apply_filters(self, query: Select, **filters: Any) -> Select:
        """
        Apply filters to query.
        
        Supports:
            - Exact match: field=value
            - List/IN: field=[val1, val2]
            - Comparison: field__gt=value, field__gte, field__lt, field__lte
            - Like: field__like=pattern, field__ilike (case insensitive)
            - Null check: field__isnull=True/False
        """
        conditions = []
        
        for key, value in filters.items():
            if value is None:
                continue
            
            # Handle special operators
            if "__" in key:
                field_name, operator = key.rsplit("__", 1)
            else:
                field_name, operator = key, "eq"
            
            # Get model attribute
            if not hasattr(self.model, field_name):
                logger.warning(f"Unknown filter field: {field_name}")
                continue
            
            column = getattr(self.model, field_name)
            
            # Apply operator
            if operator == "eq":
                if isinstance(value, list):
                    conditions.append(column.in_(value))
                else:
                    conditions.append(column == value)
            elif operator == "ne":
                conditions.append(column != value)
            elif operator == "gt":
                conditions.append(column > value)
            elif operator == "gte":
                conditions.append(column >= value)
            elif operator == "lt":
                conditions.append(column < value)
            elif operator == "lte":
                conditions.append(column <= value)
            elif operator == "like":
                conditions.append(column.like(f"%{value}%"))
            elif operator == "ilike":
                conditions.append(column.ilike(f"%{value}%"))
            elif operator == "isnull":
                if value:
                    conditions.append(column.is_(None))
                else:
                    conditions.append(column.isnot(None))
            elif operator == "in":
                if isinstance(value, list):
                    conditions.append(column.in_(value))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        return query
    
    def _apply_ordering(
        self,
        query: Select,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> Select:
        """Apply ordering to query."""
        if order_by and hasattr(self.model, order_by):
            column = getattr(self.model, order_by)
            query = query.order_by(desc(column) if order_desc else asc(column))
        elif hasattr(self.model, "created_at"):
            query = query.order_by(desc(self.model.created_at))
        
        return query
    
    def _apply_pagination(self, query: Select, skip: int = 0, limit: int = 100) -> Select:
        """Apply pagination to query."""
        return query.offset(skip).limit(min(limit, 1000))  # Cap at 1000
    
    async def get_by_id(
        self,
        id: UUID,
        load_relations: Optional[list[str]] = None
    ) -> Optional[ModelT]:
        """
        Get entity by ID with optional eager loading.
        
        Args:
            id: Entity UUID
            load_relations: List of relationship names to eager load
        
        Returns:
            Entity if found, None otherwise
        """
        query = self._build_base_query().where(self.model.id == id)
        
        # Apply eager loading
        if load_relations:
            for relation in load_relations:
                if hasattr(self.model, relation):
                    query = query.options(selectinload(getattr(self.model, relation)))
        
        result = await self.session.execute(query)
        return result.scalars().first()
    
    async def get_by_id_or_raise(
        self,
        id: UUID,
        load_relations: Optional[list[str]] = None
    ) -> ModelT:
        """Get entity by ID or raise DataNotFoundException."""
        entity = await self.get_by_id(id, load_relations)
        if entity is None:
            raise DataNotFoundException(self._model_name, id)
        return entity
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        load_relations: Optional[list[str]] = None,
        **filters: Any
    ) -> Sequence[ModelT]:
        """
        Get all entities with pagination, filtering, and ordering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            order_by: Field name to order by
            order_desc: Order descending if True
            load_relations: Relationships to eager load
            **filters: Filter conditions
        
        Returns:
            Sequence of matching entities
        """
        query = self._build_base_query()
        query = self._apply_filters(query, **filters)
        query = self._apply_ordering(query, order_by, order_desc)
        query = self._apply_pagination(query, skip, limit)
        
        # Apply eager loading
        if load_relations:
            for relation in load_relations:
                if hasattr(self.model, relation):
                    query = query.options(selectinload(getattr(self.model, relation)))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_one(self, **filters: Any) -> Optional[ModelT]:
        """Get single entity matching filters."""
        query = self._build_base_query()
        query = self._apply_filters(query, **filters)
        result = await self.session.execute(query)
        return result.scalars().first()
    
    async def exists(self, **filters: Any) -> bool:
        """Check if entity exists matching filters."""
        query = self._build_base_query()
        query = self._apply_filters(query, **filters)
        query = query.limit(1)
        result = await self.session.execute(query)
        return result.scalars().first() is not None
    
    async def create(self, data: dict[str, Any]) -> ModelT:
        """
        Create new entity.
        
        Args:
            data: Dictionary of entity attributes
        
        Returns:
            Created entity
        """
        context = LogContext(component=self._model_name, operation="create")
        
        try:
            entity = self.model(**data)
            self.session.add(entity)
            await self.session.flush()
            await self.session.refresh(entity)
            
            logger.info(f"Created {self._model_name}", context=context, id=str(entity.id))
            return entity
            
        except Exception as e:
            logger.error(f"Failed to create {self._model_name}: {e}", context=context)
            raise
    
    async def create_many(self, data_list: list[dict[str, Any]]) -> list[ModelT]:
        """Create multiple entities in a single transaction."""
        entities = [self.model(**data) for data in data_list]
        self.session.add_all(entities)
        await self.session.flush()
        
        for entity in entities:
            await self.session.refresh(entity)
        
        logger.info(f"Created {len(entities)} {self._model_name} records")
        return entities
    
    async def update(self, id: UUID, data: dict[str, Any]) -> ModelT:
        """
        Update existing entity.
        
        Args:
            id: Entity UUID
            data: Dictionary of attributes to update
        
        Returns:
            Updated entity
        """
        context = LogContext(component=self._model_name, operation="update")
        
        entity = await self.get_by_id_or_raise(id)
        
        # Update attributes
        for key, value in data.items():
            if hasattr(entity, key) and key not in ("id", "created_at"):
                setattr(entity, key, value)
        
        await self.session.flush()
        await self.session.refresh(entity)
        
        logger.info(f"Updated {self._model_name}", context=context, id=str(id))
        return entity
    
    async def delete(self, id: UUID, soft: bool = True) -> bool:
        """
        Delete entity (soft delete by default).
        
        Args:
            id: Entity UUID
            soft: Use soft delete if True and model supports it
        
        Returns:
            True if deleted
        """
        context = LogContext(component=self._model_name, operation="delete")
        
        entity = await self.get_by_id_or_raise(id)
        
        if soft and hasattr(entity, "soft_delete"):
            entity.soft_delete()
            logger.info(f"Soft deleted {self._model_name}", context=context, id=str(id))
        else:
            await self.session.delete(entity)
            logger.info(f"Hard deleted {self._model_name}", context=context, id=str(id))
        
        await self.session.flush()
        return True
    
    async def count(self, **filters: Any) -> int:
        """Count entities matching filters."""
        query = select(func.count()).select_from(self.model)
        
        # Apply soft delete filter
        if hasattr(self.model, "is_deleted"):
            query = query.where(self.model.is_deleted == False)
        
        # Apply other filters
        for key, value in filters.items():
            if value is not None and hasattr(self.model, key):
                column = getattr(self.model, key)
                if isinstance(value, list):
                    query = query.where(column.in_(value))
                else:
                    query = query.where(column == value)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def search(
        self,
        search_term: str,
        search_fields: list[str],
        skip: int = 0,
        limit: int = 100,
        **filters: Any
    ) -> Sequence[ModelT]:
        """
        Search entities across multiple fields.
        
        Args:
            search_term: Term to search for
            search_fields: List of field names to search in
            skip: Pagination offset
            limit: Pagination limit
            **filters: Additional filters
        
        Returns:
            Matching entities
        """
        query = self._build_base_query()
        query = self._apply_filters(query, **filters)
        
        # Build search conditions
        search_conditions = []
        for field in search_fields:
            if hasattr(self.model, field):
                column = getattr(self.model, field)
                search_conditions.append(column.ilike(f"%{search_term}%"))
        
        if search_conditions:
            query = query.where(or_(*search_conditions))
        
        query = self._apply_pagination(query, skip, limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
