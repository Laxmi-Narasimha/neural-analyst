# AI Enterprise Data Analyst - Caching Service
# Multi-tier caching with Redis and in-memory cache

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Callable
from functools import wraps
import hashlib
import json
import pickle

from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


# ============================================================================
# Cache Types
# ============================================================================

class CacheTier(str, Enum):
    """Cache tier levels."""
    L1_MEMORY = "memory"
    L2_REDIS = "redis"
    L3_DISK = "disk"


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Single cache entry."""
    
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache statistics."""
    
    hits: int = 0
    misses: int = 0
    size: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": self.size,
            "evictions": self.evictions
        }


# ============================================================================
# In-Memory Cache
# ============================================================================

class LRUCache:
    """In-memory LRU cache."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            self._stats.misses += 1
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired():
            del self._cache[key]
            self._stats.misses += 1
            return None
        
        entry.touch()
        self._stats.hits += 1
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = None
    ) -> None:
        """Set value in cache."""
        # Evict if necessary
        while len(self._cache) >= self.max_size:
            self._evict_lru()
        
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=len(pickle.dumps(value))
        )
        
        self._stats.size = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._stats.size = len(self._cache)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats.size = 0
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        
        del self._cache[lru_key]
        self._stats.evictions += 1
    
    @property
    def stats(self) -> CacheStats:
        return self._stats


# ============================================================================
# Redis Cache
# ============================================================================

class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        prefix: str = "aida:"
    ):
        self.host = host or settings.redis.host
        self.port = port or settings.redis.port
        self.db = db
        self.prefix = prefix
        self._client = None
        self._stats = CacheStats()
    
    def _get_client(self):
        """Get Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=False
                )
            except ImportError:
                logger.warning("redis-py not installed")
                return None
        return self._client
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        client = self._get_client()
        if client is None:
            return None
        
        try:
            data = client.get(f"{self.prefix}{key}")
            if data is None:
                self._stats.misses += 1
                return None
            
            self._stats.hits += 1
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.misses += 1
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600
    ) -> bool:
        """Set value in Redis."""
        client = self._get_client()
        if client is None:
            return False
        
        try:
            data = pickle.dumps(value)
            return client.setex(
                f"{self.prefix}{key}",
                ttl_seconds,
                data
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        client = self._get_client()
        if client is None:
            return False
        
        try:
            return client.delete(f"{self.prefix}{key}") > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear_prefix(self, prefix: str = "") -> int:
        """Clear all keys with prefix."""
        client = self._get_client()
        if client is None:
            return 0
        
        try:
            pattern = f"{self.prefix}{prefix}*"
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0
    
    @property
    def stats(self) -> CacheStats:
        return self._stats


# ============================================================================
# Multi-Tier Cache
# ============================================================================

class MultiTierCache:
    """
    Multi-tier caching system (L1: Memory, L2: Redis).
    
    Features:
    - Two-tier caching
    - Automatic promotion
    - Cache-aside pattern
    - Statistics tracking
    """
    
    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_enabled: bool = True
    ):
        self.l1 = LRUCache(max_size=l1_max_size)
        self.l2 = RedisCache() if l2_enabled else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value (check L1, then L2)."""
        # Check L1
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Check L2
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                # Promote to L1
                self.l1.set(key, value)
                return value
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        l1_only: bool = False
    ) -> None:
        """Set value in cache tiers."""
        self.l1.set(key, value, ttl_seconds)
        
        if self.l2 and not l1_only:
            self.l2.set(key, value, ttl_seconds)
    
    def delete(self, key: str) -> bool:
        """Delete from all tiers."""
        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key) if self.l2 else False
        return l1_deleted or l2_deleted
    
    def stats(self) -> dict[str, Any]:
        """Get combined stats."""
        return {
            "l1": self.l1.stats.to_dict(),
            "l2": self.l2.stats.to_dict() if self.l2 else None
        }


# ============================================================================
# Cache Decorators
# ============================================================================

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = json.dumps({"args": str(args), "kwargs": str(kwargs)}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(
    ttl_seconds: int = 3600,
    prefix: str = "",
    cache: MultiTierCache = None
):
    """Decorator to cache function results."""
    _cache = cache or MultiTierCache()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{prefix}{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            _cache.set(key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def async_cached(
    ttl_seconds: int = 3600,
    prefix: str = "",
    cache: MultiTierCache = None
):
    """Decorator to cache async function results."""
    _cache = cache or MultiTierCache()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{prefix}{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            _cache.set(key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# Cache Service
# ============================================================================

class CacheService:
    """
    Unified caching service.
    
    Features:
    - Multi-tier caching (Memory + Redis)
    - Namespace support
    - TTL management
    - Statistics
    - Memoization decorators
    """
    
    def __init__(self):
        self._caches: dict[str, MultiTierCache] = {}
        self._default = MultiTierCache()
    
    def get_cache(self, namespace: str = "default") -> MultiTierCache:
        """Get or create cache for namespace."""
        if namespace not in self._caches:
            self._caches[namespace] = MultiTierCache()
        return self._caches[namespace]
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache."""
        return self.get_cache(namespace).get(key)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        namespace: str = "default"
    ) -> None:
        """Set value in cache."""
        self.get_cache(namespace).set(key, value, ttl_seconds)
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete key from cache."""
        return self.get_cache(namespace).delete(key)
    
    def stats(self) -> dict[str, Any]:
        """Get stats for all namespaces."""
        return {
            ns: cache.stats()
            for ns, cache in self._caches.items()
        }


# Global cache instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
