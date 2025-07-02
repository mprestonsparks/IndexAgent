"""
Redis caching layer for DEAN system.
Per specifications: IndexAgent/indexagent/cache/
"""

from .redis_manager import (
    RedisManager,
    get_redis_manager,
    PatternCache,
    AgentStatusCache,
    MetricsCache
)

__all__ = [
    "RedisManager",
    "get_redis_manager", 
    "PatternCache",
    "AgentStatusCache",
    "MetricsCache"
]