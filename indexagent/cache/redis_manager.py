"""
Redis caching manager for DEAN system per Data Architecture Section 4.1.
Per specifications: IndexAgent/indexagent/cache/redis_manager.py

Redis Requirements from Specifications:
- Section 4.1: Redis for caching pattern matches and agent status (not optional)
- Pattern match results with TTL based on pattern stability
- Agent status for rapid status queries
- Evolution metrics for dashboard performance
- FR-010: Pattern propagation via Redis agent registry at redis://agent-registry:6379
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError
import time

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Redis connection and operation manager.
    Per Data Architecture Section 4.1 specifications.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis manager.
        
        Args:
            redis_url: Redis connection URL. Defaults to environment variable.
        """
        self.redis_url = redis_url or os.getenv(
            "AGENT_REGISTRY_URL",
            "redis://agent-registry:6379"
        )
        
        # Connection configuration per specifications
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        self.socket_connect_timeout = int(os.getenv("REDIS_CONNECT_TIMEOUT", "5"))
        self.retry_on_timeout = True
        self.health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client with connection pooling."""
        try:
            self._client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=self.health_check_interval,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            self._client.ping()
            logger.info(f"Redis client initialized successfully: {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    @property
    def client(self):
        """Get Redis client with automatic reconnection."""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            return self.client.ping()
        except (ConnectionError, TimeoutError, RedisError) as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Redis server info dictionary
        """
        try:
            return self.client.info()
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    def close(self):
        """Close Redis connections."""
        try:
            if self._client:
                self._client.close()
                logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

class PatternCache:
    """
    Pattern caching with TTL based on pattern stability.
    Per specification requirement for pattern match results caching.
    """
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager.client
        self.namespace = "patterns:"
        
        # TTL configuration based on pattern stability
        self.stable_pattern_ttl = int(os.getenv("STABLE_PATTERN_TTL", "3600"))  # 1 hour
        self.emerging_pattern_ttl = int(os.getenv("EMERGING_PATTERN_TTL", "600"))  # 10 minutes
        self.experimental_pattern_ttl = int(os.getenv("EXPERIMENTAL_PATTERN_TTL", "300"))  # 5 minutes
    
    def _get_ttl_for_pattern(self, pattern_data: Dict[str, Any]) -> int:
        """
        Calculate TTL based on pattern stability.
        
        Args:
            pattern_data: Pattern information including effectiveness and occurrences
            
        Returns:
            TTL in seconds
        """
        occurrences = pattern_data.get('occurrences', 1)
        effectiveness = pattern_data.get('effectiveness_score', 0.0)
        
        # Stable patterns: high occurrences and effectiveness
        if occurrences >= 10 and effectiveness >= 0.5:
            return self.stable_pattern_ttl
        
        # Emerging patterns: moderate occurrences or effectiveness
        elif occurrences >= 3 or effectiveness >= 0.2:
            return self.emerging_pattern_ttl
        
        # Experimental patterns: low occurrences and effectiveness
        else:
            return self.experimental_pattern_ttl
    
    def cache_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """
        Cache pattern with stability-based TTL.
        
        Args:
            pattern_id: Unique pattern identifier
            pattern_data: Pattern information dictionary
        """
        try:
            key = f"{self.namespace}{pattern_id}"
            ttl = self._get_ttl_for_pattern(pattern_data)
            
            # Add caching metadata
            cache_data = {
                **pattern_data,
                'cached_at': datetime.utcnow().isoformat(),
                'cache_ttl': ttl
            }
            
            # Serialize and cache
            serialized_data = json.dumps(cache_data, default=str)
            self.redis.setex(key, ttl, serialized_data)
            
            logger.debug(f"Cached pattern {pattern_id} with TTL {ttl}s")
            
        except Exception as e:
            logger.error(f"Failed to cache pattern {pattern_id}: {e}")
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached pattern.
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            Pattern data if found, None otherwise
        """
        try:
            key = f"{self.namespace}{pattern_id}"
            cached_data = self.redis.get(key)
            
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve pattern {pattern_id}: {e}")
            return None
    
    def cache_pattern_matches(self, query_hash: str, matches: List[Dict[str, Any]], ttl: int = 300):
        """
        Cache pattern search results.
        
        Args:
            query_hash: Hash of the search query
            matches: List of matching patterns
            ttl: Time to live in seconds
        """
        try:
            key = f"{self.namespace}matches:{query_hash}"
            
            cache_data = {
                'matches': matches,
                'cached_at': datetime.utcnow().isoformat(),
                'query_hash': query_hash
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            self.redis.setex(key, ttl, serialized_data)
            
            logger.debug(f"Cached {len(matches)} pattern matches for query {query_hash[:8]}")
            
        except Exception as e:
            logger.error(f"Failed to cache pattern matches: {e}")
    
    def get_pattern_matches(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached pattern matches.
        
        Args:
            query_hash: Hash of the search query
            
        Returns:
            List of matching patterns if found, None otherwise
        """
        try:
            key = f"{self.namespace}matches:{query_hash}"
            cached_data = self.redis.get(key)
            
            if cached_data:
                data = json.loads(cached_data.decode('utf-8'))
                return data.get('matches', [])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve pattern matches: {e}")
            return None
    
    def invalidate_pattern(self, pattern_id: str):
        """Remove pattern from cache."""
        try:
            key = f"{self.namespace}{pattern_id}"
            self.redis.delete(key)
            logger.debug(f"Invalidated pattern cache for {pattern_id}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern_id}: {e}")

class AgentStatusCache:
    """
    Agent status caching for rapid status queries.
    Per specification requirement for agent status caching.
    """
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager.client
        self.namespace = "agents:"
        self.status_ttl = int(os.getenv("AGENT_STATUS_TTL", "60"))  # 1 minute
        self.registry_ttl = int(os.getenv("AGENT_REGISTRY_TTL", "300"))  # 5 minutes
    
    def cache_agent_status(self, agent_id: str, status_data: Dict[str, Any]):
        """
        Cache agent status information.
        
        Args:
            agent_id: Agent identifier
            status_data: Agent status dictionary
        """
        try:
            key = f"{self.namespace}status:{agent_id}"
            
            cache_data = {
                **status_data,
                'cached_at': datetime.utcnow().isoformat(),
                'agent_id': agent_id
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            self.redis.setex(key, self.status_ttl, serialized_data)
            
            logger.debug(f"Cached status for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to cache agent status {agent_id}: {e}")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached agent status.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent status if found, None otherwise
        """
        try:
            key = f"{self.namespace}status:{agent_id}"
            cached_data = self.redis.get(key)
            
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve agent status {agent_id}: {e}")
            return None
    
    def register_agent(self, agent_id: str, agent_data: Dict[str, Any]):
        """
        Register agent in Redis registry per FR-010.
        
        Args:
            agent_id: Agent identifier
            agent_data: Agent registration data
        """
        try:
            # Add to active agents set
            self.redis.sadd("agents:active", agent_id)
            
            # Cache full agent data
            key = f"{self.namespace}registry:{agent_id}"
            
            registry_data = {
                **agent_data,
                'registered_at': datetime.utcnow().isoformat(),
                'agent_id': agent_id
            }
            
            serialized_data = json.dumps(registry_data, default=str)
            self.redis.setex(key, self.registry_ttl, serialized_data)
            
            logger.info(f"Registered agent {agent_id} in Redis registry")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
    
    def unregister_agent(self, agent_id: str):
        """
        Remove agent from registry.
        
        Args:
            agent_id: Agent identifier
        """
        try:
            # Remove from active agents set
            self.redis.srem("agents:active", agent_id)
            
            # Remove registry data
            key = f"{self.namespace}registry:{agent_id}"
            self.redis.delete(key)
            
            # Remove status cache
            status_key = f"{self.namespace}status:{agent_id}"
            self.redis.delete(status_key)
            
            logger.info(f"Unregistered agent {agent_id} from Redis registry")
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
    
    def get_active_agents(self) -> List[str]:
        """
        Get list of active agent IDs.
        
        Returns:
            List of active agent IDs
        """
        try:
            agents = self.redis.smembers("agents:active")
            return [agent.decode('utf-8') for agent in agents]
            
        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []

class MetricsCache:
    """
    Evolution metrics caching for dashboard performance.
    Per specification requirement for dashboard performance.
    """
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager.client
        self.namespace = "metrics:"
        self.metrics_ttl = int(os.getenv("METRICS_TTL", "120"))  # 2 minutes
    
    def cache_evolution_metrics(self, metrics: Dict[str, Any]):
        """
        Cache evolution metrics for dashboard.
        
        Args:
            metrics: Metrics dictionary
        """
        try:
            key = f"{self.namespace}evolution"
            
            cache_data = {
                **metrics,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            self.redis.setex(key, self.metrics_ttl, serialized_data)
            
            logger.debug("Cached evolution metrics")
            
        except Exception as e:
            logger.error(f"Failed to cache evolution metrics: {e}")
    
    def get_evolution_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached evolution metrics.
        
        Returns:
            Evolution metrics if found, None otherwise
        """
        try:
            key = f"{self.namespace}evolution"
            cached_data = self.redis.get(key)
            
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve evolution metrics: {e}")
            return None
    
    def cache_agent_efficiency_summary(self, summary: Dict[str, Any]):
        """
        Cache agent efficiency summary.
        
        Args:
            summary: Efficiency summary data
        """
        try:
            key = f"{self.namespace}efficiency_summary"
            
            cache_data = {
                **summary,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            self.redis.setex(key, self.metrics_ttl, serialized_data)
            
            logger.debug("Cached agent efficiency summary")
            
        except Exception as e:
            logger.error(f"Failed to cache efficiency summary: {e}")
    
    def get_agent_efficiency_summary(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached agent efficiency summary.
        
        Returns:
            Efficiency summary if found, None otherwise
        """
        try:
            key = f"{self.metrics}efficiency_summary"
            cached_data = self.redis.get(key)
            
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve efficiency summary: {e}")
            return None

# Global Redis manager instance
_redis_manager = None

def get_redis_manager() -> RedisManager:
    """
    Get global Redis manager instance (singleton pattern).
    
    Returns:
        RedisManager instance
    """
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager

def test_redis_connection() -> bool:
    """Test Redis connection."""
    try:
        redis_manager = get_redis_manager()
        return redis_manager.ping()
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test Redis connection when run directly
    redis_manager = RedisManager()
    
    if redis_manager.ping():
        print("✅ Redis connection successful")
        
        # Test caching functionality
        pattern_cache = PatternCache(redis_manager)
        agent_cache = AgentStatusCache(redis_manager)
        metrics_cache = MetricsCache(redis_manager)
        
        # Test pattern caching
        test_pattern = {
            'pattern_id': 'test_pattern',
            'pattern_type': 'behavioral',
            'effectiveness_score': 0.8,
            'occurrences': 5
        }
        
        pattern_cache.cache_pattern('test_pattern', test_pattern)
        retrieved_pattern = pattern_cache.get_pattern('test_pattern')
        
        if retrieved_pattern:
            print("✅ Pattern caching working")
        else:
            print("❌ Pattern caching failed")
        
        # Test agent registry
        test_agent = {
            'agent_id': 'test_agent',
            'status': 'active',
            'token_budget': 1000
        }
        
        agent_cache.register_agent('test_agent', test_agent)
        active_agents = agent_cache.get_active_agents()
        
        if 'test_agent' in active_agents:
            print("✅ Agent registry working")
        else:
            print("❌ Agent registry failed")
        
        # Cleanup
        agent_cache.unregister_agent('test_agent')
        pattern_cache.invalidate_pattern('test_pattern')
        
        print("✅ Redis caching layer validation completed")
        
    else:
        print("❌ Redis connection failed")
        exit(1)