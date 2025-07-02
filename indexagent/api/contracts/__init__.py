#!/usr/bin/env python3
"""
API Contracts Module for IndexAgent Service Communication

This module implements the exact API contracts specified in Service Communication Section 3.4
of the DEAN system architectural design document. It provides standardized communication
patterns between IndexAgent, Evolution API, and database services.

Implements:
- FR-004: Agent lineage tracking 
- FR-006: Performance metrics tracking
- FR-009: Cellular automata evolution
- FR-010: Pattern propagation via Redis
- FR-011: Evolution history tracking
- FR-015: Pattern discovery cataloging
- FR-017: Token allocation based on efficiency
- NFR-008: Agent action auditing

Service Communication Patterns:
- IndexAgent ↔ Evolution API: Agent evolution and pattern propagation
- IndexAgent ↔ Database: Data persistence and retrieval with caching
- Evolution API ↔ Database: Metrics collection and population analysis
"""

from .evolution_service import (
    EvolutionServiceClient,
    PatternPropagationRequest,
    AgentEvolutionRequest,
    PopulationMetricsRequest,
    ServiceResponse,
    ServiceCommunicationError,
    create_evolution_service_client,
    propagate_pattern_to_population,
    evolve_agent_with_ca_rule,
    check_evolution_service_health
)

from .database_service import (
    DatabaseServiceClient,
    AgentCreationRequest,
    PerformanceMetricsRequest,
    EvolutionHistoryRequest,
    PatternDiscoveryRequest,
    AuditLogRequest,
    TokenTransactionRequest,
    DatabaseServiceResponse,
    DatabaseOperationError,
    CacheOperationError,
    create_database_service_client,
    create_agent_with_lineage,
    record_agent_evolution
)

from .knowledge import (
    KnowledgeServiceClient,
    KnowledgeArtifact,
    PatternShareRequest,
    StrategyImportRequest,
    PerformanceBenchmarkRequest,
    KnowledgeDiscoveryRequest,
    KnowledgeType,
    ShareScope,
    KnowledgeStatus,
    create_knowledge_client,
    test_knowledge_api_integration
)

__all__ = [
    # Evolution Service
    "EvolutionServiceClient",
    "PatternPropagationRequest", 
    "AgentEvolutionRequest",
    "PopulationMetricsRequest",
    "ServiceResponse",
    "ServiceCommunicationError",
    "create_evolution_service_client",
    "propagate_pattern_to_population",
    "evolve_agent_with_ca_rule", 
    "check_evolution_service_health",
    
    # Database Service
    "DatabaseServiceClient",
    "AgentCreationRequest",
    "PerformanceMetricsRequest",
    "EvolutionHistoryRequest",
    "PatternDiscoveryRequest",
    "AuditLogRequest",
    "TokenTransactionRequest",
    "DatabaseServiceResponse",
    "DatabaseOperationError",
    "CacheOperationError",
    "create_database_service_client",
    "create_agent_with_lineage",
    "record_agent_evolution",
    
    # Knowledge Repository Service (FR-032)
    "KnowledgeServiceClient",
    "KnowledgeArtifact",
    "PatternShareRequest",
    "StrategyImportRequest", 
    "PerformanceBenchmarkRequest",
    "KnowledgeDiscoveryRequest",
    "KnowledgeType",
    "ShareScope",
    "KnowledgeStatus",
    "create_knowledge_client",
    "test_knowledge_api_integration"
]

# API Contract Version
API_CONTRACTS_VERSION = "1.0.0"

# Service Communication Configuration
SERVICE_COMMUNICATION_CONFIG = {
    "evolution_api": {
        "base_url": "http://agent-evolution:8080/api/v1",
        "timeout": 30,
        "retry_count": 3
    },
    "database": {
        "connection_pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 3600
    },
    "redis_cache": {
        "pattern_ttl": {
            "stable": 3600,    # 1 hour for stable patterns
            "emerging": 600,   # 10 minutes for emerging patterns  
            "experimental": 300 # 5 minutes for experimental patterns
        },
        "agent_registry_ttl": 300,  # 5 minutes for agent status
        "metrics_ttl": 120         # 2 minutes for metrics cache
    }
}

def get_service_communication_config():
    """Get service communication configuration"""
    return SERVICE_COMMUNICATION_CONFIG

def get_api_contracts_version():
    """Get API contracts version"""
    return API_CONTRACTS_VERSION