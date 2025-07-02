#!/usr/bin/env python3
"""
Database Service API Contracts
Implementation of database layer service contracts per Service Communication Section 3.4

This module implements the database service communication patterns for the DEAN system,
providing standardized access to PostgreSQL and Redis layers with proper error handling
and transaction management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field, validator

# Import database models
try:
    from IndexAgent.indexagent.database.schema import (
        Agent, EvolutionHistory, PerformanceMetrics, DiscoveredPatterns,
        StrategyEvolution, AuditLog, TokenTransaction, Base
    )
    from IndexAgent.indexagent.cache.redis_manager import (
        RedisManager, PatternCache, AgentStatusCache, MetricsCache
    )
    DATABASE_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Database models not available: {e}")
    DATABASE_MODELS_AVAILABLE = False

# Implements Service Communication Section 3.4: Database API contracts
logger = logging.getLogger(__name__)


class DatabaseOperationError(Exception):
    """Exception raised for database operation failures"""
    pass


class CacheOperationError(Exception):
    """Exception raised for cache operation failures"""
    pass


class DatabaseOperationType(str, Enum):
    """Database operation types for auditing"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EVOLVE = "evolve"
    PATTERN_DISCOVERY = "pattern_discovery"


class AgentCreationRequest(BaseModel):
    """Agent creation request per FR-004"""
    name: str = Field(..., description="Agent name")
    parent_id: Optional[str] = Field(None, description="Parent agent ID for lineage")
    generation: int = Field(default=0, ge=0, description="Generation number")
    token_budget: int = Field(..., gt=0, description="Initial token budget")
    status: str = Field(default="active", description="Agent status")
    worktree_path: Optional[str] = Field(None, description="Git worktree path")
    diversity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Diversity score")
    fitness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Fitness score")
    genome_traits: Dict[str, Any] = Field(default_factory=dict, description="Genetic traits")
    strategies: List[str] = Field(default_factory=list, description="Agent strategies")


class PerformanceMetricsRequest(BaseModel):
    """Performance metrics recording request per FR-006"""
    agent_id: str = Field(..., description="Agent ID")
    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")
    metric_unit: str = Field(..., description="Metric unit")
    task_type: Optional[str] = Field(None, description="Task type")
    measurement_context: Dict[str, Any] = Field(default_factory=dict, description="Measurement context")


class EvolutionHistoryRequest(BaseModel):
    """Evolution history recording request per FR-011"""
    agent_id: str = Field(..., description="Agent ID")
    generation: int = Field(..., ge=0, description="Generation number")
    rule_applied: str = Field(..., description="Applied cellular automata rule")
    performance_delta: float = Field(..., description="Performance change")
    strategy_changes: List[str] = Field(default_factory=list, description="Strategy modifications")
    new_patterns: List[str] = Field(default_factory=list, description="Newly discovered patterns")
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Mutation rate")
    crossover_applied: bool = Field(default=False, description="Whether crossover was applied")
    population_size: int = Field(..., gt=0, description="Population size during evolution")
    population_diversity: float = Field(..., ge=0.0, le=1.0, description="Population diversity")
    environment_context: Dict[str, Any] = Field(default_factory=dict, description="Evolution environment")


class PatternDiscoveryRequest(BaseModel):
    """Pattern discovery recording request per FR-015"""
    pattern_hash: str = Field(..., description="Unique pattern hash")
    pattern_type: str = Field(..., description="Pattern classification")
    description: str = Field(..., description="Pattern description")
    pattern_sequence: List[str] = Field(..., description="Pattern sequence")
    pattern_context: Dict[str, Any] = Field(default_factory=dict, description="Pattern context")
    effectiveness_score: float = Field(..., ge=0.0, le=1.0, description="Effectiveness score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    discovered_by_agent_id: str = Field(..., description="Discovering agent ID")
    agent_ids: List[str] = Field(default_factory=list, description="Associated agent IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AuditLogRequest(BaseModel):
    """Audit log entry request per NFR-008"""
    agent_id: str = Field(..., description="Agent ID")
    action_type: str = Field(..., description="Action type")
    action_description: str = Field(..., description="Action description")
    target_resource: Optional[str] = Field(None, description="Target resource")
    action_parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    success: bool = Field(..., description="Action success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    tokens_consumed: int = Field(default=0, ge=0, description="Tokens consumed")
    execution_time_ms: int = Field(default=0, ge=0, description="Execution time in milliseconds")


class TokenTransactionRequest(BaseModel):
    """Token transaction request per FR-017"""
    agent_id: str = Field(..., description="Agent ID")
    transaction_type: str = Field(..., description="Transaction type")
    amount: int = Field(..., description="Token amount (negative for consumption)")
    reason: str = Field(..., description="Transaction reason")
    task_description: Optional[str] = Field(None, description="Task description")
    value_generated: Optional[float] = Field(None, description="Value generated")
    balance_before: int = Field(..., ge=0, description="Balance before transaction")
    balance_after: int = Field(..., ge=0, description="Balance after transaction")


@dataclass
class DatabaseServiceResponse:
    """Database service operation response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    operation_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DatabaseServiceClient:
    """
    Database Service client implementing database API contracts per Section 3.4
    
    Provides standardized database operations for the DEAN system with proper
    transaction management, caching integration, and error handling.
    """
    
    def __init__(self, database_url: str, redis_url: str = "redis://agent-registry:6379"):
        """
        Initialize Database Service client
        
        Args:
            database_url: PostgreSQL connection URL
            redis_url: Redis connection URL
        """
        self.database_url = database_url
        self.redis_url = redis_url
        self.engine = None
        self.SessionLocal = None
        self.redis_manager = None
        self.pattern_cache = None
        self.agent_cache = None
        self.metrics_cache = None
        
    async def initialize(self):
        """Initialize database and cache connections"""
        if not DATABASE_MODELS_AVAILABLE:
            raise DatabaseOperationError("Database models not available")
        
        try:
            # Initialize PostgreSQL connection
            self.engine = create_async_engine(
                self.database_url,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.SessionLocal = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize Redis caches
            self.redis_manager = RedisManager()
            await self.redis_manager.initialize()
            
            self.pattern_cache = PatternCache(self.redis_manager)
            self.agent_cache = AgentStatusCache(self.redis_manager)
            self.metrics_cache = MetricsCache(self.redis_manager)
            
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Database service initialization failed: {e}")
            raise DatabaseOperationError(f"Initialization failed: {e}")
    
    async def close(self):
        """Close database and cache connections"""
        if self.engine:
            await self.engine.dispose()
        if self.redis_manager:
            await self.redis_manager.close()
    
    @asynccontextmanager
    async def get_session(self) -> AsyncContextManager[AsyncSession]:
        """Get database session with automatic transaction management"""
        if not self.SessionLocal:
            raise DatabaseOperationError("Database not initialized")
        
        async with self.SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def create_agent(self, request: AgentCreationRequest) -> DatabaseServiceResponse:
        """
        Create new agent per FR-004
        
        Implements agent creation with lineage tracking in PostgreSQL
        and cache registration in Redis.
        
        Args:
            request: Agent creation request
            
        Returns:
            DatabaseServiceResponse with created agent data
        """
        try:
            async with self.get_session() as session:
                # Create agent instance
                agent = Agent(
                    name=request.name,
                    parent_id=request.parent_id,
                    generation=request.generation,
                    token_budget=request.token_budget,
                    tokens_consumed=0,
                    tokens_remaining=request.token_budget,
                    status=request.status,
                    worktree_path=request.worktree_path,
                    diversity_score=request.diversity_score,
                    fitness_score=request.fitness_score,
                    genome_traits=request.genome_traits,
                    strategies=request.strategies
                )
                
                session.add(agent)
                await session.flush()  # Get agent ID
                
                # Register in Redis cache
                await self.agent_cache.register_agent(str(agent.id), {
                    'name': agent.name,
                    'status': agent.status,
                    'token_budget': agent.token_budget,
                    'generation': agent.generation,
                    'diversity_score': agent.diversity_score
                })
                
                # Log audit entry
                await self._log_audit_entry(
                    session,
                    str(agent.id),
                    "agent_created",
                    f"Agent {agent.name} created with budget {agent.token_budget}",
                    True,
                    tokens_consumed=50  # Creation cost
                )
                
                logger.info(f"Agent {agent.name} created successfully (ID: {agent.id})")
                
                return DatabaseServiceResponse(
                    success=True,
                    message=f"Agent {agent.name} created successfully",
                    data={
                        "agent_id": str(agent.id),
                        "name": agent.name,
                        "generation": agent.generation,
                        "token_budget": agent.token_budget,
                        "parent_id": agent.parent_id
                    },
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Agent creation failed: {e}")
            raise DatabaseOperationError(f"Agent creation failed: {e}")
    
    async def record_performance_metrics(self, request: PerformanceMetricsRequest) -> DatabaseServiceResponse:
        """
        Record performance metrics per FR-006
        
        Implements value-per-token metrics tracking in PostgreSQL
        with cache updates for real-time access.
        
        Args:
            request: Performance metrics request
            
        Returns:
            DatabaseServiceResponse with recording confirmation
        """
        try:
            async with self.get_session() as session:
                # Create performance metric entry
                metric = PerformanceMetrics(
                    agent_id=request.agent_id,
                    metric_name=request.metric_name,
                    metric_value=request.metric_value,
                    metric_unit=request.metric_unit,
                    task_type=request.task_type,
                    measurement_context=request.measurement_context
                )
                
                session.add(metric)
                
                # Update metrics cache
                await self.metrics_cache.update_agent_efficiency(
                    request.agent_id,
                    request.metric_value
                )
                
                logger.info(f"Performance metric recorded for agent {request.agent_id}: "
                           f"{request.metric_name}={request.metric_value}")
                
                return DatabaseServiceResponse(
                    success=True,
                    message="Performance metrics recorded successfully",
                    data={
                        "agent_id": request.agent_id,
                        "metric_name": request.metric_name,
                        "metric_value": request.metric_value,
                        "recorded_at": datetime.now().isoformat()
                    },
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Performance metrics recording failed: {e}")
            raise DatabaseOperationError(f"Performance metrics recording failed: {e}")
    
    async def record_evolution_history(self, request: EvolutionHistoryRequest) -> DatabaseServiceResponse:
        """
        Record evolution history per FR-011
        
        Implements evolutionary history tracking with cellular automata rules
        and performance delta tracking.
        
        Args:
            request: Evolution history request
            
        Returns:
            DatabaseServiceResponse with recording confirmation
        """
        try:
            async with self.get_session() as session:
                # Create evolution history entry
                evolution = EvolutionHistory(
                    agent_id=request.agent_id,
                    generation=request.generation,
                    rule_applied=request.rule_applied,
                    performance_delta=request.performance_delta,
                    strategy_changes=request.strategy_changes,
                    new_patterns=request.new_patterns,
                    mutation_rate=request.mutation_rate,
                    crossover_applied=request.crossover_applied,
                    population_size=request.population_size,
                    population_diversity=request.population_diversity,
                    environment_context=request.environment_context
                )
                
                session.add(evolution)
                
                # Update evolution metrics cache
                await self.metrics_cache.cache_evolution_metrics({
                    'population_size': request.population_size,
                    'population_diversity': request.population_diversity,
                    'performance_delta': request.performance_delta
                })
                
                logger.info(f"Evolution history recorded for agent {request.agent_id}: "
                           f"rule {request.rule_applied}, delta {request.performance_delta}")
                
                return DatabaseServiceResponse(
                    success=True,
                    message="Evolution history recorded successfully",
                    data={
                        "agent_id": request.agent_id,
                        "generation": request.generation,
                        "rule_applied": request.rule_applied,
                        "performance_delta": request.performance_delta
                    },
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Evolution history recording failed: {e}")
            raise DatabaseOperationError(f"Evolution history recording failed: {e}")
    
    async def discover_pattern(self, request: PatternDiscoveryRequest) -> DatabaseServiceResponse:
        """
        Record discovered pattern per FR-015
        
        Implements pattern discovery cataloging with effectiveness scoring
        and Redis cache propagation.
        
        Args:
            request: Pattern discovery request
            
        Returns:
            DatabaseServiceResponse with discovery confirmation
        """
        try:
            async with self.get_session() as session:
                # Create discovered pattern entry
                pattern = DiscoveredPatterns(
                    pattern_hash=request.pattern_hash,
                    pattern_type=request.pattern_type,
                    description=request.description,
                    pattern_sequence=request.pattern_sequence,
                    pattern_context=request.pattern_context,
                    effectiveness_score=request.effectiveness_score,
                    confidence_score=request.confidence_score,
                    occurrences=1,
                    reuse_count=0,
                    agent_ids=request.agent_ids,
                    discovered_by_agent_id=request.discovered_by_agent_id,
                    metadata=request.metadata
                )
                
                session.add(pattern)
                await session.flush()  # Get pattern ID
                
                # Cache pattern for propagation
                await self.pattern_cache.cache_pattern(str(pattern.id), {
                    'pattern_id': str(pattern.id),
                    'pattern_type': pattern.pattern_type,
                    'effectiveness_score': pattern.effectiveness_score,
                    'confidence_score': pattern.confidence_score,
                    'occurrences': pattern.occurrences,
                    'description': pattern.description
                })
                
                logger.info(f"Pattern discovered and cached: {pattern.description} "
                           f"(effectiveness: {pattern.effectiveness_score})")
                
                return DatabaseServiceResponse(
                    success=True,
                    message="Pattern discovery recorded successfully",
                    data={
                        "pattern_id": str(pattern.id),
                        "pattern_type": pattern.pattern_type,
                        "effectiveness_score": pattern.effectiveness_score,
                        "discovered_by": request.discovered_by_agent_id
                    },
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Pattern discovery recording failed: {e}")
            raise DatabaseOperationError(f"Pattern discovery recording failed: {e}")
    
    async def record_token_transaction(self, request: TokenTransactionRequest) -> DatabaseServiceResponse:
        """
        Record token transaction per FR-017
        
        Implements token economy tracking with balance validation
        and efficiency calculation.
        
        Args:
            request: Token transaction request
            
        Returns:
            DatabaseServiceResponse with transaction confirmation
        """
        try:
            async with self.get_session() as session:
                # Create token transaction entry
                transaction = TokenTransaction(
                    agent_id=request.agent_id,
                    transaction_type=request.transaction_type,
                    amount=request.amount,
                    reason=request.reason,
                    task_description=request.task_description,
                    value_generated=request.value_generated,
                    balance_before=request.balance_before,
                    balance_after=request.balance_after
                )
                
                session.add(transaction)
                
                # Update agent token balance
                agent_result = await session.execute(
                    select(Agent).where(Agent.id == request.agent_id)
                )
                agent = agent_result.scalar_one_or_none()
                
                if agent:
                    agent.tokens_consumed += abs(request.amount) if request.amount < 0 else 0
                    agent.tokens_remaining = request.balance_after
                    
                    # Update agent cache
                    await self.agent_cache.update_agent_status(request.agent_id, {
                        'token_budget': agent.token_budget,
                        'tokens_remaining': agent.tokens_remaining,
                        'tokens_consumed': agent.tokens_consumed
                    })
                
                logger.info(f"Token transaction recorded for agent {request.agent_id}: "
                           f"{request.amount} tokens ({request.transaction_type})")
                
                return DatabaseServiceResponse(
                    success=True,
                    message="Token transaction recorded successfully",
                    data={
                        "agent_id": request.agent_id,
                        "transaction_type": request.transaction_type,
                        "amount": request.amount,
                        "balance_after": request.balance_after
                    },
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Token transaction recording failed: {e}")
            raise DatabaseOperationError(f"Token transaction recording failed: {e}")
    
    async def get_agent_by_id(self, agent_id: str) -> DatabaseServiceResponse:
        """
        Get agent by ID with caching
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            DatabaseServiceResponse with agent data
        """
        try:
            # Try cache first
            cached_agent = await self.agent_cache.get_agent_status(agent_id)
            if cached_agent:
                return DatabaseServiceResponse(
                    success=True,
                    message="Agent retrieved from cache",
                    data={"agent": cached_agent, "source": "cache"},
                    operation_id=self._generate_operation_id()
                )
            
            # Fallback to database
            async with self.get_session() as session:
                result = await session.execute(
                    select(Agent).where(Agent.id == agent_id)
                )
                agent = result.scalar_one_or_none()
                
                if not agent:
                    return DatabaseServiceResponse(
                        success=False,
                        message=f"Agent {agent_id} not found",
                        operation_id=self._generate_operation_id()
                    )
                
                agent_data = {
                    "id": str(agent.id),
                    "name": agent.name,
                    "parent_id": agent.parent_id,
                    "generation": agent.generation,
                    "token_budget": agent.token_budget,
                    "tokens_remaining": agent.tokens_remaining,
                    "status": agent.status,
                    "diversity_score": agent.diversity_score,
                    "fitness_score": agent.fitness_score
                }
                
                # Update cache
                await self.agent_cache.register_agent(agent_id, agent_data)
                
                return DatabaseServiceResponse(
                    success=True,
                    message="Agent retrieved from database",
                    data={"agent": agent_data, "source": "database"},
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Agent retrieval failed: {e}")
            raise DatabaseOperationError(f"Agent retrieval failed: {e}")
    
    async def get_population_metrics(self) -> DatabaseServiceResponse:
        """
        Get population-wide metrics
        
        Returns:
            DatabaseServiceResponse with population metrics
        """
        try:
            # Try cache first
            cached_metrics = await self.metrics_cache.get_evolution_metrics()
            if cached_metrics:
                return DatabaseServiceResponse(
                    success=True,
                    message="Population metrics retrieved from cache",
                    data={"metrics": cached_metrics, "source": "cache"},
                    operation_id=self._generate_operation_id()
                )
            
            # Calculate from database
            async with self.get_session() as session:
                # Get agent count and diversity
                agent_stats = await session.execute(
                    select(
                        func.count(Agent.id).label('total_agents'),
                        func.avg(Agent.diversity_score).label('avg_diversity'),
                        func.avg(Agent.fitness_score).label('avg_fitness')
                    ).where(Agent.status == 'active')
                )
                stats = agent_stats.one()
                
                # Get recent performance metrics
                recent_efficiency = await session.execute(
                    select(func.avg(PerformanceMetrics.metric_value))
                    .where(
                        and_(
                            PerformanceMetrics.metric_name == 'token_efficiency',
                            PerformanceMetrics.recorded_at >= datetime.now().replace(hour=0, minute=0, second=0)
                        )
                    )
                )
                avg_efficiency = recent_efficiency.scalar() or 0.0
                
                metrics = {
                    "total_agents": stats.total_agents or 0,
                    "average_diversity": float(stats.avg_diversity or 0.0),
                    "average_fitness": float(stats.avg_fitness or 0.0),
                    "average_efficiency": float(avg_efficiency),
                    "calculated_at": datetime.now().isoformat()
                }
                
                # Cache for next time
                await self.metrics_cache.cache_evolution_metrics(metrics)
                
                return DatabaseServiceResponse(
                    success=True,
                    message="Population metrics calculated from database",
                    data={"metrics": metrics, "source": "database"},
                    operation_id=self._generate_operation_id()
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Population metrics calculation failed: {e}")
            raise DatabaseOperationError(f"Population metrics calculation failed: {e}")
    
    async def _log_audit_entry(self, session: AsyncSession, agent_id: str, action_type: str,
                             description: str, success: bool, error_message: str = None,
                             tokens_consumed: int = 0, execution_time_ms: int = 0):
        """Log audit entry per NFR-008"""
        audit_entry = AuditLog(
            agent_id=agent_id,
            action_type=action_type,
            action_description=description,
            success=success,
            error_message=error_message,
            tokens_consumed=tokens_consumed,
            execution_time_ms=execution_time_ms
        )
        session.add(audit_entry)
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID"""
        import uuid
        return str(uuid.uuid4())[:8]


# Service factory function
def create_database_service_client(database_url: str, redis_url: str = "redis://agent-registry:6379") -> DatabaseServiceClient:
    """
    Factory function to create Database Service client
    
    Args:
        database_url: PostgreSQL connection URL
        redis_url: Redis connection URL
        
    Returns:
        Configured DatabaseServiceClient
    """
    return DatabaseServiceClient(database_url, redis_url)


# Convenience functions for common operations
async def create_agent_with_lineage(name: str, parent_id: Optional[str], token_budget: int,
                                  database_client: DatabaseServiceClient) -> DatabaseServiceResponse:
    """
    Convenience function for agent creation with proper lineage tracking
    """
    generation = 0
    if parent_id:
        # Get parent generation to increment
        parent_response = await database_client.get_agent_by_id(parent_id)
        if parent_response.success and parent_response.data:
            generation = parent_response.data.get('agent', {}).get('generation', 0) + 1
    
    request = AgentCreationRequest(
        name=name,
        parent_id=parent_id,
        generation=generation,
        token_budget=token_budget,
        diversity_score=0.7,  # Default diversity target
        strategies=["exploration", "optimization"]
    )
    
    return await database_client.create_agent(request)


async def record_agent_evolution(agent_id: str, ca_rule: str, performance_delta: float,
                                population_size: int, database_client: DatabaseServiceClient) -> DatabaseServiceResponse:
    """
    Convenience function for recording agent evolution
    """
    # Get current generation
    agent_response = await database_client.get_agent_by_id(agent_id)
    generation = 0
    if agent_response.success and agent_response.data:
        generation = agent_response.data.get('agent', {}).get('generation', 0)
    
    request = EvolutionHistoryRequest(
        agent_id=agent_id,
        generation=generation,
        rule_applied=ca_rule,
        performance_delta=performance_delta,
        population_size=population_size,
        population_diversity=0.7,  # Default value
        strategy_changes=["rule_application"],
        new_patterns=[],
        environment_context={"ca_rule": ca_rule}
    )
    
    return await database_client.record_evolution_history(request)