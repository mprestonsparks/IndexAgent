"""
SQLAlchemy models for DEAN agent evolution system
Maps to agent_evolution schema defined in init_agent_evolution.sql
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Boolean, ARRAY, CheckConstraint, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

Base = declarative_base()


class Agent(Base):
    """Core agent registry tracking lineage and performance per FR-004"""
    __tablename__ = 'agents'
    __table_args__ = {'schema': 'agent_evolution'}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    name = Column(String(255), nullable=False)
    goal = Column(String, nullable=False)
    
    # Lineage tracking
    parent_ids = Column(ARRAY(UUID(as_uuid=True)), default=[], nullable=False)
    generation = Column(Integer, nullable=False, default=0)
    
    # Economic constraints per FR-005
    token_budget = Column(Integer, CheckConstraint('token_budget >= 0'), nullable=False)
    token_consumed = Column(Integer, CheckConstraint('token_consumed >= 0'), nullable=False, default=0)
    token_efficiency = Column(Float, CheckConstraint('token_efficiency >= 0 AND token_efficiency <= 2'), default=0.5)
    
    # Agent characteristics
    status = Column(String(50), nullable=False, default='active')
    worktree_path = Column(String(500))
    specialized_domain = Column(String(100))
    
    # Diversity and evolution tracking per FR-012
    diversity_weight = Column(Float, CheckConstraint('diversity_weight >= 0 AND diversity_weight <= 1'),
                            nullable=False, default=0.5)
    diversity_score = Column(Float, CheckConstraint('diversity_score >= 0 AND diversity_score <= 1'),
                           nullable=False, default=0.5)
    fitness_score = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp(), 
                       onupdate=func.current_timestamp())
    terminated_at = Column(DateTime(timezone=True))
    
    # Relationships
    evolution_history = relationship("EvolutionHistory", back_populates="agent", cascade="all, delete-orphan")
    performance_metrics = relationship("PerformanceMetric", back_populates="agent", cascade="all, delete-orphan")
    discovered_patterns = relationship("DiscoveredPattern", back_populates="agent", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="agent", cascade="all, delete-orphan")
    token_transactions = relationship("TokenTransaction", back_populates="agent", cascade="all, delete-orphan")


class EvolutionHistory(Base):
    """Maintain evolutionary history per FR-011"""
    __tablename__ = 'evolution_history'
    __table_args__ = {'schema': 'agent_evolution'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.agents.id', ondelete='CASCADE'), nullable=False)
    
    # Evolution details
    generation = Column(Integer, nullable=False)
    evolution_type = Column(String(50), nullable=False)  # cellular_automata, genetic_algorithm, mutation
    rule_applied = Column(String(50))  # rule_110, rule_30, etc.
    
    # Results
    fitness_before = Column(Float, nullable=False)
    fitness_after = Column(Float, nullable=False)
    # fitness_delta is computed column in SQL, calculate in Python
    
    # Pattern tracking
    patterns_applied = Column(ARRAY(UUID(as_uuid=True)))
    new_patterns_discovered = Column(Integer, default=0)
    
    # Context
    population_size = Column(Integer, nullable=False)
    population_diversity = Column(Float, nullable=False)
    
    # Timestamps
    evolved_at = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp())
    
    # Relationships
    agent = relationship("Agent", back_populates="evolution_history")
    
    @property
    def fitness_delta(self):
        """Computed property for fitness delta"""
        return self.fitness_after - self.fitness_before


class PerformanceMetric(Base):
    """Track value-per-token metrics per FR-006"""
    __tablename__ = 'performance_metrics'
    __table_args__ = {'schema': 'agent_evolution'}
    
    id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.agents.id', ondelete='CASCADE'), nullable=False)
    
    # Metric details
    metric_type = Column(String(50), nullable=False)  # speed, quality, efficiency, pattern_discovery
    metric_value = Column(Float, nullable=False)
    
    # Token tracking
    tokens_used = Column(Integer, nullable=False)
    # value_per_token is computed in SQL, calculate in Python
    
    # Task context
    task_type = Column(String(100))
    task_description = Column(String)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp())
    
    # Composite primary key for partitioning
    __mapper_args__ = {
        'primary_key': [id, timestamp]
    }
    
    # Relationships
    agent = relationship("Agent", back_populates="performance_metrics")
    
    @property
    def value_per_token(self):
        """Computed property for value per token"""
        return self.metric_value / self.tokens_used if self.tokens_used > 0 else 0


class DiscoveredPattern(Base):
    """Emergent behavior patterns per FR-015 and FR-027"""
    __tablename__ = 'discovered_patterns'
    __table_args__ = {'schema': 'agent_evolution'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.agents.id', ondelete='CASCADE'), nullable=False)
    
    # Pattern identification
    pattern_hash = Column(String(64), unique=True, nullable=False)
    pattern_type = Column(String(50), nullable=False)  # efficiency_optimization, collaboration, innovation
    pattern_content = Column(JSONB, nullable=False)
    
    # Effectiveness tracking per FR-027
    effectiveness_score = Column(Float, CheckConstraint('effectiveness_score >= -1 AND effectiveness_score <= 2'),
                               nullable=False, default=0.0)
    confidence_score = Column(Float, CheckConstraint('confidence_score >= 0 AND confidence_score <= 1'),
                            nullable=False, default=0.5)
    
    # Economic impact
    token_efficiency_delta = Column(Float, default=0.0)
    performance_improvement = Column(Float, default=0.0)
    
    # Usage tracking per FR-024
    reuse_count = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True))
    
    # Timestamps
    discovered_at = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp())
    
    # Pattern metadata (maps to 'metadata' column in DB)
    pattern_meta = Column('metadata', JSONB, default={})
    
    # Relationships
    agent = relationship("Agent", back_populates="discovered_patterns")
    token_transactions = relationship("TokenTransaction", back_populates="pattern")


class AuditLog(Base):
    """Complete audit trail per NFR-008: All agent actions SHALL be auditable"""
    __tablename__ = 'audit_log'
    __table_args__ = {'schema': 'agent_evolution'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.agents.id', ondelete='CASCADE'), nullable=False)
    
    # Action details
    action_type = Column(String(100), nullable=False)
    action_description = Column(String, nullable=False)
    
    # Resource tracking
    target_resource = Column(String(500))
    tokens_consumed = Column(Integer, default=0)
    
    # Results
    success = Column(Boolean, nullable=False)
    error_message = Column(String)
    execution_time_ms = Column(Integer)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp())
    
    # Relationships
    agent = relationship("Agent", back_populates="audit_logs")


class TokenTransaction(Base):
    """Economic tracking for token budgets"""
    __tablename__ = 'token_transactions'
    __table_args__ = {'schema': 'agent_evolution'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.agents.id', ondelete='CASCADE'), nullable=False)
    
    # Transaction details
    transaction_type = Column(String(50), nullable=False)  # allocation, consumption, reallocation
    amount = Column(Integer, CheckConstraint('amount != 0'), nullable=False)
    
    # Context
    reason = Column(String(255), nullable=False)
    task_id = Column(UUID(as_uuid=True))
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.discovered_patterns.id'))
    
    # Balance tracking
    balance_before = Column(Integer, nullable=False)
    balance_after = Column(Integer, nullable=False)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.current_timestamp())
    
    # Relationships
    agent = relationship("Agent", back_populates="token_transactions")
    pattern = relationship("DiscoveredPattern", back_populates="token_transactions")


# Utility functions for database operations
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)