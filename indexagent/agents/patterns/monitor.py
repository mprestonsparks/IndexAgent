"""
Emergent Behavior Monitor for DEAN system.
Captures and catalogs novel agent strategies per specifications.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from enum import Enum
import asyncio

class PatternType(Enum):
    """Types of emergent patterns that can be detected."""
    STRATEGY_INNOVATION = "strategy_innovation"
    EFFICIENCY_BREAKTHROUGH = "efficiency_breakthrough"
    COLLABORATION_PATTERN = "collaboration_pattern"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    PROBLEM_SOLVING_APPROACH = "problem_solving_approach"
    ADAPTATION_STRATEGY = "adaptation_strategy"
    META_LEARNING = "meta_learning"

class PatternSignificance(Enum):
    """Significance levels for discovered patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BehaviorSignature:
    """Unique signature for a behavior pattern."""
    action_sequence: List[str]
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, float]
    context_factors: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_hash(self) -> str:
        """Generate unique hash for this behavior signature."""
        signature_data = {
            "actions": sorted(self.action_sequence),
            "resources": sorted(self.resource_usage.items()),
            "metrics": sorted(self.performance_metrics.items()),
            "context": sorted(str(item) for item in self.context_factors.items())
        }
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]

@dataclass
class EmergentPattern:
    """Represents a discovered emergent behavior pattern."""
    id: str
    agent_id: str
    pattern_type: PatternType
    significance: PatternSignificance
    behavior_signature: BehaviorSignature
    effectiveness_score: float
    token_efficiency_delta: float
    reuse_count: int = 0
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "pattern_type": self.pattern_type.value,
            "significance": self.significance.value,
            "effectiveness_score": self.effectiveness_score,
            "token_efficiency_delta": self.token_efficiency_delta,
            "reuse_count": self.reuse_count,
            "discovered_at": self.discovered_at.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "description": self.description,
            "behavior_signature": {
                "action_sequence": self.behavior_signature.action_sequence,
                "resource_usage": self.behavior_signature.resource_usage,
                "performance_metrics": self.behavior_signature.performance_metrics,
                "context_factors": self.behavior_signature.context_factors,
                "hash": self.behavior_signature.to_hash()
            },
            "metadata": self.metadata
        }

class EmergentBehaviorMonitor:
    """
    Captures and catalogs novel agent strategies.
    Identifies strategies not explicitly programmed and evaluates their effectiveness.
    """
    
    def __init__(self, db_session=None, storage_backend: str = "memory"):
        self.db_session = db_session
        self.storage_backend = storage_backend
        self.known_patterns: Dict[str, EmergentPattern] = {}
        self.behavior_history: Dict[str, List[BehaviorSignature]] = {}
        self.pattern_catalog: List[EmergentPattern] = []
        self.observation_window = timedelta(minutes=30)
        self.significance_thresholds = {
            PatternSignificance.LOW: 0.1,
            PatternSignificance.MEDIUM: 0.3,
            PatternSignificance.HIGH: 0.6,
            PatternSignificance.CRITICAL: 0.9
        }
        self.logger = logging.getLogger(__name__)
    
    async def observe_agent_behavior(self, agent: 'FractalAgent', 
                                   action_context: Dict[str, Any]) -> Optional[EmergentPattern]:
        """
        Observe and record agent behavior for pattern detection.
        
        Args:
            agent: The agent being observed
            action_context: Context of the action being performed
            
        Returns:
            Discovered pattern if novel behavior detected, None otherwise
        """
        # Create behavior signature for this observation
        behavior_signature = await self._create_behavior_signature(agent, action_context)
        
        # Add to behavior history
        if agent.id not in self.behavior_history:
            self.behavior_history[agent.id] = []
        
        self.behavior_history[agent.id].append(behavior_signature)
        
        # Keep only recent observations within window
        cutoff_time = datetime.utcnow() - self.observation_window
        self.behavior_history[agent.id] = [
            sig for sig in self.behavior_history[agent.id] 
            if sig.timestamp > cutoff_time
        ]
        
        # Check for novel patterns
        pattern = await self._analyze_for_novel_patterns(agent, behavior_signature)
        
        if pattern:
            await self._catalog_pattern(pattern)
            self.logger.info(f"Discovered new pattern: {pattern.id} for agent {agent.id}")
        
        return pattern
    
    async def detect_novel_behaviors(self, agent: 'FractalAgent') -> List[EmergentPattern]:
        """
        Identify strategies not explicitly programmed.
        
        Args:
            agent: Agent to analyze for novel behaviors
            
        Returns:
            List of novel patterns discovered
        """
        if agent.id not in self.behavior_history:
            return []
        
        recent_behaviors = self.behavior_history[agent.id]
        novel_patterns = []
        
        # Analyze sequence patterns
        sequence_patterns = await self._detect_sequence_patterns(agent.id, recent_behaviors)
        novel_patterns.extend(sequence_patterns)
        
        # Analyze resource optimization patterns
        optimization_patterns = await self._detect_optimization_patterns(agent.id, recent_behaviors)
        novel_patterns.extend(optimization_patterns)
        
        # Analyze adaptation patterns
        adaptation_patterns = await self._detect_adaptation_patterns(agent.id, recent_behaviors)
        novel_patterns.extend(adaptation_patterns)
        
        # Analyze collaboration patterns
        collaboration_patterns = await self._detect_collaboration_patterns(agent.id, recent_behaviors)
        novel_patterns.extend(collaboration_patterns)
        
        # Filter out known patterns
        truly_novel = []
        for pattern in novel_patterns:
            pattern_hash = pattern.behavior_signature.to_hash()
            if not await self._is_known_pattern(pattern_hash):
                truly_novel.append(pattern)
                await self._catalog_pattern(pattern)
        
        return truly_novel
    
    async def _create_behavior_signature(self, agent: 'FractalAgent', 
                                       action_context: Dict[str, Any]) -> BehaviorSignature:
        """Create behavior signature from agent action."""
        # Extract action sequence
        action_sequence = action_context.get('actions', [])
        if isinstance(action_sequence, str):
            action_sequence = [action_sequence]
        
        # Extract resource usage
        resource_usage = {
            'tokens_used': action_context.get('tokens_consumed', 0),
            'time_elapsed': action_context.get('execution_time', 0),
            'memory_usage': action_context.get('memory_mb', 0),
            'api_calls': action_context.get('api_calls', 0)
        }
        
        # Extract performance metrics
        performance_metrics = {
            'success_rate': action_context.get('success_rate', 0.0),
            'efficiency_score': agent.token_budget.efficiency_score,
            'fitness_score': agent.fitness_score,
            'diversity_score': agent.diversity_score
        }
        
        # Extract context factors
        context_factors = {
            'agent_level': agent.level,
            'generation': agent.generation,
            'strategies_count': len(agent.genome.strategies),
            'emergent_patterns_count': len(agent.emergent_patterns),
            'parent_id': agent.parent_id,
            'task_type': action_context.get('task_type', 'unknown')
        }
        
        return BehaviorSignature(
            action_sequence=action_sequence,
            resource_usage=resource_usage,
            performance_metrics=performance_metrics,
            context_factors=context_factors
        )
    
    async def _analyze_for_novel_patterns(self, agent: 'FractalAgent', 
                                        behavior_signature: BehaviorSignature) -> Optional[EmergentPattern]:
        """Analyze behavior signature for novel patterns."""
        
        # Check if this exact pattern was seen before
        pattern_hash = behavior_signature.to_hash()
        if await self._is_known_pattern(pattern_hash):
            # Update reuse count for known pattern
            await self._update_pattern_reuse(pattern_hash)
            return None
        
        # Evaluate pattern significance
        significance = await self._evaluate_pattern_significance(agent, behavior_signature)
        
        if significance == PatternSignificance.LOW:
            return None  # Not significant enough to catalog
        
        # Determine pattern type
        pattern_type = await self._classify_pattern_type(behavior_signature)
        
        # Calculate effectiveness metrics
        effectiveness_score = await self._calculate_effectiveness(behavior_signature)
        token_efficiency_delta = await self._calculate_efficiency_delta(agent, behavior_signature)
        
        # Create pattern
        pattern = EmergentPattern(
            id=f"pattern_{pattern_hash}",
            agent_id=agent.id,
            pattern_type=pattern_type,
            significance=significance,
            behavior_signature=behavior_signature,
            effectiveness_score=effectiveness_score,
            token_efficiency_delta=token_efficiency_delta,
            description=await self._generate_pattern_description(pattern_type, behavior_signature)
        )
        
        return pattern
    
    async def _detect_sequence_patterns(self, agent_id: str, 
                                      behaviors: List[BehaviorSignature]) -> List[EmergentPattern]:
        """Detect patterns in action sequences."""
        patterns = []
        
        if len(behaviors) < 3:
            return patterns
        
        # Look for recurring action sequences
        sequences = [tuple(b.action_sequence) for b in behaviors if b.action_sequence]
        
        # Find sequences that appear multiple times
        sequence_counts = {}
        for seq in sequences:
            if len(seq) > 1:  # Only consider multi-step sequences
                sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
        
        # Identify significant recurring sequences
        for sequence, count in sequence_counts.items():
            if count >= 3:  # Sequence appeared at least 3 times
                # Calculate average performance for this sequence
                matching_behaviors = [b for b in behaviors if tuple(b.action_sequence) == sequence]
                avg_performance = sum(
                    b.performance_metrics.get('efficiency_score', 0) 
                    for b in matching_behaviors
                ) / len(matching_behaviors)
                
                if avg_performance > 0.5:  # Above average performance
                    pattern = EmergentPattern(
                        id=f"sequence_{hashlib.md5(str(sequence).encode()).hexdigest()[:8]}",
                        agent_id=agent_id,
                        pattern_type=PatternType.STRATEGY_INNOVATION,
                        significance=PatternSignificance.MEDIUM,
                        behavior_signature=matching_behaviors[-1],  # Use most recent
                        effectiveness_score=avg_performance,
                        token_efficiency_delta=0.1,
                        description=f"Recurring effective sequence: {' -> '.join(sequence[:3])}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_optimization_patterns(self, agent_id: str, 
                                          behaviors: List[BehaviorSignature]) -> List[EmergentPattern]:
        """Detect resource optimization patterns."""
        patterns = []
        
        if len(behaviors) < 5:
            return patterns
        
        # Analyze token efficiency trends
        efficiency_trend = [
            b.performance_metrics.get('efficiency_score', 0) / max(b.resource_usage.get('tokens_used', 1), 1)
            for b in behaviors
        ]
        
        # Look for improving efficiency
        if len(efficiency_trend) >= 5:
            recent_avg = sum(efficiency_trend[-3:]) / 3
            earlier_avg = sum(efficiency_trend[:3]) / 3
            
            if recent_avg > earlier_avg * 1.2:  # 20% improvement
                pattern = EmergentPattern(
                    id=f"optimization_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    agent_id=agent_id,
                    pattern_type=PatternType.RESOURCE_OPTIMIZATION,
                    significance=PatternSignificance.HIGH,
                    behavior_signature=behaviors[-1],
                    effectiveness_score=recent_avg,
                    token_efficiency_delta=recent_avg - earlier_avg,
                    description=f"Resource optimization: {((recent_avg/earlier_avg-1)*100):.1f}% efficiency improvement"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_adaptation_patterns(self, agent_id: str, 
                                        behaviors: List[BehaviorSignature]) -> List[EmergentPattern]:
        """Detect adaptation and learning patterns."""
        patterns = []
        
        if len(behaviors) < 4:
            return patterns
        
        # Look for context-aware adaptations
        context_adaptations = {}
        for behavior in behaviors:
            task_type = behavior.context_factors.get('task_type', 'unknown')
            if task_type not in context_adaptations:
                context_adaptations[task_type] = []
            context_adaptations[task_type].append(behavior)
        
        # Check for different strategies per context
        for task_type, task_behaviors in context_adaptations.items():
            if len(task_behaviors) >= 3:
                # Check if strategies evolved for this task type
                action_diversity = len(set(
                    tuple(b.action_sequence) for b in task_behaviors 
                    if b.action_sequence
                )) / len(task_behaviors)
                
                if action_diversity > 0.6:  # High diversity in approaches
                    avg_performance = sum(
                        b.performance_metrics.get('efficiency_score', 0) 
                        for b in task_behaviors
                    ) / len(task_behaviors)
                    
                    if avg_performance > 0.4:
                        pattern = EmergentPattern(
                            id=f"adaptation_{task_type}_{agent_id}",
                            agent_id=agent_id,
                            pattern_type=PatternType.ADAPTATION_STRATEGY,
                            significance=PatternSignificance.MEDIUM,
                            behavior_signature=task_behaviors[-1],
                            effectiveness_score=avg_performance,
                            token_efficiency_delta=0.05,
                            description=f"Context adaptation for {task_type}: {action_diversity:.1%} strategy diversity"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_collaboration_patterns(self, agent_id: str, 
                                           behaviors: List[BehaviorSignature]) -> List[EmergentPattern]:
        """Detect collaboration and social learning patterns."""
        patterns = []
        
        # Look for behaviors that indicate learning from others
        learning_indicators = []
        for behavior in behaviors:
            # Check for patterns that suggest external learning
            if 'learn' in ' '.join(behavior.action_sequence).lower():
                learning_indicators.append(behavior)
            elif behavior.performance_metrics.get('efficiency_score', 0) > 0.7:
                # High performance might indicate learned behavior
                learning_indicators.append(behavior)
        
        if len(learning_indicators) >= 2:
            avg_effectiveness = sum(
                b.performance_metrics.get('efficiency_score', 0) 
                for b in learning_indicators
            ) / len(learning_indicators)
            
            if avg_effectiveness > 0.6:
                pattern = EmergentPattern(
                    id=f"collaboration_{agent_id}_{len(learning_indicators)}",
                    agent_id=agent_id,
                    pattern_type=PatternType.COLLABORATION_PATTERN,
                    significance=PatternSignificance.MEDIUM,
                    behavior_signature=learning_indicators[-1],
                    effectiveness_score=avg_effectiveness,
                    token_efficiency_delta=0.1,
                    description=f"Collaborative learning pattern: {len(learning_indicators)} learning events"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _is_known_pattern(self, pattern_hash: str) -> bool:
        """Check if pattern is already known."""
        return pattern_hash in self.known_patterns
    
    async def _update_pattern_reuse(self, pattern_hash: str):
        """Update reuse count for known pattern."""
        if pattern_hash in self.known_patterns:
            self.known_patterns[pattern_hash].reuse_count += 1
            self.known_patterns[pattern_hash].last_observed = datetime.utcnow()
    
    async def _evaluate_pattern_significance(self, agent: 'FractalAgent', 
                                           behavior_signature: BehaviorSignature) -> PatternSignificance:
        """Evaluate the significance of a behavior pattern."""
        score = 0.0
        
        # Performance impact
        efficiency = behavior_signature.performance_metrics.get('efficiency_score', 0)
        score += efficiency * 0.4
        
        # Resource efficiency
        tokens_used = behavior_signature.resource_usage.get('tokens_used', 1)
        if tokens_used > 0:
            value_per_token = efficiency / tokens_used
            score += min(value_per_token, 1.0) * 0.3
        
        # Novelty (complexity of action sequence)
        action_complexity = len(set(behavior_signature.action_sequence)) / max(len(behavior_signature.action_sequence), 1)
        score += action_complexity * 0.2
        
        # Context appropriateness
        if behavior_signature.context_factors.get('generation', 0) > 5:
            score += 0.1  # Evolved behavior is more significant
        
        # Determine significance level
        if score >= self.significance_thresholds[PatternSignificance.CRITICAL]:
            return PatternSignificance.CRITICAL
        elif score >= self.significance_thresholds[PatternSignificance.HIGH]:
            return PatternSignificance.HIGH
        elif score >= self.significance_thresholds[PatternSignificance.MEDIUM]:
            return PatternSignificance.MEDIUM
        else:
            return PatternSignificance.LOW
    
    async def _classify_pattern_type(self, behavior_signature: BehaviorSignature) -> PatternType:
        """Classify the type of pattern based on behavior signature."""
        actions = ' '.join(behavior_signature.action_sequence).lower()
        
        # Keyword-based classification
        if any(word in actions for word in ['optimize', 'improve', 'efficient']):
            return PatternType.RESOURCE_OPTIMIZATION
        elif any(word in actions for word in ['learn', 'copy', 'adapt']):
            return PatternType.COLLABORATION_PATTERN
        elif any(word in actions for word in ['solve', 'approach', 'method']):
            return PatternType.PROBLEM_SOLVING_APPROACH
        elif any(word in actions for word in ['meta', 'abstract', 'level']):
            return PatternType.META_LEARNING
        elif behavior_signature.performance_metrics.get('efficiency_score', 0) > 0.8:
            return PatternType.EFFICIENCY_BREAKTHROUGH
        else:
            return PatternType.STRATEGY_INNOVATION
    
    async def _calculate_effectiveness(self, behavior_signature: BehaviorSignature) -> float:
        """Calculate effectiveness score for a behavior pattern."""
        efficiency = behavior_signature.performance_metrics.get('efficiency_score', 0)
        fitness = behavior_signature.performance_metrics.get('fitness_score', 0)
        
        # Weighted combination of metrics
        effectiveness = 0.6 * efficiency + 0.4 * fitness
        return min(max(effectiveness, 0.0), 1.0)
    
    async def _calculate_efficiency_delta(self, agent: 'FractalAgent', 
                                        behavior_signature: BehaviorSignature) -> float:
        """Calculate improvement in token efficiency."""
        current_efficiency = behavior_signature.performance_metrics.get('efficiency_score', 0)
        baseline_efficiency = agent.token_budget.efficiency_score
        
        return current_efficiency - baseline_efficiency
    
    async def _generate_pattern_description(self, pattern_type: PatternType, 
                                          behavior_signature: BehaviorSignature) -> str:
        """Generate human-readable description of the pattern."""
        actions = behavior_signature.action_sequence[:3]  # First 3 actions
        efficiency = behavior_signature.performance_metrics.get('efficiency_score', 0)
        
        descriptions = {
            PatternType.STRATEGY_INNOVATION: f"Novel strategy: {' -> '.join(actions)} (efficiency: {efficiency:.2f})",
            PatternType.EFFICIENCY_BREAKTHROUGH: f"Efficiency breakthrough: {efficiency:.2f} score achieved",
            PatternType.RESOURCE_OPTIMIZATION: f"Resource optimization: {' -> '.join(actions)}",
            PatternType.COLLABORATION_PATTERN: f"Collaborative behavior: {' -> '.join(actions)}",
            PatternType.PROBLEM_SOLVING_APPROACH: f"Problem-solving approach: {' -> '.join(actions)}",
            PatternType.ADAPTATION_STRATEGY: f"Adaptive strategy: context-aware behavior",
            PatternType.META_LEARNING: f"Meta-learning pattern: higher-order optimization"
        }
        
        return descriptions.get(pattern_type, f"Emergent pattern: {' -> '.join(actions)}")
    
    async def _catalog_pattern(self, pattern: EmergentPattern):
        """Store discovered pattern in catalog."""
        pattern_hash = pattern.behavior_signature.to_hash()
        self.known_patterns[pattern_hash] = pattern
        self.pattern_catalog.append(pattern)
        
        # Store in database if available
        if self.db_session:
            await self._store_pattern_in_db(pattern)
        
        self.logger.info(
            f"Cataloged pattern {pattern.id}: {pattern.pattern_type.value} "
            f"(significance: {pattern.significance.value}, effectiveness: {pattern.effectiveness_score:.3f})"
        )
    
    async def _store_pattern_in_db(self, pattern: EmergentPattern):
        """Store pattern in database using the database service client."""
        if not self.db_session:
            self.logger.warning("No database session available for pattern storage")
            return
        
        try:
            from indexagent.api.contracts.database_service import (
                PatternDiscoveryRequest, 
                create_database_service_client
            )
            
            # Create database client
            db_client = create_database_service_client(
                database_url=os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/indexagent"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
            )
            
            # Initialize if not already done
            await db_client.initialize()
            
            # Create pattern discovery request
            request = PatternDiscoveryRequest(
                pattern_hash=pattern.behavior_signature.to_hash(),
                pattern_type=pattern.pattern_type.value,
                description=pattern.description or f"Pattern {pattern.id}: {pattern.pattern_type.value}",
                pattern_sequence=pattern.behavior_signature.action_sequence,
                pattern_context={
                    "resource_usage": pattern.behavior_signature.resource_usage,
                    "performance_metrics": pattern.behavior_signature.performance_metrics,
                    "context_factors": pattern.behavior_signature.context_factors
                },
                effectiveness_score=pattern.effectiveness_score,
                confidence_score=pattern.effectiveness_score,  # Using effectiveness as confidence proxy
                discovered_by_agent_id=pattern.agent_id,
                agent_ids=[pattern.agent_id],
                metadata={
                    **pattern.metadata,
                    "significance": pattern.significance.value,
                    "token_efficiency_delta": pattern.token_efficiency_delta,
                    "reuse_count": pattern.reuse_count,
                    "discovered_at": pattern.discovered_at.isoformat(),
                    "last_observed": pattern.last_observed.isoformat()
                }
            )
            
            # Store pattern
            response = await db_client.discover_pattern(request)
            
            if response.success:
                self.logger.info(f"Pattern {pattern.id} stored in database: {response.data.get('pattern_id')}")
            else:
                self.logger.error(f"Failed to store pattern {pattern.id}: {response.message}")
            
            # Clean up
            await db_client.close()
            
        except Exception as e:
            self.logger.error(f"Error storing pattern {pattern.id} in database: {e}")
            # Don't raise - pattern is already cataloged in memory
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns."""
        total_patterns = len(self.pattern_catalog)
        
        if total_patterns == 0:
            return {"total_patterns": 0}
        
        # Group by type
        type_counts = {}
        significance_counts = {}
        effectiveness_scores = []
        
        for pattern in self.pattern_catalog:
            # Count by type
            pattern_type = pattern.pattern_type.value
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
            
            # Count by significance
            significance = pattern.significance.value
            significance_counts[significance] = significance_counts.get(significance, 0) + 1
            
            # Collect effectiveness scores
            effectiveness_scores.append(pattern.effectiveness_score)
        
        return {
            "total_patterns": total_patterns,
            "pattern_types": type_counts,
            "significance_distribution": significance_counts,
            "average_effectiveness": sum(effectiveness_scores) / len(effectiveness_scores),
            "top_patterns": [
                {
                    "id": p.id,
                    "type": p.pattern_type.value,
                    "effectiveness": p.effectiveness_score,
                    "reuse_count": p.reuse_count
                }
                for p in sorted(self.pattern_catalog, key=lambda x: x.effectiveness_score, reverse=True)[:5]
            ]
        }