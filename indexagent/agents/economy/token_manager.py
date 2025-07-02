"""
Core economic controller for the DEAN system.
Manages global token budget with Prometheus metrics per specifications.
Per specifications: IndexAgent/indexagent/agents/economy/token_manager.py
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json

# Optional Prometheus client for metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock prometheus classes for graceful degradation
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def set(self, *args, **kwargs): pass
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

@dataclass
class TokenBudget:
    """Represents a token allocation for an agent."""
    total: int
    used: int = 0
    reserved: int = 0
    
    @property
    def available(self) -> int:
        """Tokens available for immediate use."""
        return self.total - self.used - self.reserved
    
    @property
    def utilization_rate(self) -> float:
        """Percentage of tokens used."""
        return self.used / self.total if self.total > 0 else 0.0
    
    def can_afford(self, cost: int) -> bool:
        """Check if budget can cover a cost."""
        return self.available >= cost
    
    def consume(self, amount: int) -> None:
        """Consume tokens from the budget."""
        if not self.can_afford(amount):
            raise ValueError(f"Insufficient tokens: need {amount}, have {self.available}")
        self.used += amount
    
    def reserve(self, amount: int) -> None:
        """Reserve tokens for future use."""
        if self.available < amount:
            raise ValueError(f"Cannot reserve {amount} tokens, only {self.available} available")
        self.reserved += amount
    
    def release_reservation(self, amount: int) -> None:
        """Release reserved tokens back to available pool."""
        self.reserved = max(0, self.reserved - amount)

@dataclass  
class TokenAllocation:
    """Represents token allocation for an agent with performance tracking."""
    agent_id: str
    allocated: int
    consumed: int
    reserved: int
    efficiency_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def remaining(self) -> int:
        return self.allocated - self.consumed - self.reserved
    
    @property
    def utilization_rate(self) -> float:
        return (self.consumed / self.allocated) if self.allocated > 0 else 0.0

@dataclass
class PerformanceMetrics:
    """Performance metrics for token efficiency tracking."""
    tokens_per_task: float = 0.0
    value_per_token: float = 0.0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    efficiency_trend: List[float] = field(default_factory=list)

class TokenEconomyManager:
    """
    Core economic controller - implement this before any agent logic.
    
    Manages global token budget, allocates tokens to agents based on
    historical performance, and enforces hard limits to prevent
    runaway consumption.
    """
    
    def __init__(self, global_budget: int, safety_margin: float = 0.1, prometheus_enabled: bool = True):
        """
        Initialize the token economy.
        
        Args:
            global_budget: Total tokens available for all agents
            safety_margin: Fraction of budget to reserve for emergencies
            prometheus_enabled: Enable Prometheus metrics collection
        """
        self.global_budget = global_budget
        self.safety_margin = safety_margin
        self.safety_reserve = int(global_budget * safety_margin)
        self.available_budget = global_budget - self.safety_reserve
        self.prometheus_enabled = prometheus_enabled
        
        self.agent_allocations: Dict[str, TokenBudget] = {}
        self.efficiency_history: Dict[str, List[float]] = defaultdict(list)
        self.consumption_log: List[Dict] = []
        self.allocation_history: List[Dict] = []
        
        # Enhanced metrics tracking
        self.efficiency_metrics: Dict[str, PerformanceMetrics] = {}
        self.budget_history: List[Dict[str, Any]] = []
        
        self._start_time = time.time()
        self._terminated_agents: Set[str] = set()
        
        # Configuration
        self.base_allocation = 1000  # Base tokens per agent
        self.max_allocation = 8000   # Maximum tokens per agent
        self.efficiency_multiplier = 2.0  # Max efficiency boost
        self.decay_rate = 0.95  # Budget decay for long-running agents
        self.reserve_ratio = 0.1  # Reserve 10% of global budget
        
        # Initialize Prometheus metrics
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        elif self.prometheus_enabled and not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus metrics requested but prometheus_client not available")
        
        logger.info(f"Initialized TokenEconomyManager with budget: {global_budget}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for monitoring."""
        self.tokens_allocated = Counter(
            'dean_tokens_allocated_total', 
            'Total tokens allocated to agents',
            ['agent_id', 'allocation_type']
        )
        
        self.tokens_consumed = Counter(
            'dean_tokens_consumed_total',
            'Total tokens consumed by agents',
            ['agent_id', 'task_type']
        )
        
        self.efficiency_gauge = Gauge(
            'dean_agent_efficiency',
            'Agent efficiency score (value per token)',
            ['agent_id']
        )
        
        self.budget_utilization = Gauge(
            'dean_budget_utilization',
            'Global budget utilization rate'
        )
        
        self.allocation_histogram = Histogram(
            'dean_allocation_size_tokens',
            'Distribution of token allocations',
            buckets=[100, 500, 1000, 2000, 4000, 8000, 16000]
        )
        
        self.efficiency_histogram = Histogram(
            'dean_token_efficiency',
            'Token efficiency distribution',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.system_info = Info(
            'dean_token_economy_info',
            'Token economy system information'
        )
        
        # Set system info
        self.system_info.info({
            'global_budget': str(self.global_budget),
            'base_allocation': str(self.base_allocation),
            'max_allocation': str(self.max_allocation),
            'decay_rate': str(self.decay_rate)
        })
    
    async def allocate_tokens(self, agent_id: str, base_amount: Optional[int] = None,
                            task_type: str = "general") -> int:
        """
        Dynamic allocation based on historical efficiency.
        
        Args:
            agent_id: Unique identifier for the agent
            base_amount: Base allocation amount (uses default if None)
            task_type: Type of task for metrics tracking
            
        Returns:
            Number of tokens allocated
        """
        if base_amount is None:
            base_amount = self.base_allocation
        
        # Get agent's historical efficiency
        efficiency = await self._get_agent_efficiency(agent_id)
        
        # Calculate performance multiplier (1.0 to efficiency_multiplier)
        performance_multiplier = min(self.efficiency_multiplier, max(1.0, efficiency))
        
        # Apply performance multiplier
        requested_allocation = int(base_amount * performance_multiplier)
        
        # Apply global budget constraints
        available_budget = await self._get_available_budget()
        allocation = min(requested_allocation, available_budget)
        
        # Check if agent already has allocation
        if agent_id in self.agent_allocations:
            # Add to existing allocation
            self.agent_allocations[agent_id].allocated += allocation
        else:
            # Create new allocation
            self.agent_allocations[agent_id] = TokenAllocation(
                agent_id=agent_id,
                allocated=allocation,
                consumed=0,
                reserved=0,
                efficiency_score=efficiency
            )
        
        # Update metrics
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            self.tokens_allocated.labels(
                agent_id=agent_id, 
                allocation_type=task_type
            ).inc(allocation)
            self.allocation_histogram.observe(allocation)
            self._update_budget_utilization()
        
        # Log allocation
        logger.info(
            f"Allocated {allocation} tokens to agent {agent_id} "
            f"(efficiency: {efficiency:.3f}, multiplier: {performance_multiplier:.2f})"
        )
        
        return allocation
    
    async def consume_tokens(self, agent_id: str, amount: int, 
                           task_type: str = "general", 
                           value_generated: float = 0.0) -> bool:
        """
        Consume tokens for agent operation.
        
        Args:
            agent_id: Agent consuming tokens
            amount: Number of tokens to consume
            task_type: Type of task for metrics
            value_generated: Value/output generated by the operation
            
        Returns:
            True if consumption successful, False if insufficient tokens
        """
        if agent_id not in self.agent_allocations:
            logger.warning(f"Agent {agent_id} has no token allocation")
            return False
        
        allocation = self.agent_allocations[agent_id]
        
        # Check if agent has enough tokens
        if allocation.remaining < amount:
            logger.warning(
                f"Agent {agent_id} attempted to consume {amount} tokens "
                f"but only has {allocation.remaining} remaining"
            )
            return False
        
        # Consume tokens
        allocation.consumed += amount
        allocation.last_updated = datetime.utcnow()
        
        # Update efficiency metrics
        if value_generated > 0:
            await self._update_efficiency_metrics(agent_id, amount, value_generated)
        
        # Update Prometheus metrics
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            self.tokens_consumed.labels(
                agent_id=agent_id,
                task_type=task_type
            ).inc(amount)
            self._update_budget_utilization()
        
        logger.debug(
            f"Agent {agent_id} consumed {amount} tokens "
            f"({allocation.remaining} remaining)"
        )
        
        return True
    
    async def reserve_tokens(self, agent_id: str, amount: int) -> bool:
        """Reserve tokens for future operations."""
        if agent_id not in self.agent_allocations:
            return False
        
        allocation = self.agent_allocations[agent_id]
        
        if allocation.remaining < amount:
            return False
        
        allocation.reserved += amount
        allocation.last_updated = datetime.utcnow()
        
        return True
    
    async def release_reservation(self, agent_id: str, amount: int) -> bool:
        """Release previously reserved tokens."""
        if agent_id not in self.agent_allocations:
            return False
        
        allocation = self.agent_allocations[agent_id]
        release_amount = min(amount, allocation.reserved)
        allocation.reserved -= release_amount
        allocation.last_updated = datetime.utcnow()
        
        return True
    
    async def get_agent_budget_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed budget status for an agent."""
        if agent_id not in self.agent_allocations:
            return None
        
        allocation = self.agent_allocations[agent_id]
        metrics = self.efficiency_metrics.get(agent_id, PerformanceMetrics())
        
        return {
            "agent_id": agent_id,
            "allocated": allocation.allocated,
            "consumed": allocation.consumed,
            "reserved": allocation.reserved,
            "remaining": allocation.remaining,
            "utilization_rate": allocation.utilization_rate,
            "efficiency_score": allocation.efficiency_score,
            "last_updated": allocation.last_updated.isoformat(),
            "performance_metrics": {
                "tokens_per_task": metrics.tokens_per_task,
                "value_per_token": metrics.value_per_token,
                "completion_rate": metrics.completion_rate,
                "error_rate": metrics.error_rate,
                "efficiency_trend": metrics.efficiency_trend[-10:]  # Last 10 measurements
            }
        }
    
    async def get_global_budget_status(self) -> Dict[str, Any]:
        """Get global budget and allocation status."""
        total_allocated = sum(alloc.allocated for alloc in self.agent_allocations.values())
        total_consumed = sum(alloc.consumed for alloc in self.agent_allocations.values())
        total_reserved = sum(alloc.reserved for alloc in self.agent_allocations.values())
        
        utilization_rate = total_consumed / self.global_budget if self.global_budget > 0 else 0.0
        allocation_rate = total_allocated / self.global_budget if self.global_budget > 0 else 0.0
        
        return {
            "global_budget": self.global_budget,
            "total_allocated": total_allocated,
            "total_consumed": total_consumed,
            "total_reserved": total_reserved,
            "available": self.global_budget - total_allocated,
            "utilization_rate": utilization_rate,
            "allocation_rate": allocation_rate,
            "active_agents": len(self.agent_allocations),
            "average_efficiency": await self._calculate_average_efficiency()
        }
    
    async def apply_budget_decay(self, agent_id: str) -> float:
        """
        Apply budget decay for long-running agents.
        
        Returns:
            Decay factor applied (0.0 to 1.0)
        """
        if agent_id not in self.agent_allocations:
            return 1.0
        
        allocation = self.agent_allocations[agent_id]
        time_since_update = datetime.utcnow() - allocation.last_updated
        
        # Apply decay based on time inactive (hours)
        hours_inactive = time_since_update.total_seconds() / 3600
        
        if hours_inactive > 1:  # Start decay after 1 hour
            decay_factor = self.decay_rate ** hours_inactive
            original_allocation = allocation.allocated
            allocation.allocated = int(allocation.allocated * decay_factor)
            
            logger.info(
                f"Applied budget decay to agent {agent_id}: "
                f"{original_allocation} -> {allocation.allocated} "
                f"(factor: {decay_factor:.3f})"
            )
            
            return decay_factor
        
        return 1.0
    
    async def rebalance_allocations(self) -> Dict[str, Any]:
        """Rebalance token allocations based on performance."""
        if not self.agent_allocations:
            return {"rebalanced_agents": 0, "total_adjustments": 0}
        
        # Calculate efficiency scores for all agents
        efficiency_scores = {}
        for agent_id in self.agent_allocations:
            efficiency_scores[agent_id] = await self._get_agent_efficiency(agent_id)
        
        # Sort agents by efficiency
        sorted_agents = sorted(
            efficiency_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        rebalanced_count = 0
        total_adjustments = 0
        
        # Redistribute from low performers to high performers
        low_performers = [aid for aid, eff in sorted_agents if eff < 0.5]
        high_performers = [aid for aid, eff in sorted_agents if eff > 1.5]
        
        for low_agent in low_performers:
            if not high_performers:
                break
                
            # Take 20% from low performer
            low_allocation = self.agent_allocations[low_agent]
            reduction = int(low_allocation.allocated * 0.2)
            
            if reduction > 0:
                low_allocation.allocated -= reduction
                
                # Give to best high performer
                high_agent = high_performers[0]
                high_allocation = self.agent_allocations[high_agent]
                high_allocation.allocated += reduction
                
                rebalanced_count += 2
                total_adjustments += reduction
                
                logger.info(
                    f"Rebalanced {reduction} tokens from {low_agent} "
                    f"(efficiency: {efficiency_scores[low_agent]:.3f}) "
                    f"to {high_agent} (efficiency: {efficiency_scores[high_agent]:.3f})"
                )
        
        return {
            "rebalanced_agents": rebalanced_count,
            "total_adjustments": total_adjustments,
            "efficiency_scores": efficiency_scores
        }
    
    async def _get_available_budget(self) -> int:
        """Calculate available budget for new allocations."""
        total_allocated = sum(alloc.allocated for alloc in self.agent_allocations.values())
        reserved_budget = int(self.global_budget * self.reserve_ratio)
        return max(0, self.global_budget - total_allocated - reserved_budget)
    
    async def _get_agent_efficiency(self, agent_id: str) -> float:
        """Get agent's efficiency score."""
        if agent_id in self.efficiency_metrics:
            metrics = self.efficiency_metrics[agent_id]
            return metrics.value_per_token
        return 1.0  # Default efficiency for new agents
    
    async def _update_efficiency_metrics(self, agent_id: str, tokens_consumed: int, 
                                       value_generated: float):
        """Update efficiency metrics for an agent."""
        if agent_id not in self.efficiency_metrics:
            self.efficiency_metrics[agent_id] = PerformanceMetrics()
        
        metrics = self.efficiency_metrics[agent_id]
        
        # Update efficiency score (value per token)
        if tokens_consumed > 0:
            current_efficiency = value_generated / tokens_consumed
            
            # Update running average
            if metrics.value_per_token == 0:
                metrics.value_per_token = current_efficiency
            else:
                # Exponential moving average
                alpha = 0.3
                metrics.value_per_token = (alpha * current_efficiency + 
                                         (1 - alpha) * metrics.value_per_token)
            
            # Track efficiency trend
            metrics.efficiency_trend.append(current_efficiency)
            if len(metrics.efficiency_trend) > 20:  # Keep last 20 measurements
                metrics.efficiency_trend = metrics.efficiency_trend[-20:]
        
        # Update allocation efficiency score
        if agent_id in self.agent_allocations:
            self.agent_allocations[agent_id].efficiency_score = metrics.value_per_token
        
        # Update Prometheus metrics
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            self.efficiency_gauge.labels(agent_id=agent_id).set(metrics.value_per_token)
            self.efficiency_histogram.observe(metrics.value_per_token)
    
    async def _calculate_average_efficiency(self) -> float:
        """Calculate average efficiency across all agents."""
        if not self.efficiency_metrics:
            return 1.0
        
        efficiencies = [m.value_per_token for m in self.efficiency_metrics.values()]
        return sum(efficiencies) / len(efficiencies)
    
    def _update_budget_utilization(self):
        """Update global budget utilization metric."""
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            total_consumed = sum(alloc.consumed for alloc in self.agent_allocations.values())
            utilization = total_consumed / self.global_budget if self.global_budget > 0 else 0.0
            self.budget_utilization.set(utilization)
    
    async def cleanup_inactive_agents(self, inactive_threshold_hours: int = 24) -> List[str]:
        """Remove allocations for agents inactive beyond threshold."""
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_threshold_hours)
        inactive_agents = []
        
        for agent_id, allocation in list(self.agent_allocations.items()):
            if allocation.last_updated < cutoff_time:
                # Return unused allocation to global budget
                unused_tokens = allocation.remaining
                
                del self.agent_allocations[agent_id]
                if agent_id in self.efficiency_metrics:
                    del self.efficiency_metrics[agent_id]
                
                inactive_agents.append(agent_id)
                
                logger.info(
                    f"Cleaned up inactive agent {agent_id} "
                    f"(returned {unused_tokens} tokens to global budget)"
                )
        
        return inactive_agents
    
    def allocate_tokens_basic(self, agent_id: str, historical_efficiency: Optional[float] = None) -> int:
        """
        Dynamic allocation based on past performance with safety controls.
        
        Uses a performance-weighted algorithm that rewards efficient agents
        with larger budgets while ensuring minimum viable allocations for
        new or struggling agents.
        """
        if agent_id in self._terminated_agents:
            logger.warning(f"Attempted to allocate tokens to terminated agent: {agent_id}")
            return 0
        
        # Calculate remaining budget
        total_allocated = sum(
            b.total for aid, b in self.agent_allocations.items() 
            if aid not in self._terminated_agents
        )
        remaining_budget = self.available_budget - total_allocated
        
        # Calculate base allocation
        active_agents = len([a for a in self.agent_allocations if a not in self._terminated_agents])
        
        # For initial agents, distribute evenly with future reserve
        if active_agents < 5:
            # Assume we might have up to 10 agents total
            base_allocation = self.available_budget // 10
        else:
            # For later agents, use remaining budget
            base_allocation = remaining_budget // 2  # Save half for future agents
        
        # Apply performance multiplier
        if historical_efficiency is not None and historical_efficiency > 0:
            # Agents with better efficiency get up to 3x base allocation
            if historical_efficiency < 0.5:
                performance_multiplier = 0.5 + historical_efficiency
            elif historical_efficiency < 1.0:
                performance_multiplier = 1.0 + (historical_efficiency - 0.5) * 2
            else:
                performance_multiplier = 2.0 + min(1.0, (historical_efficiency - 1.0))
        else:
            # New agents get 80% of base to encourage competition
            performance_multiplier = 0.8
        
        allocation = int(base_allocation * performance_multiplier)
        
        # Ensure minimum viable allocation (1% of available budget)
        min_allocation = max(100, int(self.available_budget * 0.01))
        allocation = max(allocation, min_allocation)
        
        # Ensure allocation fits within remaining budget
        if allocation > remaining_budget:
            allocation = min(allocation, remaining_budget)
        
        if allocation > 0:
            self.agent_allocations[agent_id] = TokenBudget(total=allocation)
            self.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'allocation': allocation,
                'efficiency': historical_efficiency,
                'multiplier': performance_multiplier
            })
            logger.info(f"Allocated {allocation} tokens to agent {agent_id}")
        
        return allocation
    
    def track_consumption_basic(self, agent_id: str, tokens_used: int, value_generated: float) -> None:
        """
        Track token consumption and value generation for an agent.
        
        This data drives future allocation decisions and identifies
        agents that should be terminated for inefficiency.
        """
        if agent_id not in self.agent_allocations:
            logger.error(f"Unknown agent: {agent_id}")
            return
        
        budget = self.agent_allocations[agent_id]
        
        try:
            budget.consume(tokens_used)
        except ValueError as e:
            logger.error(f"Agent {agent_id} exceeded budget: {e}")
            self.terminate_agent(agent_id, reason="budget_exceeded")
            return
        
        # Calculate efficiency
        efficiency = value_generated / tokens_used if tokens_used > 0 else 0.0
        self.efficiency_history[agent_id].append(efficiency)
        
        # Log consumption
        self.consumption_log.append({
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'tokens_used': tokens_used,
            'value_generated': value_generated,
            'efficiency': efficiency,
            'budget_remaining': budget.available
        })
        
        # Check for inefficiency termination
        if len(self.efficiency_history[agent_id]) >= 3:
            recent_efficiency = self.efficiency_history[agent_id][-3:]
            avg_efficiency = sum(recent_efficiency) / len(recent_efficiency)
            
            # Terminate if consistently inefficient
            if avg_efficiency < 0.1 and budget.utilization_rate > 0.5:
                self.terminate_agent(agent_id, reason="persistent_inefficiency")
    
    def terminate_agent(self, agent_id: str, reason: str) -> None:
        """
        Terminate an agent and reclaim its unused tokens.
        
        Terminated agents cannot receive new allocations and their
        unused budget is returned to the global pool.
        """
        if agent_id in self._terminated_agents:
            return
        
        logger.warning(f"Terminating agent {agent_id}: {reason}")
        self._terminated_agents.add(agent_id)
        
        # Reclaim unused tokens
        if agent_id in self.agent_allocations:
            budget = self.agent_allocations[agent_id]
            reclaimed = budget.available
            self.available_budget += reclaimed
            logger.info(f"Reclaimed {reclaimed} tokens from {agent_id}")
    
    def get_agent_metrics_basic(self, agent_id: str) -> Dict:
        """Get comprehensive metrics for an agent."""
        if agent_id not in self.agent_allocations:
            return {}
        
        budget = self.agent_allocations[agent_id]
        efficiency_hist = self.efficiency_history.get(agent_id, [])
        
        return {
            'agent_id': agent_id,
            'total_budget': budget.total,
            'tokens_used': budget.used,
            'tokens_available': budget.available,
            'utilization_rate': budget.utilization_rate,
            'efficiency_history': efficiency_hist,
            'average_efficiency': sum(efficiency_hist) / len(efficiency_hist) if efficiency_hist else 0.0,
            'is_terminated': agent_id in self._terminated_agents
        }
    
    def get_global_metrics_basic(self) -> Dict:
        """Get system-wide economic metrics."""
        total_allocated = sum(b.total for b in self.agent_allocations.values())
        total_used = sum(b.used for b in self.agent_allocations.values())
        
        all_efficiencies = []
        for hist in self.efficiency_history.values():
            all_efficiencies.extend(hist)
        
        return {
            'global_budget': self.global_budget,
            'safety_reserve': self.safety_reserve,
            'total_allocated': total_allocated,
            'total_used': total_used,
            'available_budget': self.available_budget,
            'active_agents': len([a for a in self.agent_allocations if a not in self._terminated_agents]),
            'terminated_agents': len(self._terminated_agents),
            'average_efficiency': sum(all_efficiencies) / len(all_efficiencies) if all_efficiencies else 0.0,
            'runtime_seconds': time.time() - self._start_time
        }
    
    def export_metrics_basic(self, filepath: str) -> None:
        """Export all metrics to a JSON file for analysis."""
        metrics = {
            'global': self.get_global_metrics_basic(),
            'agents': {aid: self.get_agent_metrics_basic(aid) for aid in self.agent_allocations},
            'consumption_log': self.consumption_log,
            'allocation_history': self.allocation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath}")
    
    def calculate_roi(self, tokens_spent: int, value_generated: float) -> float:
        """Calculate return on investment for token spending."""
        return value_generated / tokens_spent if tokens_spent > 0 else 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "token_economy": {
                "global_budget": self.global_budget,
                "active_agents": len(self.agent_allocations),
                "total_allocated": sum(a.allocated for a in self.agent_allocations.values()),
                "total_consumed": sum(a.consumed for a in self.agent_allocations.values()),
                "average_efficiency": asyncio.run(self._calculate_average_efficiency()) if hasattr(self, '_calculate_average_efficiency') else 0.0
            },
            "top_performers": sorted(
                [(aid, m.value_per_token) for aid, m in self.efficiency_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "budget_utilization": {
                "allocation_rate": sum(a.allocated for a in self.agent_allocations.values()) / self.global_budget,
                "consumption_rate": sum(a.consumed for a in self.agent_allocations.values()) / self.global_budget,
                "efficiency_distribution": {
                    "high": len([m for m in self.efficiency_metrics.values() if m.value_per_token > 2.0]),
                    "medium": len([m for m in self.efficiency_metrics.values() if 0.5 <= m.value_per_token <= 2.0]),
                    "low": len([m for m in self.efficiency_metrics.values() if m.value_per_token < 0.5])
                }
            }
        }