"""
Base agent implementation for DEAN system.
Per specifications: IndexAgent/indexagent/agents/base_agent.py
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import uuid4, UUID
from datetime import datetime
import asyncio

class AgentGenome(BaseModel):
    """Genetic representation of agent capabilities."""
    traits: Dict[str, float] = Field(default_factory=dict)
    strategies: List[str] = Field(default_factory=list)
    mutation_rate: float = 0.1
    crossover_points: List[int] = Field(default_factory=list)

class TokenBudget(BaseModel):
    """Economic constraint system for agents."""
    total: int
    used: int = 0
    reserved: int = 0
    efficiency_score: float = 0.0
    
    @property
    def remaining(self) -> int:
        return self.total - self.used - self.reserved
    
    async def can_afford(self, estimated_cost: int) -> bool:
        """Check if agent can afford operation."""
        return self.remaining >= estimated_cost
    
    async def consume(self, cost: int) -> bool:
        """Consume tokens for operation."""
        if await self.can_afford(cost):
            self.used += cost
            return True
        return False

class FractalAgent(BaseModel):
    """
    Base agent class with economic and diversity constraints.
    Implements cellular automata evolution rules and emergent behavior capture.
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    genome: AgentGenome
    level: int = 0
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    token_budget: TokenBudget
    diversity_score: float = 0.0
    emergent_patterns: List[str] = Field(default_factory=list)
    worktree_path: Optional[str] = None
    fitness_score: float = 0.0
    generation: int = 0
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    async def evolve(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolution with economic constraints and diversity maintenance.
        Implements cellular automata rules for strategic evolution.
        """
        # Check token budget before any operation
        if not await self.token_budget.can_afford(estimated_cost=100):
            return {"status": "budget_exhausted", "reason": "Insufficient tokens"}
        
        # Consume tokens for evolution cycle
        await self.token_budget.consume(100)
        
        # Maintain diversity from population
        if 'population' in environment:
            await self._maintain_diversity(environment['population'])
        
        # Execute evolution cycle using cellular automata rules
        result = await self._execute_evolution(environment)
        
        # Track emergent patterns
        patterns = await self._detect_patterns(result)
        self.emergent_patterns.extend(patterns)
        
        # Update fitness and generation
        self.generation += 1
        self.updated_at = datetime.utcnow()
        
        return result
    
    async def _maintain_diversity(self, population: List['FractalAgent']):
        """Ensure genetic diversity within population."""
        # Calculate diversity score based on genome similarity
        if len(population) <= 1:
            self.diversity_score = 1.0
            return
        
        similarity_scores = []
        for other in population:
            if other.id != self.id:
                similarity = await self._calculate_genome_similarity(other.genome)
                similarity_scores.append(similarity)
        
        # Diversity score is inverse of average similarity
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            self.diversity_score = 1.0 - avg_similarity
        else:
            self.diversity_score = 1.0
    
    async def _calculate_genome_similarity(self, other_genome: AgentGenome) -> float:
        """Calculate similarity between two genomes."""
        # Simple trait-based similarity calculation
        common_traits = set(self.genome.traits.keys()) & set(other_genome.traits.keys())
        if not common_traits:
            return 0.0
        
        trait_diffs = []
        for trait in common_traits:
            diff = abs(self.genome.traits[trait] - other_genome.traits[trait])
            trait_diffs.append(diff)
        
        # Average difference (0 = identical, 1 = completely different)
        avg_diff = sum(trait_diffs) / len(trait_diffs) if trait_diffs else 0.0
        return 1.0 - avg_diff  # Convert to similarity
    
    async def _execute_evolution(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cellular automata evolution rules."""
        evolution_result = {
            "rule_applied": None,
            "strategy_changes": [],
            "performance_delta": 0.0,
            "new_patterns": []
        }
        
        # Apply cellular automata rules based on environment
        fitness_threshold = environment.get('fitness_threshold', 0.7)
        population_diversity = environment.get('population_diversity', 0.5)
        
        if self.fitness_score < fitness_threshold:
            # Rule 110: Create improved neighbors when detecting imperfections
            evolution_result["rule_applied"] = "rule_110"
            await self._apply_rule_110(evolution_result)
        elif population_diversity < 0.3:
            # Rule 30: Fork into parallel strategies when bottlenecked
            evolution_result["rule_applied"] = "rule_30"
            await self._apply_rule_30(evolution_result)
        elif len(self.emergent_patterns) > 5:
            # Rule 90: Abstract patterns into reusable components
            evolution_result["rule_applied"] = "rule_90"
            await self._apply_rule_90(evolution_result)
        else:
            # Rule 184: Learn from higher-performing neighbors
            evolution_result["rule_applied"] = "rule_184"
            await self._apply_rule_184(evolution_result, environment)
        
        return evolution_result
    
    async def _apply_rule_110(self, result: Dict[str, Any]):
        """Rule 110: Create improved neighbors when detecting imperfections."""
        # Mutate genome to improve performance
        for trait, value in self.genome.traits.items():
            mutation = (self.genome.mutation_rate * 2 - self.genome.mutation_rate) * 0.1
            self.genome.traits[trait] = max(0.0, min(1.0, value + mutation))
            result["strategy_changes"].append(f"Mutated {trait}: {value:.3f} -> {self.genome.traits[trait]:.3f}")
        
        result["performance_delta"] = 0.05  # Expected improvement
    
    async def _apply_rule_30(self, result: Dict[str, Any]):
        """Rule 30: Fork into parallel strategies when bottlenecked."""
        # Add new strategies to genome
        new_strategies = [f"parallel_strategy_{len(self.genome.strategies)}", "diversity_boost"]
        self.genome.strategies.extend(new_strategies)
        result["strategy_changes"] = [f"Added strategy: {s}" for s in new_strategies]
        result["performance_delta"] = 0.02
    
    async def _apply_rule_90(self, result: Dict[str, Any]):
        """Rule 90: Abstract patterns into reusable components."""
        # Create pattern abstractions from emergent behaviors
        if self.emergent_patterns:
            abstracted = f"pattern_abstraction_{len(self.genome.strategies)}"
            self.genome.strategies.append(abstracted)
            result["strategy_changes"].append(f"Abstracted pattern: {abstracted}")
            result["new_patterns"] = [abstracted]
            result["performance_delta"] = 0.03
    
    async def _apply_rule_184(self, result: Dict[str, Any], environment: Dict[str, Any]):
        """Rule 184: Learn from higher-performing neighbors."""
        population = environment.get('population', [])
        high_performers = [a for a in population if a.fitness_score > self.fitness_score]
        
        if high_performers:
            # Copy strategies from best performer
            best_performer = max(high_performers, key=lambda a: a.fitness_score)
            learned_strategies = best_performer.genome.strategies[-2:]  # Copy last 2 strategies
            self.genome.strategies.extend(learned_strategies)
            result["strategy_changes"] = [f"Learned from {best_performer.id}: {s}" for s in learned_strategies]
            result["performance_delta"] = 0.04
    
    async def _detect_patterns(self, evolution_result: Dict[str, Any]) -> List[str]:
        """Detect emergent patterns from evolution results."""
        patterns = []
        
        # Pattern detection based on strategy changes
        if evolution_result.get("performance_delta", 0) > 0.03:
            patterns.append(f"high_performance_evolution_{self.generation}")
        
        if len(evolution_result.get("strategy_changes", [])) > 2:
            patterns.append(f"multi_strategy_adaptation_{self.generation}")
        
        if evolution_result.get("rule_applied") == "rule_110":
            patterns.append(f"improvement_seeking_{self.generation}")
        
        return patterns
    
    async def track_token_efficiency(self) -> float:
        """Calculate value generated per token consumed."""
        if self.token_budget.used == 0:
            return 0.0
        
        # Simple efficiency calculation: fitness per token
        efficiency = self.fitness_score / self.token_budget.used
        self.token_budget.efficiency_score = efficiency
        return efficiency
    
    async def create_child(self, partner: Optional['FractalAgent'] = None) -> 'FractalAgent':
        """Create child agent through reproduction."""
        if not await self.token_budget.can_afford(200):
            raise ValueError("Insufficient tokens for reproduction")
        
        await self.token_budget.consume(200)
        
        # Create child genome through crossover
        child_genome = AgentGenome()
        
        if partner:
            # Crossover traits
            for trait in self.genome.traits:
                if trait in partner.genome.traits:
                    # Average parent traits with mutation
                    avg_value = (self.genome.traits[trait] + partner.genome.traits[trait]) / 2
                    mutation = (self.genome.mutation_rate * 2 - self.genome.mutation_rate) * 0.05
                    child_genome.traits[trait] = max(0.0, min(1.0, avg_value + mutation))
                else:
                    child_genome.traits[trait] = self.genome.traits[trait]
            
            # Combine strategies
            child_genome.strategies = self.genome.strategies + partner.genome.strategies
        else:
            # Asexual reproduction with mutation
            child_genome.traits = self.genome.traits.copy()
            child_genome.strategies = self.genome.strategies.copy()
            
            # Apply mutations
            for trait in child_genome.traits:
                mutation = (self.genome.mutation_rate * 2 - self.genome.mutation_rate) * 0.1
                child_genome.traits[trait] = max(0.0, min(1.0, child_genome.traits[trait] + mutation))
        
        # Create child agent
        child = FractalAgent(
            name=f"{self.name}_child_{len(self.children)}",
            genome=child_genome,
            level=self.level + 1,
            parent_id=self.id,
            token_budget=TokenBudget(total=self.token_budget.total // 2),
            generation=self.generation + 1
        )
        
        self.children.append(child.id)
        return child

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "genome": {
                "traits": self.genome.traits,
                "strategies": self.genome.strategies,
                "mutation_rate": self.genome.mutation_rate
            },
            "level": self.level,
            "parent_id": self.parent_id,
            "children": self.children,
            "token_budget": {
                "total": self.token_budget.total,
                "used": self.token_budget.used,
                "remaining": self.token_budget.remaining,
                "efficiency_score": self.token_budget.efficiency_score
            },
            "diversity_score": self.diversity_score,
            "emergent_patterns": self.emergent_patterns,
            "fitness_score": self.fitness_score,
            "generation": self.generation,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

# Alias for backward compatibility 
Agent = FractalAgent