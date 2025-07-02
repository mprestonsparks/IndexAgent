"""
Genetic Diversity Manager for DEAN system.
Maintains population diversity through active intervention per specifications.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import asyncio
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class DiversityMetrics:
    """Metrics for population diversity assessment."""
    variance_score: float
    entropy_score: float
    distance_matrix: List[List[float]]
    convergence_risk: float
    intervention_recommended: bool

class GeneticDiversityManager:
    """
    Maintains population diversity through active intervention.
    Detects convergence and injects mutations to prevent monocultures.
    """
    
    def __init__(self, min_diversity_threshold: float = 0.3):
        self.min_diversity_threshold = min_diversity_threshold
        self.convergence_threshold = 0.8
        self.mutation_strength = 0.1
        self.intervention_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    async def enforce_diversity(self, population: List['FractalAgent']) -> Dict[str, Any]:
        """
        Detect convergence and inject mutations to maintain diversity.
        
        Args:
            population: List of FractalAgent instances
            
        Returns:
            Dictionary with intervention results
        """
        if len(population) < 2:
            return {"intervention_needed": False, "reason": "Population too small"}
        
        # Calculate current diversity metrics
        diversity_metrics = await self._calculate_population_diversity(population)
        
        intervention_result = {
            "diversity_score": diversity_metrics.variance_score,
            "entropy_score": diversity_metrics.entropy_score,
            "convergence_risk": diversity_metrics.convergence_risk,
            "intervention_needed": diversity_metrics.intervention_recommended,
            "actions_taken": []
        }
        
        if diversity_metrics.intervention_recommended:
            # Apply diversity interventions
            actions = await self._apply_diversity_interventions(population, diversity_metrics)
            intervention_result["actions_taken"] = actions
            
            # Log intervention
            self.intervention_history.append({
                "timestamp": datetime.utcnow(),
                "population_size": len(population),
                "diversity_before": diversity_metrics.variance_score,
                "actions": actions
            })
            
            self.logger.info(
                f"Diversity intervention applied: {len(actions)} actions, "
                f"diversity score: {diversity_metrics.variance_score:.3f}"
            )
        
        return intervention_result
    
    async def _calculate_population_diversity(self, population: List['FractalAgent']) -> DiversityMetrics:
        """Calculate comprehensive diversity metrics for the population."""
        
        # Calculate pairwise genome distances
        distance_matrix = await self._calculate_distance_matrix(population)
        
        # Calculate variance score (average pairwise distance)
        total_distance = sum(sum(row) for row in distance_matrix)
        num_pairs = len(population) * (len(population) - 1)
        variance_score = total_distance / num_pairs if num_pairs > 0 else 0.0
        
        # Calculate entropy based on strategy distributions
        entropy_score = await self._calculate_strategy_entropy(population)
        
        # Assess convergence risk
        convergence_risk = 1.0 - variance_score
        
        # Determine if intervention is needed
        intervention_recommended = (
            variance_score < self.min_diversity_threshold or
            convergence_risk > self.convergence_threshold
        )
        
        return DiversityMetrics(
            variance_score=variance_score,
            entropy_score=entropy_score,
            distance_matrix=distance_matrix,
            convergence_risk=convergence_risk,
            intervention_recommended=intervention_recommended
        )
    
    async def _calculate_distance_matrix(self, population: List['FractalAgent']) -> List[List[float]]:
        """Calculate pairwise distances between agents in population."""
        n = len(population)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = await self._calculate_agent_distance(population[i], population[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        return distance_matrix
    
    async def _calculate_agent_distance(self, agent1: 'FractalAgent', agent2: 'FractalAgent') -> float:
        """Calculate distance between two agents based on genome similarity."""
        
        # Trait distance
        trait_distance = 0.0
        common_traits = set(agent1.genome.traits.keys()) & set(agent2.genome.traits.keys())
        
        if common_traits:
            trait_diffs = [
                abs(agent1.genome.traits[trait] - agent2.genome.traits[trait])
                for trait in common_traits
            ]
            trait_distance = sum(trait_diffs) / len(trait_diffs)
        
        # Strategy distance (Jaccard distance)
        strategies1 = set(agent1.genome.strategies)
        strategies2 = set(agent2.genome.strategies)
        
        if strategies1 or strategies2:
            intersection = len(strategies1 & strategies2)
            union = len(strategies1 | strategies2)
            strategy_similarity = intersection / union if union > 0 else 0.0
            strategy_distance = 1.0 - strategy_similarity
        else:
            strategy_distance = 0.0
        
        # Performance distance
        performance_distance = abs(agent1.fitness_score - agent2.fitness_score)
        
        # Weighted combination
        total_distance = (
            0.4 * trait_distance +
            0.4 * strategy_distance +
            0.2 * performance_distance
        )
        
        return total_distance
    
    async def _calculate_strategy_entropy(self, population: List['FractalAgent']) -> float:
        """Calculate entropy based on strategy distribution."""
        # Collect all strategies
        all_strategies = []
        for agent in population:
            all_strategies.extend(agent.genome.strategies)
        
        if not all_strategies:
            return 0.0
        
        # Count strategy frequencies
        strategy_counts = {}
        for strategy in all_strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Calculate entropy
        total_strategies = len(all_strategies)
        entropy = 0.0
        
        for count in strategy_counts.values():
            probability = count / total_strategies
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(strategy_counts)) if len(strategy_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    async def _apply_diversity_interventions(self, population: List['FractalAgent'], 
                                           metrics: DiversityMetrics) -> List[str]:
        """Apply various interventions to increase population diversity."""
        actions = []
        
        # Identify agents that are too similar
        similar_pairs = await self._find_similar_agents(population, metrics.distance_matrix)
        
        if similar_pairs:
            # Mutate one agent from each similar pair
            mutated_agents = await self._apply_targeted_mutations(population, similar_pairs)
            actions.extend([f"Mutated agent {agent_id}" for agent_id in mutated_agents])
        
        # If diversity is still too low, apply broader interventions
        if metrics.variance_score < self.min_diversity_threshold * 0.5:
            # Inject random mutations across population
            random_mutations = await self._apply_random_mutations(population)
            actions.extend([f"Random mutation to agent {agent_id}" for agent_id in random_mutations])
            
            # Import foreign patterns if available
            imported_patterns = await self._import_foreign_patterns(population)
            if imported_patterns:
                actions.append(f"Imported {len(imported_patterns)} foreign patterns")
        
        return actions
    
    async def _find_similar_agents(self, population: List['FractalAgent'], 
                                 distance_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """Find pairs of agents that are too similar."""
        similar_threshold = 0.2  # Agents closer than this are considered too similar
        similar_pairs = []
        
        n = len(population)
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i][j] < similar_threshold:
                    similar_pairs.append((i, j))
        
        return similar_pairs
    
    async def _apply_targeted_mutations(self, population: List['FractalAgent'], 
                                      similar_pairs: List[Tuple[int, int]]) -> List[str]:
        """Apply mutations to agents that are too similar."""
        mutated_agents = []
        
        for i, j in similar_pairs:
            # Choose the agent with lower fitness to mutate
            if population[i].fitness_score <= population[j].fitness_score:
                target_agent = population[i]
            else:
                target_agent = population[j]
            
            await self._mutate_agent(target_agent, strength=self.mutation_strength)
            mutated_agents.append(target_agent.id)
        
        return mutated_agents
    
    async def _apply_random_mutations(self, population: List['FractalAgent']) -> List[str]:
        """Apply random mutations to increase diversity."""
        mutated_agents = []
        
        # Mutate 20% of population randomly
        num_to_mutate = max(1, len(population) // 5)
        mutation_targets = np.random.choice(len(population), num_to_mutate, replace=False)
        
        for idx in mutation_targets:
            agent = population[idx]
            await self._mutate_agent(agent, strength=self.mutation_strength * 1.5)
            mutated_agents.append(agent.id)
        
        return mutated_agents
    
    async def _mutate_agent(self, agent: 'FractalAgent', strength: float = 0.1):
        """Apply mutations to an agent's genome."""
        
        # Mutate traits
        for trait in agent.genome.traits:
            if np.random.random() < 0.3:  # 30% chance to mutate each trait
                mutation = np.random.normal(0, strength)
                agent.genome.traits[trait] = max(0.0, min(1.0, agent.genome.traits[trait] + mutation))
        
        # Add new strategies
        if np.random.random() < 0.5:  # 50% chance to add new strategy
            new_strategy = f"diversity_injection_{len(agent.genome.strategies)}_{int(datetime.utcnow().timestamp())}"
            agent.genome.strategies.append(new_strategy)
        
        # Modify existing strategies
        if agent.genome.strategies and np.random.random() < 0.3:
            # Replace a random strategy
            idx = np.random.randint(len(agent.genome.strategies))
            agent.genome.strategies[idx] = f"mutated_{agent.genome.strategies[idx]}"
        
        # Update mutation rate
        if np.random.random() < 0.2:  # 20% chance to modify mutation rate
            rate_change = np.random.normal(0, 0.01)
            agent.genome.mutation_rate = max(0.01, min(0.5, agent.genome.mutation_rate + rate_change))
        
        # Update agent metadata
        agent.updated_at = datetime.utcnow()
    
    async def _import_foreign_patterns(self, population: List['FractalAgent']) -> List[str]:
        """Import patterns from external sources to increase diversity."""
        foreign_patterns = [
            "external_optimization_pattern",
            "cross_domain_strategy", 
            "imported_solution_approach",
            "foreign_collaboration_pattern"
        ]
        
        imported = []
        
        # Add foreign patterns to random agents
        for pattern in foreign_patterns[:2]:  # Import up to 2 patterns
            if population:
                target_agent = np.random.choice(population)
                target_agent.genome.strategies.append(pattern)
                imported.append(pattern)
        
        return imported
    
    async def calculate_diversity_trend(self) -> Dict[str, Any]:
        """Calculate diversity trend over time based on intervention history."""
        if len(self.intervention_history) < 2:
            return {"trend": "insufficient_data", "interventions": len(self.intervention_history)}
        
        # Calculate intervention frequency
        recent_interventions = [
            entry for entry in self.intervention_history
            if (datetime.utcnow() - entry["timestamp"]).days <= 7
        ]
        
        # Calculate average diversity before interventions
        diversity_scores = [entry.get("diversity_before", 0) for entry in self.intervention_history]
        avg_diversity = sum(diversity_scores) / len(diversity_scores)
        
        trend_analysis = {
            "total_interventions": len(self.intervention_history),
            "recent_interventions": len(recent_interventions),
            "average_diversity": avg_diversity,
            "intervention_frequency": len(recent_interventions) / 7,  # per day
            "trend": "improving" if avg_diversity > self.min_diversity_threshold else "concerning"
        }
        
        return trend_analysis
    
    def get_diversity_summary(self) -> Dict[str, Any]:
        """Get summary of diversity management status."""
        return {
            "min_diversity_threshold": self.min_diversity_threshold,
            "convergence_threshold": self.convergence_threshold,
            "total_interventions": len(self.intervention_history),
            "recent_interventions": len([
                entry for entry in self.intervention_history
                if (datetime.utcnow() - entry["timestamp"]).hours <= 24
            ]),
            "intervention_history": self.intervention_history[-5:]  # Last 5 interventions
        }

# Alias for backward compatibility
DiversityManager = GeneticDiversityManager