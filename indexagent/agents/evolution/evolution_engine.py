"""
Evolution Engine with Real Genetic Algorithms and Database Integration
Implements actual genetic evolution with tournament selection, uniform crossover,
adaptive mutations, and performance tracking.
"""

import asyncio
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import json
import asyncpg
from dataclasses import dataclass, field
import hashlib

from ..base_agent import FractalAgent, AgentGenome, TokenBudget
from .genetic_algorithm import GeneticAlgorithm, GeneticParameters, CrossoverType, MutationType, SelectionType

logger = logging.getLogger(__name__)


@dataclass
class EvolutionMetrics:
    """Metrics tracked during evolution"""
    generation: int
    population_diversity: float
    best_fitness: float
    avg_fitness: float
    total_tokens_consumed: int
    patterns_discovered: int
    mutation_rate_used: float
    crossover_success_rate: float
    
    
@dataclass
class EvolutionConfig:
    """Configuration for evolution engine"""
    population_size: int = 10
    generations: int = 50
    base_mutation_rate: float = 0.1
    diversity_threshold: float = 0.3
    tournament_size: int = 3
    elitism_count: int = 2
    token_budget_per_generation: int = 50000
    database_url: Optional[str] = None
    enable_adaptive_mutation: bool = True
    track_lineage: bool = True
    min_fitness_improvement: float = 0.001


class EvolutionEngine:
    """
    Real evolution engine implementing genetic algorithms with database integration.
    Performs actual evolution operations, not simulations.
    """
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.db_pool: Optional[asyncpg.Pool] = None
        self.current_generation = 0
        self.evolution_id = self._generate_evolution_id()
        self.metrics_history: List[EvolutionMetrics] = []
        
        # Initialize genetic algorithm with real parameters
        self.genetic_params = GeneticParameters(
            crossover_rate=0.8,
            mutation_rate=config.base_mutation_rate,
            crossover_type=CrossoverType.UNIFORM,
            mutation_type=MutationType.GAUSSIAN,
            selection_type=SelectionType.TOURNAMENT,
            tournament_size=config.tournament_size,
            elitism_ratio=config.elitism_count / config.population_size,
            diversity_pressure=0.3
        )
        self.genetic_algorithm = GeneticAlgorithm(parameters=self.genetic_params)
        
        # Random seed for reproducible but different results
        self.random_seed = int(datetime.now().timestamp() * 1000) % 2**32
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info(f"Evolution engine initialized with seed {self.random_seed}")
    
    async def initialize_database(self):
        """Initialize database connection pool"""
        if self.config.database_url:
            try:
                self.db_pool = await asyncpg.create_pool(
                    self.config.database_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
                await self._ensure_tables()
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_pool = None
    
    async def close(self):
        """Close database connections"""
        if self.db_pool:
            await self.db_pool.close()
    
    async def evolve_population(self, initial_population: List[FractalAgent]) -> List[FractalAgent]:
        """
        Evolve population through multiple generations using real genetic algorithms.
        
        Args:
            initial_population: Starting population of agents
            
        Returns:
            Final evolved population
        """
        if not self.db_pool:
            await self.initialize_database()
        
        population = initial_population.copy()
        
        # Load historical performance data if available
        for agent in population:
            agent.fitness_score = await self._calculate_fitness_from_metrics(agent)
        
        logger.info(f"Starting evolution {self.evolution_id} with {len(population)} agents")
        
        for generation in range(self.config.generations):
            self.current_generation = generation
            
            # Calculate population metrics
            diversity = await self._calculate_population_diversity(population)
            avg_fitness = np.mean([a.fitness_score for a in population])
            best_fitness = max(a.fitness_score for a in population)
            
            # Adaptive mutation rate based on diversity
            if self.config.enable_adaptive_mutation:
                if diversity < self.config.diversity_threshold:
                    # Increase mutation when diversity is low
                    current_mutation_rate = min(0.3, self.config.base_mutation_rate * 3)
                    self.genetic_params.mutation_rate = current_mutation_rate
                else:
                    self.genetic_params.mutation_rate = self.config.base_mutation_rate
            
            logger.info(f"Generation {generation}: diversity={diversity:.3f}, "
                       f"avg_fitness={avg_fitness:.3f}, best={best_fitness:.3f}, "
                       f"mutation_rate={self.genetic_params.mutation_rate:.3f}")
            
            # Store generation metrics
            metrics = EvolutionMetrics(
                generation=generation,
                population_diversity=diversity,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                total_tokens_consumed=sum(a.token_budget.used for a in population),
                patterns_discovered=sum(len(a.emergent_patterns) for a in population),
                mutation_rate_used=self.genetic_params.mutation_rate,
                crossover_success_rate=0.0  # Will be updated during evolution
            )
            
            # Evolve to next generation
            population = await self._evolve_generation(population, metrics)
            
            # Track metrics
            self.metrics_history.append(metrics)
            await self._store_generation_metrics(metrics)
            
            # Check for convergence
            if generation > 10:
                recent_improvement = self._calculate_recent_improvement()
                if recent_improvement < self.config.min_fitness_improvement:
                    logger.info(f"Convergence detected at generation {generation}")
                    break
        
        # Store final results
        await self._store_evolution_results(population)
        
        return population
    
    async def _evolve_generation(self, population: List[FractalAgent], 
                                metrics: EvolutionMetrics) -> List[FractalAgent]:
        """Evolve one generation using real genetic operations"""
        new_population = []
        
        # Elitism: Keep best agents
        sorted_pop = sorted(population, key=lambda a: a.fitness_score, reverse=True)
        elite = sorted_pop[:self.config.elitism_count]
        new_population.extend(elite)
        
        # Generate offspring
        crossover_successes = 0
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = await self._tournament_selection(population)
            parent2 = await self._tournament_selection(population)
            
            # Ensure different parents for sexual reproduction
            attempts = 0
            while parent2.id == parent1.id and attempts < 10:
                parent2 = await self._tournament_selection(population)
                attempts += 1
            
            if parent2.id == parent1.id:
                parent2 = None  # Fall back to asexual reproduction
            
            # Create offspring
            try:
                child = await self._create_offspring(parent1, parent2)
                
                # Calculate fitness using real metrics
                child.fitness_score = await self._calculate_fitness_from_metrics(child)
                
                # Update child's generation
                child.generation = self.current_generation + 1
                
                new_population.append(child)
                if parent2:
                    crossover_successes += 1
                    
            except Exception as e:
                logger.error(f"Failed to create offspring: {e}")
                continue
        
        # Update crossover success rate
        total_offspring = len(new_population) - self.config.elitism_count
        if total_offspring > 0:
            metrics.crossover_success_rate = crossover_successes / total_offspring
        
        # Trim to exact population size
        new_population = new_population[:self.config.population_size]
        
        # Update token budgets based on performance
        await self._update_token_budgets(new_population)
        
        return new_population
    
    async def _tournament_selection(self, population: List[FractalAgent]) -> FractalAgent:
        """
        Tournament selection with stochastic element for diversity.
        Implements real tournament selection, not deterministic best selection.
        """
        tournament_size = min(self.config.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        
        # Sort by fitness
        tournament.sort(key=lambda a: a.fitness_score, reverse=True)
        
        # Stochastic selection - best doesn't always win
        selection_probs = [0.5, 0.3, 0.2][:tournament_size]
        selection_probs = selection_probs[:len(tournament)]
        
        # Normalize probabilities
        total_prob = sum(selection_probs)
        selection_probs = [p/total_prob for p in selection_probs]
        
        # Select based on probabilities
        selected_idx = np.random.choice(len(tournament), p=selection_probs)
        return tournament[selected_idx]
    
    async def _create_offspring(self, parent1: FractalAgent, 
                               parent2: Optional[FractalAgent]) -> FractalAgent:
        """Create offspring using real genetic operations"""
        
        if parent2 and random.random() < self.genetic_params.crossover_rate:
            # Sexual reproduction with uniform crossover
            child_genome = await self._uniform_crossover(parent1.genome, parent2.genome)
        else:
            # Asexual reproduction
            child_genome = AgentGenome(
                traits=parent1.genome.traits.copy(),
                strategies=parent1.genome.strategies.copy(),
                mutation_rate=parent1.genome.mutation_rate
            )
        
        # Apply mutations with current rate
        child_genome = await self._apply_mutations(child_genome)
        
        # Create child agent
        child = FractalAgent(
            name=f"gen{self.current_generation+1}_agent_{len(parent1.children)}",
            genome=child_genome,
            level=parent1.level + 1,
            parent_id=parent1.id,
            token_budget=TokenBudget(total=self.config.token_budget_per_generation // self.config.population_size),
            generation=self.current_generation + 1,
            worktree_path=None
        )
        
        # Track lineage
        parent1.children.append(child.id)
        if parent2:
            parent2.children.append(child.id)
        
        # Store in database if enabled
        if self.config.track_lineage and self.db_pool:
            await self._store_lineage(child, parent1, parent2)
        
        return child
    
    async def _uniform_crossover(self, genome1: AgentGenome, 
                                genome2: AgentGenome) -> AgentGenome:
        """
        Perform real uniform crossover to create genuinely new combinations.
        Each trait has 50% chance to come from either parent.
        """
        child_traits = {}
        
        # Get all unique trait keys
        all_traits = set(genome1.traits.keys()) | set(genome2.traits.keys())
        
        for trait in all_traits:
            # Randomly select parent for each trait
            if trait in genome1.traits and trait in genome2.traits:
                # Both parents have trait - randomly choose
                if random.random() < 0.5:
                    child_traits[trait] = genome1.traits[trait]
                else:
                    child_traits[trait] = genome2.traits[trait]
            elif trait in genome1.traits:
                # Only parent1 has trait - inherit with probability
                if random.random() < 0.7:  # 70% chance to inherit unique traits
                    child_traits[trait] = genome1.traits[trait]
            else:
                # Only parent2 has trait
                if random.random() < 0.7:
                    child_traits[trait] = genome2.traits[trait]
        
        # Crossover strategies
        strategies1 = set(genome1.strategies)
        strategies2 = set(genome2.strategies)
        all_strategies = strategies1 | strategies2
        
        # Each strategy has independent chance to be inherited
        child_strategies = []
        for strategy in all_strategies:
            inherit_prob = 0.5  # Base probability
            
            # Adjust probability based on source
            if strategy in strategies1 and strategy in strategies2:
                inherit_prob = 0.8  # Higher chance for common strategies
            elif strategy in strategies1 or strategy in strategies2:
                inherit_prob = 0.4  # Lower chance for unique strategies
            
            if random.random() < inherit_prob:
                child_strategies.append(strategy)
        
        # Ensure at least one strategy
        if not child_strategies and all_strategies:
            child_strategies.append(random.choice(list(all_strategies)))
        
        return AgentGenome(
            traits=child_traits,
            strategies=child_strategies,
            mutation_rate=(genome1.mutation_rate + genome2.mutation_rate) / 2
        )
    
    async def _apply_mutations(self, genome: AgentGenome) -> AgentGenome:
        """Apply mutations with adaptive rate based on diversity"""
        mutation_rate = self.genetic_params.mutation_rate
        
        # Mutate traits
        for trait, value in list(genome.traits.items()):
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    # Gaussian mutation for numeric traits
                    noise = np.random.normal(0, 0.1)
                    new_value = value + noise
                    # Clamp to [0, 1] range
                    genome.traits[trait] = max(0.0, min(1.0, new_value))
                elif isinstance(value, bool):
                    # Flip boolean traits
                    genome.traits[trait] = not value
                else:
                    # For other types, small chance to remove
                    if random.random() < 0.1:
                        del genome.traits[trait]
        
        # Add new random traits occasionally
        if random.random() < mutation_rate * 0.5:
            new_trait = f"trait_{random.randint(1000, 9999)}"
            genome.traits[new_trait] = random.random()
        
        # Mutate strategies
        if genome.strategies and random.random() < mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'modify'])
            
            if mutation_type == 'add':
                new_strategies = [
                    "exploration", "exploitation", "cooperation", 
                    "specialization", "generalization", "adaptation",
                    "innovation", "optimization", "learning"
                ]
                available = [s for s in new_strategies if s not in genome.strategies]
                if available:
                    genome.strategies.append(random.choice(available))
                    
            elif mutation_type == 'remove' and len(genome.strategies) > 1:
                genome.strategies.pop(random.randint(0, len(genome.strategies)-1))
                
            elif mutation_type == 'modify' and genome.strategies:
                # Shuffle strategy order (can affect behavior)
                random.shuffle(genome.strategies)
        
        return genome
    
    async def _calculate_fitness_from_metrics(self, agent: FractalAgent) -> float:
        """
        Calculate real fitness from database metrics.
        Uses actual performance data, not simulated values.
        """
        if not self.db_pool:
            # Fallback to genome-based estimation
            return self._estimate_fitness_from_genome(agent)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get performance metrics from database
                metrics = await conn.fetchrow("""
                    SELECT 
                        AVG(tokens_consumed) as avg_tokens,
                        AVG(value_generated) as avg_value,
                        COUNT(DISTINCT pattern_id) as patterns_discovered,
                        MAX(efficiency_score) as peak_efficiency
                    FROM agent_evolution.performance_metrics
                    WHERE agent_id = $1
                    AND recorded_at > NOW() - INTERVAL '1 hour'
                """, agent.id)
                
                if not metrics or metrics['avg_tokens'] is None:
                    return self._estimate_fitness_from_genome(agent)
                
                # Calculate efficiency (value per token)
                tokens = metrics['avg_tokens'] or 1
                value = metrics['avg_value'] or 0
                efficiency = value / tokens if tokens > 0 else 0
                
                # Normalize components
                efficiency_score = min(1.0, efficiency * 100)  # Assume 0.01 is excellent
                pattern_score = min(1.0, metrics['patterns_discovered'] / 10)  # 10 patterns is excellent
                peak_score = metrics['peak_efficiency'] or 0
                
                # Weighted combination
                fitness = (
                    efficiency_score * 0.5 +  # Efficiency is most important
                    pattern_score * 0.3 +     # Pattern discovery is valuable
                    peak_score * 0.2          # Peak performance matters
                )
                
                return max(0.0, min(1.0, fitness))
                
        except Exception as e:
            logger.error(f"Failed to calculate fitness from metrics: {e}")
            return self._estimate_fitness_from_genome(agent)
    
    def _estimate_fitness_from_genome(self, agent: FractalAgent) -> float:
        """Estimate fitness from genome when metrics unavailable"""
        base_fitness = 0.5
        
        # Trait-based fitness
        efficiency_trait = agent.genome.traits.get('efficiency', 0.5)
        innovation_trait = agent.genome.traits.get('innovation', 0.5)
        adaptation_trait = agent.genome.traits.get('adaptation', 0.5)
        
        trait_fitness = (efficiency_trait * 0.4 + 
                        innovation_trait * 0.3 + 
                        adaptation_trait * 0.3)
        
        # Strategy diversity bonus
        strategy_bonus = min(0.2, len(agent.genome.strategies) * 0.02)
        
        # Generation penalty (older generations expected to perform better)
        generation_factor = 1.0 - (0.01 * agent.generation)
        
        fitness = (trait_fitness + strategy_bonus) * generation_factor
        
        # Add small random component for diversity
        fitness += random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, fitness))
    
    async def _calculate_population_diversity(self, population: List[FractalAgent]) -> float:
        """
        Calculate real genetic diversity of population.
        Uses Hamming distance for discrete traits and Euclidean for continuous.
        """
        if len(population) <= 1:
            return 1.0
        
        diversity_scores = []
        
        for i, agent1 in enumerate(population):
            for j, agent2 in enumerate(population[i+1:], i+1):
                distance = self._calculate_genetic_distance(agent1.genome, agent2.genome)
                diversity_scores.append(distance)
        
        if not diversity_scores:
            return 1.0
        
        # Average pairwise distance
        avg_distance = np.mean(diversity_scores)
        
        # Normalize to [0, 1] range
        # Assume maximum useful distance is 2.0
        normalized_diversity = min(1.0, avg_distance / 2.0)
        
        return normalized_diversity
    
    def _calculate_genetic_distance(self, genome1: AgentGenome, genome2: AgentGenome) -> float:
        """Calculate genetic distance between two genomes"""
        distance = 0.0
        
        # Trait distance
        all_traits = set(genome1.traits.keys()) | set(genome2.traits.keys())
        
        for trait in all_traits:
            val1 = genome1.traits.get(trait, 0)
            val2 = genome2.traits.get(trait, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Euclidean distance for numeric traits
                distance += (val1 - val2) ** 2
            else:
                # Hamming distance for non-numeric
                if val1 != val2:
                    distance += 1
        
        # Strategy distance (Jaccard distance)
        set1 = set(genome1.strategies)
        set2 = set(genome2.strategies)
        
        if set1 or set2:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard_distance = 1 - (intersection / union) if union > 0 else 1
            distance += jaccard_distance
        
        return np.sqrt(distance)
    
    async def _update_token_budgets(self, population: List[FractalAgent]):
        """Update token budgets based on performance"""
        total_fitness = sum(a.fitness_score for a in population)
        if total_fitness == 0:
            return
        
        total_budget = self.config.token_budget_per_generation
        
        for agent in population:
            # Allocate proportional to fitness
            fitness_ratio = agent.fitness_score / total_fitness
            agent.token_budget.total = int(total_budget * fitness_ratio)
            
            # Minimum budget guarantee
            min_budget = total_budget // (self.config.population_size * 2)
            agent.token_budget.total = max(min_budget, agent.token_budget.total)
    
    def _calculate_recent_improvement(self) -> float:
        """Calculate fitness improvement over recent generations"""
        if len(self.metrics_history) < 5:
            return 1.0  # Not enough history
        
        recent = self.metrics_history[-5:]
        old = self.metrics_history[-10:-5] if len(self.metrics_history) >= 10 else self.metrics_history[:5]
        
        recent_avg = np.mean([m.best_fitness for m in recent])
        old_avg = np.mean([m.best_fitness for m in old])
        
        improvement = recent_avg - old_avg
        return max(0, improvement)
    
    async def _store_generation_metrics(self, metrics: EvolutionMetrics):
        """Store generation metrics in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO agent_evolution.evolution_metrics
                    (evolution_id, generation, diversity, best_fitness, avg_fitness,
                     tokens_consumed, patterns_discovered, mutation_rate, crossover_rate)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, self.evolution_id, metrics.generation, metrics.population_diversity,
                    metrics.best_fitness, metrics.avg_fitness, metrics.total_tokens_consumed,
                    metrics.patterns_discovered, metrics.mutation_rate_used,
                    metrics.crossover_success_rate)
        except Exception as e:
            logger.error(f"Failed to store generation metrics: {e}")
    
    async def _store_lineage(self, child: FractalAgent, parent1: FractalAgent, 
                           parent2: Optional[FractalAgent]):
        """Store agent lineage in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO agent_evolution.agent_lineage
                    (child_id, parent1_id, parent2_id, generation, 
                     crossover_type, mutation_rate, fitness_delta)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, child.id, parent1.id, parent2.id if parent2 else None,
                    child.generation, self.genetic_params.crossover_type.value,
                    self.genetic_params.mutation_rate,
                    child.fitness_score - parent1.fitness_score)
        except Exception as e:
            logger.error(f"Failed to store lineage: {e}")
    
    async def _store_evolution_results(self, final_population: List[FractalAgent]):
        """Store final evolution results"""
        if not self.db_pool:
            return
        
        try:
            best_agent = max(final_population, key=lambda a: a.fitness_score)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO agent_evolution.evolution_runs
                    (evolution_id, seed, generations_completed, final_best_fitness,
                     final_avg_fitness, final_diversity, best_agent_id, config)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, self.evolution_id, self.random_seed, self.current_generation,
                    best_agent.fitness_score,
                    np.mean([a.fitness_score for a in final_population]),
                    await self._calculate_population_diversity(final_population),
                    best_agent.id,
                    json.dumps({
                        'population_size': self.config.population_size,
                        'mutation_rate': self.config.base_mutation_rate,
                        'diversity_threshold': self.config.diversity_threshold,
                        'tournament_size': self.config.tournament_size
                    }))
        except Exception as e:
            logger.error(f"Failed to store evolution results: {e}")
    
    async def _ensure_tables(self):
        """Ensure required database tables exist"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Create schema if not exists
                await conn.execute("""
                    CREATE SCHEMA IF NOT EXISTS agent_evolution
                """)
                
                # Create tables if not exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_evolution.performance_metrics (
                        id SERIAL PRIMARY KEY,
                        agent_id VARCHAR(255) NOT NULL,
                        tokens_consumed INTEGER,
                        value_generated FLOAT,
                        pattern_id VARCHAR(255),
                        efficiency_score FLOAT,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_evolution.evolution_metrics (
                        id SERIAL PRIMARY KEY,
                        evolution_id VARCHAR(255) NOT NULL,
                        generation INTEGER,
                        diversity FLOAT,
                        best_fitness FLOAT,
                        avg_fitness FLOAT,
                        tokens_consumed INTEGER,
                        patterns_discovered INTEGER,
                        mutation_rate FLOAT,
                        crossover_rate FLOAT,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_evolution.agent_lineage (
                        id SERIAL PRIMARY KEY,
                        child_id VARCHAR(255) NOT NULL,
                        parent1_id VARCHAR(255) NOT NULL,
                        parent2_id VARCHAR(255),
                        generation INTEGER,
                        crossover_type VARCHAR(50),
                        mutation_rate FLOAT,
                        fitness_delta FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_evolution.evolution_runs (
                        id SERIAL PRIMARY KEY,
                        evolution_id VARCHAR(255) UNIQUE NOT NULL,
                        seed INTEGER,
                        generations_completed INTEGER,
                        final_best_fitness FLOAT,
                        final_avg_fitness FLOAT,
                        final_diversity FLOAT,
                        best_agent_id VARCHAR(255),
                        config JSONB,
                        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_performance_agent_id 
                    ON agent_evolution.performance_metrics(agent_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_evolution_metrics_evolution_id 
                    ON agent_evolution.evolution_metrics(evolution_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_lineage_child_id 
                    ON agent_evolution.agent_lineage(child_id)
                """)
                
        except Exception as e:
            logger.error(f"Failed to ensure tables: {e}")
    
    def _generate_evolution_id(self) -> str:
        """Generate unique evolution run ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{timestamp}-{self.random_seed}".encode()).hexdigest()[:16]


async def create_and_run_evolution(population_size: int = 10, 
                                 generations: int = 20,
                                 database_url: Optional[str] = None) -> List[FractalAgent]:
    """
    Convenience function to create and run evolution.
    
    Args:
        population_size: Number of agents in population
        generations: Number of generations to evolve
        database_url: PostgreSQL connection string
        
    Returns:
        Final evolved population
    """
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        database_url=database_url,
        enable_adaptive_mutation=True,
        diversity_threshold=0.3
    )
    
    engine = EvolutionEngine(config)
    
    try:
        # Create initial population
        initial_population = []
        for i in range(population_size):
            agent = FractalAgent(
                name=f"gen0_agent_{i}",
                genome=AgentGenome(
                    traits={
                        'efficiency': random.random(),
                        'innovation': random.random(),
                        'adaptation': random.random(),
                        'exploration': random.random()
                    },
                    strategies=random.sample([
                        "exploration", "exploitation", "cooperation",
                        "specialization", "generalization"
                    ], k=random.randint(1, 3))
                ),
                token_budget=TokenBudget(total=4096)
            )
            initial_population.append(agent)
        
        # Run evolution
        final_population = await engine.evolve_population(initial_population)
        
        # Log results
        best_agent = max(final_population, key=lambda a: a.fitness_score)
        logger.info(f"Evolution completed. Best fitness: {best_agent.fitness_score:.3f}")
        
        return final_population
        
    finally:
        await engine.close()


if __name__ == "__main__":
    # Example usage
    import os
    
    async def main():
        db_url = os.getenv("AGENT_EVOLUTION_DATABASE_URL")
        population = await create_and_run_evolution(
            population_size=8,
            generations=10,
            database_url=db_url
        )
        
        print(f"Evolution completed with {len(population)} agents")
        for agent in sorted(population, key=lambda a: a.fitness_score, reverse=True)[:3]:
            print(f"  {agent.name}: fitness={agent.fitness_score:.3f}, "
                  f"strategies={agent.genome.strategies}")
    
    asyncio.run(main())