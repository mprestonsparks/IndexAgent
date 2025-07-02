#!/usr/bin/env python3
"""
Genetic Algorithm Module for DEAN Agent Evolution
Implementation of FR-003: Child Agent Creation per Section 2.1

This module implements the genetic algorithm specified in Section 2.1 for creating
child agents through crossover operations, mutation operations, and fitness evaluation
based on performance metrics. Integrates with the existing cellular automata engine.
"""

import random
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import existing components
try:
    from IndexAgent.indexagent.agents.base_agent import FractalAgent, AgentGenome
    from IndexAgent.indexagent.agents.evolution.cellular_automata import CellularAutomataEngine
    from IndexAgent.indexagent.agents.economy.token_manager import TokenEconomyManager
    from IndexAgent.indexagent.agents.patterns.detector import PatternDetector
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    
    # Mock classes for standalone validation
    class AgentGenome:
        def __init__(self, traits=None, strategies=None):
            self.traits = traits or {}
            self.strategies = strategies or []
    
    class FractalAgent:
        def __init__(self, **kwargs):
            self.agent_id = kwargs.get('agent_id', str(uuid.uuid4()))
            self.genome = kwargs.get('genome', AgentGenome())
            self.fitness_score = kwargs.get('fitness_score', 0.0)
            self.diversity_score = kwargs.get('diversity_score', 0.5)

# Implements FR-003: Child agent creation through genetic algorithm per specification
logger = logging.getLogger(__name__)


class CrossoverType(str, Enum):
    """Crossover operation types per Section 2.1"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"


class MutationType(str, Enum):
    """Mutation operation types per Section 2.1"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    BIT_FLIP = "bit_flip"
    STRATEGY_SHUFFLE = "strategy_shuffle"


class SelectionType(str, Enum):
    """Selection methods for genetic algorithm"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITIST = "elitist"


@dataclass
class GeneticParameters:
    """Genetic algorithm configuration parameters"""
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    crossover_type: CrossoverType = CrossoverType.UNIFORM
    mutation_type: MutationType = MutationType.GAUSSIAN
    selection_type: SelectionType = SelectionType.TOURNAMENT
    tournament_size: int = 3
    elitism_ratio: float = 0.1
    diversity_pressure: float = 0.3
    fitness_threshold: float = 0.7
    max_generations: int = 50


@dataclass
class EvolutionResult:
    """Result of genetic algorithm evolution"""
    child_agent: 'FractalAgent'
    generation: int
    fitness_improvement: float
    diversity_score: float
    genetic_distance: float
    applied_operations: List[str]
    performance_prediction: float
    creation_metadata: Dict[str, Any] = field(default_factory=dict)


class GeneticAlgorithm:
    """
    Genetic Algorithm Engine for DEAN Agent Evolution
    
    Implements FR-003: Child agent creation through crossover operations,
    mutation operations, and fitness evaluation based on performance metrics.
    Integrates with cellular automata engine per architectural specifications.
    """
    
    def __init__(self, parameters: Optional[GeneticParameters] = None,
                 ca_engine: Optional['CellularAutomataEngine'] = None,
                 token_manager: Optional['TokenEconomyManager'] = None,
                 pattern_detector: Optional['PatternDetector'] = None):
        """
        Initialize genetic algorithm engine
        
        Args:
            parameters: Genetic algorithm configuration
            ca_engine: Cellular automata engine for rule integration
            token_manager: Token economy manager for budget constraints
            pattern_detector: Pattern detector for emergent behavior analysis
        """
        self.parameters = parameters or GeneticParameters()
        self.ca_engine = ca_engine
        self.token_manager = token_manager
        self.pattern_detector = pattern_detector
        
        # Evolution tracking
        self.generation_count = 0
        self.evolution_history: List[EvolutionResult] = []
        self.fitness_statistics = {
            "mean": [],
            "max": [],
            "min": [],
            "std": []
        }
        
        # Performance metrics
        self.total_crossovers = 0
        self.total_mutations = 0
        self.successful_evolutions = 0
        
        logger.info(f"Genetic algorithm initialized with parameters: {self.parameters}")
    
    def create_child_agent(self, parent1: FractalAgent, parent2: Optional[FractalAgent] = None,
                          target_fitness: Optional[float] = None) -> EvolutionResult:
        """
        Create child agent through genetic algorithm per FR-003
        
        Implements crossover operations for combining successful agent strategies,
        mutation operations for introducing controlled variations, and fitness
        evaluation based on performance metrics.
        
        Args:
            parent1: Primary parent agent
            parent2: Secondary parent agent (None for asexual reproduction)
            target_fitness: Target fitness score for child agent
            
        Returns:
            EvolutionResult with child agent and evolution metadata
            
        Raises:
            ValueError: If parents have incompatible genomes
            RuntimeError: If evolution fails to meet constraints
        """
        start_time = datetime.now()
        
        try:
            # Validate parent agents
            self._validate_parents(parent1, parent2)
            
            # Determine reproduction strategy
            if parent2 is None:
                # Asexual reproduction with mutation
                child_genome = self._asexual_reproduction(parent1.genome)
                genetic_distance = self._calculate_mutation_distance(parent1.genome, child_genome)
                applied_operations = ["asexual_reproduction", "mutation"]
            else:
                # Sexual reproduction with crossover and mutation
                child_genome = self._sexual_reproduction(parent1.genome, parent2.genome)
                genetic_distance = self._calculate_crossover_distance(
                    parent1.genome, parent2.genome, child_genome
                )
                applied_operations = ["crossover", "mutation"]
            
            # Apply cellular automata rules if available
            if self.ca_engine:
                ca_rule = self._select_ca_rule(parent1, parent2)
                child_genome = self._apply_ca_rule(child_genome, ca_rule)
                applied_operations.append(f"ca_rule_{ca_rule}")
            
            # Create child agent
            child_agent = self._instantiate_child_agent(
                child_genome, parent1, parent2
            )
            
            # Evaluate fitness and performance prediction
            fitness_score = self._evaluate_fitness(child_agent, parent1, parent2)
            performance_prediction = self._predict_performance(child_agent)
            
            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(child_agent, parent1, parent2)
            
            # Calculate fitness improvement
            parent_fitness = parent1.fitness_score
            if parent2:
                parent_fitness = max(parent1.fitness_score, parent2.fitness_score)
            fitness_improvement = fitness_score - parent_fitness
            
            # Update child agent properties
            child_agent.fitness_score = fitness_score
            child_agent.diversity_score = diversity_score
            
            # Create evolution result
            evolution_result = EvolutionResult(
                child_agent=child_agent,
                generation=self.generation_count + 1,
                fitness_improvement=fitness_improvement,
                diversity_score=diversity_score,
                genetic_distance=genetic_distance,
                applied_operations=applied_operations,
                performance_prediction=performance_prediction,
                creation_metadata={
                    "parent1_id": parent1.agent_id,
                    "parent2_id": parent2.agent_id if parent2 else None,
                    "creation_time": start_time.isoformat(),
                    "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                    "genetic_parameters": self.parameters.__dict__,
                    "target_fitness": target_fitness
                }
            )
            
            # Update statistics and history
            self._update_evolution_statistics(evolution_result)
            self.evolution_history.append(evolution_result)
            self.successful_evolutions += 1
            
            logger.info(f"Child agent created successfully: {child_agent.agent_id} "
                       f"(fitness: {fitness_score:.3f}, diversity: {diversity_score:.3f})")
            
            return evolution_result
            
        except Exception as e:
            logger.error(f"Child agent creation failed: {e}")
            raise RuntimeError(f"Genetic algorithm evolution failed: {e}")
    
    def evolve_population(self, population: List[FractalAgent], 
                         target_size: int, generations: int = 1) -> List[EvolutionResult]:
        """
        Evolve entire population through genetic algorithm
        
        Args:
            population: Current agent population
            target_size: Target population size
            generations: Number of generations to evolve
            
        Returns:
            List of evolution results for new agents
        """
        evolution_results = []
        
        for generation in range(generations):
            self.generation_count += 1
            generation_results = []
            
            # Select parents for reproduction
            parent_pairs = self._select_parent_pairs(population, target_size)
            
            # Create children from selected parents
            for parent1, parent2 in parent_pairs:
                try:
                    result = self.create_child_agent(parent1, parent2)
                    generation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to create child from parents "
                                 f"{parent1.agent_id}, {parent2.agent_id if parent2 else 'None'}: {e}")
            
            # Update population for next generation
            new_agents = [result.child_agent for result in generation_results]
            population = self._select_survivors(population + new_agents, target_size)
            
            evolution_results.extend(generation_results)
            
            logger.info(f"Generation {self.generation_count} completed: "
                       f"{len(generation_results)} new agents created")
        
        return evolution_results
    
    def _validate_parents(self, parent1: FractalAgent, parent2: Optional[FractalAgent]):
        """Validate parent agents for genetic compatibility"""
        if not isinstance(parent1, FractalAgent):
            raise ValueError("Parent1 must be a FractalAgent instance")
        
        if parent2 and not isinstance(parent2, FractalAgent):
            raise ValueError("Parent2 must be a FractalAgent instance")
        
        if not hasattr(parent1, 'genome') or parent1.genome is None:
            raise ValueError("Parent1 must have a valid genome")
        
        if parent2 and (not hasattr(parent2, 'genome') or parent2.genome is None):
            raise ValueError("Parent2 must have a valid genome")
    
    def _asexual_reproduction(self, parent_genome: AgentGenome) -> AgentGenome:
        """
        Asexual reproduction through genome copying and mutation
        
        Args:
            parent_genome: Parent agent genome
            
        Returns:
            Child genome with mutations applied
        """
        # Copy parent genome
        child_genome = AgentGenome(
            traits=parent_genome.traits.copy(),
            strategies=parent_genome.strategies.copy()
        )
        
        # Apply mutations
        child_genome = self._apply_mutations(child_genome)
        
        return child_genome
    
    def _sexual_reproduction(self, parent1_genome: AgentGenome, 
                           parent2_genome: AgentGenome) -> AgentGenome:
        """
        Sexual reproduction through crossover and mutation
        
        Args:
            parent1_genome: First parent genome
            parent2_genome: Second parent genome
            
        Returns:
            Child genome from crossover and mutation
        """
        # Apply crossover operation
        if random.random() < self.parameters.crossover_rate:
            child_genome = self._apply_crossover(parent1_genome, parent2_genome)
            self.total_crossovers += 1
        else:
            # Use first parent as base if no crossover
            child_genome = AgentGenome(
                traits=parent1_genome.traits.copy(),
                strategies=parent1_genome.strategies.copy()
            )
        
        # Apply mutations
        child_genome = self._apply_mutations(child_genome)
        
        return child_genome
    
    def _apply_crossover(self, genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
        """
        Apply crossover operation to combine parent genomes
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            
        Returns:
            Child genome from crossover operation
        """
        if self.parameters.crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(genome1, genome2)
        elif self.parameters.crossover_type == CrossoverType.SINGLE_POINT:
            return self._single_point_crossover(genome1, genome2)
        elif self.parameters.crossover_type == CrossoverType.TWO_POINT:
            return self._two_point_crossover(genome1, genome2)
        elif self.parameters.crossover_type == CrossoverType.ARITHMETIC:
            return self._arithmetic_crossover(genome1, genome2)
        else:
            return self._uniform_crossover(genome1, genome2)  # Default
    
    def _uniform_crossover(self, genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
        """Uniform crossover: randomly select traits from each parent"""
        child_traits = {}
        
        # Combine traits from both parents
        all_trait_keys = set(genome1.traits.keys()) | set(genome2.traits.keys())
        
        for trait_key in all_trait_keys:
            if trait_key in genome1.traits and trait_key in genome2.traits:
                # Both parents have trait - randomly select or average
                if random.random() < 0.5:
                    child_traits[trait_key] = genome1.traits[trait_key]
                else:
                    child_traits[trait_key] = genome2.traits[trait_key]
            elif trait_key in genome1.traits:
                child_traits[trait_key] = genome1.traits[trait_key]
            else:
                child_traits[trait_key] = genome2.traits[trait_key]
        
        # Combine strategies
        combined_strategies = list(set(genome1.strategies + genome2.strategies))
        child_strategies = random.sample(
            combined_strategies, 
            min(len(combined_strategies), max(len(genome1.strategies), len(genome2.strategies)))
        )
        
        return AgentGenome(traits=child_traits, strategies=child_strategies)
    
    def _single_point_crossover(self, genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
        """Single-point crossover: split at one point and combine"""
        trait_keys = list(set(genome1.traits.keys()) | set(genome2.traits.keys()))
        
        if len(trait_keys) <= 1:
            return self._uniform_crossover(genome1, genome2)
        
        crossover_point = random.randint(1, len(trait_keys) - 1)
        
        child_traits = {}
        for i, trait_key in enumerate(trait_keys):
            source_genome = genome1 if i < crossover_point else genome2
            if trait_key in source_genome.traits:
                child_traits[trait_key] = source_genome.traits[trait_key]
        
        # Combine strategies similarly
        all_strategies = list(set(genome1.strategies + genome2.strategies))
        strategy_point = random.randint(0, len(all_strategies))
        child_strategies = all_strategies[:strategy_point] + all_strategies[strategy_point:]
        child_strategies = list(set(child_strategies))  # Remove duplicates
        
        return AgentGenome(traits=child_traits, strategies=child_strategies)
    
    def _two_point_crossover(self, genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
        """Two-point crossover: split at two points"""
        trait_keys = list(set(genome1.traits.keys()) | set(genome2.traits.keys()))
        
        if len(trait_keys) <= 2:
            return self._uniform_crossover(genome1, genome2)
        
        point1 = random.randint(1, len(trait_keys) - 2)
        point2 = random.randint(point1 + 1, len(trait_keys) - 1)
        
        child_traits = {}
        for i, trait_key in enumerate(trait_keys):
            if i < point1 or i >= point2:
                source_genome = genome1
            else:
                source_genome = genome2
            
            if trait_key in source_genome.traits:
                child_traits[trait_key] = source_genome.traits[trait_key]
        
        # Combine strategies
        child_strategies = list(set(genome1.strategies + genome2.strategies))
        
        return AgentGenome(traits=child_traits, strategies=child_strategies)
    
    def _arithmetic_crossover(self, genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
        """Arithmetic crossover: blend numeric traits"""
        child_traits = {}
        
        # Blend numeric traits
        all_trait_keys = set(genome1.traits.keys()) | set(genome2.traits.keys())
        
        for trait_key in all_trait_keys:
            trait1 = genome1.traits.get(trait_key, 0)
            trait2 = genome2.traits.get(trait_key, 0)
            
            if isinstance(trait1, (int, float)) and isinstance(trait2, (int, float)):
                # Arithmetic blend for numeric traits
                alpha = random.random()
                child_traits[trait_key] = alpha * trait1 + (1 - alpha) * trait2
            else:
                # Random selection for non-numeric traits
                child_traits[trait_key] = random.choice([trait1, trait2])
        
        # Combine strategies
        child_strategies = list(set(genome1.strategies + genome2.strategies))
        
        return AgentGenome(traits=child_traits, strategies=child_strategies)
    
    def _apply_mutations(self, genome: AgentGenome) -> AgentGenome:
        """
        Apply mutation operations to introduce controlled variations
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        if random.random() < self.parameters.mutation_rate:
            if self.parameters.mutation_type == MutationType.GAUSSIAN:
                genome = self._gaussian_mutation(genome)
            elif self.parameters.mutation_type == MutationType.UNIFORM:
                genome = self._uniform_mutation(genome)
            elif self.parameters.mutation_type == MutationType.BIT_FLIP:
                genome = self._bit_flip_mutation(genome)
            elif self.parameters.mutation_type == MutationType.STRATEGY_SHUFFLE:
                genome = self._strategy_shuffle_mutation(genome)
            
            self.total_mutations += 1
        
        return genome
    
    def _gaussian_mutation(self, genome: AgentGenome) -> AgentGenome:
        """Apply Gaussian noise to numeric traits"""
        mutated_traits = genome.traits.copy()
        
        for trait_key, trait_value in mutated_traits.items():
            if isinstance(trait_value, (int, float)):
                noise = random.gauss(0, 0.1)  # 10% standard deviation
                mutated_traits[trait_key] = max(0, min(1, trait_value + noise))
        
        return AgentGenome(traits=mutated_traits, strategies=genome.strategies.copy())
    
    def _uniform_mutation(self, genome: AgentGenome) -> AgentGenome:
        """Apply uniform random mutations"""
        mutated_traits = genome.traits.copy()
        
        # Randomly modify one trait
        if mutated_traits:
            trait_key = random.choice(list(mutated_traits.keys()))
            trait_value = mutated_traits[trait_key]
            
            if isinstance(trait_value, (int, float)):
                mutated_traits[trait_key] = random.uniform(0, 1)
            else:
                # For non-numeric traits, add small random variation
                mutated_traits[f"{trait_key}_variant"] = random.random()
        
        return AgentGenome(traits=mutated_traits, strategies=genome.strategies.copy())
    
    def _bit_flip_mutation(self, genome: AgentGenome) -> AgentGenome:
        """Flip binary-like traits"""
        mutated_traits = genome.traits.copy()
        
        for trait_key, trait_value in mutated_traits.items():
            if isinstance(trait_value, bool):
                if random.random() < 0.1:  # 10% chance to flip
                    mutated_traits[trait_key] = not trait_value
            elif isinstance(trait_value, (int, float)) and 0 <= trait_value <= 1:
                if random.random() < 0.1:
                    mutated_traits[trait_key] = 1 - trait_value
        
        return AgentGenome(traits=mutated_traits, strategies=genome.strategies.copy())
    
    def _strategy_shuffle_mutation(self, genome: AgentGenome) -> AgentGenome:
        """Shuffle or modify strategy list"""
        mutated_strategies = genome.strategies.copy()
        
        if mutated_strategies and random.random() < 0.3:
            # Randomly shuffle strategies
            random.shuffle(mutated_strategies)
            
            # Occasionally add or remove a strategy
            if random.random() < 0.2:
                available_strategies = [
                    "exploration", "optimization", "learning", "collaboration",
                    "efficiency", "innovation", "adaptation", "specialization"
                ]
                new_strategy = random.choice(available_strategies)
                if new_strategy not in mutated_strategies:
                    mutated_strategies.append(new_strategy)
            
            elif len(mutated_strategies) > 1 and random.random() < 0.1:
                mutated_strategies.pop(random.randint(0, len(mutated_strategies) - 1))
        
        return AgentGenome(traits=genome.traits.copy(), strategies=mutated_strategies)
    
    def _select_ca_rule(self, parent1: FractalAgent, parent2: Optional[FractalAgent]) -> int:
        """Select appropriate cellular automata rule based on parent characteristics"""
        # Rule selection based on parent fitness and diversity
        avg_fitness = parent1.fitness_score
        if parent2:
            avg_fitness = (parent1.fitness_score + parent2.fitness_score) / 2
        
        if avg_fitness > 0.8:
            return 110  # High-performing parents get complexity generation
        elif avg_fitness < 0.4:
            return 1    # Low-performing parents trigger cleanup
        else:
            return random.choice([30, 90, 184])  # Medium performance gets varied rules
    
    def _apply_ca_rule(self, genome: AgentGenome, ca_rule: int) -> AgentGenome:
        """Apply cellular automata rule to genome evolution"""
        if not self.ca_engine:
            return genome
        
        try:
            # Convert genome to CA state and apply rule
            enhanced_genome = self.ca_engine.apply_rule_to_genome(genome, ca_rule)
            return enhanced_genome
        except Exception as e:
            logger.warning(f"Failed to apply CA rule {ca_rule}: {e}")
            return genome
    
    def _instantiate_child_agent(self, genome: AgentGenome, parent1: FractalAgent, 
                                parent2: Optional[FractalAgent]) -> FractalAgent:
        """Create child agent instance from genome"""
        child_id = str(uuid.uuid4())
        
        # Determine generation
        generation = parent1.generation + 1 if hasattr(parent1, 'generation') else 1
        
        # Create child agent
        child_agent = FractalAgent(
            agent_id=child_id,
            genome=genome,
            parent_id=parent1.agent_id,
            generation=generation,
            token_budget=self._calculate_child_token_budget(parent1, parent2),
            status="active",
            diversity_score=0.5,  # Will be calculated later
            fitness_score=0.0     # Will be calculated later
        )
        
        return child_agent
    
    def _calculate_child_token_budget(self, parent1: FractalAgent, 
                                    parent2: Optional[FractalAgent]) -> int:
        """Calculate token budget for child agent"""
        if self.token_manager:
            return self.token_manager.calculate_child_budget(parent1, parent2)
        
        # Default budget calculation
        base_budget = getattr(parent1, 'token_budget', 4096)
        if parent2:
            base_budget = max(base_budget, getattr(parent2, 'token_budget', 4096))
        
        # Child gets 50% of parent budget by default
        return int(base_budget * 0.5)
    
    def _evaluate_fitness(self, child_agent: FractalAgent, parent1: FractalAgent,
                         parent2: Optional[FractalAgent]) -> float:
        """
        Evaluate fitness based on performance metrics per specification
        
        Args:
            child_agent: Child agent to evaluate
            parent1: First parent agent
            parent2: Second parent agent (optional)
            
        Returns:
            Fitness score between 0.0 and 1.0
        """
        # Base fitness from genome traits
        efficiency = child_agent.genome.traits.get('efficiency', 0.5)
        creativity = child_agent.genome.traits.get('creativity', 0.5)
        adaptation = child_agent.genome.traits.get('adaptation', 0.5)
        
        # Calculate weighted fitness
        base_fitness = (
            efficiency * 0.4 +
            creativity * 0.3 +
            adaptation * 0.3
        )
        
        # Bonus for strategy diversity
        strategy_bonus = min(len(child_agent.genome.strategies) * 0.05, 0.2)
        
        # Parent fitness influence
        parent_influence = parent1.fitness_score * 0.1
        if parent2:
            parent_influence = max(parent_influence, parent2.fitness_score * 0.1)
        
        total_fitness = base_fitness + strategy_bonus + parent_influence
        return min(1.0, max(0.0, total_fitness))
    
    def _predict_performance(self, child_agent: FractalAgent) -> float:
        """Predict agent performance based on genome characteristics"""
        if self.pattern_detector:
            try:
                return self.pattern_detector.predict_agent_performance(child_agent)
            except Exception as e:
                logger.warning(f"Pattern detector performance prediction failed: {e}")
        
        # Simple performance prediction based on fitness and diversity
        fitness_factor = child_agent.fitness_score
        diversity_factor = child_agent.diversity_score
        strategy_factor = min(len(child_agent.genome.strategies) / 5.0, 1.0)
        
        return (fitness_factor * 0.5 + diversity_factor * 0.3 + strategy_factor * 0.2)
    
    def _calculate_diversity_score(self, child_agent: FractalAgent, parent1: FractalAgent,
                                 parent2: Optional[FractalAgent]) -> float:
        """Calculate diversity score relative to parents"""
        # Calculate genetic distance from parents
        parent1_distance = self._calculate_genetic_distance(child_agent.genome, parent1.genome)
        
        if parent2:
            parent2_distance = self._calculate_genetic_distance(child_agent.genome, parent2.genome)
            avg_distance = (parent1_distance + parent2_distance) / 2
        else:
            avg_distance = parent1_distance
        
        # Normalize to 0-1 range
        diversity_score = min(1.0, avg_distance / 2.0)
        
        # Apply diversity pressure
        diversity_score *= (1 + self.parameters.diversity_pressure)
        
        return min(1.0, max(0.0, diversity_score))
    
    def _calculate_genetic_distance(self, genome1: AgentGenome, genome2: AgentGenome) -> float:
        """Calculate genetic distance between two genomes"""
        trait_distance = 0.0
        trait_count = 0
        
        # Compare traits
        all_traits = set(genome1.traits.keys()) | set(genome2.traits.keys())
        
        for trait_key in all_traits:
            val1 = genome1.traits.get(trait_key, 0)
            val2 = genome2.traits.get(trait_key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                trait_distance += abs(val1 - val2)
                trait_count += 1
        
        # Compare strategies
        strategies1 = set(genome1.strategies)
        strategies2 = set(genome2.strategies)
        strategy_distance = len(strategies1.symmetric_difference(strategies2))
        
        # Combine distances
        if trait_count > 0:
            avg_trait_distance = trait_distance / trait_count
        else:
            avg_trait_distance = 0.0
        
        total_distance = avg_trait_distance + strategy_distance * 0.1
        return total_distance
    
    def _calculate_mutation_distance(self, parent_genome: AgentGenome, 
                                   child_genome: AgentGenome) -> float:
        """Calculate distance introduced by mutation"""
        return self._calculate_genetic_distance(parent_genome, child_genome)
    
    def _calculate_crossover_distance(self, parent1_genome: AgentGenome,
                                    parent2_genome: AgentGenome,
                                    child_genome: AgentGenome) -> float:
        """Calculate distance from crossover operation"""
        parent1_distance = self._calculate_genetic_distance(parent1_genome, child_genome)
        parent2_distance = self._calculate_genetic_distance(parent2_genome, child_genome)
        
        return (parent1_distance + parent2_distance) / 2
    
    def _select_parent_pairs(self, population: List[FractalAgent], 
                           target_offspring: int) -> List[Tuple[FractalAgent, Optional[FractalAgent]]]:
        """Select parent pairs for reproduction"""
        parent_pairs = []
        
        for _ in range(target_offspring):
            if self.parameters.selection_type == SelectionType.TOURNAMENT:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population) if len(population) > 1 else None
            elif self.parameters.selection_type == SelectionType.ROULETTE:
                parent1 = self._roulette_selection(population)
                parent2 = self._roulette_selection(population) if len(population) > 1 else None
            else:
                # Random selection fallback
                parent1 = random.choice(population)
                parent2 = random.choice(population) if len(population) > 1 else None
            
            # Avoid self-mating
            if parent2 and parent1.agent_id == parent2.agent_id:
                parent2 = None
            
            parent_pairs.append((parent1, parent2))
        
        return parent_pairs
    
    def _tournament_selection(self, population: List[FractalAgent]) -> FractalAgent:
        """Tournament selection for parent selection"""
        tournament_size = min(self.parameters.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        
        # Select best from tournament
        return max(tournament, key=lambda agent: agent.fitness_score)
    
    def _roulette_selection(self, population: List[FractalAgent]) -> FractalAgent:
        """Roulette wheel selection based on fitness"""
        total_fitness = sum(agent.fitness_score for agent in population)
        
        if total_fitness == 0:
            return random.choice(population)
        
        selection_point = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        
        for agent in population:
            cumulative_fitness += agent.fitness_score
            if cumulative_fitness >= selection_point:
                return agent
        
        return population[-1]  # Fallback
    
    def _select_survivors(self, population: List[FractalAgent], 
                         target_size: int) -> List[FractalAgent]:
        """Select survivors for next generation"""
        if len(population) <= target_size:
            return population
        
        # Sort by fitness
        sorted_population = sorted(population, key=lambda agent: agent.fitness_score, reverse=True)
        
        # Apply elitism
        elite_count = int(target_size * self.parameters.elitism_ratio)
        survivors = sorted_population[:elite_count]
        
        # Fill remaining slots with diverse selection
        remaining_slots = target_size - elite_count
        remaining_population = sorted_population[elite_count:]
        
        # Select remaining agents with diversity consideration
        for _ in range(remaining_slots):
            if not remaining_population:
                break
            
            # Select agent that maximizes diversity
            best_candidate = max(
                remaining_population,
                key=lambda agent: agent.fitness_score + agent.diversity_score * self.parameters.diversity_pressure
            )
            
            survivors.append(best_candidate)
            remaining_population.remove(best_candidate)
        
        return survivors
    
    def _update_evolution_statistics(self, evolution_result: EvolutionResult):
        """Update evolution statistics for monitoring"""
        fitness_scores = [evolution_result.child_agent.fitness_score]
        
        if len(self.evolution_history) > 0:
            recent_fitness = [result.child_agent.fitness_score for result in self.evolution_history[-10:]]
            fitness_scores.extend(recent_fitness)
        
        import statistics
        
        self.fitness_statistics["mean"].append(statistics.mean(fitness_scores))
        self.fitness_statistics["max"].append(max(fitness_scores))
        self.fitness_statistics["min"].append(min(fitness_scores))
        
        if len(fitness_scores) > 1:
            self.fitness_statistics["std"].append(statistics.stdev(fitness_scores))
        else:
            self.fitness_statistics["std"].append(0.0)
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        return {
            "generation_count": self.generation_count,
            "total_evolutions": len(self.evolution_history),
            "successful_evolutions": self.successful_evolutions,
            "total_crossovers": self.total_crossovers,
            "total_mutations": self.total_mutations,
            "fitness_statistics": self.fitness_statistics,
            "success_rate": self.successful_evolutions / max(1, len(self.evolution_history)),
            "parameters": self.parameters.__dict__
        }
    
    def reset_statistics(self):
        """Reset evolution statistics"""
        self.generation_count = 0
        self.evolution_history.clear()
        self.fitness_statistics = {"mean": [], "max": [], "min": [], "std": []}
        self.total_crossovers = 0
        self.total_mutations = 0
        self.successful_evolutions = 0


# Factory functions for easy instantiation
def create_genetic_algorithm(crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                           diversity_pressure: float = 0.3) -> GeneticAlgorithm:
    """
    Factory function to create genetic algorithm with common parameters
    
    Args:
        crossover_rate: Probability of crossover operation
        mutation_rate: Probability of mutation operation
        diversity_pressure: Weight for diversity in selection
        
    Returns:
        Configured GeneticAlgorithm instance
    """
    parameters = GeneticParameters(
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        diversity_pressure=diversity_pressure
    )
    
    return GeneticAlgorithm(parameters=parameters)


def create_high_diversity_ga() -> GeneticAlgorithm:
    """Create genetic algorithm optimized for high diversity"""
    parameters = GeneticParameters(
        crossover_rate=0.9,
        mutation_rate=0.2,
        diversity_pressure=0.5,
        crossover_type=CrossoverType.UNIFORM,
        mutation_type=MutationType.GAUSSIAN,
        selection_type=SelectionType.TOURNAMENT
    )
    
    return GeneticAlgorithm(parameters=parameters)


def create_high_performance_ga() -> GeneticAlgorithm:
    """Create genetic algorithm optimized for high performance"""
    parameters = GeneticParameters(
        crossover_rate=0.8,
        mutation_rate=0.05,
        diversity_pressure=0.1,
        crossover_type=CrossoverType.ARITHMETIC,
        mutation_type=MutationType.GAUSSIAN,
        selection_type=SelectionType.ELITIST,
        elitism_ratio=0.2
    )
    
    return GeneticAlgorithm(parameters=parameters)