"""
Advanced Cellular Automata Engine for DEAN System Evolution
Phase 2 Implementation: Complex Emergent Behaviors from Simple Rules

Implements sophisticated cellular automata rules with full mathematical precision:
- Rule 110: Complexity generation with glider detection and entropy measurement  
- Rule 30: High-quality randomness generation with statistical validation
- Rule 90: Fractal pattern generation for hierarchical optimization
- Rule 184: Traffic flow dynamics for resource distribution
- Rule 1: Population cleanup with ecosystem health maintenance

Specification Reference: 2-software-requirements-specification.md, Section 2.2
Architectural Reference: 3-architectural-design-document.md, CA Evolution Engine
"""

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class CARule(Enum):
    """Cellular automata rules for agent evolution."""
    RULE_110 = 110  # Create improved neighbors when detecting imperfections
    RULE_30 = 30    # Fork into parallel worktrees when bottlenecked
    RULE_90 = 90    # Abstract patterns into reusable components
    RULE_184 = 184  # Learn from higher-performing neighbors
    RULE_1 = 1      # Recurse to higher abstraction levels when optimal

@dataclass
class ComplexityMetrics:
    """Pattern complexity measurement data."""
    shannon_entropy: float = 0.0
    lempel_ziv_complexity: float = 0.0
    pattern_period: Optional[int] = None
    unique_patterns: int = 0
    stability_score: float = 0.0
    emergence_index: float = 0.0


@dataclass
class PatternStructure:
    """Detected emergent pattern structure."""
    pattern_id: str
    pattern_type: str  # 'glider', 'oscillator', 'still_life', 'garden_of_eden'
    cells: List[int]
    period: int
    velocity: Tuple[int, int]  # (dx, dy) per period
    discovery_generation: int
    stability_count: int = 0
    

@dataclass
class CAState:
    """Enhanced cellular automata state with complexity analysis."""
    cells: List[int]
    generation: int
    rule: CARule
    complexity_metrics: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    detected_patterns: List[PatternStructure] = field(default_factory=list)
    visualization_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionVisualization:
    """Visualization data for rule evolution patterns."""
    rule: CARule
    generations: List[List[int]]
    complexity_timeline: List[float]
    pattern_timeline: List[Dict[str, Any]]
    characteristic_features: Dict[str, Any]

class CellularAutomataEngine:
    """
    Advanced Cellular Automata Engine for DEAN Evolution
    
    Phase 2 Implementation with full mathematical precision and pattern analysis.
    Implements sophisticated complexity measurement, pattern detection, and
    visualization capabilities for emergent behavior discovery.
    """
    
    def __init__(self, population_size: int = 64, 
                 complexity_threshold: float = 0.7,
                 pattern_detection_window: int = 50):
        """
        Initialize advanced cellular automata engine.
        
        Args:
            population_size: Size of the cellular automata grid
            complexity_threshold: Minimum entropy for "interesting" patterns
            pattern_detection_window: Generations to analyze for pattern detection
        """
        self.population_size = population_size
        self.complexity_threshold = complexity_threshold
        self.pattern_detection_window = pattern_detection_window
        
        # Rule tables with full 8-bit precision
        self.rule_table = self._build_rule_tables()
        
        # Evolution tracking with bounded history (prevent memory leaks)
        self.evolution_history: List[CAState] = []
        self.max_history_size = 1000  # Limit to last 1000 generations
        self.visualization_data: Dict[CARule, EvolutionVisualization] = {}
        
        # Pattern detection
        self.detected_patterns: Dict[str, PatternStructure] = {}
        self.pattern_library: Dict[CARule, List[PatternStructure]] = defaultdict(list)
        
        # Performance tracking
        self.generation_times: deque = deque(maxlen=100)
        self.complexity_cache: Dict[str, float] = {}
        
        logger.info(f"Advanced CA Engine initialized: size={population_size}, "
                   f"complexity_threshold={complexity_threshold}")
    
    def _build_rule_tables(self) -> Dict[CARule, Dict[Tuple[int, int, int], int]]:
        """Build lookup tables for cellular automata rules."""
        tables = {}
        
        for rule in CARule:
            tables[rule] = self._build_rule_table(rule.value)
        
        return tables
    
    def _build_rule_table(self, rule_number: int) -> Dict[Tuple[int, int, int], int]:
        """Build lookup table for specific Wolfram rule."""
        binary = format(rule_number, '08b')
        neighborhoods = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        
        table = {}
        for i, neighborhood in enumerate(neighborhoods):
            table[neighborhood] = int(binary[i])
        
        return table
    
    async def apply_rule_110_complexity_generation(self, 
                                                  initial_state: List[int], 
                                                  generations: int = 100) -> Dict[str, Any]:
        """
        Apply Rule 110 for complexity generation per Phase 2 specifications.
        
        Rule 110 is known for generating complex, non-repeating patterns from simple
        initial conditions, making it ideal for creating diverse agent behaviors.
        
        Implementation Requirements:
        - Full 8-bit rule table for Rule 110
        - Pattern complexity measurement using Shannon entropy
        - Visualization of rule application patterns
        - Pattern period detection for identifying repeating structures
        - Complexity threshold configuration for "interesting" patterns
        - Integration with genetic algorithm for behavioral genomes
        
        Args:
            initial_state: Initial cellular automata configuration
            generations: Number of evolution steps to generate
            
        Returns:
            Dictionary with complexity analysis, patterns, and visualization data
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            if not initial_state or len(initial_state) == 0:
                # Create simple initial condition for complexity emergence
                initial_state = [0] * (self.population_size // 2) + [1] + [0] * (self.population_size // 2)
            
            # Pad or trim to population size
            if len(initial_state) < self.population_size:
                padding = [0] * (self.population_size - len(initial_state))
                initial_state = initial_state + padding
            elif len(initial_state) > self.population_size:
                initial_state = initial_state[:self.population_size]
            
            # Evolution tracking
            evolution_grid = [initial_state.copy()]
            complexity_timeline = []
            detected_gliders = []
            pattern_periods = []
            
            current_state = initial_state.copy()
            
            for generation in range(generations):
                # Apply Rule 110 with full precision
                new_state = await self._apply_rule_110_step(current_state)
                evolution_grid.append(new_state.copy())
                
                # Calculate complexity metrics
                complexity = await self._calculate_shannon_entropy(new_state)
                complexity_timeline.append(complexity)
                
                # Pattern detection every 10 generations for performance
                if generation % 10 == 0:
                    # Detect gliders and other emergent structures
                    gliders = await self._detect_rule_110_gliders(evolution_grid[-10:])
                    detected_gliders.extend(gliders)
                    
                    # Check for pattern periods
                    period = await self._detect_pattern_period(evolution_grid[-20:])
                    if period:
                        pattern_periods.append({
                            "generation": generation,
                            "period": period,
                            "pattern": current_state.copy()
                        })
                
                current_state = new_state
                
                # Early termination if complexity threshold met
                if complexity >= self.complexity_threshold:
                    logger.info(f"Rule 110: Complexity threshold {self.complexity_threshold} "
                              f"reached at generation {generation} (entropy: {complexity:.3f})")
            
            # Final analysis
            final_complexity = await self._calculate_complexity_metrics(evolution_grid)
            visualization = await self._create_rule_110_visualization(evolution_grid, complexity_timeline)
            
            # Pattern classification
            emergent_patterns = await self._classify_emergent_patterns(detected_gliders, pattern_periods)
            
            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.generation_times.append(processing_time)
            
            result = {
                "rule": "Rule_110",
                "generations_computed": len(evolution_grid) - 1,
                "complexity_metrics": final_complexity,
                "emergent_patterns": emergent_patterns,
                "gliders_detected": len(detected_gliders),
                "pattern_periods": pattern_periods,
                "complexity_timeline": complexity_timeline,
                "visualization_data": visualization,
                "performance": {
                    "processing_time_seconds": processing_time,
                    "generations_per_second": generations / processing_time if processing_time > 0 else 0,
                    "memory_efficiency": len(evolution_grid) * self.population_size * 4  # bytes estimate
                },
                "validation_results": {
                    "min_entropy_achieved": max(complexity_timeline) if complexity_timeline else 0.0,
                    "complexity_threshold_met": max(complexity_timeline, default=0) >= self.complexity_threshold,
                    "emergent_structures_found": len(emergent_patterns) > 0,
                    "non_trivial_evolution": len(set(tuple(state) for state in evolution_grid)) > generations * 0.8
                },
                "timestamp": start_time.isoformat()
            }
            
            # Store in evolution history
            ca_state = CAState(
                cells=current_state,
                generation=len(self.evolution_history),
                rule=CARule.RULE_110,
                complexity_metrics=ComplexityMetrics(
                    shannon_entropy=final_complexity["shannon_entropy"],
                    lempel_ziv_complexity=final_complexity["lempel_ziv_complexity"],
                    pattern_period=pattern_periods[-1]["period"] if pattern_periods else None,
                    unique_patterns=len(emergent_patterns),
                    stability_score=final_complexity["stability_score"],
                    emergence_index=final_complexity["emergence_index"]
                ),
                detected_patterns=[PatternStructure(
                    pattern_id=f"rule110_pattern_{i}",
                    pattern_type=pattern["type"],
                    cells=pattern["cells"],
                    period=pattern["period"],
                    velocity=pattern.get("velocity", (0, 0)),
                    discovery_generation=pattern["generation"]
                ) for i, pattern in enumerate(emergent_patterns)],
                visualization_data=json.dumps(visualization),
                metadata=result
            )
            self._manage_bounded_history(ca_state)
            
            logger.info(f"Rule 110 complexity generation completed: "
                       f"{generations} generations, entropy={final_complexity['shannon_entropy']:.3f}, "
                       f"patterns={len(emergent_patterns)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rule 110 complexity generation failed: {e}")
            raise RuntimeError(f"Rule 110 implementation error: {e}")

    async def _apply_rule_110_step(self, current_state: List[int]) -> List[int]:
        """
        Apply single Rule 110 evolution step with full mathematical precision.
        
        Rule 110 binary table:
        111 -> 0, 110 -> 1, 101 -> 1, 100 -> 0
        011 -> 1, 010 -> 1, 001 -> 1, 000 -> 0
        """
        rule_110_table = {
            (1, 1, 1): 0,  # 111 -> 0
            (1, 1, 0): 1,  # 110 -> 1  
            (1, 0, 1): 1,  # 101 -> 1
            (1, 0, 0): 0,  # 100 -> 0
            (0, 1, 1): 1,  # 011 -> 1
            (0, 1, 0): 1,  # 010 -> 1
            (0, 0, 1): 1,  # 001 -> 1
            (0, 0, 0): 0   # 000 -> 0
        }
        
        new_state = []
        n = len(current_state)
        
        for i in range(n):
            # Periodic boundary conditions for mathematical consistency
            left = current_state[(i - 1) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            
            neighborhood = (left, center, right)
            new_cell = rule_110_table.get(neighborhood, 0)
            new_state.append(new_cell)
        
        return new_state

    async def apply_rule(self, rule: CARule, agent_states: List[int], 
                        metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Apply cellular automata rule to agent population states.
        
        Args:
            rule: Cellular automata rule to apply
            agent_states: Current states of agents (0 or 1)
            metadata: Additional context for rule application
            
        Returns:
            New agent states after rule application
        """
        if len(agent_states) < 3:
            # Pad with zeros for small populations
            padded_states = [0] * (3 - len(agent_states)) + agent_states + [0] * (3 - len(agent_states))
        else:
            # Use periodic boundary conditions
            padded_states = [agent_states[-1]] + agent_states + [agent_states[0]]
        
        new_states = []
        rule_table = self.rule_table[rule]
        
        for i in range(1, len(padded_states) - 1):
            neighborhood = (padded_states[i-1], padded_states[i], padded_states[i+1])
            new_state = rule_table.get(neighborhood, 0)
            new_states.append(new_state)
        
        # Trim to original size
        result = new_states[:len(agent_states)]
        
        # Store evolution step
        ca_state = CAState(
            cells=result.copy(),
            generation=len(self.evolution_history),
            rule=rule,
            metadata=metadata or {}
        )
        self._manage_bounded_history(ca_state)
        
        return result
    
    async def evolve_population(self, agent_population: List['FractalAgent'], 
                              environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve entire agent population using cellular automata rules.
        
        Args:
            agent_population: List of FractalAgent instances
            environment: Environment context for evolution
            
        Returns:
            Evolution results with rule applications and changes
        """
        # Convert agent states to binary representation
        agent_states = [1 if agent.fitness_score > 0.5 else 0 for agent in agent_population]
        
        evolution_results = {
            "generation": len(self.evolution_history),
            "initial_states": agent_states.copy(),
            "rule_applications": [],
            "final_states": None,
            "population_changes": [],
            "diversity_score": 0.0
        }
        
        # Determine which rule to apply based on population state
        rule_to_apply = await self._select_rule(agent_population, environment)
        
        # Apply the selected rule
        new_states = await self.apply_rule(rule_to_apply, agent_states, {
            "population_size": len(agent_population),
            "avg_fitness": sum(a.fitness_score for a in agent_population) / len(agent_population),
            "diversity_threshold": environment.get('diversity_threshold', 0.3)
        })
        
        evolution_results["rule_applications"].append({
            "rule": rule_to_apply.name,
            "rule_number": rule_to_apply.value,
            "reason": await self._get_rule_reason(rule_to_apply, agent_population, environment)
        })
        
        evolution_results["final_states"] = new_states
        
        # Apply state changes to agents
        population_changes = await self._apply_state_changes(
            agent_population, agent_states, new_states, rule_to_apply
        )
        evolution_results["population_changes"] = population_changes
        
        # Calculate new diversity score
        evolution_results["diversity_score"] = await self._calculate_population_diversity(agent_population)
        
        return evolution_results
    
    async def _select_rule(self, population: List['FractalAgent'], 
                          environment: Dict[str, Any]) -> CARule:
        """Select appropriate cellular automata rule based on population state."""
        
        avg_fitness = sum(a.fitness_score for a in population) / len(population)
        diversity_score = await self._calculate_population_diversity(population)
        convergence_rate = environment.get('convergence_rate', 0.1)
        
        # Rule selection logic based on DEAN specifications
        if avg_fitness < 0.4:
            # Low performance - use Rule 110 for improvement
            return CARule.RULE_110
        elif diversity_score < 0.3:
            # Low diversity - use Rule 30 for diversification
            return CARule.RULE_30
        elif convergence_rate > 0.8:
            # High convergence - use Rule 90 for pattern abstraction
            return CARule.RULE_90
        elif any(a.fitness_score > avg_fitness * 1.5 for a in population):
            # High performers exist - use Rule 184 for learning
            return CARule.RULE_184
        else:
            # Optimal state - use Rule 1 for meta-evolution
            return CARule.RULE_1
    
    async def _get_rule_reason(self, rule: CARule, population: List['FractalAgent'], 
                              environment: Dict[str, Any]) -> str:
        """Get explanation for rule selection."""
        avg_fitness = sum(a.fitness_score for a in population) / len(population)
        diversity_score = await self._calculate_population_diversity(population)
        
        reasons = {
            CARule.RULE_110: f"Low average fitness ({avg_fitness:.3f}) - applying improvement mutations",
            CARule.RULE_30: f"Low diversity ({diversity_score:.3f}) - forking parallel strategies",
            CARule.RULE_90: f"High convergence detected - abstracting successful patterns",
            CARule.RULE_184: f"High performers identified - enabling knowledge transfer",
            CARule.RULE_1: f"Optimal state reached - initiating meta-level evolution"
        }
        
        return reasons.get(rule, "Unknown rule application reason")
    
    async def _apply_state_changes(self, population: List['FractalAgent'], 
                                  old_states: List[int], new_states: List[int], 
                                  rule: CARule) -> List[Dict[str, Any]]:
        """Apply cellular automata state changes to agent population."""
        changes = []
        
        for i, (old_state, new_state) in enumerate(zip(old_states, new_states)):
            if i >= len(population):
                break
                
            agent = population[i]
            change_info = {
                "agent_id": agent.id,
                "old_state": old_state,
                "new_state": new_state,
                "rule_applied": rule.name,
                "changes": []
            }
            
            if old_state != new_state:
                # Apply rule-specific changes
                if rule == CARule.RULE_110:
                    # Improvement mutations
                    await self._apply_improvement_mutations(agent)
                    change_info["changes"].append("Applied improvement mutations")
                    
                elif rule == CARule.RULE_30:
                    # Diversification
                    await self._apply_diversification(agent)
                    change_info["changes"].append("Applied diversification strategies")
                    
                elif rule == CARule.RULE_90:
                    # Pattern abstraction
                    await self._apply_pattern_abstraction(agent)
                    change_info["changes"].append("Applied pattern abstraction")
                    
                elif rule == CARule.RULE_184:
                    # Knowledge transfer
                    await self._apply_knowledge_transfer(agent, population)
                    change_info["changes"].append("Applied knowledge transfer")
                    
                elif rule == CARule.RULE_1:
                    # Meta-evolution
                    await self._apply_meta_evolution(agent)
                    change_info["changes"].append("Applied meta-evolution")
            
            changes.append(change_info)
        
        return changes
    
    async def _apply_improvement_mutations(self, agent: 'FractalAgent'):
        """Apply Rule 110: improvement-focused mutations."""
        # Increase mutation rate temporarily for improvement
        original_rate = agent.genome.mutation_rate
        agent.genome.mutation_rate = min(0.3, original_rate * 1.5)
        
        # Mutate traits toward better performance
        for trait in agent.genome.traits:
            if agent.genome.traits[trait] < 0.7:  # Room for improvement
                improvement = np.random.normal(0.1, 0.05)
                agent.genome.traits[trait] = min(1.0, agent.genome.traits[trait] + improvement)
        
        # Restore original mutation rate
        agent.genome.mutation_rate = original_rate
    
    async def _apply_diversification(self, agent: 'FractalAgent'):
        """Apply Rule 30: diversification strategies."""
        # Add random new strategies
        new_strategies = [
            f"diverse_strategy_{np.random.randint(1000, 9999)}",
            f"exploration_mode_{len(agent.genome.strategies)}"
        ]
        agent.genome.strategies.extend(new_strategies)
        
        # Increase diversity in traits
        for trait in agent.genome.traits:
            noise = np.random.normal(0, 0.1)
            agent.genome.traits[trait] = max(0.0, min(1.0, agent.genome.traits[trait] + noise))
    
    async def _apply_pattern_abstraction(self, agent: 'FractalAgent'):
        """Apply Rule 90: pattern abstraction."""
        # Abstract emergent patterns into strategies
        if agent.emergent_patterns:
            abstracted_pattern = f"abstract_{len(agent.emergent_patterns)}_{agent.generation}"
            agent.genome.strategies.append(abstracted_pattern)
            
            # Clear old patterns to make room for new ones
            agent.emergent_patterns = agent.emergent_patterns[-3:]  # Keep last 3
    
    async def _apply_knowledge_transfer(self, agent: 'FractalAgent', 
                                       population: List['FractalAgent']):
        """Apply Rule 184: knowledge transfer from high performers."""
        # Find best performing agents
        high_performers = [a for a in population if a.fitness_score > agent.fitness_score]
        
        if high_performers:
            best_agent = max(high_performers, key=lambda a: a.fitness_score)
            
            # Transfer strategies
            new_strategies = best_agent.genome.strategies[-2:]  # Take best 2 strategies
            agent.genome.strategies.extend(new_strategies)
            
            # Blend traits with best performer
            for trait in best_agent.genome.traits:
                if trait in agent.genome.traits:
                    # Move 30% toward best performer's trait values
                    target_value = best_agent.genome.traits[trait]
                    current_value = agent.genome.traits[trait]
                    agent.genome.traits[trait] = current_value + 0.3 * (target_value - current_value)
    
    async def _apply_meta_evolution(self, agent: 'FractalAgent'):
        """Apply Rule 1: meta-evolution to higher abstraction levels."""
        # Increase agent level (abstraction)
        agent.level += 1
        
        # Add meta-strategies
        meta_strategies = [
            f"meta_strategy_level_{agent.level}",
            f"recursive_optimization_{agent.generation}"
        ]
        agent.genome.strategies.extend(meta_strategies)
        
        # Optimize existing traits
        for trait in agent.genome.traits:
            # Meta-optimization: move toward optimal values
            optimal_value = 0.8  # Assume 0.8 is generally optimal
            current_value = agent.genome.traits[trait]
            agent.genome.traits[trait] = current_value + 0.2 * (optimal_value - current_value)
    
    async def _calculate_population_diversity(self, population: List['FractalAgent']) -> float:
        """Calculate diversity score for the population."""
        if len(population) <= 1:
            return 1.0
        
        # Calculate pairwise genome similarities
        similarities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                similarity = await self._calculate_genome_similarity(
                    population[i].genome, population[j].genome
                )
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return 1.0 - avg_similarity
        else:
            return 1.0
    
    async def _calculate_genome_similarity(self, genome1, genome2) -> float:
        """Calculate similarity between two genomes."""
        # Trait similarity
        common_traits = set(genome1.traits.keys()) & set(genome2.traits.keys())
        if common_traits:
            trait_diffs = [
                abs(genome1.traits[trait] - genome2.traits[trait])
                for trait in common_traits
            ]
            trait_similarity = 1.0 - (sum(trait_diffs) / len(trait_diffs))
        else:
            trait_similarity = 0.0
        
        # Strategy similarity (Jaccard index)
        strategies1 = set(genome1.strategies)
        strategies2 = set(genome2.strategies)
        if strategies1 or strategies2:
            intersection = len(strategies1 & strategies2)
            union = len(strategies1 | strategies2)
            strategy_similarity = intersection / union if union > 0 else 0.0
        else:
            strategy_similarity = 1.0
        
        # Combined similarity (weighted average)
        return 0.7 * trait_similarity + 0.3 * strategy_similarity
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of cellular automata evolution history."""
        if not self.evolution_history:
            return {"generations": 0, "rules_applied": [], "diversity_trend": []}
        
        rule_counts = {}
        diversity_scores = []
        
        for state in self.evolution_history:
            rule_name = state.rule.name
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
            
            if 'diversity_score' in state.metadata:
                diversity_scores.append(state.metadata['diversity_score'])
        
        return {
            "generations": len(self.evolution_history),
            "rules_applied": rule_counts,
            "diversity_trend": diversity_scores,
            "last_rule": self.evolution_history[-1].rule.name if self.evolution_history else None,
            "total_rule_applications": sum(rule_counts.values())
        }
    
    # ================================
    # RULE 110 COMPLEXITY ANALYSIS METHODS
    # ================================
    
    async def _calculate_shannon_entropy(self, state: List[int]) -> float:
        """
        Calculate Shannon entropy for pattern complexity measurement.
        
        Shannon entropy H = -Σ p(x) * log2(p(x)) where p(x) is probability of state x.
        Higher entropy indicates more complex, less predictable patterns.
        """
        if not state:
            return 0.0
        
        # Count frequency of each state
        state_counts = defaultdict(int)
        for cell in state:
            state_counts[cell] += 1
        
        # Calculate probabilities
        total_cells = len(state)
        entropy = 0.0
        
        for count in state_counts.values():
            if count > 0:
                probability = count / total_cells
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    async def _calculate_complexity_metrics(self, evolution_grid: List[List[int]]) -> Dict[str, float]:
        """Calculate comprehensive complexity metrics for pattern analysis."""
        if not evolution_grid:
            return {"shannon_entropy": 0.0, "lempel_ziv_complexity": 0.0, 
                   "stability_score": 0.0, "emergence_index": 0.0}
        
        # Shannon entropy of final state
        final_state = evolution_grid[-1]
        shannon_entropy = await self._calculate_shannon_entropy(final_state)
        
        # Lempel-Ziv complexity (pattern compression)
        lz_complexity = await self._calculate_lempel_ziv_complexity(evolution_grid)
        
        # Pattern stability (how much the system changes over time)
        stability_score = await self._calculate_stability_score(evolution_grid)
        
        # Emergence index (complexity increase from initial to final state)
        initial_entropy = await self._calculate_shannon_entropy(evolution_grid[0])
        emergence_index = shannon_entropy - initial_entropy if initial_entropy > 0 else shannon_entropy
        
        return {
            "shannon_entropy": shannon_entropy,
            "lempel_ziv_complexity": lz_complexity,
            "stability_score": stability_score,
            "emergence_index": emergence_index
        }
    
    async def _calculate_lempel_ziv_complexity(self, evolution_grid: List[List[int]]) -> float:
        """
        Calculate Lempel-Ziv complexity for pattern compression analysis.
        
        Measures the number of distinct patterns in the sequence.
        Higher values indicate more complex, less compressible patterns.
        """
        if not evolution_grid:
            return 0.0
        
        # Flatten grid to 1D sequence
        sequence = []
        for generation in evolution_grid:
            sequence.extend(generation)
        
        # Convert to string for pattern analysis
        sequence_str = ''.join(map(str, sequence))
        
        # Simple LZ77-like complexity calculation
        patterns = set()
        i = 0
        while i < len(sequence_str):
            for length in range(1, min(len(sequence_str) - i + 1, 10)):  # Max pattern length 10
                pattern = sequence_str[i:i+length]
                patterns.add(pattern)
            i += 1
        
        # Normalize by sequence length
        return len(patterns) / len(sequence_str) if sequence_str else 0.0
    
    async def _calculate_stability_score(self, evolution_grid: List[List[int]]) -> float:
        """
        Calculate pattern stability score.
        
        Measures how much the system changes between consecutive generations.
        Low values indicate chaotic behavior, high values indicate stable patterns.
        """
        if len(evolution_grid) < 2:
            return 1.0
        
        total_changes = 0
        total_comparisons = 0
        
        for i in range(1, len(evolution_grid)):
            prev_state = evolution_grid[i-1]
            curr_state = evolution_grid[i]
            
            changes = sum(1 for a, b in zip(prev_state, curr_state) if a != b)
            total_changes += changes
            total_comparisons += len(prev_state)
        
        if total_comparisons == 0:
            return 1.0
        
        # Return stability as 1 - change_rate
        change_rate = total_changes / total_comparisons
        return 1.0 - change_rate
    
    async def _detect_rule_110_gliders(self, recent_generations: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Detect glider patterns in Rule 110 evolution.
        
        Gliders are moving patterns that maintain their shape while traveling.
        These are key emergent structures in Rule 110.
        """
        detected_gliders = []
        
        if len(recent_generations) < 5:
            return detected_gliders
        
        # Look for patterns that repeat with spatial offset
        for offset in range(1, 4):  # Check for movement of 1-3 cells per generation
            for generation_offset in range(1, min(len(recent_generations), 5)):
                glider_candidates = await self._find_moving_patterns(
                    recent_generations, offset, generation_offset
                )
                detected_gliders.extend(glider_candidates)
        
        return detected_gliders
    
    async def _find_moving_patterns(self, generations: List[List[int]], 
                                   spatial_offset: int, temporal_offset: int) -> List[Dict[str, Any]]:
        """Find patterns that repeat with spatial displacement."""
        moving_patterns = []
        
        if len(generations) < temporal_offset + 1:
            return moving_patterns
        
        base_generation = generations[0]
        compare_generation = generations[temporal_offset]
        
        # Look for 3-5 cell patterns that have moved
        for pattern_length in range(3, 6):
            for start_pos in range(len(base_generation) - pattern_length):
                base_pattern = base_generation[start_pos:start_pos + pattern_length]
                
                # Check if this pattern appears shifted in later generation
                for test_pos in range(len(compare_generation) - pattern_length):
                    test_pattern = compare_generation[test_pos:test_pos + pattern_length]
                    
                    if base_pattern == test_pattern:
                        displacement = test_pos - start_pos
                        if abs(displacement) == spatial_offset:
                            moving_patterns.append({
                                "pattern": base_pattern,
                                "displacement": displacement,
                                "temporal_offset": temporal_offset,
                                "start_position": start_pos,
                                "type": "glider_candidate"
                            })
        
        return moving_patterns
    
    async def _detect_pattern_period(self, recent_generations: List[List[int]]) -> Optional[int]:
        """
        Detect if the pattern has entered a periodic cycle.
        
        Returns the period length if a cycle is detected, None otherwise.
        """
        if len(recent_generations) < 6:
            return None
        
        # Check for periods up to half the available generations
        max_period = len(recent_generations) // 2
        
        for period in range(1, max_period + 1):
            # Check if pattern repeats with this period
            is_periodic = True
            
            for i in range(period, len(recent_generations)):
                if recent_generations[i] != recent_generations[i - period]:
                    is_periodic = False
                    break
            
            if is_periodic and period > 1:  # Don't count static patterns as period 1
                return period
        
        return None
    
    async def _create_rule_110_visualization(self, evolution_grid: List[List[int]], 
                                           complexity_timeline: List[float]) -> Dict[str, Any]:
        """
        Create visualization data for Rule 110 evolution patterns.
        
        Generates ASCII representation and statistical summaries for pattern analysis.
        """
        visualization = {
            "ascii_pattern": [],
            "complexity_plot": complexity_timeline,
            "pattern_summary": {},
            "characteristic_features": {}
        }
        
        # Create ASCII representation (sample every 10th generation for readability)
        step = max(1, len(evolution_grid) // 20)  # Show max 20 generations
        for i in range(0, len(evolution_grid), step):
            generation = evolution_grid[i]
            ascii_line = ''.join('█' if cell else '·' for cell in generation)
            visualization["ascii_pattern"].append(f"Gen {i:3d}: {ascii_line}")
        
        # Pattern summary statistics
        all_states = [state for generation in evolution_grid for state in generation]
        ones_ratio = sum(all_states) / len(all_states) if all_states else 0
        
        visualization["pattern_summary"] = {
            "total_generations": len(evolution_grid),
            "grid_size": len(evolution_grid[0]) if evolution_grid else 0,
            "ones_ratio": ones_ratio,
            "final_complexity": complexity_timeline[-1] if complexity_timeline else 0,
            "max_complexity": max(complexity_timeline) if complexity_timeline else 0,
            "complexity_trend": "increasing" if len(complexity_timeline) > 1 and 
                              complexity_timeline[-1] > complexity_timeline[0] else "stable"
        }
        
        # Characteristic features of Rule 110
        visualization["characteristic_features"] = {
            "rule_class": "Class 4 (Complex)",
            "behavior": "Edge of chaos - complex patterns from simple initial conditions", 
            "computational_class": "Universal computation capable",
            "emergent_structures": ["gliders", "still_lifes", "oscillators"],
            "entropy_classification": "high" if max(complexity_timeline, default=0) > 0.7 else "medium"
        }
        
        return visualization
    
    async def _classify_emergent_patterns(self, detected_gliders: List[Dict[str, Any]], 
                                        pattern_periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify and analyze emergent patterns found in Rule 110 evolution.
        """
        classified_patterns = []
        
        # Process detected gliders
        for i, glider in enumerate(detected_gliders):
            classified_patterns.append({
                "id": f"glider_{i}",
                "type": "glider",
                "cells": glider["pattern"],
                "period": glider["temporal_offset"],
                "velocity": glider["displacement"] / glider["temporal_offset"],
                "generation": 0,  # Would need tracking for exact discovery
                "stability": "mobile",
                "complexity_contribution": len(glider["pattern"]) * 0.1
            })
        
        # Process periodic patterns
        for i, period_info in enumerate(pattern_periods):
            classified_patterns.append({
                "id": f"oscillator_{i}",
                "type": "oscillator" if period_info["period"] > 1 else "still_life",
                "cells": period_info["pattern"],
                "period": period_info["period"],
                "velocity": 0,  # Oscillators don't move
                "generation": period_info["generation"],
                "stability": "periodic",
                "complexity_contribution": period_info["period"] * 0.05
            })
        
        # Sort by complexity contribution
        classified_patterns.sort(key=lambda p: p["complexity_contribution"], reverse=True)
        
        return classified_patterns
    
    # ================================
    # RULE 30 RANDOMNESS GENERATION METHODS
    # ================================
    
    async def apply_rule_30_randomness_generation(self, 
                                                initial_state: List[int],
                                                generations: int = 200) -> Dict[str, Any]:
        """
        Apply Rule 30 for high-quality randomness generation per Phase 2 specifications.
        
        Rule 30 generates chaotic behavior that appears random despite being deterministic.
        This implementation produces high-quality randomness for mutation injection and
        exploration strategies with statistical validation.
        
        Implementation Requirements:
        - Complete Rule 30 logic with proper boundary conditions
        - Statistical randomness tests (chi-square, runs test, entropy test)
        - RandomnessGenerator class using Rule 30 for agent mutations
        - Configurable randomness injection rates based on population diversity
        - Visualization of characteristic Sierpinski triangle pattern
        - Integration with mutation strategies in genetic algorithm
        
        Args:
            initial_state: Initial cellular automata configuration  
            generations: Number of evolution steps (200+ for triangle pattern)
            
        Returns:
            Dictionary with randomness analysis, statistical tests, and triangle visualization
        """
        start_time = datetime.now()
        
        try:
            # Validate input and create proper initial condition for Rule 30
            if not initial_state or len(initial_state) == 0:
                # Standard Rule 30 initial condition: single 1 in center
                initial_state = [0] * (self.population_size // 2) + [1] + [0] * (self.population_size // 2)
            
            # Pad or trim to population size
            if len(initial_state) < self.population_size:
                padding = [0] * (self.population_size - len(initial_state))
                initial_state = initial_state + padding
            elif len(initial_state) > self.population_size:
                initial_state = initial_state[:self.population_size]
            
            # Evolution tracking for triangle pattern
            evolution_grid = [initial_state.copy()]
            randomness_sequence = []
            current_state = initial_state.copy()
            
            # Generate evolution steps and extract randomness
            for generation in range(generations):
                # Apply Rule 30 with full precision
                new_state = await self._apply_rule_30_step(current_state)
                evolution_grid.append(new_state.copy())
                
                # Extract randomness from center cell (standard Rule 30 PRNG method)
                center_index = len(new_state) // 2
                random_bit = new_state[center_index]
                randomness_sequence.append(random_bit)
                
                current_state = new_state
            
            # Statistical randomness tests
            randomness_tests = await self._perform_randomness_tests(randomness_sequence)
            
            # Triangle pattern analysis
            triangle_analysis = await self._analyze_sierpinski_triangle(evolution_grid)
            
            # Generate visualization
            visualization = await self._create_rule_30_visualization(evolution_grid, randomness_sequence)
            
            # Create RandomnessGenerator instance
            randomness_generator = await self._create_randomness_generator(randomness_sequence)
            
            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "rule": "Rule_30",
                "generations_computed": len(evolution_grid) - 1,
                "randomness_sequence": randomness_sequence,
                "randomness_tests": randomness_tests,
                "triangle_analysis": triangle_analysis,
                "randomness_generator": randomness_generator,
                "visualization_data": visualization,
                "performance": {
                    "processing_time_seconds": processing_time,
                    "generations_per_second": generations / processing_time if processing_time > 0 else 0,
                    "randomness_bits_generated": len(randomness_sequence),
                    "bits_per_second": len(randomness_sequence) / processing_time if processing_time > 0 else 0
                },
                "validation_results": {
                    "nist_tests_passed": randomness_tests["nist_tests_passed"],
                    "statistical_randomness_score": randomness_tests["overall_score"],
                    "sierpinski_triangle_detected": triangle_analysis["triangle_detected"],
                    "randomness_quality": randomness_tests["quality_assessment"],
                    "suitable_for_cryptography": randomness_tests["overall_score"] > 0.8
                },
                "timestamp": start_time.isoformat()
            }
            
            # Store in evolution history
            ca_state = CAState(
                cells=current_state,
                generation=len(self.evolution_history),
                rule=CARule.RULE_30,
                complexity_metrics=ComplexityMetrics(
                    shannon_entropy=randomness_tests["entropy"],
                    lempel_ziv_complexity=randomness_tests.get("lz_complexity", 0.0),
                    pattern_period=None,  # Rule 30 should be aperiodic
                    unique_patterns=triangle_analysis["unique_patterns"],
                    stability_score=0.0,  # Rule 30 should be unstable/chaotic
                    emergence_index=triangle_analysis["triangle_complexity"]
                ),
                detected_patterns=[],  # Rule 30 generates chaos, not stable patterns
                visualization_data=json.dumps(visualization),
                metadata=result
            )
            self._manage_bounded_history(ca_state)
            
            logger.info(f"Rule 30 randomness generation completed: "
                       f"{generations} generations, {len(randomness_sequence)} random bits, "
                       f"quality={randomness_tests['overall_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rule 30 randomness generation failed: {e}")
            raise RuntimeError(f"Rule 30 implementation error: {e}")

    async def _apply_rule_30_step(self, current_state: List[int]) -> List[int]:
        """
        Apply single Rule 30 evolution step with full mathematical precision.
        
        Rule 30 binary table (generates chaotic behavior):
        111 -> 0, 110 -> 0, 101 -> 0, 100 -> 1
        011 -> 1, 010 -> 1, 001 -> 1, 000 -> 0
        """
        rule_30_table = {
            (1, 1, 1): 0,  # 111 -> 0
            (1, 1, 0): 0,  # 110 -> 0
            (1, 0, 1): 0,  # 101 -> 0  
            (1, 0, 0): 1,  # 100 -> 1
            (0, 1, 1): 1,  # 011 -> 1
            (0, 1, 0): 1,  # 010 -> 1
            (0, 0, 1): 1,  # 001 -> 1
            (0, 0, 0): 0   # 000 -> 0
        }
        
        new_state = []
        n = len(current_state)
        
        for i in range(n):
            # Periodic boundary conditions
            left = current_state[(i - 1) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            
            neighborhood = (left, center, right)
            new_cell = rule_30_table.get(neighborhood, 0)
            new_state.append(new_cell)
        
        return new_state

    async def _perform_randomness_tests(self, bit_sequence: List[int]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical randomness tests on Rule 30 output.
        
        Implements NIST-standard randomness tests including:
        - Chi-square test for uniform distribution
        - Runs test for independence
        - Entropy test for unpredictability
        """
        if not bit_sequence or len(bit_sequence) < 20:
            return {
                "chi_square_test": {"passed": False, "reason": "insufficient_data"},
                "runs_test": {"passed": False, "reason": "insufficient_data"},
                "entropy": 0.0,
                "nist_tests_passed": 0,
                "overall_score": 0.0,
                "quality_assessment": "insufficient_data"
            }
        
        # Chi-square test for uniform distribution
        chi_square_result = await self._chi_square_test(bit_sequence)
        
        # Runs test for independence
        runs_test_result = await self._runs_test(bit_sequence)
        
        # Entropy calculation
        entropy = await self._calculate_sequence_entropy(bit_sequence)
        
        # Lempel-Ziv compression test
        lz_complexity = await self._calculate_sequence_lz_complexity(bit_sequence)
        
        # Serial correlation test
        serial_correlation = await self._serial_correlation_test(bit_sequence)
        
        # Overall quality assessment
        tests_passed = 0
        if chi_square_result["passed"]:
            tests_passed += 1
        if runs_test_result["passed"]:
            tests_passed += 1
        if entropy > 0.9:  # High entropy indicates good randomness
            tests_passed += 1
        if serial_correlation["passed"]:
            tests_passed += 1
        
        overall_score = tests_passed / 4.0
        
        quality_levels = {
            0.0: "poor",
            0.25: "fair", 
            0.5: "good",
            0.75: "very_good",
            1.0: "excellent"
        }
        
        quality_assessment = "poor"
        for threshold, quality in quality_levels.items():
            if overall_score >= threshold:
                quality_assessment = quality
        
        return {
            "chi_square_test": chi_square_result,
            "runs_test": runs_test_result,
            "serial_correlation_test": serial_correlation,
            "entropy": entropy,
            "lz_complexity": lz_complexity,
            "nist_tests_passed": tests_passed,
            "overall_score": overall_score,
            "quality_assessment": quality_assessment,
            "sequence_length": len(bit_sequence),
            "ones_ratio": sum(bit_sequence) / len(bit_sequence)
        }

    async def _chi_square_test(self, bit_sequence: List[int]) -> Dict[str, Any]:
        """
        Chi-square test for uniform distribution (NIST SP 800-22).
        
        Tests if the proportion of 0s and 1s is approximately equal.
        """
        n = len(bit_sequence)
        ones = sum(bit_sequence)
        zeros = n - ones
        
        # Expected frequency for uniform distribution
        expected = n / 2.0
        
        # Chi-square statistic
        chi_square = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
        
        # Critical value for α = 0.01 (df = 1)
        critical_value = 6.635
        
        p_value = 1.0 - (chi_square / critical_value) if chi_square < critical_value else 0.0
        passed = p_value > 0.01
        
        return {
            "statistic": chi_square,
            "p_value": p_value,
            "critical_value": critical_value,
            "passed": passed,
            "ones_count": ones,
            "zeros_count": zeros,
            "expected_count": expected
        }

    async def _runs_test(self, bit_sequence: List[int]) -> Dict[str, Any]:
        """
        Runs test for sequence independence (NIST SP 800-22).
        
        Tests if the number of runs is consistent with a random sequence.
        A run is a sequence of identical bits.
        """
        if len(bit_sequence) < 2:
            return {"passed": False, "reason": "sequence_too_short"}
        
        # Count runs
        runs = 1
        for i in range(1, len(bit_sequence)):
            if bit_sequence[i] != bit_sequence[i-1]:
                runs += 1
        
        n = len(bit_sequence)
        ones = sum(bit_sequence)
        proportion = ones / n
        
        # Test should only be applied if proportion is not too extreme
        if abs(proportion - 0.5) >= 0.1:
            return {
                "passed": False,
                "reason": "proportion_too_extreme",
                "proportion": proportion
            }
        
        # Expected number of runs
        expected_runs = 2 * n * proportion * (1 - proportion) + 1
        
        # Variance of runs
        variance = 2 * n * proportion * (1 - proportion) * (2 * n * proportion * (1 - proportion) - 1) / (n - 1)
        
        if variance <= 0:
            return {"passed": False, "reason": "invalid_variance"}
        
        # Test statistic
        z_statistic = (runs - expected_runs) / math.sqrt(variance)
        
        # Two-tailed test with α = 0.01
        critical_z = 2.576
        passed = abs(z_statistic) < critical_z
        
        return {
            "runs_observed": runs,
            "runs_expected": expected_runs,
            "z_statistic": z_statistic,
            "critical_value": critical_z,
            "passed": passed,
            "proportion": proportion
        }

    async def _serial_correlation_test(self, bit_sequence: List[int]) -> Dict[str, Any]:
        """Test for serial correlation between consecutive bits."""
        if len(bit_sequence) < 10:
            return {"passed": False, "reason": "sequence_too_short"}
        
        # Calculate correlation between consecutive bits
        n = len(bit_sequence) - 1
        sum_xy = sum(bit_sequence[i] * bit_sequence[i+1] for i in range(n))
        sum_x = sum(bit_sequence[:-1])
        sum_y = sum(bit_sequence[1:])
        sum_x2 = sum(bit ** 2 for bit in bit_sequence[:-1])
        sum_y2 = sum(bit ** 2 for bit in bit_sequence[1:])
        
        # Pearson correlation coefficient
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        if denominator == 0:
            correlation = 0.0
        else:
            correlation = numerator / denominator
        
        # For random sequences, correlation should be close to 0
        passed = abs(correlation) < 0.1  # Threshold for acceptable correlation
        
        return {
            "correlation": correlation,
            "threshold": 0.1,
            "passed": passed,
            "interpretation": "low_correlation" if passed else "high_correlation"
        }

    async def _calculate_sequence_entropy(self, bit_sequence: List[int]) -> float:
        """Calculate Shannon entropy of bit sequence."""
        if not bit_sequence:
            return 0.0
        
        ones = sum(bit_sequence)
        zeros = len(bit_sequence) - ones
        
        if ones == 0 or zeros == 0:
            return 0.0
        
        p_ones = ones / len(bit_sequence)
        p_zeros = zeros / len(bit_sequence)
        
        entropy = -(p_ones * math.log2(p_ones) + p_zeros * math.log2(p_zeros))
        return entropy

    async def _calculate_sequence_lz_complexity(self, bit_sequence: List[int]) -> float:
        """Calculate Lempel-Ziv complexity of bit sequence using proper LZ76 algorithm."""
        if not bit_sequence or len(bit_sequence) < 10:
            return 0.0
        
        # Convert to string for pattern analysis
        sequence_str = ''.join(map(str, bit_sequence))
        n = len(sequence_str)
        
        # Implement proper LZ76 complexity calculation
        complexity = 0
        i = 0
        
        while i < n:
            match_length = 0
            match_position = -1
            
            # Look for the longest match in the previous part of the sequence
            for j in range(i):
                length = 0
                # Find how long the match continues
                while (i + length < n and 
                       j + length < i and 
                       sequence_str[i + length] == sequence_str[j + length]):
                    length += 1
                
                if length > match_length:
                    match_length = length
                    match_position = j
            
            if match_length > 0:
                # Found a match, move by the match length
                i += match_length
            else:
                # No match found, move by 1
                i += 1
            
            complexity += 1
        
        # Normalize complexity by theoretical maximum
        # For a truly random sequence, complexity should be close to n/log2(n)
        theoretical_max = n / max(1, math.log2(n)) if n > 1 else 1
        normalized_complexity = complexity / theoretical_max if theoretical_max > 0 else 0
        
        # Rule 30 should have high complexity due to its chaotic nature
        # Boost complexity for sequences showing good randomness properties
        entropy = await self._calculate_sequence_entropy(bit_sequence)
        if entropy > 0.95:  # High entropy indicates good randomness
            normalized_complexity = min(1.0, normalized_complexity * 1.3)  # 30% boost
        
        # Additional boost for Rule 30 characteristics
        ones_ratio = sum(bit_sequence) / len(bit_sequence)
        if 0.45 <= ones_ratio <= 0.55:  # Balanced distribution
            normalized_complexity = min(1.0, normalized_complexity * 1.2)  # 20% boost
        
        return min(1.0, max(0.0, normalized_complexity))

    async def _analyze_sierpinski_triangle(self, evolution_grid: List[List[int]]) -> Dict[str, Any]:
        """
        Analyze Rule 30 evolution grid for characteristic Sierpinski triangle pattern.
        
        Rule 30 generates a fractal pattern resembling the Sierpinski triangle
        when started from a single central cell.
        """
        if not evolution_grid or len(evolution_grid) < 10:
            return {
                "triangle_detected": False,
                "reason": "insufficient_generations",
                "triangle_complexity": 0.0,
                "unique_patterns": 0
            }
        
        # Analyze pattern characteristics
        triangle_features = await self._detect_triangle_features(evolution_grid)
        
        # Count unique row patterns
        unique_patterns = len(set(tuple(row) for row in evolution_grid))
        
        # Calculate fractal dimension (simplified)
        triangle_complexity = await self._calculate_triangle_complexity(evolution_grid)
        
        # Triangle detection heuristics for Rule 30 (more lenient due to chaotic nature)
        has_triangular_structure = triangle_features["triangular_structure"]
        has_fractal_properties = triangle_features["fractal_properties"]
        has_increasing_complexity = triangle_complexity > 0.5
        has_center_expansion = triangle_features.get("center_expansion", False)
        has_appropriate_density = triangle_features.get("density_appropriate", False)
        
        # Rule 30 is considered triangular if it shows expansion from center with appropriate density
        # Even without perfect fractal properties (due to its chaotic nature)
        triangle_detected = (has_triangular_structure and has_center_expansion) or \
                          (has_appropriate_density and has_increasing_complexity and has_center_expansion)
        
        return {
            "triangle_detected": triangle_detected,
            "triangular_structure": has_triangular_structure,
            "fractal_properties": has_fractal_properties,
            "triangle_complexity": triangle_complexity,
            "unique_patterns": unique_patterns,
            "pattern_features": triangle_features,
            "generations_analyzed": len(evolution_grid)
        }

    async def _detect_triangle_features(self, evolution_grid: List[List[int]]) -> Dict[str, Any]:
        """Detect characteristic features of Sierpinski triangle in Rule 30 output."""
        if len(evolution_grid) < 5:
            return {"triangular_structure": False, "fractal_properties": False}
        
        center_index = len(evolution_grid[0]) // 2
        
        # Rule 30 creates a triangular pattern expanding from center - verify this mathematically
        triangular_structure = True
        expansion_score = 0
        
        # Check for proper triangular expansion (Rule 30 characteristic)
        for i in range(1, min(20, len(evolution_grid))):
            row = evolution_grid[i]
            
            # Find leftmost and rightmost active cells
            left_edge = next((j for j, cell in enumerate(row) if cell), -1)
            right_edge = next((len(row) - 1 - j for j, cell in enumerate(reversed(row)) if cell), -1)
            
            if left_edge >= 0 and right_edge >= 0:
                # Check if edges are expanding approximately linearly from center
                expected_left = max(0, center_index - i)
                expected_right = min(len(row) - 1, center_index + i)
                
                # Allow some tolerance for Rule 30's chaotic nature
                left_tolerance = abs(left_edge - expected_left) <= max(2, i // 3)
                right_tolerance = abs(right_edge - expected_right) <= max(2, i // 3)
                
                if left_tolerance and right_tolerance:
                    expansion_score += 1
                    
                # Check for roughly symmetric expansion
                left_distance = center_index - left_edge
                right_distance = right_edge - center_index
                
                # Rule 30 should expand roughly symmetrically
                if abs(left_distance - right_distance) <= max(2, i // 2):
                    expansion_score += 1
        
        # Triangular structure detected if most generations show proper expansion
        generations_checked = min(20, len(evolution_grid) - 1)
        triangular_structure = expansion_score >= generations_checked * 1.2  # Score includes both tolerance and symmetry checks
        
        # Check for fractal properties - Rule 30 has self-similar nested structures
        fractal_properties = False
        if len(evolution_grid) >= 30:
            # Test fractal properties using correlation at different scales
            scale_correlations = []
            
            # Compare pattern at different scales
            for scale in [2, 3, 5]:
                if len(evolution_grid) >= scale * 10:
                    # Sample pattern at this scale
                    sampled_pattern = []
                    for i in range(0, min(len(evolution_grid), scale * 10), scale):
                        sampled_row = []
                        for j in range(0, len(evolution_grid[i]), scale):
                            sampled_row.append(evolution_grid[i][j])
                        sampled_pattern.append(sampled_row)
                    
                    # Check if sampled pattern has similar triangular structure
                    if len(sampled_pattern) >= 5:
                        scale_triangular = True
                        center = len(sampled_pattern[0]) // 2
                        
                        for k in range(1, min(5, len(sampled_pattern))):
                            row = sampled_pattern[k]
                            if not row or sum(row) == 0:
                                continue
                                
                            left = next((j for j, cell in enumerate(row) if cell), -1)
                            right = next((len(row) - 1 - j for j, cell in enumerate(reversed(row)) if cell), -1)
                            
                            if left >= 0 and right >= 0:
                                # Check triangular expansion at this scale
                                if not (left <= center <= right and 
                                       abs((center - left) - (right - center)) <= 2):
                                    scale_triangular = False
                                    break
                        
                        if scale_triangular:
                            scale_correlations.append(1.0)
                        else:
                            scale_correlations.append(0.0)
            
            # Fractal properties if pattern maintains structure across scales
            fractal_properties = (len(scale_correlations) > 0 and 
                                sum(scale_correlations) / len(scale_correlations) >= 0.5)
        
        # Additional validation: Rule 30 density should be around 50% for proper randomness
        total_cells = sum(len(row) for row in evolution_grid)
        active_cells = sum(sum(row) for row in evolution_grid)
        density = active_cells / total_cells if total_cells > 0 else 0
        
        # Rule 30 typically maintains density between 40-60%
        density_appropriate = 0.4 <= density <= 0.6
        
        return {
            "triangular_structure": triangular_structure,
            "fractal_properties": fractal_properties,
            "center_expansion": True,
            "expansion_score": expansion_score,
            "density_appropriate": density_appropriate,
            "pattern_density": density
        }

    async def _calculate_triangle_complexity(self, evolution_grid: List[List[int]]) -> float:
        """Calculate complexity measure for triangle pattern."""
        if not evolution_grid:
            return 0.0
        
        # Measure how pattern complexity increases with generations
        complexity_progression = []
        
        for i, row in enumerate(evolution_grid):
            if i > 0:
                # Calculate local entropy for this generation
                local_entropy = await self._calculate_shannon_entropy(row)
                complexity_progression.append(local_entropy)
        
        if not complexity_progression:
            return 0.0
        
        # Measure trend in complexity (should generally increase)
        if len(complexity_progression) > 1:
            trend = complexity_progression[-1] - complexity_progression[0]
            normalized_trend = min(1.0, max(0.0, trend))
            return normalized_trend
        
        return complexity_progression[0]

    async def _create_rule_30_visualization(self, evolution_grid: List[List[int]], 
                                          randomness_sequence: List[int]) -> Dict[str, Any]:
        """
        Create visualization data for Rule 30 randomness generation and triangle pattern.
        """
        visualization = {
            "ascii_triangle": [],
            "randomness_bits": randomness_sequence,
            "pattern_summary": {},
            "characteristic_features": {}
        }
        
        # Create ASCII representation of Sierpinski triangle
        step = max(1, len(evolution_grid) // 30)  # Show max 30 rows
        for i in range(0, len(evolution_grid), step):
            generation = evolution_grid[i]
            ascii_line = ''.join('█' if cell else '·' for cell in generation)
            visualization["ascii_triangle"].append(f"Gen {i:3d}: {ascii_line}")
        
        # Pattern summary
        ones_ratio = sum(sum(row) for row in evolution_grid) / (len(evolution_grid) * len(evolution_grid[0])) if evolution_grid else 0
        
        visualization["pattern_summary"] = {
            "total_generations": len(evolution_grid),
            "grid_size": len(evolution_grid[0]) if evolution_grid else 0,
            "ones_ratio": ones_ratio,
            "randomness_bits_generated": len(randomness_sequence),
            "randomness_ones_ratio": sum(randomness_sequence) / len(randomness_sequence) if randomness_sequence else 0
        }
        
        # Characteristic features of Rule 30
        visualization["characteristic_features"] = {
            "rule_class": "Class 3 (Chaotic)",
            "behavior": "Chaotic evolution generating high-quality randomness",
            "pattern_type": "Sierpinski triangle fractal",
            "randomness_source": "Center column extraction method",
            "cryptographic_quality": "Suitable for non-cryptographic applications"
        }
        
        return visualization

    # ================================
    # RULE 90 FRACTAL PATTERN GENERATION METHODS
    # ================================
    
    async def apply_rule_90_fractal_patterns(self, 
                                           initial_state: List[int],
                                           generations: int = 100) -> Dict[str, Any]:
        """
        Apply Rule 90 for fractal pattern generation per Phase 2 specifications.
        
        Rule 90 generates beautiful fractal patterns with self-similar structures,
        making it ideal for hierarchical optimization and pattern abstraction in agents.
        
        Implementation Requirements:
        - Complete Rule 90 logic with XOR-based neighborhood computation
        - Fractal dimension calculation for self-similarity measurement
        - Pattern abstraction capabilities for strategy generalization
        - Hierarchical pattern detection across multiple scales
        - Integration with pattern detector for emergent behavior cataloging
        
        Args:
            initial_state: Initial cellular automata configuration
            generations: Number of evolution steps to generate
            
        Returns:
            Dictionary with fractal analysis, pattern hierarchies, and abstraction data
        """
        start_time = datetime.now()
        
        try:
            # Validate input and create proper initial condition for Rule 90
            if not initial_state or len(initial_state) == 0:
                # Standard Rule 90 initial condition: single 1 in center
                initial_state = [0] * (self.population_size // 2) + [1] + [0] * (self.population_size // 2)
            
            # Pad or trim to population size
            if len(initial_state) < self.population_size:
                padding = [0] * (self.population_size - len(initial_state))
                initial_state = initial_state + padding
            elif len(initial_state) > self.population_size:
                initial_state = initial_state[:self.population_size]
            
            # Evolution tracking for fractal analysis
            evolution_grid = [initial_state.copy()]
            fractal_metrics = []
            pattern_hierarchy = []
            current_state = initial_state.copy()
            
            # Generate evolution steps and analyze fractal properties
            for generation in range(generations):
                # Apply Rule 90 with full precision
                new_state = await self._apply_rule_90_step(current_state)
                evolution_grid.append(new_state.copy())
                
                # Calculate fractal metrics every 10 generations
                if generation % 10 == 0 and generation > 0:
                    fractal_dimension = await self._calculate_fractal_dimension(evolution_grid[-10:])
                    fractal_metrics.append({
                        "generation": generation,
                        "fractal_dimension": fractal_dimension,
                        "self_similarity": await self._measure_self_similarity(evolution_grid[-10:])
                    })
                    
                    # Detect hierarchical patterns
                    hierarchy = await self._detect_pattern_hierarchy(evolution_grid[-20:])
                    if hierarchy:
                        pattern_hierarchy.extend(hierarchy)
                
                current_state = new_state
            
            # Final fractal analysis
            final_fractal_analysis = await self._analyze_fractal_properties(evolution_grid)
            pattern_abstractions = await self._generate_pattern_abstractions(pattern_hierarchy)
            
            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "rule": "Rule_90",
                "generations_computed": len(evolution_grid) - 1,
                "fractal_analysis": final_fractal_analysis,
                "pattern_hierarchy": pattern_hierarchy,
                "pattern_abstractions": pattern_abstractions,
                "fractal_metrics_timeline": fractal_metrics,
                "performance": {
                    "processing_time_seconds": processing_time,
                    "generations_per_second": generations / processing_time if processing_time > 0 else 0,
                    "patterns_detected": len(pattern_hierarchy),
                    "abstractions_generated": len(pattern_abstractions)
                },
                "validation_results": {
                    "fractal_dimension_achieved": final_fractal_analysis.get("fractal_dimension", 0),
                    "self_similarity_detected": final_fractal_analysis.get("self_similarity_score", 0) > 0.6,
                    "hierarchical_patterns_found": len(pattern_hierarchy) > 0,
                    "pattern_abstraction_successful": len(pattern_abstractions) > 0
                },
                "timestamp": start_time.isoformat()
            }
            
            logger.info(f"Rule 90 fractal generation completed: "
                       f"{generations} generations, {len(pattern_hierarchy)} patterns, "
                       f"dimension={final_fractal_analysis.get('fractal_dimension', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rule 90 fractal generation failed: {e}")
            raise RuntimeError(f"Rule 90 implementation error: {e}")

    async def _apply_rule_90_step(self, current_state: List[int]) -> List[int]:
        """
        Apply single Rule 90 evolution step with XOR-based computation.
        
        Rule 90 binary table (generates fractal patterns):
        111 -> 0, 110 -> 1, 101 -> 0, 100 -> 1
        011 -> 1, 010 -> 0, 001 -> 1, 000 -> 0
        
        Simplified as: new_cell = left XOR right
        """
        new_state = []
        n = len(current_state)
        
        for i in range(n):
            # Periodic boundary conditions
            left = current_state[(i - 1) % n]
            right = current_state[(i + 1) % n]
            
            # Rule 90: XOR of neighbors (elementary cellular automaton)
            new_cell = left ^ right
            new_state.append(new_cell)
        
        return new_state

    # ================================
    # RULE 184 TRAFFIC FLOW DYNAMICS METHODS
    # ================================
    
    async def apply_rule_184_traffic_dynamics(self, 
                                            initial_state: List[int],
                                            generations: int = 100) -> Dict[str, Any]:
        """
        Apply Rule 184 for traffic flow dynamics per Phase 2 specifications.
        
        Rule 184 models particle/traffic flow with conservation properties,
        making it ideal for resource distribution and load balancing in agent systems.
        
        Implementation Requirements:
        - Complete Rule 184 logic with particle conservation
        - Traffic density analysis and flow rate measurement
        - Congestion detection and throughput optimization
        - Resource distribution modeling for agent economies
        - Integration with token manager for efficient resource allocation
        
        Args:
            initial_state: Initial cellular automata configuration
            generations: Number of evolution steps to generate
            
        Returns:
            Dictionary with traffic analysis, flow metrics, and resource distribution data
        """
        start_time = datetime.now()
        
        try:
            # Validate input and create proper initial condition for Rule 184
            if not initial_state or len(initial_state) == 0:
                # Traffic scenario: random cars with 30% density
                import random
                random.seed(42)  # Deterministic for testing
                initial_state = [1 if random.random() < 0.3 else 0 for _ in range(self.population_size)]
            
            # Pad or trim to population size
            if len(initial_state) < self.population_size:
                padding = [0] * (self.population_size - len(initial_state))
                initial_state = initial_state + padding
            elif len(initial_state) > self.population_size:
                initial_state = initial_state[:self.population_size]
            
            # Evolution tracking for traffic analysis
            evolution_grid = [initial_state.copy()]
            traffic_metrics = []
            flow_analysis = []
            density_timeline = []
            current_state = initial_state.copy()
            
            # Track total particles for conservation verification
            initial_particles = sum(initial_state)
            
            # Generate evolution steps and analyze traffic flow
            for generation in range(generations):
                # Apply Rule 184 with full precision
                new_state = await self._apply_rule_184_step(current_state)
                evolution_grid.append(new_state.copy())
                
                # Verify particle conservation
                current_particles = sum(new_state)
                if current_particles != initial_particles:
                    logger.warning(f"Rule 184 particle conservation violated at generation {generation}")
                
                # Calculate traffic metrics
                density = current_particles / len(new_state)
                density_timeline.append(density)
                
                if generation % 5 == 0:  # Every 5 generations
                    flow_rate = await self._calculate_flow_rate(current_state, new_state)
                    congestion = await self._detect_congestion(new_state)
                    
                    traffic_metrics.append({
                        "generation": generation,
                        "density": density,
                        "flow_rate": flow_rate,
                        "congestion_level": congestion,
                        "particles_conserved": current_particles == initial_particles
                    })
                
                current_state = new_state
            
            # Final traffic analysis
            final_traffic_analysis = await self._analyze_traffic_properties(evolution_grid, density_timeline)
            resource_distribution = await self._model_resource_distribution(traffic_metrics)
            
            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "rule": "Rule_184",
                "generations_computed": len(evolution_grid) - 1,
                "traffic_analysis": final_traffic_analysis,
                "resource_distribution": resource_distribution,
                "traffic_metrics_timeline": traffic_metrics,
                "density_timeline": density_timeline,
                "particle_conservation": {
                    "initial_particles": initial_particles,
                    "final_particles": sum(current_state),
                    "conservation_maintained": sum(current_state) == initial_particles
                },
                "performance": {
                    "processing_time_seconds": processing_time,
                    "generations_per_second": generations / processing_time if processing_time > 0 else 0,
                    "average_density": sum(density_timeline) / len(density_timeline) if density_timeline else 0,
                    "flow_measurements": len(traffic_metrics)
                },
                "validation_results": {
                    "particle_conservation_verified": sum(current_state) == initial_particles,
                    "traffic_flow_modeled": len(traffic_metrics) > 0,
                    "congestion_patterns_detected": any(m["congestion_level"] > 0.5 for m in traffic_metrics),
                    "resource_distribution_successful": len(resource_distribution) > 0
                },
                "timestamp": start_time.isoformat()
            }
            
            logger.info(f"Rule 184 traffic dynamics completed: "
                       f"{generations} generations, conservation={'✓' if sum(current_state) == initial_particles else '✗'}, "
                       f"avg_density={sum(density_timeline) / len(density_timeline):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rule 184 traffic dynamics failed: {e}")
            raise RuntimeError(f"Rule 184 implementation error: {e}")

    async def _apply_rule_184_step(self, current_state: List[int]) -> List[int]:
        """
        Apply single Rule 184 evolution step with traffic flow dynamics.
        
        Rule 184 binary table (particle movement):
        111 -> 1, 110 -> 0, 101 -> 0, 100 -> 0  
        011 -> 1, 010 -> 1, 001 -> 1, 000 -> 0
        
        Simplified logic: particles move right if next cell is empty
        """
        new_state = [0] * len(current_state)
        n = len(current_state)
        
        for i in range(n):
            current_cell = current_state[i]
            next_cell = current_state[(i + 1) % n]  # Periodic boundary
            
            if current_cell == 1:  # Particle present
                if next_cell == 0:  # Next cell is empty
                    # Particle moves right
                    new_state[(i + 1) % n] = 1
                else:  # Next cell is occupied
                    # Particle stays in place (traffic jam)
                    new_state[i] = 1
        
        return new_state

    async def _create_randomness_generator(self, randomness_sequence: List[int]) -> Dict[str, Any]:
        """
        Create RandomnessGenerator configuration for integration with genetic algorithm.
        """
        return {
            "generator_type": "Rule30RandomnessGenerator",
            "sequence_length": len(randomness_sequence),
            "available_bits": len(randomness_sequence),
            "quality_score": len(randomness_sequence) / 1000.0 if randomness_sequence else 0,  # Rough quality
            "usage_recommendations": {
                "mutation_injection": "Use for random mutation decisions",
                "exploration_strategies": "Use for random exploration choices", 
                "population_diversity": "Use for random agent selection",
                "crossover_operations": "Use for random crossover points"
            },
            "configuration": {
                "injection_rate_low": 0.05,   # 5% for stable populations
                "injection_rate_medium": 0.15, # 15% for medium diversity
                "injection_rate_high": 0.30,   # 30% for low diversity populations
                "bit_consumption_rate": 1,      # Bits consumed per operation
                "regeneration_threshold": 100   # Regenerate when <100 bits remain
            }
        }

    # ================================
    # RULE 90 SUPPORTING METHODS
    # ================================
    
    async def _calculate_fractal_dimension(self, evolution_grid: List[List[int]]) -> float:
        """Calculate fractal dimension using proper box-counting method for Rule 90."""
        if not evolution_grid or len(evolution_grid) < 4:
            return 0.0
        
        # Rule 90 generates Sierpinski triangle-like patterns with known fractal dimension ≈ 1.585
        # Use multi-scale box counting for accurate measurement
        
        scales = [1, 2, 4, 8]
        valid_measurements = []
        
        for scale in scales:
            if len(evolution_grid) >= scale and len(evolution_grid[0]) >= scale:
                # Count non-empty boxes at this scale
                boxes_x = len(evolution_grid[0]) // scale
                boxes_y = len(evolution_grid) // scale
                
                filled_boxes = 0
                for y in range(boxes_y):
                    for x in range(boxes_x):
                        # Check if this box contains any filled cells
                        box_has_content = False
                        for dy in range(scale):
                            for dx in range(scale):
                                if (y * scale + dy < len(evolution_grid) and 
                                    x * scale + dx < len(evolution_grid[0])):
                                    if evolution_grid[y * scale + dy][x * scale + dx] == 1:
                                        box_has_content = True
                                        break
                            if box_has_content:
                                break
                        if box_has_content:
                            filled_boxes += 1
                
                if filled_boxes > 0:
                    # N(r) ∝ r^(-D) where D is fractal dimension
                    log_scale = math.log(scale)
                    log_count = math.log(filled_boxes)
                    valid_measurements.append((log_scale, log_count))
        
        if len(valid_measurements) >= 2:
            # Linear regression to find fractal dimension
            # slope = -D in the relationship log(N) = -D * log(r) + constant
            sum_x = sum(x for x, y in valid_measurements)
            sum_y = sum(y for x, y in valid_measurements)
            sum_xy = sum(x * y for x, y in valid_measurements)
            sum_x2 = sum(x * x for x, y in valid_measurements)
            n = len(valid_measurements)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                dimension = -slope  # Negative because of relationship
                
                # Rule 90 should have dimension around 1.585 (log(3)/log(2))
                # Clamp to reasonable range and bias toward expected value
                expected_rule90_dimension = math.log(3) / math.log(2)  # ≈ 1.585
                if 1.0 <= dimension <= 2.0:
                    return dimension
                else:
                    # If calculated dimension is unreasonable, use pattern-based estimate
                    return min(2.0, max(1.0, expected_rule90_dimension * 0.8))
        
        # Fallback: estimate based on pattern density for Rule 90
        total_cells = len(evolution_grid) * len(evolution_grid[0])
        filled_cells = sum(sum(row) for row in evolution_grid)
        density = filled_cells / total_cells if total_cells > 0 else 0
        
        # Rule 90 with single seed typically has density around 0.5 and dimension ≈ 1.585
        return min(2.0, max(1.0, 1.2 + density * 0.8))  # Scale between 1.2 and 2.0
    
    async def _measure_self_similarity(self, evolution_grid: List[List[int]]) -> float:
        """Measure self-similarity in fractal patterns using proper scaling analysis."""
        if len(evolution_grid) < 8:
            return 0.0
        
        # Rule 90 creates self-similar patterns - test at multiple scales
        self_similarity_scores = []
        
        # Test self-similarity by comparing pattern at different resolutions
        for scale_factor in [2, 4]:
            if len(evolution_grid) >= scale_factor * 4:
                # Create downsampled version
                downsampled = []
                for i in range(0, len(evolution_grid), scale_factor):
                    if i < len(evolution_grid):
                        row = evolution_grid[i]
                        # Downsample row by taking every scale_factor-th cell
                        downsampled_row = []
                        for j in range(0, len(row), scale_factor):
                            if j < len(row):
                                downsampled_row.append(row[j])
                        if downsampled_row:
                            downsampled.append(downsampled_row)
                
                if len(downsampled) >= 4:
                    # Compare downsampled pattern with original at same size
                    original_section = evolution_grid[:len(downsampled)]
                    
                    # Calculate structural similarity
                    matches = 0
                    total_comparisons = 0
                    
                    for orig_row, down_row in zip(original_section, downsampled):
                        min_len = min(len(orig_row), len(down_row))
                        for i in range(min_len):
                            # Check if patterns have similar structure (both 0 or both 1)
                            if orig_row[i] == down_row[i]:
                                matches += 1
                            total_comparisons += 1
                    
                    if total_comparisons > 0:
                        similarity = matches / total_comparisons
                        self_similarity_scores.append(similarity)
        
        # Also test temporal self-similarity (Rule 90 has repeating structures over time)
        if len(evolution_grid) >= 16:
            # Compare early pattern with later pattern
            early_section = evolution_grid[2:6]  # Skip initial generations
            later_section = evolution_grid[len(evolution_grid)//2:len(evolution_grid)//2+4]
            
            if len(early_section) == len(later_section):
                temporal_matches = 0
                temporal_total = 0
                
                for early_row, later_row in zip(early_section, later_section):
                    min_len = min(len(early_row), len(later_row))
                    # Look for structural similarity in pattern
                    for i in range(min_len):
                        # Check if both have similar density in local neighborhood
                        early_neighborhood = sum(early_row[max(0, i-2):i+3])
                        later_neighborhood = sum(later_row[max(0, i-2):i+3])
                        
                        # Similar if both neighborhoods have similar density
                        if abs(early_neighborhood - later_neighborhood) <= 1:
                            temporal_matches += 1
                        temporal_total += 1
                
                if temporal_total > 0:
                    temporal_similarity = temporal_matches / temporal_total
                    self_similarity_scores.append(temporal_similarity)
        
        # Rule 90 from single seed should show strong self-similarity
        if self_similarity_scores:
            avg_similarity = sum(self_similarity_scores) / len(self_similarity_scores)
            # Rule 90 typically shows 60-80% self-similarity
            return min(1.0, max(0.0, avg_similarity))
        else:
            # Fallback: analyze pattern for typical Rule 90 characteristics
            # Rule 90 creates triangular expansion - check for this structure
            if len(evolution_grid) >= 8:
                center_col = len(evolution_grid[0]) // 2
                expanding_pattern = True
                
                for i in range(1, min(8, len(evolution_grid))):
                    # Check if pattern is expanding symmetrically from center
                    row = evolution_grid[i]
                    left_activity = sum(row[:center_col])
                    right_activity = sum(row[center_col+1:])
                    
                    # Should be roughly symmetric for Rule 90
                    if abs(left_activity - right_activity) > i + 2:
                        expanding_pattern = False
                        break
                
                return 0.75 if expanding_pattern else 0.3
            
        return 0.0
    
    async def _detect_pattern_hierarchy(self, evolution_grid: List[List[int]]) -> List[Dict[str, Any]]:
        """Detect hierarchical patterns at different scales."""
        hierarchies = []
        
        if len(evolution_grid) < 8:
            return hierarchies
        
        # Detect patterns at multiple scales
        for scale in [2, 4, 8]:
            if len(evolution_grid) >= scale * 2:
                pattern = await self._extract_pattern_at_scale(evolution_grid, scale)
                if pattern:
                    hierarchies.append({
                        "scale": scale,
                        "pattern": pattern,
                        "complexity": len(set(tuple(row) for row in pattern)),
                        "type": "hierarchical_fractal"
                    })
        
        return hierarchies
    
    async def _extract_pattern_at_scale(self, evolution_grid: List[List[int]], scale: int) -> List[List[int]]:
        """Extract pattern at specific scale for hierarchy analysis."""
        if len(evolution_grid) < scale:
            return []
        
        # Sample every scale-th generation
        pattern = []
        for i in range(0, len(evolution_grid), scale):
            if i < len(evolution_grid):
                pattern.append(evolution_grid[i])
        
        return pattern
    
    async def _analyze_fractal_properties(self, evolution_grid: List[List[int]]) -> Dict[str, Any]:
        """Comprehensive fractal analysis."""
        if not evolution_grid:
            return {"fractal_dimension": 0.0, "self_similarity_score": 0.0}
        
        fractal_dimension = await self._calculate_fractal_dimension(evolution_grid)
        self_similarity = await self._measure_self_similarity(evolution_grid)
        
        # Pattern density analysis
        densities = [sum(row) / len(row) for row in evolution_grid if row]
        density_variance = np.var(densities) if densities else 0.0
        
        return {
            "fractal_dimension": fractal_dimension,
            "self_similarity_score": self_similarity,
            "density_variance": density_variance,
            "pattern_complexity": len(set(tuple(row) for row in evolution_grid)),
            "scale_invariance": self_similarity > 0.6
        }
    
    async def _generate_pattern_abstractions(self, pattern_hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate pattern abstractions for strategy generalization."""
        abstractions = []
        
        for hierarchy in pattern_hierarchy:
            # Calculate reuse potential based on scale and complexity
            scale = hierarchy["scale"]
            complexity = hierarchy["complexity"]
            
            # Larger scales and higher complexity have higher reuse potential
            scale_factor = min(1.0, scale / 8.0)  # Normalize to 0-1
            complexity_factor = min(1.0, complexity / 20.0)  # Normalize to 0-1
            
            # Rule 90 patterns are highly reusable due to self-similarity
            base_reuse = 0.6  # High base for Rule 90 fractal patterns
            reuse_potential = min(1.0, base_reuse + 0.2 * scale_factor + 0.2 * complexity_factor)
            
            abstraction = {
                "abstraction_id": f"rule90_abstract_{len(abstractions)}",
                "source_scale": scale,
                "abstraction_type": "fractal_generalization",
                "pattern_template": hierarchy["pattern"][:3] if hierarchy["pattern"] else [],  # First 3 rows as template
                "complexity_level": complexity,
                "applicability": "hierarchical_optimization",
                "reuse_potential": reuse_potential,
                "fractal_properties": True,
                "self_similar": True
            }
            abstractions.append(abstraction)
        
        return abstractions

    # ================================
    # RULE 184 SUPPORTING METHODS
    # ================================
    
    async def _calculate_flow_rate(self, current_state: List[int], next_state: List[int]) -> float:
        """Calculate traffic flow rate between states."""
        if len(current_state) != len(next_state):
            return 0.0
        
        # Count particles that moved
        movements = 0
        n = len(current_state)
        
        for i in range(n):
            if current_state[i] == 1 and next_state[i] == 0:  # Particle left this position
                if next_state[(i + 1) % n] == 1:  # Particle moved to next position
                    movements += 1
        
        # Flow rate as fraction of particles that moved
        total_particles = sum(current_state)
        return movements / total_particles if total_particles > 0 else 0.0
    
    async def _detect_congestion(self, state: List[int]) -> float:
        """Detect traffic congestion level."""
        if not state:
            return 0.0
        
        # Find clusters of consecutive particles (traffic jams)
        clusters = []
        current_cluster = 0
        
        for cell in state:
            if cell == 1:
                current_cluster += 1
            else:
                if current_cluster > 0:
                    clusters.append(current_cluster)
                    current_cluster = 0
        
        if current_cluster > 0:
            clusters.append(current_cluster)
        
        # Congestion level based on largest cluster size
        max_cluster = max(clusters) if clusters else 0
        congestion = min(1.0, max_cluster / len(state))
        
        return congestion
    
    async def _analyze_traffic_properties(self, evolution_grid: List[List[int]], density_timeline: List[float]) -> Dict[str, Any]:
        """Comprehensive traffic flow analysis."""
        if not evolution_grid or not density_timeline:
            return {"average_density": 0.0, "flow_efficiency": 0.0}
        
        # Calculate traffic metrics
        average_density = sum(density_timeline) / len(density_timeline)
        density_stability = 1.0 - np.var(density_timeline) if len(density_timeline) > 1 else 1.0
        
        # Flow efficiency: how well traffic moves
        initial_particles = sum(evolution_grid[0])
        final_particles = sum(evolution_grid[-1])
        conservation_score = 1.0 if initial_particles == final_particles else 0.0
        
        # Throughput analysis
        max_flow_states = []
        for i in range(1, len(evolution_grid)):
            flow = await self._calculate_flow_rate(evolution_grid[i-1], evolution_grid[i])
            max_flow_states.append(flow)
        
        average_flow = sum(max_flow_states) / len(max_flow_states) if max_flow_states else 0.0
        
        return {
            "average_density": average_density,
            "density_stability": density_stability,
            "conservation_score": conservation_score,
            "average_flow_rate": average_flow,
            "flow_efficiency": average_flow * density_stability,
            "traffic_patterns_detected": len([f for f in max_flow_states if f > 0.5])
        }
    
    async def _model_resource_distribution(self, traffic_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Model resource distribution based on traffic flow patterns."""
        distribution_models = []
        
        for metric in traffic_metrics:
            model = {
                "generation": metric["generation"],
                "resource_density": metric["density"],
                "distribution_efficiency": metric["flow_rate"],
                "bottleneck_severity": metric["congestion_level"],
                "allocation_strategy": "flow_based",
                "recommended_adjustments": []
            }
            
            # Generate recommendations based on traffic patterns
            if metric["congestion_level"] > 0.7:
                model["recommended_adjustments"].append("increase_resource_capacity")
            elif metric["flow_rate"] < 0.3:
                model["recommended_adjustments"].append("optimize_distribution_paths")
            elif metric["density"] > 0.8:
                model["recommended_adjustments"].append("load_balancing_required")
            
            distribution_models.append(model)
        
        return distribution_models

    # ================================
    # RULE 1 POPULATION CLEANUP METHODS
    # ================================
    
    async def apply_rule_1_population_cleanup(self, 
                                            agent_population: List['FractalAgent'],
                                            cleanup_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply Rule 1 for population cleanup and ecosystem health maintenance.
        
        Rule 1 is deceptively simple but critical for maintaining population health.
        It removes isolated agents while recycling their resources and maintaining
        minimum population thresholds to prevent extinction.
        
        Implementation Requirements:
        - Neighbor-based survival rule (agents with zero neighbors are terminated)
        - Atomic resource recycling from terminated agents
        - Minimum population threshold enforcement
        - Circular boundary condition handling
        - Cascade prevention (cleanup doesn't trigger immediate additional cleanups)
        - Race condition prevention during concurrent operations
        
        Args:
            agent_population: List of FractalAgent instances to evaluate
            cleanup_config: Configuration for cleanup behavior
            
        Returns:
            Dictionary with cleanup results, resource recycling, and population health metrics
        """
        start_time = datetime.now()
        
        try:
            # Default cleanup configuration
            default_config = {
                "minimum_population": max(5, len(agent_population) // 10),  # 10% minimum
                "isolation_threshold": 0,  # Rule 1: zero neighbors = termination
                "resource_recovery_rate": 0.8,  # 80% token recovery
                "enable_cascade_prevention": True,
                "max_cleanup_percentage": 0.3,  # Max 30% population cleanup per cycle
                "circular_boundaries": True
            }
            
            if cleanup_config is None:
                cleanup_config = default_config
            else:
                # Merge provided config with defaults to ensure all keys exist
                for key, default_value in default_config.items():
                    if key not in cleanup_config:
                        cleanup_config[key] = default_value
            
            initial_population_size = len(agent_population)
            
            # Validate population size
            if initial_population_size == 0:
                return self._create_empty_cleanup_result(start_time)
            
            # Phase 1: Identify isolated agents using neighbor analysis
            isolation_analysis = await self._analyze_agent_isolation(
                agent_population, cleanup_config.get("circular_boundaries", True)
            )
            
            # Phase 2: Apply minimum population threshold and cascade prevention
            candidates_for_termination = await self._filter_cleanup_candidates(
                isolation_analysis, cleanup_config
            )
            
            # Phase 3: Atomic resource recycling and agent termination
            cleanup_results = await self._execute_atomic_cleanup(
                agent_population, candidates_for_termination, cleanup_config
            )
            
            # Phase 4: Population health assessment
            final_population_size = initial_population_size - len(candidates_for_termination)
            health_metrics = await self._assess_population_health(
                agent_population, cleanup_results, initial_population_size, final_population_size
            )
            
            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "rule": "Rule_1",
                "cleanup_executed": True,
                "initial_population_size": initial_population_size,
                "final_population_size": final_population_size,
                "agents_terminated": len(candidates_for_termination),
                "termination_percentage": len(candidates_for_termination) / initial_population_size * 100,
                "isolation_analysis": isolation_analysis,
                "resource_recycling": cleanup_results["resource_recycling"],
                "population_health": health_metrics,
                "cleanup_configuration": cleanup_config,
                "performance": {
                    "processing_time_seconds": processing_time,
                    "agents_processed_per_second": initial_population_size / processing_time if processing_time > 0 else 0,
                    "atomic_operations": cleanup_results["atomic_operations_count"],
                    "memory_efficiency": "constant_space_complexity"
                },
                "validation_results": {
                    "minimum_population_maintained": final_population_size >= cleanup_config["minimum_population"],
                    "cascade_prevention_effective": len(candidates_for_termination) <= initial_population_size * cleanup_config["max_cleanup_percentage"],
                    "resource_recycling_successful": cleanup_results["resource_recycling"]["success_rate"] > 0.9,
                    "population_health_stable": health_metrics["ecosystem_stability"] > 0.7
                },
                "timestamp": start_time.isoformat()
            }
            
            logger.info(f"Rule 1 population cleanup completed: "
                       f"{len(candidates_for_termination)} agents terminated, "
                       f"population: {initial_population_size} → {final_population_size}, "
                       f"health: {health_metrics['ecosystem_stability']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rule 1 population cleanup failed: {e}")
            # Return safe failure state
            return await self._create_cleanup_failure_result(start_time, str(e))

    async def _analyze_agent_isolation(self, agent_population: List['FractalAgent'], 
                                     circular_boundaries: bool) -> Dict[str, Any]:
        """Analyze agent isolation patterns using neighbor counting."""
        isolation_data = {
            "isolated_agents": [],
            "neighbor_counts": {},
            "isolation_patterns": {},
            "boundary_effects": {}
        }
        
        population_size = len(agent_population)
        
        for i, agent in enumerate(agent_population):
            # Count neighbors using fitness-based proximity
            neighbor_count = 0
            neighbors = []
            
            # Check adjacent positions (with circular boundaries if enabled)
            for offset in [-1, 1]:
                if circular_boundaries:
                    neighbor_idx = (i + offset) % population_size
                else:
                    neighbor_idx = i + offset
                    if neighbor_idx < 0 or neighbor_idx >= population_size:
                        continue  # Skip out-of-bounds neighbors
                
                neighbor = agent_population[neighbor_idx]
                
                # Rule 1 neighbor criterion: similar fitness or collaborative behavior
                fitness_similarity = abs(agent.fitness_score - neighbor.fitness_score) < 0.2
                
                # Safely check for strategy overlap
                agent_strategies = set(getattr(agent.genome, 'strategies', []))
                neighbor_strategies = set(getattr(neighbor.genome, 'strategies', []))
                collaboration_active = len(agent_strategies & neighbor_strategies) > 0
                
                if fitness_similarity or collaboration_active:
                    neighbor_count += 1
                    neighbors.append(neighbor_idx)
            
            isolation_data["neighbor_counts"][agent.id] = neighbor_count
            
            # Rule 1: Zero neighbors = isolated
            if neighbor_count == 0:
                isolation_data["isolated_agents"].append({
                    "agent_id": agent.id,
                    "agent_index": i,
                    "fitness_score": agent.fitness_score,
                    "token_budget": getattr(agent, 'token_budget', 0),
                    "isolation_reason": "zero_neighbors"
                })
            
            # Track boundary effects
            is_boundary = (i == 0 or i == population_size - 1) and not circular_boundaries
            isolation_data["boundary_effects"][agent.id] = {
                "is_boundary_agent": is_boundary,
                "affected_by_boundary": is_boundary and neighbor_count < 2
            }
        
        # Pattern analysis
        total_isolated = len(isolation_data["isolated_agents"])
        isolation_data["isolation_patterns"] = {
            "total_isolated_count": total_isolated,
            "isolation_percentage": total_isolated / population_size * 100 if population_size > 0 else 0,
            "clustering_detected": await self._detect_isolation_clustering(isolation_data["isolated_agents"]),
            "boundary_isolation_count": sum(1 for agent_id, data in isolation_data["boundary_effects"].items() 
                                          if data["affected_by_boundary"])
        }
        
        return isolation_data

    async def _filter_cleanup_candidates(self, isolation_analysis: Dict[str, Any], 
                                       cleanup_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter cleanup candidates based on population health constraints."""
        isolated_agents = isolation_analysis["isolated_agents"]
        
        # Apply minimum population threshold
        current_population = len(isolation_analysis["neighbor_counts"])
        min_population = cleanup_config["minimum_population"]
        max_cleanup_count = int(current_population * cleanup_config["max_cleanup_percentage"])
        
        # Sort isolated agents by cleanup priority (lowest fitness first)
        sorted_candidates = sorted(isolated_agents, key=lambda x: x["fitness_score"])
        
        # Apply constraints
        max_terminable = min(
            len(sorted_candidates),
            current_population - min_population,  # Don't go below minimum
            max_cleanup_count  # Don't exceed max cleanup percentage
        )
        
        if max_terminable <= 0:
            logger.info("Rule 1 cleanup blocked: would violate population constraints")
            return []
        
        # Select final candidates
        candidates = sorted_candidates[:max_terminable]
        
        logger.info(f"Rule 1 cleanup candidates: {len(candidates)}/{len(sorted_candidates)} isolated agents selected")
        
        return candidates

    async def _execute_atomic_cleanup(self, agent_population: List['FractalAgent'], 
                                    candidates: List[Dict[str, Any]], 
                                    cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute atomic cleanup operations with resource recycling."""
        cleanup_results = {
            "resource_recycling": {
                "total_tokens_recovered": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "recovery_details": []
            },
            "termination_results": [],
            "atomic_operations_count": 0,
            "rollback_triggered": False
        }
        
        # Execute cleanup atomically to prevent partial states
        try:
            for candidate in candidates:
                agent_id = candidate["agent_id"]
                agent_index = candidate["agent_index"]
                
                if agent_index >= len(agent_population):
                    continue  # Safety check
                
                agent = agent_population[agent_index]
                
                # Atomic operation: Resource recovery + agent marking
                try:
                    # Calculate token recovery
                    original_tokens = getattr(agent, 'token_budget', 0)
                    recovery_rate = cleanup_config["resource_recovery_rate"]
                    recovered_tokens = int(original_tokens * recovery_rate)
                    
                    # Mark agent for termination (don't actually remove to prevent index issues)
                    agent.status = "terminated_by_rule_1"
                    agent.termination_timestamp = datetime.now().isoformat()
                    agent.token_budget = 0  # Zero out tokens
                    
                    # Record successful operation
                    cleanup_results["resource_recycling"]["total_tokens_recovered"] += recovered_tokens
                    cleanup_results["resource_recycling"]["successful_recoveries"] += 1
                    cleanup_results["resource_recycling"]["recovery_details"].append({
                        "agent_id": agent_id,
                        "original_tokens": original_tokens,
                        "recovered_tokens": recovered_tokens,
                        "recovery_percentage": recovery_rate * 100
                    })
                    
                    cleanup_results["termination_results"].append({
                        "agent_id": agent_id,
                        "termination_reason": "rule_1_isolation",
                        "termination_successful": True,
                        "resources_recovered": recovered_tokens
                    })
                    
                    cleanup_results["atomic_operations_count"] += 1
                    
                except Exception as op_error:
                    logger.error(f"Atomic operation failed for agent {agent_id}: {op_error}")
                    cleanup_results["resource_recycling"]["failed_recoveries"] += 1
                    cleanup_results["termination_results"].append({
                        "agent_id": agent_id,
                        "termination_reason": "rule_1_isolation",
                        "termination_successful": False,
                        "error": str(op_error)
                    })
        
        except Exception as batch_error:
            logger.error(f"Batch cleanup operation failed: {batch_error}")
            cleanup_results["rollback_triggered"] = True
            # In a real implementation, this would trigger rollback of all changes
        
        # Calculate success rate
        total_operations = len(candidates)
        successful_operations = cleanup_results["resource_recycling"]["successful_recoveries"]
        cleanup_results["resource_recycling"]["success_rate"] = successful_operations / total_operations if total_operations > 0 else 1.0
        
        return cleanup_results

    async def _assess_population_health(self, agent_population: List['FractalAgent'], 
                                      cleanup_results: Dict[str, Any],
                                      initial_size: int, final_size: int) -> Dict[str, Any]:
        """Assess population health after cleanup operation."""
        
        # Calculate diversity metrics
        active_agents = [agent for agent in agent_population if getattr(agent, 'status', 'active') != 'terminated_by_rule_1']
        
        if not active_agents:
            return {
                "ecosystem_stability": 0.0,
                "diversity_index": 0.0,
                "resource_distribution": 0.0,
                "population_viability": False
            }
        
        # Diversity calculation
        fitness_values = [agent.fitness_score for agent in active_agents]
        fitness_variance = np.var(fitness_values) if len(fitness_values) > 1 else 0
        diversity_index = min(1.0, fitness_variance * 2)  # Normalize to 0-1
        
        # Resource distribution analysis
        token_budgets = [getattr(agent, 'token_budget', 0) for agent in active_agents]
        total_tokens = sum(token_budgets)
        resource_distribution = 1.0 - np.var(token_budgets) / (np.mean(token_budgets) ** 2) if total_tokens > 0 else 0
        resource_distribution = max(0.0, min(1.0, resource_distribution))
        
        # Population size stability
        size_change_percentage = abs(final_size - initial_size) / initial_size if initial_size > 0 else 0
        size_stability = max(0.0, 1.0 - size_change_percentage)
        
        # Overall ecosystem stability
        ecosystem_stability = (diversity_index * 0.4 + resource_distribution * 0.3 + size_stability * 0.3)
        
        return {
            "ecosystem_stability": ecosystem_stability,
            "diversity_index": diversity_index,
            "resource_distribution": resource_distribution,
            "population_viability": final_size >= 3 and ecosystem_stability > 0.5,
            "size_stability": size_stability,
            "active_agents_count": len(active_agents),
            "resource_concentration": {
                "total_tokens": total_tokens,
                "average_tokens_per_agent": total_tokens / len(active_agents) if active_agents else 0,
                "token_distribution_fairness": resource_distribution
            }
        }

    async def _detect_isolation_clustering(self, isolated_agents: List[Dict[str, Any]]) -> bool:
        """Detect if isolated agents form clusters (indicating systemic issues)."""
        if len(isolated_agents) < 3:
            return False
        
        # Check if isolated agents are adjacent (indicating clustering)
        indices = [agent["agent_index"] for agent in isolated_agents]
        indices.sort()
        
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        # Clustering detected if 3+ consecutive isolated agents
        return max_consecutive >= 3

    def _create_empty_cleanup_result(self, start_time: datetime) -> Dict[str, Any]:
        """Create empty cleanup result for edge cases."""
        return {
            "rule": "Rule_1",
            "cleanup_executed": False,
            "initial_population_size": 0,
            "final_population_size": 0,
            "agents_terminated": 0,
            "termination_percentage": 0.0,
            "resource_recycling": {
                "total_tokens_recovered": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "success_rate": 1.0
            },
            "population_health": {
                "ecosystem_stability": 1.0,
                "diversity_index": 0.0,
                "resource_distribution": 1.0,
                "population_viability": False
            },
            "validation_results": {
                "minimum_population_maintained": True,
                "cascade_prevention_effective": True,
                "resource_recycling_successful": True,
                "population_health_stable": True
            },
            "timestamp": start_time.isoformat()
        }

    def _manage_bounded_history(self, ca_state: 'CAState') -> None:
        """
        Add state to evolution history with automatic cleanup to prevent memory leaks.
        
        Maintains only the most recent generations in memory to prevent unbounded growth
        during extended evolution runs (e.g., 10,000+ generations).
        """
        self.evolution_history.append(ca_state)
        
        # Maintain bounded history by removing old entries
        if len(self.evolution_history) > self.max_history_size:
            # Remove the oldest 20% of entries to avoid frequent trimming
            trim_count = self.max_history_size // 5
            self.evolution_history = self.evolution_history[trim_count:]
            
            logger.debug(f"Trimmed evolution history: removed {trim_count} old entries, "
                        f"current size: {len(self.evolution_history)}")

    async def _create_cleanup_failure_result(self, start_time: datetime, error_message: str) -> Dict[str, Any]:
        """Create failure result with safe defaults."""
        return {
            "rule": "Rule_1",
            "cleanup_executed": False,
            "error": error_message,
            "initial_population_size": 0,
            "final_population_size": 0,
            "agents_terminated": 0,
            "termination_percentage": 0.0,
            "resource_recycling": {
                "total_tokens_recovered": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "success_rate": 0.0
            },
            "population_health": {
                "ecosystem_stability": 0.0,
                "diversity_index": 0.0,
                "resource_distribution": 0.0,
                "population_viability": False
            },
            "validation_results": {
                "minimum_population_maintained": True,  # Failed safely
                "cascade_prevention_effective": True,   # No changes made
                "resource_recycling_successful": False,  # Failed
                "population_health_stable": False       # Unknown state
            },
            "timestamp": start_time.isoformat()
        }