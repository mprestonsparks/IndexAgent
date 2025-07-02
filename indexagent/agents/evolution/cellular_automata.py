"""
Cellular Automata Implementation for DEAN Agent Evolution

This module implements cellular automata rules that create specific, measurable 
behavioral changes in agents:

- Rule 110: Increase exploration rate by 10% and mutation variance by 0.05
- Rule 30: Introduce bounded stochastic elements to decision-making
- Rule 90: Create self-similar patterns at different abstraction levels
- Rule 184: Optimize pathways between agent components

The implementation applies cellular automata principles to agent genomes to create
observable, non-random behavioral changes while maintaining state history for analysis.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class CARule(Enum):
    """Cellular automata rules for agent evolution."""
    RULE_110 = 110  # Increase exploration and mutation variance
    RULE_30 = 30    # Introduce bounded stochastic elements
    RULE_90 = 90    # Create self-similar patterns
    RULE_184 = 184  # Optimize pathways between components


@dataclass
class BehavioralChange:
    """Represents a specific behavioral change applied to an agent."""
    rule: CARule
    timestamp: datetime
    parameters: Dict[str, Any]
    magnitude: float
    description: str


@dataclass
class CAState:
    """State of the cellular automaton at a given time."""
    cells: np.ndarray
    generation: int
    rule: CARule
    behavioral_changes: List[BehavioralChange] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'cells': self.cells.tolist(),
            'generation': self.generation,
            'rule': self.rule.value,
            'behavioral_changes': [
                {
                    'rule': bc.rule.value,
                    'timestamp': bc.timestamp.isoformat(),
                    'parameters': bc.parameters,
                    'magnitude': bc.magnitude,
                    'description': bc.description
                }
                for bc in self.behavioral_changes
            ],
            'metrics': self.metrics
        }


class CellularAutomataEngine:
    """
    Cellular Automata Engine for DEAN Agent Evolution
    
    Applies cellular automata principles to agent genomes to create specific,
    measurable behavioral changes that enhance agent capabilities.
    """
    
    def __init__(self, grid_size: int = 64, neighborhood_size: int = 3):
        """
        Initialize the cellular automata engine.
        
        Args:
            grid_size: Size of the CA grid
            neighborhood_size: Size of the neighborhood for rule application
        """
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.state_history: List[CAState] = []
        self.rule_lookup = self._generate_rule_lookup()
        
    def _generate_rule_lookup(self) -> Dict[CARule, np.ndarray]:
        """Generate lookup tables for each CA rule."""
        lookup = {}
        
        # Rule 110: Creates complex patterns and increases exploration
        rule_110 = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
        lookup[CARule.RULE_110] = rule_110
        
        # Rule 30: Chaotic behavior for stochastic elements
        rule_30 = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
        lookup[CARule.RULE_30] = rule_30
        
        # Rule 90: Self-similar fractal patterns
        rule_90 = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
        lookup[CARule.RULE_90] = rule_90
        
        # Rule 184: Traffic flow for pathway optimization
        rule_184 = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
        lookup[CARule.RULE_184] = rule_184
        
        return lookup
    
    def _apply_ca_rule(self, cells: np.ndarray, rule: CARule) -> np.ndarray:
        """Apply a cellular automaton rule to a cell array."""
        rule_array = self.rule_lookup[rule]
        new_cells = np.zeros_like(cells)
        
        for i in range(len(cells)):
            # Get neighborhood (with wraparound)
            left = cells[(i - 1) % len(cells)]
            center = cells[i]
            right = cells[(i + 1) % len(cells)]
            
            # Convert to index
            index = (left << 2) | (center << 1) | right
            new_cells[i] = rule_array[index]
            
        return new_cells
    
    async def apply_rule_110(self, agent_genome: Dict[str, Any], 
                           current_state: Optional[CAState] = None) -> Tuple[Dict[str, Any], BehavioralChange]:
        """
        Apply Rule 110: Increase exploration rate by 10% and mutation variance by 0.05
        
        This rule generates complex patterns that enhance agent exploration capabilities.
        """
        # Initialize or get current state
        if current_state is None:
            cells = self._genome_to_cells(agent_genome)
        else:
            cells = current_state.cells
            
        # Apply Rule 110
        new_cells = self._apply_ca_rule(cells, CARule.RULE_110)
        
        # Calculate pattern complexity as a measure of exploration potential
        complexity = self._calculate_complexity(new_cells)
        
        # Create behavioral change based on complexity
        exploration_increase = 0.1 * (complexity / self.grid_size)  # Normalized by grid size
        mutation_variance_increase = 0.05 * (complexity / self.grid_size)
        
        # Apply changes to genome
        modified_genome = agent_genome.copy()
        
        # Increase exploration rate
        current_exploration = modified_genome.get('exploration_rate', 0.1)
        modified_genome['exploration_rate'] = min(1.0, current_exploration * (1 + exploration_increase))
        
        # Increase mutation variance
        current_variance = modified_genome.get('mutation_variance', 0.1)
        modified_genome['mutation_variance'] = min(0.5, current_variance + mutation_variance_increase)
        
        # Record behavioral change
        behavioral_change = BehavioralChange(
            rule=CARule.RULE_110,
            timestamp=datetime.now(),
            parameters={
                'exploration_increase': exploration_increase,
                'mutation_variance_increase': mutation_variance_increase,
                'complexity_score': complexity
            },
            magnitude=exploration_increase + mutation_variance_increase,
            description=f"Increased exploration by {exploration_increase:.2%} and mutation variance by {mutation_variance_increase:.3f}"
        )
        
        # Update state
        new_state = CAState(
            cells=new_cells,
            generation=(current_state.generation + 1) if current_state else 0,
            rule=CARule.RULE_110,
            behavioral_changes=[behavioral_change],
            metrics={'complexity': complexity}
        )
        
        self.state_history.append(new_state)
        
        return modified_genome, behavioral_change
    
    async def apply_rule_30(self, agent_genome: Dict[str, Any],
                          current_state: Optional[CAState] = None) -> Tuple[Dict[str, Any], BehavioralChange]:
        """
        Apply Rule 30: Introduce bounded stochastic elements to decision-making
        
        This rule generates high-quality randomness for stochastic exploration.
        """
        # Initialize or get current state
        if current_state is None:
            cells = self._genome_to_cells(agent_genome)
        else:
            cells = current_state.cells
            
        # Apply Rule 30
        new_cells = self._apply_ca_rule(cells, CARule.RULE_30)
        
        # Extract random values from the chaotic pattern
        random_values = self._extract_random_values(new_cells, count=5)
        
        # Create bounded stochastic parameters
        stochastic_params = {
            'decision_noise': 0.1 * random_values[0],  # 0-10% noise
            'exploration_epsilon': 0.2 * random_values[1],  # 0-20% epsilon
            'action_temperature': 0.5 + 0.5 * random_values[2],  # 0.5-1.0 temperature
            'learning_rate_variance': 0.05 * random_values[3],  # 0-5% variance
            'momentum_stochasticity': 0.1 * random_values[4]  # 0-10% momentum noise
        }
        
        # Apply changes to genome
        modified_genome = agent_genome.copy()
        
        # Add stochastic elements
        if 'stochastic_params' not in modified_genome:
            modified_genome['stochastic_params'] = {}
        
        for param, value in stochastic_params.items():
            modified_genome['stochastic_params'][param] = value
        
        # Record behavioral change
        total_stochasticity = sum(stochastic_params.values())
        behavioral_change = BehavioralChange(
            rule=CARule.RULE_30,
            timestamp=datetime.now(),
            parameters=stochastic_params,
            magnitude=total_stochasticity,
            description=f"Added bounded stochastic elements with total magnitude {total_stochasticity:.3f}"
        )
        
        # Update state
        new_state = CAState(
            cells=new_cells,
            generation=(current_state.generation + 1) if current_state else 0,
            rule=CARule.RULE_30,
            behavioral_changes=[behavioral_change],
            metrics={'entropy': self._calculate_entropy(new_cells)}
        )
        
        self.state_history.append(new_state)
        
        return modified_genome, behavioral_change
    
    async def apply_rule_90(self, agent_genome: Dict[str, Any],
                          current_state: Optional[CAState] = None) -> Tuple[Dict[str, Any], BehavioralChange]:
        """
        Apply Rule 90: Create self-similar patterns at different abstraction levels
        
        This rule generates fractal patterns for hierarchical optimization.
        """
        # Initialize or get current state
        if current_state is None:
            cells = self._genome_to_cells(agent_genome)
        else:
            cells = current_state.cells
            
        # Apply Rule 90
        new_cells = self._apply_ca_rule(cells, CARule.RULE_90)
        
        # Detect self-similar patterns at multiple scales
        patterns = self._detect_self_similar_patterns(new_cells)
        
        # Create hierarchical abstraction levels
        abstraction_levels = []
        for scale, pattern_info in patterns.items():
            abstraction_levels.append({
                'scale': scale,
                'pattern_strength': pattern_info['strength'],
                'optimization_weight': pattern_info['strength'] * 0.2,  # Up to 20% weight per level
                'feature_extraction_depth': min(5, int(pattern_info['strength'] * 5))
            })
        
        # Apply changes to genome
        modified_genome = agent_genome.copy()
        
        # Add hierarchical processing
        modified_genome['hierarchical_processing'] = {
            'enabled': True,
            'abstraction_levels': abstraction_levels[:3],  # Use top 3 levels
            'cross_level_connections': 0.3,  # 30% cross-level information flow
            'recursive_depth': min(4, len(abstraction_levels))
        }
        
        # Record behavioral change
        total_levels = len(abstraction_levels[:3])
        behavioral_change = BehavioralChange(
            rule=CARule.RULE_90,
            timestamp=datetime.now(),
            parameters={
                'abstraction_levels': total_levels,
                'pattern_scales': list(patterns.keys())[:3],
                'max_pattern_strength': max(p['strength'] for p in patterns.values()) if patterns else 0
            },
            magnitude=total_levels * 0.3,
            description=f"Created {total_levels} self-similar abstraction levels"
        )
        
        # Update state
        new_state = CAState(
            cells=new_cells,
            generation=(current_state.generation + 1) if current_state else 0,
            rule=CARule.RULE_90,
            behavioral_changes=[behavioral_change],
            metrics={'fractal_dimension': self._estimate_fractal_dimension(new_cells)}
        )
        
        self.state_history.append(new_state)
        
        return modified_genome, behavioral_change
    
    async def apply_rule_184(self, agent_genome: Dict[str, Any],
                           current_state: Optional[CAState] = None) -> Tuple[Dict[str, Any], BehavioralChange]:
        """
        Apply Rule 184: Optimize pathways between agent components
        
        This rule models traffic flow dynamics for efficient information routing.
        """
        # Initialize or get current state
        if current_state is None:
            cells = self._genome_to_cells(agent_genome)
        else:
            cells = current_state.cells
            
        # Apply Rule 184
        new_cells = self._apply_ca_rule(cells, CARule.RULE_184)
        
        # Analyze traffic flow patterns
        flow_metrics = self._analyze_traffic_flow(new_cells)
        
        # Create optimized pathways based on flow analysis
        pathway_optimizations = {
            'input_processing_priority': flow_metrics['density_clusters'],
            'component_routing': {
                'perception_to_decision': 0.8 + 0.2 * flow_metrics['flow_efficiency'],
                'decision_to_action': 0.7 + 0.3 * flow_metrics['flow_efficiency'],
                'memory_to_decision': 0.6 + 0.4 * flow_metrics['flow_efficiency'],
                'learning_feedback': 0.5 + 0.5 * flow_metrics['flow_efficiency']
            },
            'parallel_pathways': max(1, int(flow_metrics['parallel_flows'] * 3)),
            'bottleneck_bypasses': flow_metrics['bottleneck_count']
        }
        
        # Apply changes to genome
        modified_genome = agent_genome.copy()
        
        # Optimize information pathways
        modified_genome['information_routing'] = pathway_optimizations
        
        # Add flow control parameters
        modified_genome['flow_control'] = {
            'adaptive_routing': True,
            'congestion_threshold': 0.7,
            'reroute_probability': 0.3 * flow_metrics['flow_efficiency'],
            'pathway_learning_rate': 0.1
        }
        
        # Record behavioral change
        optimization_score = flow_metrics['flow_efficiency'] + \
                           (flow_metrics['parallel_flows'] * 0.2) - \
                           (flow_metrics['bottleneck_count'] * 0.1)
        
        behavioral_change = BehavioralChange(
            rule=CARule.RULE_184,
            timestamp=datetime.now(),
            parameters={
                'flow_efficiency': flow_metrics['flow_efficiency'],
                'parallel_pathways': pathway_optimizations['parallel_pathways'],
                'bottlenecks_removed': flow_metrics['bottleneck_count']
            },
            magnitude=optimization_score,
            description=f"Optimized pathways with {flow_metrics['flow_efficiency']:.2%} efficiency"
        )
        
        # Update state
        new_state = CAState(
            cells=new_cells,
            generation=(current_state.generation + 1) if current_state else 0,
            rule=CARule.RULE_184,
            behavioral_changes=[behavioral_change],
            metrics=flow_metrics
        )
        
        self.state_history.append(new_state)
        
        return modified_genome, behavioral_change
    
    async def apply_rule(self, rule: CARule, agent_genome: Dict[str, Any],
                        current_state: Optional[CAState] = None) -> Tuple[Dict[str, Any], BehavioralChange]:
        """
        Apply a specific cellular automaton rule to an agent genome.
        
        Args:
            rule: The CA rule to apply
            agent_genome: The agent's genome to modify
            current_state: Optional current CA state
            
        Returns:
            Tuple of (modified_genome, behavioral_change)
        """
        if rule == CARule.RULE_110:
            return await self.apply_rule_110(agent_genome, current_state)
        elif rule == CARule.RULE_30:
            return await self.apply_rule_30(agent_genome, current_state)
        elif rule == CARule.RULE_90:
            return await self.apply_rule_90(agent_genome, current_state)
        elif rule == CARule.RULE_184:
            return await self.apply_rule_184(agent_genome, current_state)
        else:
            raise ValueError(f"Unknown rule: {rule}")
    
    def _genome_to_cells(self, genome: Dict[str, Any]) -> np.ndarray:
        """Convert agent genome to cellular automaton cells."""
        # Create a hash of the genome for deterministic conversion
        genome_str = json.dumps(genome, sort_keys=True)
        genome_hash = hashlib.sha256(genome_str.encode()).digest()
        
        # Convert hash to binary cells
        cells = np.zeros(self.grid_size, dtype=np.uint8)
        for i in range(min(len(genome_hash), self.grid_size)):
            cells[i] = genome_hash[i] % 2
            
        return cells
    
    def _calculate_complexity(self, cells: np.ndarray) -> float:
        """Calculate the complexity of a cell pattern."""
        # Count transitions
        transitions = np.sum(np.abs(np.diff(cells)))
        
        # Calculate local pattern diversity
        patterns = set()
        for i in range(len(cells) - 2):
            pattern = tuple(cells[i:i+3])
            patterns.add(pattern)
        
        # Complexity score combines transitions and pattern diversity
        complexity = transitions + len(patterns)
        return float(complexity)
    
    def _calculate_entropy(self, cells: np.ndarray) -> float:
        """Calculate Shannon entropy of the cell state."""
        unique, counts = np.unique(cells, return_counts=True)
        probabilities = counts / len(cells)
        
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def _extract_random_values(self, cells: np.ndarray, count: int) -> List[float]:
        """Extract pseudo-random values from chaotic CA pattern."""
        values = []
        
        # Use overlapping windows to extract values
        window_size = 8
        for i in range(count):
            start = (i * 5) % (len(cells) - window_size)
            window = cells[start:start + window_size]
            
            # Convert binary window to float in [0, 1]
            value = 0.0
            for j, bit in enumerate(window):
                value += bit * (2 ** -(j + 1))
            
            values.append(value)
            
        return values
    
    def _detect_self_similar_patterns(self, cells: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Detect self-similar patterns at different scales."""
        patterns = {}
        
        # Check patterns at different scales (2, 4, 8, 16)
        for scale in [2, 4, 8, 16]:
            if scale > len(cells) // 4:
                continue
                
            # Downsample cells by scale
            downsampled = cells[::scale]
            
            # Compare with original pattern structure
            similarity = self._calculate_pattern_similarity(cells, downsampled, scale)
            
            if similarity > 0.5:  # Threshold for self-similarity
                patterns[scale] = {
                    'strength': similarity,
                    'period': self._find_pattern_period(downsampled),
                    'symmetry': self._calculate_symmetry(downsampled)
                }
                
        return patterns
    
    def _calculate_pattern_similarity(self, original: np.ndarray, 
                                    downsampled: np.ndarray, scale: int) -> float:
        """Calculate similarity between original and scaled patterns."""
        # Resize downsampled to match a portion of original
        comparison_length = min(len(downsampled) * scale, len(original))
        
        similarity_scores = []
        for offset in range(scale):
            score = 0.0
            for i in range(len(downsampled)):
                if i * scale + offset < len(original):
                    if downsampled[i] == original[i * scale + offset]:
                        score += 1.0
            
            similarity_scores.append(score / len(downsampled))
        
        return max(similarity_scores)
    
    def _find_pattern_period(self, cells: np.ndarray) -> int:
        """Find the period of a repeating pattern."""
        for period in range(1, len(cells) // 2):
            is_periodic = True
            for i in range(len(cells) - period):
                if cells[i] != cells[i + period]:
                    is_periodic = False
                    break
            
            if is_periodic:
                return period
                
        return len(cells)  # No period found
    
    def _calculate_symmetry(self, cells: np.ndarray) -> float:
        """Calculate the symmetry score of a pattern."""
        symmetry_score = 0.0
        
        # Check for reflection symmetry
        for i in range(len(cells) // 2):
            if cells[i] == cells[-(i + 1)]:
                symmetry_score += 1.0
                
        return symmetry_score / (len(cells) // 2)
    
    def _estimate_fractal_dimension(self, cells: np.ndarray) -> float:
        """Estimate the fractal dimension using box-counting method."""
        # Simplified box-counting for 1D pattern
        box_sizes = [1, 2, 4, 8]
        counts = []
        
        for size in box_sizes:
            if size > len(cells):
                break
                
            count = 0
            for i in range(0, len(cells), size):
                box = cells[i:i+size]
                if np.any(box):  # Box contains at least one active cell
                    count += 1
            
            counts.append(count)
        
        # Estimate dimension from log-log plot slope
        if len(counts) > 1:
            log_sizes = np.log(box_sizes[:len(counts)])
            log_counts = np.log(counts)
            
            # Linear regression for slope
            slope = -np.polyfit(log_sizes, log_counts, 1)[0]
            return slope
        
        return 1.0  # Default dimension
    
    def _analyze_traffic_flow(self, cells: np.ndarray) -> Dict[str, float]:
        """Analyze traffic flow patterns in the CA."""
        # Count clusters of active cells (vehicles/packets)
        clusters = []
        in_cluster = False
        cluster_size = 0
        
        for cell in cells:
            if cell == 1:
                if not in_cluster:
                    in_cluster = True
                    cluster_size = 1
                else:
                    cluster_size += 1
            else:
                if in_cluster:
                    clusters.append(cluster_size)
                    in_cluster = False
                    cluster_size = 0
        
        if in_cluster:
            clusters.append(cluster_size)
        
        # Calculate flow metrics
        total_active = np.sum(cells)
        density = total_active / len(cells)
        
        # Flow efficiency based on cluster distribution
        if clusters:
            avg_cluster_size = np.mean(clusters)
            cluster_variance = np.var(clusters)
            flow_efficiency = 1.0 / (1.0 + cluster_variance / (avg_cluster_size + 1))
        else:
            flow_efficiency = 0.0
        
        # Identify bottlenecks (large clusters)
        bottleneck_threshold = len(cells) * 0.1
        bottleneck_count = sum(1 for c in clusters if c > bottleneck_threshold)
        
        # Estimate parallel flows
        parallel_flows = len(clusters) / max(1, total_active) if total_active > 0 else 0
        
        return {
            'density': density,
            'flow_efficiency': flow_efficiency,
            'bottleneck_count': bottleneck_count,
            'parallel_flows': parallel_flows,
            'density_clusters': clusters[:5]  # Top 5 clusters for priority routing
        }
    
    def get_state_history(self) -> List[CAState]:
        """Get the complete state history for analysis."""
        return self.state_history
    
    def analyze_behavioral_impact(self) -> Dict[str, Any]:
        """Analyze the cumulative behavioral impact of CA applications."""
        if not self.state_history:
            return {'total_changes': 0, 'rules_applied': [], 'cumulative_magnitude': 0.0}
        
        total_changes = len(self.state_history)
        rules_applied = [state.rule.value for state in self.state_history]
        
        cumulative_magnitude = 0.0
        rule_impacts = {}
        
        for state in self.state_history:
            for change in state.behavioral_changes:
                cumulative_magnitude += change.magnitude
                
                if change.rule not in rule_impacts:
                    rule_impacts[change.rule.value] = {
                        'count': 0,
                        'total_magnitude': 0.0,
                        'parameters': []
                    }
                
                rule_impacts[change.rule.value]['count'] += 1
                rule_impacts[change.rule.value]['total_magnitude'] += change.magnitude
                rule_impacts[change.rule.value]['parameters'].append(change.parameters)
        
        return {
            'total_changes': total_changes,
            'rules_applied': rules_applied,
            'cumulative_magnitude': cumulative_magnitude,
            'rule_impacts': rule_impacts,
            'average_magnitude_per_change': cumulative_magnitude / total_changes if total_changes > 0 else 0
        }
    
    def export_visualization_data(self) -> Dict[str, Any]:
        """Export data for visualizing CA evolution."""
        if not self.state_history:
            return {'generations': 0, 'states': []}
        
        visualization_data = {
            'generations': len(self.state_history),
            'grid_size': self.grid_size,
            'states': []
        }
        
        for state in self.state_history:
            state_data = {
                'generation': state.generation,
                'rule': state.rule.value,
                'cells': state.cells.tolist(),
                'metrics': state.metrics,
                'behavioral_changes': [
                    {
                        'description': bc.description,
                        'magnitude': bc.magnitude
                    }
                    for bc in state.behavioral_changes
                ]
            }
            visualization_data['states'].append(state_data)
        
        return visualization_data