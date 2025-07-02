"""
Core pattern detection functionality for DEAN system.

Identifies recurring patterns, optimization strategies, and
emergent behaviors in agent evolution.
Per specifications: IndexAgent/indexagent/agents/patterns/detector.py
"""

import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
from enum import Enum
from .sliding_window_analyzer import SlidingWindowAnalyzer

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected."""
    BEHAVIORAL = "behavioral"  # Recurring action sequences
    OPTIMIZATION = "optimization"  # Performance improvement strategies
    STRUCTURAL = "structural"  # Genome organization patterns
    TEMPORAL = "temporal"  # Time-based patterns
    EMERGENT = "emergent"  # Unexpected beneficial behaviors
    ADVERSARIAL = "adversarial"  # Gaming or exploitation patterns


@dataclass
class Pattern:
    """Represents a detected pattern."""
    pattern_id: str
    pattern_type: PatternType
    description: str
    occurrences: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    effectiveness: float = 0.0  # How beneficial the pattern is
    confidence: float = 0.0  # Detection confidence
    
    # Pattern data
    sequence: List[Any] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Agents that exhibited this pattern
    agent_ids: Set[str] = field(default_factory=set)
    
    def update_occurrence(self, agent_id: str, effectiveness: float = None) -> None:
        """Update pattern with new occurrence."""
        self.occurrences += 1
        self.last_seen = datetime.now()
        self.agent_ids.add(agent_id)
        
        if effectiveness is not None:
            # Running average of effectiveness
            self.effectiveness = (
                (self.effectiveness * (self.occurrences - 1) + effectiveness) / 
                self.occurrences
            )
    
    def calculate_hash(self) -> str:
        """Calculate unique hash for pattern matching."""
        pattern_data = {
            'type': self.pattern_type.value,
            'sequence': str(self.sequence),
            'context_keys': sorted(self.context.keys())
        }
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Serialize pattern to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'description': self.description,
            'occurrences': self.occurrences,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'effectiveness': self.effectiveness,
            'confidence': self.confidence,
            'sequence': self.sequence,
            'context': self.context,
            'metadata': self.metadata,
            'agent_ids': list(self.agent_ids)
        }


class PatternDetector:
    """
    Detects patterns in agent behavior and evolution.
    
    Uses sequence analysis, statistical methods, and heuristics
    to identify recurring patterns and emergent strategies.
    """
    
    def __init__(self,
                 min_occurrences: int = 3,
                 confidence_threshold: float = 0.7,
                 sequence_length: int = 5):
        """
        Initialize pattern detector.
        
        Args:
            min_occurrences: Minimum times pattern must occur
            confidence_threshold: Minimum confidence for pattern recognition
            sequence_length: Default length for sequence patterns
        """
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        
        # Pattern storage
        self.detected_patterns: Dict[str, Pattern] = {}
        self.pattern_sequences: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        
        # Tracking for pattern detection
        self.action_history: Dict[str, List[Dict]] = defaultdict(list)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.genome_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Initialize sliding window analyzer
        self.sliding_window = SlidingWindowAnalyzer(
            min_window_size=3,
            max_window_size=10,
            confidence_threshold=confidence_threshold
        )
    
    def track_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Track an agent action for pattern detection."""
        timestamped_action = {
            **action,
            'timestamp': datetime.now()
        }
        self.action_history[agent_id].append(timestamped_action)
        
        # Keep history bounded
        if len(self.action_history[agent_id]) > 1000:
            self.action_history[agent_id] = self.action_history[agent_id][-1000:]
    
    def track_performance(self, agent_id: str, performance: float) -> None:
        """Track agent performance metrics."""
        self.performance_history[agent_id].append(performance)
        
        # Keep history bounded
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
    
    def track_genome_state(self, agent_id: str, genome_summary: Dict) -> None:
        """Track genome evolution for structural patterns."""
        self.genome_history[agent_id].append({
            **genome_summary,
            'timestamp': datetime.now()
        })
        
        # Keep history bounded
        if len(self.genome_history[agent_id]) > 50:
            self.genome_history[agent_id] = self.genome_history[agent_id][-50:]
    
    def detect_patterns_for_agent(self, agent_id: str) -> List[Pattern]:
        """
        Detect all types of patterns for an agent.
        
        Returns list of newly detected or updated patterns.
        """
        patterns = []
        
        # Detect different pattern types
        patterns.extend(self._detect_behavioral_patterns(agent_id))
        patterns.extend(self._detect_optimization_patterns(agent_id))
        patterns.extend(self._detect_structural_patterns(agent_id))
        patterns.extend(self._detect_temporal_patterns(agent_id))
        
        # Add sliding window pattern detection
        patterns.extend(self.detect_sliding_window_patterns(agent_id))
        
        # Filter by confidence
        confident_patterns = [
            p for p in patterns 
            if p.confidence >= self.confidence_threshold
        ]
        
        # Update pattern registry
        for pattern in confident_patterns:
            pattern_hash = pattern.calculate_hash()
            
            if pattern_hash in self.detected_patterns:
                # Update existing pattern
                existing = self.detected_patterns[pattern_hash]
                existing.update_occurrence(agent_id, pattern.effectiveness)
            else:
                # New pattern
                pattern.pattern_id = pattern_hash
                self.detected_patterns[pattern_hash] = pattern
                logger.info(f"New pattern detected: {pattern.description}")
        
        return confident_patterns
    
    def _detect_behavioral_patterns(self, agent_id: str) -> List[Pattern]:
        """Detect recurring action sequences."""
        patterns = []
        
        if agent_id not in self.action_history:
            return patterns
        
        actions = self.action_history[agent_id]
        if len(actions) < self.sequence_length * 2:
            return patterns
        
        # Extract action sequences
        sequences = []
        for i in range(len(actions) - self.sequence_length + 1):
            seq = tuple(a.get('action_type', 'unknown') 
                       for a in actions[i:i + self.sequence_length])
            sequences.append(seq)
        
        # Find recurring sequences
        seq_counter = Counter(sequences)
        
        for seq, count in seq_counter.items():
            if count >= self.min_occurrences:
                # Calculate effectiveness based on performance after sequence
                effectiveness = self._calculate_sequence_effectiveness(
                    agent_id, seq, actions
                )
                
                pattern = Pattern(
                    pattern_id="",  # Will be set by hash
                    pattern_type=PatternType.BEHAVIORAL,
                    description=f"Action sequence: {' -> '.join(seq)}",
                    occurrences=count,
                    effectiveness=effectiveness,
                    confidence=min(1.0, count / 10),  # More occurrences = higher confidence
                    sequence=list(seq),
                    context={'agent_id': agent_id}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_optimization_patterns(self, agent_id: str) -> List[Pattern]:
        """Detect performance improvement strategies."""
        patterns = []
        
        if agent_id not in self.performance_history:
            return patterns
        
        performance = self.performance_history[agent_id]
        if len(performance) < 10:
            return patterns
        
        # Detect improvement trends
        improvements = []
        for i in range(1, len(performance)):
            if performance[i] > performance[i-1] * 1.1:  # 10% improvement
                # Look for associated actions
                if i < len(self.action_history[agent_id]):
                    action = self.action_history[agent_id][i]
                    improvements.append({
                        'improvement': performance[i] / performance[i-1],
                        'action': action,
                        'performance': performance[i]
                    })
        
        # Group similar improvements
        if len(improvements) >= self.min_occurrences:
            # Simple clustering by action type
            action_groups = defaultdict(list)
            for imp in improvements:
                action_type = imp['action'].get('action_type', 'unknown')
                action_groups[action_type].append(imp)
            
            for action_type, group in action_groups.items():
                if len(group) >= self.min_occurrences:
                    avg_improvement = np.mean([g['improvement'] for g in group])
                    
                    pattern = Pattern(
                        pattern_id="",
                        pattern_type=PatternType.OPTIMIZATION,
                        description=f"Performance optimization via {action_type}",
                        occurrences=len(group),
                        effectiveness=avg_improvement - 1.0,  # Improvement factor
                        confidence=min(1.0, len(group) / 5),
                        sequence=[action_type],
                        context={
                            'agent_id': agent_id,
                            'avg_improvement': avg_improvement
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_structural_patterns(self, agent_id: str) -> List[Pattern]:
        """Detect patterns in genome evolution."""
        patterns = []
        
        if agent_id not in self.genome_history:
            return patterns
        
        genome_states = self.genome_history[agent_id]
        if len(genome_states) < 5:
            return patterns
        
        # Detect gene combination patterns
        gene_combinations = []
        for state in genome_states:
            if 'gene_types' in state:
                combo = tuple(sorted(state['gene_types']))
                gene_combinations.append(combo)
        
        combo_counter = Counter(gene_combinations)
        
        for combo, count in combo_counter.items():
            if count >= self.min_occurrences:
                # Check if this combination correlates with performance
                perf_correlation = self._calculate_structure_performance_correlation(
                    agent_id, combo
                )
                
                if abs(perf_correlation) > 0.5:  # Significant correlation
                    pattern = Pattern(
                        pattern_id="",
                        pattern_type=PatternType.STRUCTURAL,
                        description=f"Gene combination: {', '.join(combo)}",
                        occurrences=count,
                        effectiveness=perf_correlation,
                        confidence=abs(perf_correlation),
                        sequence=list(combo),
                        context={
                            'agent_id': agent_id,
                            'correlation': perf_correlation
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_temporal_patterns(self, agent_id: str) -> List[Pattern]:
        """Detect time-based patterns."""
        patterns = []
        
        if agent_id not in self.action_history:
            return patterns
        
        actions = self.action_history[agent_id]
        if len(actions) < 20:
            return patterns
        
        # Detect periodic behaviors
        timestamps = [a['timestamp'] for a in actions]
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        # Simple periodicity detection
        if len(intervals) > 10:
            # Check for regular intervals
            interval_groups = defaultdict(int)
            for interval in intervals:
                # Group into buckets (e.g., 0-5s, 5-10s, etc.)
                bucket = int(interval / 5) * 5
                interval_groups[bucket] += 1
            
            # Find dominant interval
            if interval_groups:
                dominant_interval = max(interval_groups, key=interval_groups.get)
                frequency = interval_groups[dominant_interval] / len(intervals)
                
                if frequency > 0.3:  # 30% of actions follow this interval
                    pattern = Pattern(
                        pattern_id="",
                        pattern_type=PatternType.TEMPORAL,
                        description=f"Periodic behavior (~{dominant_interval}s intervals)",
                        occurrences=interval_groups[dominant_interval],
                        effectiveness=0.0,  # Neutral unless proven otherwise
                        confidence=frequency,
                        sequence=[dominant_interval],
                        context={
                            'agent_id': agent_id,
                            'interval': dominant_interval,
                            'frequency': frequency
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_sequence_effectiveness(self,
                                        agent_id: str,
                                        sequence: Tuple[str, ...],
                                        actions: List[Dict]) -> float:
        """Calculate effectiveness of an action sequence."""
        if agent_id not in self.performance_history:
            return 0.0
        
        performance = self.performance_history[agent_id]
        effectiveness_scores = []
        
        # Find sequence occurrences and check performance after
        for i in range(len(actions) - len(sequence)):
            current_seq = tuple(
                actions[j].get('action_type', 'unknown') 
                for j in range(i, i + len(sequence))
            )
            
            if current_seq == sequence:
                # Check performance change after sequence
                perf_idx = min(i + len(sequence), len(performance) - 1)
                if perf_idx > 0:
                    perf_change = performance[perf_idx] - performance[perf_idx - 1]
                    effectiveness_scores.append(perf_change)
        
        return np.mean(effectiveness_scores) if effectiveness_scores else 0.0
    
    def _calculate_structure_performance_correlation(self,
                                                   agent_id: str,
                                                   gene_combo: Tuple[str, ...]) -> float:
        """Calculate correlation between genome structure and performance."""
        if agent_id not in self.performance_history:
            return 0.0
        
        genome_states = self.genome_history[agent_id]
        performance = self.performance_history[agent_id]
        
        # Match genome states with performance
        has_combo = []
        perf_values = []
        
        for i, state in enumerate(genome_states):
            if i < len(performance):
                if 'gene_types' in state:
                    current_combo = tuple(sorted(state['gene_types']))
                    has_combo.append(1 if current_combo == gene_combo else 0)
                    perf_values.append(performance[i])
        
        if len(has_combo) < 5:
            return 0.0
        
        # Calculate correlation
        return np.corrcoef(has_combo, perf_values)[0, 1]
    
    def get_top_patterns(self, 
                        pattern_type: Optional[PatternType] = None,
                        min_effectiveness: float = 0.0,
                        limit: int = 10) -> List[Pattern]:
        """Get most effective/common patterns."""
        patterns = list(self.detected_patterns.values())
        
        # Filter by type if specified
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        # Filter by effectiveness
        patterns = [p for p in patterns if p.effectiveness >= min_effectiveness]
        
        # Sort by combination of effectiveness and occurrences
        patterns.sort(
            key=lambda p: p.effectiveness * np.log1p(p.occurrences),
            reverse=True
        )
        
        return patterns[:limit]
    
    def export_patterns(self, filepath: str) -> None:
        """Export detected patterns to file."""
        patterns_data = {
            'metadata': {
                'total_patterns': len(self.detected_patterns),
                'export_time': datetime.now().isoformat(),
                'settings': {
                    'min_occurrences': self.min_occurrences,
                    'confidence_threshold': self.confidence_threshold
                }
            },
            'patterns': [p.to_dict() for p in self.detected_patterns.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"Exported {len(self.detected_patterns)} patterns to {filepath}")
    
    async def detect_patterns(self, agents: List['Agent']) -> List[Pattern]:
        """
        Detect patterns across a population of agents.
        Compatible with evolution engine usage.
        """
        all_patterns = []
        
        for agent in agents:
            # Track agent data
            self.track_genome_state(agent.id, {
                'gene_types': list(agent.genome.strategies),
                'traits': agent.genome.traits
            })
            self.track_performance(agent.id, agent.fitness_score)
            
            # Detect patterns for this single agent
            patterns_for_agent = self.detect_patterns_for_agent(agent.id)
            all_patterns.extend(patterns_for_agent)
        
        return all_patterns
    
    def detect_sliding_window_patterns(self, agent_id: str) -> List[Pattern]:
        """
        Detect patterns using sliding window analysis.
        
        Returns:
            List of patterns detected through sliding window analysis
        """
        patterns = []
        
        # Analyze action sequences
        if agent_id in self.action_history:
            actions = self.action_history[agent_id]
            if len(actions) >= 3:
                # Extract action types and timestamps
                action_sequence = [a.get('action_type', 'unknown') for a in actions]
                timestamps = [a.get('timestamp', datetime.now()) for a in actions]
                
                # Run sliding window analysis
                window_patterns = self.sliding_window.analyze_sequence(
                    action_sequence, 
                    f"agent_{agent_id}_actions",
                    timestamps
                )
                
                # Convert to Pattern objects
                for wp in window_patterns:
                    pattern = Pattern(
                        pattern_id=wp.pattern_hash,
                        pattern_type=PatternType.BEHAVIORAL,
                        description=f"Sliding window pattern: {wp.sequence}",
                        occurrences=wp.frequency,
                        effectiveness=wp.confidence,
                        confidence=wp.confidence,
                        sequence=wp.sequence,
                        context={
                            'agent_id': agent_id,
                            'window_size': wp.window_size,
                            'temporal_consistency': wp.temporal_consistency
                        },
                        metadata=wp.metadata
                    )
                    patterns.append(pattern)
        
        # Analyze performance sequences
        if agent_id in self.performance_history:
            performance = self.performance_history[agent_id]
            if len(performance) >= 5:
                # Create timestamps based on history length
                timestamps = [
                    datetime.now() - timedelta(minutes=len(performance)-i) 
                    for i in range(len(performance))
                ]
                
                # Run analysis
                perf_patterns = self.sliding_window.analyze_sequence(
                    performance,
                    f"agent_{agent_id}_performance",
                    timestamps
                )
                
                # Look for temporal patterns
                temporal_patterns = self.sliding_window.temporal_patterns
                
                for tp in temporal_patterns:
                    if tp.pattern_type == "trending" and tp.trend_slope:
                        pattern = Pattern(
                            pattern_id=tp.pattern_id,
                            pattern_type=PatternType.OPTIMIZATION if tp.trend_slope > 0 else PatternType.TEMPORAL,
                            description=f"Performance trend: {'improving' if tp.trend_slope > 0 else 'declining'}",
                            occurrences=len(performance),
                            effectiveness=abs(tp.trend_slope),
                            confidence=tp.confidence,
                            sequence=[],
                            context={
                                'agent_id': agent_id,
                                'trend_slope': tp.trend_slope,
                                'pattern_subtype': 'trend'
                            }
                        )
                        patterns.append(pattern)
        
        return patterns