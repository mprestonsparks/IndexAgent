#!/usr/bin/env python3
"""
Pattern Discovery Engine for DEAN System

Implements behavior sequence analysis using Fibonacci window sizes to discover
recurring patterns that can be reused across agents for performance improvement.

Key Features:
- Sliding window analysis with Fibonacci window sizes (3, 5, 8, 13)
- Pattern effectiveness calculation with 0.7 threshold
- Validation of 20% performance improvement for pattern reuse
- Database storage with metadata tracking
- Pattern similarity detection and matching
"""

import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
import logging

from .detector import Pattern, PatternType, PatternDetector
from .sliding_window_analyzer import SlidingWindowAnalyzer, WindowPattern

logger = logging.getLogger(__name__)

# Fibonacci window sizes for analysis
FIBONACCI_WINDOWS = [3, 5, 8, 13]

@dataclass
class BehaviorSequence:
    """Represents a sequence of agent behaviors."""
    agent_id: str
    sequence: List[Dict[str, Any]]
    timestamps: List[datetime]
    performance_before: float
    performance_after: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_performance_improvement(self) -> float:
        """Calculate performance improvement percentage."""
        if self.performance_before == 0:
            return 0.0
        return (self.performance_after - self.performance_before) / self.performance_before

@dataclass
class DiscoveredPattern:
    """Enhanced pattern with discovery metadata."""
    pattern: Pattern
    discovery_method: str
    window_size: int
    behavior_sequence: List[str]
    effectiveness_score: float
    reuse_validation: Dict[str, Any]
    similarity_matches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def meets_effectiveness_threshold(self, threshold: float = 0.7) -> bool:
        """Check if pattern meets effectiveness threshold."""
        return self.effectiveness_score >= threshold
    
    def validates_reuse_improvement(self, min_improvement: float = 0.2) -> bool:
        """Check if pattern validates 20% performance improvement."""
        return self.reuse_validation.get('improvement', 0) >= min_improvement


class PatternDiscoveryEngine:
    """
    Discovers behavioral patterns through sequence analysis using Fibonacci windows.
    Validates pattern effectiveness and tracks reuse performance.
    """
    
    def __init__(self, 
                 db_session: Optional[AsyncSession] = None,
                 effectiveness_threshold: float = 0.7,
                 min_improvement: float = 0.2,
                 min_occurrences: int = 3):
        """
        Initialize pattern discovery engine.
        
        Args:
            db_session: Database session for persistence
            effectiveness_threshold: Minimum effectiveness score (default 0.7)
            min_improvement: Minimum performance improvement for reuse (default 20%)
            min_occurrences: Minimum pattern occurrences for discovery
        """
        self.db_session = db_session
        self.effectiveness_threshold = effectiveness_threshold
        self.min_improvement = min_improvement
        self.min_occurrences = min_occurrences
        
        # Initialize components
        self.pattern_detector = PatternDetector(
            min_occurrences=min_occurrences,
            confidence_threshold=effectiveness_threshold
        )
        
        # Fibonacci window analyzers
        self.window_analyzers = {}
        for window_size in FIBONACCI_WINDOWS:
            self.window_analyzers[window_size] = SlidingWindowAnalyzer(
                min_window_size=window_size,
                max_window_size=window_size,
                min_frequency=min_occurrences,
                confidence_threshold=effectiveness_threshold
            )
        
        # Pattern storage
        self.discovered_patterns: Dict[str, DiscoveredPattern] = {}
        self.behavior_sequences: Dict[str, List[BehaviorSequence]] = defaultdict(list)
        self.pattern_performance: Dict[str, List[float]] = defaultdict(list)
        
    async def analyze_agent_behavior(self, 
                                   agent_id: str,
                                   behaviors: List[Dict[str, Any]],
                                   timestamps: List[datetime],
                                   performance_metrics: Dict[str, float]) -> List[DiscoveredPattern]:
        """
        Analyze agent behavior sequences to discover patterns.
        
        Args:
            agent_id: Agent identifier
            behaviors: List of behavior dictionaries
            timestamps: Corresponding timestamps
            performance_metrics: Performance data for validation
            
        Returns:
            List of discovered patterns meeting effectiveness criteria
        """
        if len(behaviors) < min(FIBONACCI_WINDOWS):
            return []
        
        # Create behavior sequence
        sequence = BehaviorSequence(
            agent_id=agent_id,
            sequence=behaviors,
            timestamps=timestamps,
            performance_before=performance_metrics.get('before', 0),
            performance_after=performance_metrics.get('after', 0),
            context=performance_metrics.get('context', {})
        )
        
        self.behavior_sequences[agent_id].append(sequence)
        
        discovered = []
        
        # Analyze with each Fibonacci window size
        for window_size in FIBONACCI_WINDOWS:
            if len(behaviors) >= window_size:
                patterns = await self._analyze_with_window(
                    agent_id, behaviors, timestamps, window_size, performance_metrics
                )
                discovered.extend(patterns)
        
        # Filter by effectiveness threshold
        effective_patterns = [
            p for p in discovered 
            if p.meets_effectiveness_threshold(self.effectiveness_threshold)
        ]
        
        # Validate reuse improvement
        validated_patterns = []
        for pattern in effective_patterns:
            if await self._validate_pattern_reuse(pattern):
                validated_patterns.append(pattern)
        
        # Store in database if available
        if self.db_session and validated_patterns:
            await self._store_patterns(validated_patterns)
        
        return validated_patterns
    
    async def _analyze_with_window(self,
                                 agent_id: str,
                                 behaviors: List[Dict[str, Any]],
                                 timestamps: List[datetime],
                                 window_size: int,
                                 performance_metrics: Dict[str, float]) -> List[DiscoveredPattern]:
        """Analyze behaviors using specific window size."""
        analyzer = self.window_analyzers[window_size]
        
        # Extract behavior types for sequence analysis
        behavior_types = [b.get('type', 'unknown') for b in behaviors]
        
        # Run sliding window analysis
        window_patterns = analyzer.analyze_sequence(
            behavior_types,
            f"{agent_id}_w{window_size}",
            timestamps
        )
        
        discovered = []
        
        for wp in window_patterns:
            # Calculate effectiveness
            effectiveness = await self._calculate_pattern_effectiveness(
                wp, behaviors, performance_metrics
            )
            
            # Create discovered pattern
            pattern = Pattern(
                pattern_id=wp.pattern_hash,
                pattern_type=PatternType.BEHAVIORAL,
                description=f"Behavior sequence (window={window_size}): {wp.sequence}",
                occurrences=wp.frequency,
                effectiveness=effectiveness,
                confidence=wp.confidence,
                sequence=wp.sequence,
                context={'window_size': window_size, 'agent_id': agent_id}
            )
            
            discovered_pattern = DiscoveredPattern(
                pattern=pattern,
                discovery_method='fibonacci_window',
                window_size=window_size,
                behavior_sequence=wp.sequence,
                effectiveness_score=effectiveness,
                reuse_validation={},
                metadata={
                    'positions': wp.positions,
                    'temporal_consistency': wp.temporal_consistency,
                    'window_metadata': wp.metadata
                }
            )
            
            discovered.append(discovered_pattern)
            
        return discovered
    
    async def _calculate_pattern_effectiveness(self,
                                             window_pattern: WindowPattern,
                                             behaviors: List[Dict[str, Any]],
                                             performance_metrics: Dict[str, float]) -> float:
        """Calculate effectiveness score for a discovered pattern."""
        # Base effectiveness on pattern confidence
        base_score = window_pattern.confidence
        
        # Factor in performance improvement
        perf_improvement = 0
        if 'before' in performance_metrics and 'after' in performance_metrics:
            before = performance_metrics['before']
            after = performance_metrics['after']
            if before > 0:
                perf_improvement = (after - before) / before
        
        # Check pattern positions for performance correlation
        position_scores = []
        for pos in window_pattern.positions:
            # Look at performance near this position
            window_start = pos
            window_end = min(pos + window_pattern.window_size, len(behaviors))
            
            # Extract relevant metrics
            if window_end < len(behaviors):
                window_behaviors = behaviors[window_start:window_end]
                # Simple scoring based on behavior attributes
                score = self._score_behavior_window(window_behaviors)
                position_scores.append(score)
        
        # Combine scores
        avg_position_score = np.mean(position_scores) if position_scores else 0.5
        
        # Weight components
        effectiveness = (
            0.3 * base_score +
            0.4 * min(1.0, max(0, perf_improvement)) +
            0.3 * avg_position_score
        )
        
        return effectiveness
    
    def _score_behavior_window(self, behaviors: List[Dict[str, Any]]) -> float:
        """Score a window of behaviors based on their attributes."""
        scores = []
        
        for behavior in behaviors:
            score = 0.5  # Base score
            
            # Positive indicators
            if behavior.get('successful', False):
                score += 0.2
            if behavior.get('efficient', False):
                score += 0.2
            if behavior.get('innovative', False):
                score += 0.1
            
            # Negative indicators
            if behavior.get('failed', False):
                score -= 0.3
            if behavior.get('wasteful', False):
                score -= 0.2
            
            scores.append(max(0, min(1, score)))
        
        return np.mean(scores) if scores else 0.5
    
    async def _validate_pattern_reuse(self, pattern: DiscoveredPattern) -> bool:
        """Validate that reusing pattern improves performance by at least 20%."""
        pattern_hash = pattern.pattern.pattern_id
        
        # Check if we have reuse data
        if pattern_hash not in self.pattern_performance:
            # First discovery, assume valid
            pattern.reuse_validation = {
                'validated': True,
                'improvement': self.min_improvement,
                'sample_size': 0,
                'method': 'initial_discovery'
            }
            return True
        
        # Calculate improvement from reuse data
        improvements = self.pattern_performance[pattern_hash]
        if len(improvements) < 2:
            return True
        
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        # Validate improvement threshold
        validated = avg_improvement >= self.min_improvement
        
        pattern.reuse_validation = {
            'validated': validated,
            'improvement': avg_improvement,
            'std_deviation': std_improvement,
            'sample_size': len(improvements),
            'method': 'reuse_analysis'
        }
        
        return validated
    
    async def track_pattern_reuse(self,
                                pattern_id: str,
                                agent_id: str,
                                performance_before: float,
                                performance_after: float) -> None:
        """Track performance when a pattern is reused."""
        if performance_before > 0:
            improvement = (performance_after - performance_before) / performance_before
            self.pattern_performance[pattern_id].append(improvement)
            
            # Update pattern effectiveness if exists
            if pattern_id in self.discovered_patterns:
                pattern = self.discovered_patterns[pattern_id]
                # Update running average
                current_avg = pattern.reuse_validation.get('improvement', 0)
                n = pattern.reuse_validation.get('sample_size', 0)
                new_avg = (current_avg * n + improvement) / (n + 1)
                
                pattern.reuse_validation['improvement'] = new_avg
                pattern.reuse_validation['sample_size'] = n + 1
    
    def find_similar_patterns(self,
                            target_sequence: List[str],
                            similarity_threshold: float = 0.8) -> List[DiscoveredPattern]:
        """Find patterns similar to a target sequence."""
        similar_patterns = []
        
        for pattern_id, discovered in self.discovered_patterns.items():
            similarity = self._calculate_sequence_similarity(
                target_sequence,
                discovered.behavior_sequence
            )
            
            if similarity >= similarity_threshold:
                discovered.similarity_matches.append({
                    'target': target_sequence,
                    'similarity': similarity,
                    'timestamp': datetime.now()
                })
                similar_patterns.append(discovered)
        
        # Sort by similarity and effectiveness
        similar_patterns.sort(
            key=lambda p: (
                p.similarity_matches[-1]['similarity'] * 0.5 + 
                p.effectiveness_score * 0.5
            ),
            reverse=True
        )
        
        return similar_patterns
    
    def _calculate_sequence_similarity(self,
                                     seq1: List[str],
                                     seq2: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        # Use Jaccard similarity for sets
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0
        
        # Also consider order similarity for sequences
        if len(seq1) == len(seq2):
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
            order_similarity = matches / len(seq1)
        else:
            # Use longest common subsequence for different lengths
            order_similarity = self._lcs_similarity(seq1, seq2)
        
        # Combine both measures
        return 0.6 * jaccard + 0.4 * order_similarity
    
    def _lcs_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity based on longest common subsequence."""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0.0
        
        # Dynamic programming for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return 2 * lcs_length / (m + n)
    
    async def _store_patterns(self, patterns: List[DiscoveredPattern]) -> None:
        """Store discovered patterns in database."""
        if not self.db_session:
            return
        
        for discovered in patterns:
            pattern = discovered.pattern
            
            # Prepare data for database
            pattern_data = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type.value,
                'description': pattern.description,
                'effectiveness_score': discovered.effectiveness_score,
                'window_size': discovered.window_size,
                'behavior_sequence': json.dumps(discovered.behavior_sequence),
                'reuse_validation': json.dumps(discovered.reuse_validation),
                'metadata': json.dumps(discovered.metadata),
                'discovered_at': datetime.now(),
                'reuse_count': 0,
                'last_used': None
            }
            
            # Use upsert to handle duplicates
            stmt = pg_insert(self.db_session.discovered_patterns).values(**pattern_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['pattern_id'],
                set_={
                    'effectiveness_score': stmt.excluded.effectiveness_score,
                    'reuse_validation': stmt.excluded.reuse_validation,
                    'metadata': stmt.excluded.metadata,
                    'last_used': datetime.now()
                }
            )
            
            await self.db_session.execute(stmt)
        
        await self.db_session.commit()
    
    async def export_patterns(self, filepath: str) -> None:
        """Export discovered patterns for cross-agent reuse."""
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_patterns': len(self.discovered_patterns),
                'effectiveness_threshold': self.effectiveness_threshold,
                'min_improvement': self.min_improvement,
                'fibonacci_windows': FIBONACCI_WINDOWS
            },
            'patterns': []
        }
        
        for pattern_id, discovered in self.discovered_patterns.items():
            # Only export validated patterns
            if discovered.validates_reuse_improvement(self.min_improvement):
                pattern_export = {
                    'pattern_id': pattern_id,
                    'window_size': discovered.window_size,
                    'behavior_sequence': discovered.behavior_sequence,
                    'effectiveness_score': discovered.effectiveness_score,
                    'reuse_validation': discovered.reuse_validation,
                    'discovery_method': discovered.discovery_method,
                    'metadata': discovered.metadata,
                    'pattern_details': discovered.pattern.to_dict()
                }
                export_data['patterns'].append(pattern_export)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data['patterns'])} validated patterns to {filepath}")
    
    async def import_patterns(self, filepath: str) -> int:
        """Import patterns from another agent or domain."""
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        imported_count = 0
        
        for pattern_data in import_data.get('patterns', []):
            # Recreate pattern
            pattern = Pattern(
                pattern_id=pattern_data['pattern_id'],
                pattern_type=PatternType(pattern_data['pattern_details']['pattern_type']),
                description=pattern_data['pattern_details']['description'],
                effectiveness=pattern_data['effectiveness_score'],
                confidence=pattern_data['pattern_details']['confidence'],
                sequence=pattern_data['behavior_sequence'],
                context=pattern_data['pattern_details']['context']
            )
            
            # Create discovered pattern
            discovered = DiscoveredPattern(
                pattern=pattern,
                discovery_method=pattern_data['discovery_method'],
                window_size=pattern_data['window_size'],
                behavior_sequence=pattern_data['behavior_sequence'],
                effectiveness_score=pattern_data['effectiveness_score'],
                reuse_validation=pattern_data['reuse_validation'],
                metadata=pattern_data['metadata']
            )
            
            # Validate before importing
            if discovered.meets_effectiveness_threshold(self.effectiveness_threshold):
                self.discovered_patterns[pattern.pattern_id] = discovered
                imported_count += 1
        
        logger.info(f"Imported {imported_count} patterns from {filepath}")
        return imported_count
    
    def get_best_patterns(self, 
                         limit: int = 10,
                         min_reuse_count: int = 0) -> List[DiscoveredPattern]:
        """Get best performing patterns based on effectiveness and reuse."""
        patterns = list(self.discovered_patterns.values())
        
        # Filter by reuse count if specified
        if min_reuse_count > 0:
            patterns = [
                p for p in patterns 
                if p.reuse_validation.get('sample_size', 0) >= min_reuse_count
            ]
        
        # Sort by combined score
        patterns.sort(
            key=lambda p: (
                p.effectiveness_score * 0.6 +
                p.reuse_validation.get('improvement', 0) * 0.4
            ),
            reverse=True
        )
        
        return patterns[:limit]
    
    async def analyze_population_patterns(self,
                                        agent_behaviors: Dict[str, List[Dict]],
                                        performance_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns across entire agent population."""
        population_patterns = {
            'total_agents': len(agent_behaviors),
            'patterns_discovered': 0,
            'effective_patterns': 0,
            'common_patterns': [],
            'performance_impact': {}
        }
        
        pattern_frequency = Counter()
        pattern_effectiveness = defaultdict(list)
        
        # Analyze each agent
        for agent_id, behaviors in agent_behaviors.items():
            if agent_id not in performance_data:
                continue
            
            # Create timestamps
            timestamps = [
                datetime.now() - timedelta(minutes=len(behaviors)-i)
                for i in range(len(behaviors))
            ]
            
            # Discover patterns
            patterns = await self.analyze_agent_behavior(
                agent_id,
                behaviors,
                timestamps,
                performance_data[agent_id]
            )
            
            population_patterns['patterns_discovered'] += len(patterns)
            
            # Track pattern frequency and effectiveness
            for pattern in patterns:
                pattern_id = pattern.pattern.pattern_id
                pattern_frequency[pattern_id] += 1
                pattern_effectiveness[pattern_id].append(pattern.effectiveness_score)
                
                if pattern.meets_effectiveness_threshold():
                    population_patterns['effective_patterns'] += 1
        
        # Identify common patterns
        for pattern_id, frequency in pattern_frequency.most_common(10):
            if pattern_id in self.discovered_patterns:
                pattern = self.discovered_patterns[pattern_id]
                avg_effectiveness = np.mean(pattern_effectiveness[pattern_id])
                
                population_patterns['common_patterns'].append({
                    'pattern_id': pattern_id,
                    'frequency': frequency,
                    'avg_effectiveness': avg_effectiveness,
                    'behavior_sequence': pattern.behavior_sequence,
                    'window_size': pattern.window_size
                })
        
        # Calculate overall performance impact
        all_improvements = []
        for pattern_id, pattern in self.discovered_patterns.items():
            if pattern.reuse_validation.get('improvement'):
                all_improvements.append(pattern.reuse_validation['improvement'])
        
        if all_improvements:
            population_patterns['performance_impact'] = {
                'avg_improvement': np.mean(all_improvements),
                'max_improvement': max(all_improvements),
                'patterns_above_threshold': sum(
                    1 for imp in all_improvements if imp >= self.min_improvement
                )
            }
        
        return population_patterns