#!/usr/bin/env python3
"""
Meta-Learning Module for DEAN System
Extracts high-level patterns and injects them back into the evolution process
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .detector import PatternDetector
from .cataloger import PatternCataloger, CatalogedPattern
from infra.services.optimization.dspy_optimizer import DEANOptimizer, Example, Task

logger = logging.getLogger(__name__)


@dataclass
class MetaPattern:
    """High-level pattern extracted from multiple successful patterns"""
    meta_id: str
    pattern_type: str
    abstraction_level: int  # 1=direct, 2=generalized, 3=meta
    component_patterns: List[str]
    success_threshold: float
    description: str
    strategy_template: str


class MetaLearner:
    """Performs meta-learning on discovered patterns"""
    
    def __init__(self, detector: PatternDetector, cataloger: PatternCataloger,
                 optimizer: DEANOptimizer, generation_window: int = 5):
        """
        Initialize meta-learner
        
        Args:
            detector: Pattern detector instance
            cataloger: Pattern cataloger instance
            optimizer: DSPy optimizer instance
            generation_window: Generations to analyze for meta-patterns
        """
        self.detector = detector
        self.cataloger = cataloger
        self.optimizer = optimizer
        self.generation_window = generation_window
        
        # Meta-pattern storage
        self.meta_patterns: Dict[str, MetaPattern] = {}
        self.injection_history: List[Dict[str, Any]] = []
        
        logger.info(f"MetaLearner initialized with window={generation_window}")
    
    def extract_meta_patterns(self, current_generation: int) -> List[MetaPattern]:
        """Extract meta-patterns from recent successful patterns"""
        # Get top patterns from recent generations
        top_patterns = self.cataloger.get_top_patterns_by_generation(
            current_generation, 
            window=self.generation_window,
            limit=20
        )
        
        if len(top_patterns) < 3:
            return []
        
        meta_patterns = []
        
        # Group patterns by type
        patterns_by_type = {}
        for pattern in top_patterns:
            if pattern.pattern_type not in patterns_by_type:
                patterns_by_type[pattern.pattern_type] = []
            patterns_by_type[pattern.pattern_type].append(pattern)
        
        # Extract meta-patterns for each type
        for pattern_type, patterns in patterns_by_type.items():
            if len(patterns) >= 2:
                meta_pattern = self._extract_type_meta_pattern(pattern_type, patterns)
                if meta_pattern:
                    meta_patterns.append(meta_pattern)
                    self.meta_patterns[meta_pattern.meta_id] = meta_pattern
        
        # Extract cross-type meta-patterns
        if len(top_patterns) >= 5:
            cross_pattern = self._extract_cross_type_pattern(top_patterns)
            if cross_pattern:
                meta_patterns.append(cross_pattern)
                self.meta_patterns[cross_pattern.meta_id] = cross_pattern
        
        logger.info(f"Extracted {len(meta_patterns)} meta-patterns from generation {current_generation}")
        return meta_patterns
    
    def inject_patterns_to_dspy(self, meta_patterns: List[MetaPattern]) -> int:
        """Inject meta-patterns into DSPy training set"""
        injected_count = 0
        
        for meta_pattern in meta_patterns:
            # Create training examples from meta-pattern
            examples = self._create_training_examples(meta_pattern)
            
            if examples:
                # Add to DSPy optimizer
                self.optimizer.compile(examples)
                injected_count += len(examples)
                
                # Record injection
                self.injection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'meta_pattern': meta_pattern.meta_id,
                    'examples_injected': len(examples),
                    'abstraction_level': meta_pattern.abstraction_level
                })
        
        logger.info(f"Injected {injected_count} training examples from {len(meta_patterns)} meta-patterns")
        return injected_count
    
    def propagate_to_population(self, meta_patterns: List[MetaPattern],
                              agent_ids: List[str]) -> Dict[str, List[str]]:
        """Propagate meta-patterns to agent population"""
        propagation_map = {}
        
        for agent_id in agent_ids:
            assigned_patterns = []
            
            # Assign patterns based on agent performance
            # In production, this would consider agent history
            for meta_pattern in meta_patterns:
                if self._should_assign_pattern(agent_id, meta_pattern):
                    assigned_patterns.append(meta_pattern.strategy_template)
            
            if assigned_patterns:
                propagation_map[agent_id] = assigned_patterns
        
        logger.info(f"Propagated patterns to {len(propagation_map)} agents")
        return propagation_map
    
    def _extract_type_meta_pattern(self, pattern_type: str, 
                                  patterns: List[CatalogedPattern]) -> Optional[MetaPattern]:
        """Extract meta-pattern from patterns of same type"""
        if not patterns:
            return None
        
        # Calculate common success threshold
        success_scores = [p.avg_success_delta for p in patterns]
        success_threshold = np.percentile(success_scores, 75)
        
        # Find common elements
        component_ids = [p.pattern_id for p in patterns]
        
        # Generate strategy template
        if pattern_type == "optimization":
            strategy_template = "systematic_optimization_v2"
        elif pattern_type == "refactoring":
            strategy_template = "structured_refactoring_v2"
        else:
            strategy_template = f"meta_{pattern_type}_strategy"
        
        meta_pattern = MetaPattern(
            meta_id=f"meta_{pattern_type}_{len(self.meta_patterns)}",
            pattern_type=pattern_type,
            abstraction_level=2,
            component_patterns=component_ids[:5],
            success_threshold=success_threshold,
            description=f"Meta-pattern for {pattern_type} with {len(patterns)} components",
            strategy_template=strategy_template
        )
        
        return meta_pattern
    
    def _extract_cross_type_pattern(self, patterns: List[CatalogedPattern]) -> Optional[MetaPattern]:
        """Extract pattern that crosses types"""
        # Look for patterns that work well together
        success_pairs = []
        
        for i, p1 in enumerate(patterns):
            for j, p2 in enumerate(patterns[i+1:], i+1):
                if p1.pattern_type != p2.pattern_type:
                    combined_success = (p1.avg_success_delta + p2.avg_success_delta) / 2
                    if combined_success > 0.8:
                        success_pairs.append((p1, p2, combined_success))
        
        if not success_pairs:
            return None
        
        # Take best pair
        best_pair = max(success_pairs, key=lambda x: x[2])
        p1, p2, success = best_pair
        
        meta_pattern = MetaPattern(
            meta_id=f"meta_cross_{len(self.meta_patterns)}",
            pattern_type="hybrid",
            abstraction_level=3,
            component_patterns=[p1.pattern_id, p2.pattern_id],
            success_threshold=success,
            description=f"Hybrid pattern combining {p1.pattern_type} and {p2.pattern_type}",
            strategy_template="hybrid_strategy_advanced"
        )
        
        return meta_pattern
    
    def _create_training_examples(self, meta_pattern: MetaPattern) -> List[Example]:
        """Create DSPy training examples from meta-pattern"""
        examples = []
        
        # Get component patterns
        for pattern_id in meta_pattern.component_patterns[:3]:
            pattern = self.cataloger.get_pattern(pattern_id)
            if not pattern:
                continue
            
            # Create synthetic task
            task = Task(
                task_id=f"meta_{pattern_id}",
                description=pattern.description,
                target_files=["synthetic.py"],
                constraints={},
                performance_target={"success_rate": meta_pattern.success_threshold}
            )
            
            # Create example prompt based on pattern
            if pattern.pattern_type == "optimization":
                prompt = f"Optimize code using {meta_pattern.strategy_template} strategy"
            elif pattern.pattern_type == "refactoring":
                prompt = f"Refactor following {meta_pattern.strategy_template} principles"
            else:
                prompt = f"Apply {meta_pattern.strategy_template} to improve code"
            
            # Create training example
            example = Example(
                task=task,
                prompt=prompt,
                result_metrics={
                    "tokens_used": 1500,
                    "success_achieved": pattern.avg_success_delta
                },
                success=True,
                task_success_score=pattern.avg_success_delta,
                quality_score=0.85
            )
            
            examples.append(example)
        
        return examples
    
    def _should_assign_pattern(self, agent_id: str, meta_pattern: MetaPattern) -> bool:
        """Determine if pattern should be assigned to agent"""
        # Simple heuristic - assign based on abstraction level
        # In production, consider agent's current strategies and performance
        
        # Higher abstraction patterns go to more experienced agents
        if meta_pattern.abstraction_level >= 3:
            # Only assign to agents with sufficient experience
            # This would check agent history in production
            return agent_id.endswith(('1', '2', '3'))  # Simple demo logic
        
        return True
    
    def get_meta_learning_report(self) -> Dict[str, Any]:
        """Get report on meta-learning activities"""
        return {
            'total_meta_patterns': len(self.meta_patterns),
            'patterns_by_level': self._count_by_abstraction_level(),
            'recent_injections': self.injection_history[-10:],
            'top_meta_patterns': self._get_top_meta_patterns(5),
            'injection_summary': {
                'total_injections': len(self.injection_history),
                'total_examples': sum(h['examples_injected'] for h in self.injection_history)
            }
        }
    
    def _count_by_abstraction_level(self) -> Dict[int, int]:
        """Count meta-patterns by abstraction level"""
        counts = {1: 0, 2: 0, 3: 0}
        for pattern in self.meta_patterns.values():
            counts[pattern.abstraction_level] = counts.get(pattern.abstraction_level, 0) + 1
        return counts
    
    def _get_top_meta_patterns(self, limit: int) -> List[Dict[str, Any]]:
        """Get top meta-patterns by success threshold"""
        sorted_patterns = sorted(
            self.meta_patterns.values(),
            key=lambda p: p.success_threshold,
            reverse=True
        )
        
        return [
            {
                'meta_id': p.meta_id,
                'type': p.pattern_type,
                'level': p.abstraction_level,
                'success_threshold': p.success_threshold,
                'description': p.description
            }
            for p in sorted_patterns[:limit]
        ]