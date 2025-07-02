#!/usr/bin/env python3
"""
Capability Achievement System
DEAN Phase 2: Evolution Validation Through Concrete Accomplishments

This module validates evolution through concrete capabilities rather than mechanism novelty.
Success is measured by solving optimization challenges that were previously impossible,
not by having unique transformation sequences.

Core Innovation: "Can this strategy optimize code that nothing else can?"
"""

import ast
import re
import time
import statistics
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChallengeResult:
    """Result of testing a strategy against an optimization challenge."""
    solved: bool
    improvement_factor: float
    performance_metrics: Dict[str, float]
    novel_approach: bool
    optimization_technique: Optional[str] = None
    execution_time: float = 0.0
    memory_improvement: float = 0.0
    code_quality_score: float = 0.0

@dataclass
class CapabilityEmergence:
    """Records when a genuinely new capability emerges."""
    capability_type: str
    strategy_description: str
    demonstration_code: str
    emergence_generation: int
    performance_achievement: Dict[str, float]
    first_solved_challenge: str
    timestamp: datetime = field(default_factory=datetime.now)

class OptimizationChallenge(ABC):
    """
    Represents a code optimization challenge that tests specific capabilities.
    
    Challenges are designed to be impossible for basic strategies but solvable
    through evolved capabilities like dynamic programming, vectorization, etc.
    """
    
    def __init__(self, name: str, challenge_code: str, optimal_characteristics: Dict[str, Any]):
        self.name = name
        self.challenge_code = challenge_code
        self.optimal_characteristics = optimal_characteristics
        self.solved_by_strategies = []
        self.first_solved_generation = None
        self.baseline_performance = None
        
    @abstractmethod
    def meets_optimal_characteristics(self, optimized_result) -> bool:
        """Check if the optimization meets the challenge requirements."""
        pass
    
    @abstractmethod
    def measure_improvement(self, optimized_result) -> float:
        """Measure concrete improvement achieved."""
        pass
    
    def test_strategy(self, strategy) -> ChallengeResult:
        """Tests if a strategy can solve this challenge."""
        try:
            # Apply strategy to challenge code
            optimization_result = strategy.apply_to_code(self.challenge_code)
            
            # Measure performance characteristics
            performance_metrics = self._analyze_performance(optimization_result)
            
            # Check if it meets optimal characteristics
            solved = self.meets_optimal_characteristics(optimization_result)
            improvement = self.measure_improvement(optimization_result)
            
            # Detect if this uses a novel approach
            novel_approach = self._is_approach_novel(optimization_result)
            
            return ChallengeResult(
                solved=solved,
                improvement_factor=improvement,
                performance_metrics=performance_metrics,
                novel_approach=novel_approach,
                optimization_technique=self._detect_technique(optimization_result),
                execution_time=performance_metrics.get('execution_time', 0.0),
                memory_improvement=performance_metrics.get('memory_improvement', 0.0)
            )
            
        except Exception as e:
            logger.debug(f"Challenge {self.name} failed for strategy: {e}")
            return ChallengeResult(
                solved=False,
                improvement_factor=0.0,
                performance_metrics={},
                novel_approach=False
            )
    
    def _analyze_performance(self, optimization_result) -> Dict[str, float]:
        """Analyze performance characteristics of the optimization."""
        metrics = {}
        
        try:
            # Time complexity analysis (simplified)
            original_complexity = self._estimate_complexity(self.challenge_code)
            optimized_complexity = self._estimate_complexity(optimization_result.optimized_code)
            
            metrics['complexity_improvement'] = original_complexity / max(optimized_complexity, 1)
            
            # Execution time improvement
            metrics['execution_time'] = optimization_result.execution_time_optimized
            metrics['time_improvement'] = (optimization_result.execution_time_original - 
                                         optimization_result.execution_time_optimized) / \
                                         max(optimization_result.execution_time_original, 0.001)
            
            # Memory usage delta
            metrics['memory_improvement'] = abs(optimization_result.memory_usage_delta)
            
            # Code quality metrics
            metrics['code_length_ratio'] = len(self.challenge_code) / max(len(optimization_result.optimized_code), 1)
            
        except Exception as e:
            logger.debug(f"Performance analysis failed: {e}")
            
        return metrics
    
    def _estimate_complexity(self, code: str) -> float:
        """Estimate computational complexity of code (simplified heuristic)."""
        try:
            tree = ast.parse(code)
            
            loop_count = 0
            nested_depth = 0
            recursive_calls = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    loop_count += 1
                    # Estimate nesting depth
                    current_depth = len([n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))])
                    nested_depth = max(nested_depth, current_depth)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Check for recursive calls (simplified)
                        if 'recurs' in node.func.id.lower() or any(
                            isinstance(parent, ast.FunctionDef) and parent.name == node.func.id
                            for parent in ast.walk(tree) if isinstance(parent, ast.FunctionDef)
                        ):
                            recursive_calls += 1
            
            # Simplified complexity estimation
            if recursive_calls > 0:
                return 2 ** (recursive_calls + nested_depth)  # Exponential for recursion
            elif nested_depth > 1:
                return (loop_count + 1) ** nested_depth  # Polynomial for nested loops
            else:
                return loop_count + 1  # Linear for single loops
                
        except:
            return 10.0  # Default moderate complexity
    
    def _is_approach_novel(self, optimization_result) -> bool:
        """Determine if the optimization uses a novel approach."""
        # Check for known optimization patterns
        optimized_code = optimization_result.optimized_code.lower()
        
        novel_indicators = [
            'memoiz' in optimized_code,  # Memoization
            'dynamic' in optimized_code or 'dp' in optimized_code,  # Dynamic programming
            'vectoriz' in optimized_code or 'numpy' in optimized_code,  # Vectorization
            'async' in optimized_code or 'await' in optimized_code,  # Asynchronous
            'parallel' in optimized_code or 'multiprocess' in optimized_code,  # Parallelization
        ]
        
        return any(novel_indicators)
    
    def _detect_technique(self, optimization_result) -> Optional[str]:
        """Detect the optimization technique used."""
        optimized_code = optimization_result.optimized_code.lower()
        
        if 'memoiz' in optimized_code or 'lru_cache' in optimized_code:
            return "memoization"
        elif 'dynamic' in optimized_code or '_dp' in optimized_code:
            return "dynamic_programming"
        elif 'vectoriz' in optimized_code or 'numpy' in optimized_code:
            return "vectorization"
        elif 'async' in optimized_code:
            return "asynchronous"
        elif 'parallel' in optimized_code:
            return "parallelization"
        elif len(optimization_result.transformations_applied) > 1:
            return "multi_transformation"
        else:
            return "basic_optimization"

class RecursiveToDPChallenge(OptimizationChallenge):
    """Challenge: Convert exponential recursive algorithm to polynomial dynamic programming."""
    
    def __init__(self):
        challenge_code = '''
def count_paths(n, m):
    """Count paths in grid - exponential time without optimization"""
    if n == 0 or m == 0:
        return 1
    return count_paths(n-1, m) + count_paths(n, m-1)
        '''
        
        optimal_characteristics = {
            "time_complexity": "O(n*m)",
            "space_complexity": "O(n*m)", 
            "technique": "dynamic_programming",
            "min_improvement_factor": 100.0  # Should be dramatically faster
        }
        
        super().__init__("recursive_to_dp", challenge_code, optimal_characteristics)
    
    def meets_optimal_characteristics(self, optimized_result) -> bool:
        """Check if the solution uses dynamic programming or memoization."""
        code = optimized_result.optimized_code.lower()
        
        # Check for DP indicators
        dp_indicators = [
            'memoiz' in code,
            'lru_cache' in code,
            'dp' in code and 'table' in code,
            'dynamic' in code,
            'cache' in code
        ]
        
        # Must have significant performance improvement
        improvement_sufficient = optimized_result.improvement_factor >= self.optimal_characteristics["min_improvement_factor"]
        
        return any(dp_indicators) and improvement_sufficient
    
    def measure_improvement(self, optimized_result) -> float:
        """Measure improvement from exponential to polynomial time."""
        return optimized_result.improvement_factor

class NestedLoopVectorizationChallenge(OptimizationChallenge):
    """Challenge: Optimize nested loops through vectorization or algorithmic improvement."""
    
    def __init__(self):
        challenge_code = '''
def matrix_multiply(A, B):
    """Naive O(n^3) matrix multiplication"""
    n, m, p = len(A), len(A[0]), len(B[0])
    result = [[0] * p for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
    
    return result
        '''
        
        optimal_characteristics = {
            "time_complexity": "O(n^2.8)",  # Better than naive O(n^3)
            "technique": "vectorization_or_strassen",
            "min_improvement_factor": 2.0
        }
        
        super().__init__("nested_loop_vectorization", challenge_code, optimal_characteristics)
    
    def meets_optimal_characteristics(self, optimized_result) -> bool:
        """Check for vectorization or algorithmic improvements."""
        code = optimized_result.optimized_code.lower()
        
        optimization_indicators = [
            'numpy' in code and 'dot' in code,
            'vectoriz' in code,
            'strassen' in code,
            'block' in code and 'multiplic' in code,
            '@' in code,  # Matrix multiplication operator
            'einsum' in code
        ]
        
        improvement_sufficient = optimized_result.improvement_factor >= self.optimal_characteristics["min_improvement_factor"]
        
        return any(optimization_indicators) and improvement_sufficient
    
    def measure_improvement(self, optimized_result) -> float:
        """Measure improvement in matrix multiplication efficiency."""
        return optimized_result.improvement_factor

class PatternMemoizationChallenge(OptimizationChallenge):
    """Challenge: Optimize complex recursive pattern matching with memoization."""
    
    def __init__(self):
        challenge_code = '''
def is_match(s, p):
    """Regular expression matching - exponential without memoization"""
    if not p:
        return not s
    
    first_match = bool(s) and (p[0] == s[0] or p[0] == '.')
    
    if len(p) >= 2 and p[1] == '*':
        return (is_match(s, p[2:]) or 
                (first_match and is_match(s[1:], p)))
    else:
        return first_match and is_match(s[1:], p[1:])
        '''
        
        optimal_characteristics = {
            "time_complexity": "O(m*n)",
            "space_complexity": "O(m*n)",
            "technique": "memoized_recursion",
            "min_improvement_factor": 50.0
        }
        
        super().__init__("pattern_memoization", challenge_code, optimal_characteristics)
    
    def meets_optimal_characteristics(self, optimized_result) -> bool:
        """Check for memoization in recursive pattern matching."""
        code = optimized_result.optimized_code.lower()
        
        memoization_indicators = [
            'memoiz' in code,
            'lru_cache' in code,
            'cache' in code and 'dict' in code,
            'memo' in code and 'def' in code,
            '@' in code and 'cache' in code
        ]
        
        improvement_sufficient = optimized_result.improvement_factor >= self.optimal_characteristics["min_improvement_factor"]
        
        return any(memoization_indicators) and improvement_sufficient
    
    def measure_improvement(self, optimized_result) -> float:
        """Measure improvement in pattern matching efficiency."""
        return optimized_result.improvement_factor

class AlgorithmSubstitutionChallenge(OptimizationChallenge):
    """Challenge: Replace brute force with efficient algorithm."""
    
    def __init__(self):
        challenge_code = '''
def find_two_sum(nums, target):
    """Brute force O(n^2) two sum solution"""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
        '''
        
        optimal_characteristics = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "technique": "hash_table_optimization",
            "min_improvement_factor": 10.0
        }
        
        super().__init__("algorithm_substitution", challenge_code, optimal_characteristics)
    
    def meets_optimal_characteristics(self, optimized_result) -> bool:
        """Check for hash table or other O(n) optimization."""
        code = optimized_result.optimized_code.lower()
        
        optimization_indicators = [
            'dict' in code and 'target' in code,
            'hash' in code,
            'set' in code and 'complement' in code,
            '{' in code and '}' in code and 'target' in code,  # Dictionary usage
            'enumerate' in code and 'dict' in code
        ]
        
        improvement_sufficient = optimized_result.improvement_factor >= self.optimal_characteristics["min_improvement_factor"]
        
        return any(optimization_indicators) and improvement_sufficient
    
    def measure_improvement(self, optimized_result) -> float:
        """Measure improvement from O(n^2) to O(n)."""
        return optimized_result.improvement_factor

class EvolutionaryAchievementTracker:
    """
    Tracks genuine capability evolution through concrete achievements.
    
    Success is measured by solving optimization challenges that were previously
    impossible, not by mechanism novelty.
    """
    
    def __init__(self):
        self.challenge_suite = self._create_challenge_suite()
        self.achievement_history = []
        self.capability_emergence_timeline = []
        self.baseline_established = False
        self.unsolved_challenges = set(c.name for c in self.challenge_suite)
        
    def _create_challenge_suite(self) -> List[OptimizationChallenge]:
        """Creates challenges that basic optimizers cannot solve."""
        return [
            RecursiveToDPChallenge(),
            NestedLoopVectorizationChallenge(), 
            PatternMemoizationChallenge(),
            AlgorithmSubstitutionChallenge()
        ]
    
    def establish_baseline(self, initial_strategies: List) -> Dict[str, Any]:
        """Establish that initial strategies cannot solve the challenges."""
        baseline_results = {
            "generation": 0,
            "strategies_tested": len(initial_strategies),
            "challenges_solved": 0,
            "total_challenges": len(self.challenge_suite),
            "unsolvable_challenges": []
        }
        
        for challenge in self.challenge_suite:
            challenge_solved = False
            
            for strategy in initial_strategies:
                result = challenge.test_strategy(strategy)
                if result.solved:
                    challenge_solved = True
                    break
            
            if not challenge_solved:
                baseline_results["unsolvable_challenges"].append(challenge.name)
        
        baseline_results["challenges_solved"] = len(self.challenge_suite) - len(baseline_results["unsolvable_challenges"])
        
        self.baseline_established = True
        logger.info(f"Baseline established: {baseline_results['challenges_solved']}/{baseline_results['total_challenges']} "
                   f"challenges solvable by initial strategies")
        
        return baseline_results
    
    def evaluate_generation(self, strategies: List, generation: int) -> List[Dict[str, Any]]:
        """Evaluates which challenges are newly solved by current generation."""
        if not self.baseline_established:
            logger.warning("Baseline not established - call establish_baseline() first")
            return []
        
        achievements = []
        newly_solved = []
        
        for challenge in self.challenge_suite:
            if challenge.name in self.unsolved_challenges:  # Still unsolved
                
                for strategy in strategies:
                    result = challenge.test_strategy(strategy)
                    
                    if result.solved:
                        # BREAKTHROUGH! This challenge was impossible before
                        achievement = {
                            "generation": generation,
                            "challenge_name": challenge.name,
                            "challenge_description": challenge.optimal_characteristics,
                            "strategy_description": self._describe_strategy(strategy),
                            "improvement_factor": result.improvement_factor,
                            "performance_metrics": result.performance_metrics,
                            "optimization_technique": result.optimization_technique,
                            "novel_approach": result.novel_approach,
                            "breakthrough": True,  # This was impossible for initial strategies
                            "timestamp": datetime.now()
                        }
                        
                        achievements.append(achievement)
                        newly_solved.append(challenge.name)
                        
                        # Record capability emergence
                        capability_emergence = CapabilityEmergence(
                            capability_type=result.optimization_technique or "unknown_technique",
                            strategy_description=achievement["strategy_description"],
                            demonstration_code=challenge.challenge_code,
                            emergence_generation=generation,
                            performance_achievement=result.performance_metrics,
                            first_solved_challenge=challenge.name
                        )
                        
                        self.capability_emergence_timeline.append(capability_emergence)
                        
                        # Mark challenge as solved
                        challenge.solved_by_strategies.append(strategy)
                        challenge.first_solved_generation = generation
                        
                        logger.info(f"🎯 BREAKTHROUGH: Challenge '{challenge.name}' solved in generation {generation} "
                                   f"with {result.improvement_factor:.1f}x improvement!")
                        
                        break  # One solution per challenge per generation
        
        # Update unsolved challenges
        for solved in newly_solved:
            self.unsolved_challenges.discard(solved)
        
        self.achievement_history.extend(achievements)
        
        return achievements
    
    def _describe_strategy(self, strategy) -> str:
        """Generate human-readable description of a strategy."""
        if hasattr(strategy, 'transformation_sequence'):
            transforms = [t.__class__.__name__ for t in strategy.transformation_sequence]
            return f"Strategy with {len(transforms)} transformations: {' → '.join(transforms)}"
        else:
            return "Unknown strategy structure"
    
    def get_achievement_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of achievements."""
        total_challenges = len(self.challenge_suite)
        solved_challenges = total_challenges - len(self.unsolved_challenges)
        
        summary = {
            "total_challenges": total_challenges,
            "solved_challenges": solved_challenges,
            "unsolved_challenges": len(self.unsolved_challenges),
            "success_rate": solved_challenges / total_challenges,
            "breakthrough_generations": [],
            "capabilities_emerged": len(self.capability_emergence_timeline),
            "achievement_timeline": []
        }
        
        # Track when breakthroughs occurred
        for achievement in self.achievement_history:
            if achievement["breakthrough"]:
                summary["breakthrough_generations"].append({
                    "generation": achievement["generation"],
                    "challenge": achievement["challenge_name"],
                    "improvement": achievement["improvement_factor"]
                })
        
        # Sort achievement timeline
        summary["achievement_timeline"] = sorted(
            self.achievement_history,
            key=lambda a: a["generation"]
        )
        
        return summary
    
    def get_unsolved_challenges(self) -> List[str]:
        """Get list of challenges that remain unsolved."""
        return list(self.unsolved_challenges)
    
    def get_capability_emergence_timeline(self) -> List[CapabilityEmergence]:
        """Get timeline of capability emergence events."""
        return self.capability_emergence_timeline.copy()
    
    def is_evolution_successful(self) -> bool:
        """Determine if evolution has been successful based on concrete achievements."""
        summary = self.get_achievement_summary()
        
        # Success criteria:
        # 1. At least 50% of challenges solved
        # 2. At least 2 different capabilities emerged
        # 3. Significant performance improvements demonstrated
        
        success_criteria = {
            "challenge_success_rate": summary["success_rate"] >= 0.5,
            "capability_diversity": summary["capabilities_emerged"] >= 2,
            "performance_improvements": any(
                a["improvement_factor"] >= 10.0 for a in summary["achievement_timeline"]
            )
        }
        
        return all(success_criteria.values())

class CapabilityEmergenceDetector:
    """
    Detects when genuinely new capabilities emerge through evolution.
    
    Focuses on what agents can accomplish rather than how they do it.
    """
    
    def __init__(self):
        self.capability_taxonomy = {
            "recursion_optimization": {
                "techniques": ["tail_recursion", "recursion_to_iteration", "memoization"],
                "challenges": ["recursive_to_dp", "pattern_memoization"]
            },
            "loop_optimization": {
                "techniques": ["loop_fusion", "loop_unrolling", "vectorization"],
                "challenges": ["nested_loop_vectorization"]
            },
            "algorithm_substitution": {
                "techniques": ["brute_force_to_dp", "naive_to_efficient", "complexity_reduction"],
                "challenges": ["algorithm_substitution"]
            },
            "parallelization": {
                "techniques": ["async_detection", "parallel_map", "work_distribution"],
                "challenges": []  # Future challenges
            }
        }
        
        self.emerged_capabilities = set()
        self.initial_capabilities = set()
    
    def establish_initial_capabilities(self, initial_strategies: List, tracker: EvolutionaryAchievementTracker):
        """Establish what capabilities exist initially."""
        baseline = tracker.establish_baseline(initial_strategies)
        
        # Record which challenges were solvable initially
        solvable_challenges = set()
        for challenge in tracker.challenge_suite:
            if challenge.name not in baseline["unsolvable_challenges"]:
                solvable_challenges.add(challenge.name)
        
        # Map solvable challenges to capabilities
        for capability_type, info in self.capability_taxonomy.items():
            if any(challenge in solvable_challenges for challenge in info["challenges"]):
                self.initial_capabilities.add(capability_type)
        
        logger.info(f"Initial capabilities established: {self.initial_capabilities}")
    
    def detect_emerged_capability(self, achievements: List[Dict[str, Any]]) -> List[CapabilityEmergence]:
        """Identifies if achievements demonstrate previously unseen capabilities."""
        emergences = []
        
        for achievement in achievements:
            challenge_name = achievement["challenge_name"]
            technique = achievement["optimization_technique"]
            
            # Find which capability category this belongs to
            for capability_type, info in self.capability_taxonomy.items():
                if (challenge_name in info["challenges"] and 
                    capability_type not in self.initial_capabilities and
                    capability_type not in self.emerged_capabilities):
                    
                    # This is a genuinely new capability!
                    self.emerged_capabilities.add(capability_type)
                    
                    emergence = CapabilityEmergence(
                        capability_type=capability_type,
                        strategy_description=achievement["strategy_description"],
                        demonstration_code=f"Solved {challenge_name} with {technique}",
                        emergence_generation=achievement["generation"],
                        performance_achievement=achievement["performance_metrics"],
                        first_solved_challenge=challenge_name
                    )
                    
                    emergences.append(emergence)
                    
                    logger.info(f"🧬 CAPABILITY EMERGENCE: '{capability_type}' capability emerged in "
                               f"generation {achievement['generation']}")
        
        return emergences
    
    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of capability emergence."""
        return {
            "initial_capabilities": list(self.initial_capabilities),
            "emerged_capabilities": list(self.emerged_capabilities),
            "total_capabilities": len(self.initial_capabilities) + len(self.emerged_capabilities),
            "emergence_count": len(self.emerged_capabilities),
            "capability_growth_factor": len(self.emerged_capabilities) / max(len(self.initial_capabilities), 1)
        }