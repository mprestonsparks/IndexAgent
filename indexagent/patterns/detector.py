#!/usr/bin/env python3
"""
Pattern Detector for DEAN System
Identifies successful patterns in agent behavior and code modifications
"""

import re
import ast
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """Represents a detected code pattern"""
    pattern_id: str
    pattern_type: str  # 'optimization', 'refactoring', 'error_handling', etc.
    description: str
    code_before: str
    code_after: str
    agent_id: str
    action_type: str
    performance_delta: float
    occurrences: int = 1
    first_seen: datetime = None
    last_seen: datetime = None
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.now()
        if self.last_seen is None:
            self.last_seen = datetime.now()


@dataclass
class PromptPattern:
    """Represents a successful prompt pattern"""
    pattern_id: str
    pattern_text: str
    action_type: str
    success_rate: float
    avg_tokens: int
    avg_task_success: float
    usage_count: int = 1
    example_prompts: List[str] = None
    
    def __post_init__(self):
        if self.example_prompts is None:
            self.example_prompts = []


class PatternDetector:
    """Detects and analyzes patterns in agent behaviors and modifications"""
    
    def __init__(self, min_occurrences: int = 2, success_threshold: float = 0.7):
        """
        Initialize pattern detector
        
        Args:
            min_occurrences: Minimum times pattern must occur to be significant
            success_threshold: Minimum success rate for pattern consideration
        """
        self.min_occurrences = min_occurrences
        self.success_threshold = success_threshold
        
        # Pattern storage
        self.code_patterns: Dict[str, CodePattern] = {}
        self.prompt_patterns: Dict[str, PromptPattern] = {}
        
        # Tracking
        self.modifications_analyzed = 0
        self.patterns_detected = 0
        
        logger.info(f"PatternDetector initialized: min_occurrences={min_occurrences}")
    
    def analyze_modification(self, agent_id: str, action_type: str,
                           git_diff: str, prompt: str,
                           success_score: float, tokens_used: int) -> List[str]:
        """
        Analyze a code modification for patterns
        
        Returns:
            List of detected pattern IDs
        """
        self.modifications_analyzed += 1
        detected_patterns = []
        
        # Analyze code patterns in diff
        code_patterns = self._extract_code_patterns(git_diff, agent_id, action_type, success_score)
        for pattern in code_patterns:
            pattern_id = self._add_or_update_code_pattern(pattern)
            if pattern_id:
                detected_patterns.append(pattern_id)
        
        # Analyze prompt patterns
        if success_score >= self.success_threshold:
            prompt_pattern = self._extract_prompt_pattern(prompt, action_type, success_score, tokens_used)
            if prompt_pattern:
                pattern_id = self._add_or_update_prompt_pattern(prompt_pattern)
                if pattern_id:
                    detected_patterns.append(pattern_id)
        
        return detected_patterns
    
    def _extract_code_patterns(self, git_diff: str, agent_id: str, 
                             action_type: str, success_score: float) -> List[CodePattern]:
        """Extract code patterns from git diff"""
        patterns = []
        
        # Parse diff into chunks
        chunks = self._parse_git_diff(git_diff)
        
        for chunk in chunks:
            # Detect optimization patterns
            if pattern := self._detect_optimization_pattern(chunk):
                pattern.agent_id = agent_id
                pattern.action_type = action_type
                pattern.performance_delta = success_score
                patterns.append(pattern)
            
            # Detect refactoring patterns
            if pattern := self._detect_refactoring_pattern(chunk):
                pattern.agent_id = agent_id
                pattern.action_type = action_type
                pattern.performance_delta = success_score
                patterns.append(pattern)
            
            # Detect error handling patterns
            if pattern := self._detect_error_handling_pattern(chunk):
                pattern.agent_id = agent_id
                pattern.action_type = action_type
                pattern.performance_delta = success_score
                patterns.append(pattern)
        
        return patterns
    
    def _detect_optimization_pattern(self, chunk: Dict[str, Any]) -> Optional[CodePattern]:
        """Detect optimization patterns in code changes"""
        removed = chunk['removed']
        added = chunk['added']
        
        # Pattern: Loop optimization
        if self._is_loop_optimization(removed, added):
            return CodePattern(
                pattern_id="opt_loop_vectorization",
                pattern_type="optimization",
                description="Loop vectorization or comprehension",
                code_before='\n'.join(removed),
                code_after='\n'.join(added),
                agent_id="",  # Will be set by caller
                action_type="",
                performance_delta=0.0
            )
        
        # Pattern: Caching/memoization
        if self._is_caching_pattern(removed, added):
            return CodePattern(
                pattern_id="opt_caching",
                pattern_type="optimization",
                description="Added caching or memoization",
                code_before='\n'.join(removed),
                code_after='\n'.join(added),
                agent_id="",
                action_type="",
                performance_delta=0.0
            )
        
        # Pattern: Algorithm substitution
        if self._is_algorithm_substitution(removed, added):
            return CodePattern(
                pattern_id="opt_algorithm",
                pattern_type="optimization", 
                description="Algorithm substitution for efficiency",
                code_before='\n'.join(removed),
                code_after='\n'.join(added),
                agent_id="",
                action_type="",
                performance_delta=0.0
            )
        
        return None
    
    def _detect_refactoring_pattern(self, chunk: Dict[str, Any]) -> Optional[CodePattern]:
        """Detect refactoring patterns in code changes"""
        removed = chunk['removed']
        added = chunk['added']
        
        # Pattern: Extract method
        if self._is_extract_method(removed, added):
            return CodePattern(
                pattern_id="ref_extract_method",
                pattern_type="refactoring",
                description="Extract method refactoring",
                code_before='\n'.join(removed),
                code_after='\n'.join(added),
                agent_id="",
                action_type="",
                performance_delta=0.0
            )
        
        # Pattern: Simplify conditionals
        if self._is_simplify_conditional(removed, added):
            return CodePattern(
                pattern_id="ref_simplify_conditional",
                pattern_type="refactoring",
                description="Simplified conditional logic",
                code_before='\n'.join(removed),
                code_after='\n'.join(added),
                agent_id="",
                action_type="",
                performance_delta=0.0
            )
        
        return None
    
    def _detect_error_handling_pattern(self, chunk: Dict[str, Any]) -> Optional[CodePattern]:
        """Detect error handling patterns"""
        added = chunk['added']
        
        # Pattern: Try-except addition
        if any('try:' in line and 'except' in '\n'.join(added) for line in added):
            return CodePattern(
                pattern_id="err_try_except",
                pattern_type="error_handling",
                description="Added try-except error handling",
                code_before='\n'.join(chunk['removed']),
                code_after='\n'.join(added),
                agent_id="",
                action_type="",
                performance_delta=0.0
            )
        
        # Pattern: Validation addition
        if self._is_validation_pattern(added):
            return CodePattern(
                pattern_id="err_validation",
                pattern_type="error_handling",
                description="Added input validation",
                code_before='\n'.join(chunk['removed']),
                code_after='\n'.join(added),
                agent_id="",
                action_type="",
                performance_delta=0.0
            )
        
        return None
    
    def _extract_prompt_pattern(self, prompt: str, action_type: str,
                              success_score: float, tokens_used: int) -> Optional[PromptPattern]:
        """Extract patterns from successful prompts"""
        # Normalize prompt
        normalized = self._normalize_prompt(prompt)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(normalized)
        
        if not key_phrases:
            return None
        
        # Create pattern from key phrases
        pattern_text = " | ".join(sorted(key_phrases))
        pattern_id = f"prompt_{action_type}_{hash(pattern_text) % 10000}"
        
        return PromptPattern(
            pattern_id=pattern_id,
            pattern_text=pattern_text,
            action_type=action_type,
            success_rate=success_score,
            avg_tokens=tokens_used,
            avg_task_success=success_score,
            example_prompts=[prompt[:200]]  # Store truncated example
        )
    
    def _add_or_update_code_pattern(self, pattern: CodePattern) -> Optional[str]:
        """Add or update a code pattern"""
        pattern_key = f"{pattern.pattern_id}_{pattern.pattern_type}"
        
        if pattern_key in self.code_patterns:
            # Update existing pattern
            existing = self.code_patterns[pattern_key]
            existing.occurrences += 1
            existing.last_seen = datetime.now()
            existing.performance_delta = (
                (existing.performance_delta * (existing.occurrences - 1) + 
                 pattern.performance_delta) / existing.occurrences
            )
            
            if existing.occurrences >= self.min_occurrences:
                self.patterns_detected += 1
                return pattern_key
        else:
            # Add new pattern
            self.code_patterns[pattern_key] = pattern
            if pattern.occurrences >= self.min_occurrences:
                self.patterns_detected += 1
                return pattern_key
        
        return None
    
    def _add_or_update_prompt_pattern(self, pattern: PromptPattern) -> Optional[str]:
        """Add or update a prompt pattern"""
        pattern_key = pattern.pattern_id
        
        if pattern_key in self.prompt_patterns:
            # Update existing pattern
            existing = self.prompt_patterns[pattern_key]
            existing.usage_count += 1
            
            # Update averages
            existing.avg_tokens = (
                (existing.avg_tokens * (existing.usage_count - 1) + 
                 pattern.avg_tokens) / existing.usage_count
            )
            existing.avg_task_success = (
                (existing.avg_task_success * (existing.usage_count - 1) + 
                 pattern.avg_task_success) / existing.usage_count
            )
            
            # Add example if not too many
            if len(existing.example_prompts) < 5:
                existing.example_prompts.extend(pattern.example_prompts)
            
            if existing.usage_count >= self.min_occurrences:
                return pattern_key
        else:
            # Add new pattern
            self.prompt_patterns[pattern_key] = pattern
            if pattern.usage_count >= self.min_occurrences:
                return pattern_key
        
        return None
    
    def get_top_patterns(self, pattern_type: Optional[str] = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing patterns"""
        # Get code patterns
        code_patterns = []
        for pattern in self.code_patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            if pattern.occurrences >= self.min_occurrences:
                code_patterns.append({
                    'type': 'code',
                    'pattern': asdict(pattern),
                    'score': pattern.performance_delta * pattern.occurrences
                })
        
        # Get prompt patterns
        prompt_patterns = []
        for pattern in self.prompt_patterns.values():
            if pattern_type and pattern_type != 'prompt':
                continue
            if pattern.usage_count >= self.min_occurrences:
                prompt_patterns.append({
                    'type': 'prompt',
                    'pattern': asdict(pattern),
                    'score': pattern.avg_task_success * pattern.usage_count
                })
        
        # Combine and sort
        all_patterns = code_patterns + prompt_patterns
        all_patterns.sort(key=lambda x: x['score'], reverse=True)
        
        return all_patterns[:limit]
    
    def export_patterns(self, output_dir: str = "logs/patterns"):
        """Export detected patterns to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export code patterns
        code_patterns_data = {
            pattern_id: asdict(pattern)
            for pattern_id, pattern in self.code_patterns.items()
            if pattern.occurrences >= self.min_occurrences
        }
        
        with open(output_path / f"code_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(code_patterns_data, f, indent=2, default=str)
        
        # Export prompt patterns
        prompt_patterns_data = {
            pattern_id: asdict(pattern)
            for pattern_id, pattern in self.prompt_patterns.items()
            if pattern.usage_count >= self.min_occurrences
        }
        
        with open(output_path / f"prompt_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(prompt_patterns_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(code_patterns_data)} code patterns and {len(prompt_patterns_data)} prompt patterns")
    
    def _parse_git_diff(self, diff: str) -> List[Dict[str, Any]]:
        """Parse git diff into analyzable chunks"""
        chunks = []
        current_chunk = {'file': '', 'removed': [], 'added': []}
        
        for line in diff.split('\n'):
            if line.startswith('diff --git'):
                if current_chunk['removed'] or current_chunk['added']:
                    chunks.append(current_chunk)
                current_chunk = {'file': '', 'removed': [], 'added': []}
            elif line.startswith('+++'):
                current_chunk['file'] = line[4:]
            elif line.startswith('-') and not line.startswith('---'):
                current_chunk['removed'].append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                current_chunk['added'].append(line[1:])
        
        if current_chunk['removed'] or current_chunk['added']:
            chunks.append(current_chunk)
        
        return chunks
    
    def _is_loop_optimization(self, removed: List[str], added: List[str]) -> bool:
        """Check if changes represent loop optimization"""
        removed_text = '\n'.join(removed)
        added_text = '\n'.join(added)
        
        # Check for loop to comprehension
        if ('for ' in removed_text and 'append' in removed_text and
            any(comp in added_text for comp in ['[', '{', '('])):
            return True
        
        # Check for vectorization
        if ('for ' in removed_text and 
            any(vec in added_text for vec in ['np.', 'pandas.', 'torch.'])):
            return True
        
        return False
    
    def _is_caching_pattern(self, removed: List[str], added: List[str]) -> bool:
        """Check if changes add caching"""
        added_text = '\n'.join(added)
        
        cache_indicators = ['@cache', '@lru_cache', 'functools.cache',
                          'memoize', 'cache[', 'cache.get']
        
        return any(indicator in added_text for indicator in cache_indicators)
    
    def _is_algorithm_substitution(self, removed: List[str], added: List[str]) -> bool:
        """Check if algorithm was substituted"""
        # Simple heuristic: significant change in complexity
        if len(removed) > 10 and len(added) < len(removed) / 2:
            return True
        
        # Check for known algorithm patterns
        algorithms = ['sort', 'search', 'find', 'filter', 'map', 'reduce']
        removed_has_algo = any(algo in '\n'.join(removed).lower() for algo in algorithms)
        added_has_algo = any(algo in '\n'.join(added).lower() for algo in algorithms)
        
        return removed_has_algo and added_has_algo and len(added) < len(removed)
    
    def _is_extract_method(self, removed: List[str], added: List[str]) -> bool:
        """Check if code was extracted into a method"""
        # New function definition added
        has_new_func = any('def ' in line for line in added)
        
        # Code moved (removed lines similar to added function body)
        if has_new_func and removed:
            similarity = difflib.SequenceMatcher(None, removed, added).ratio()
            return similarity > 0.6
        
        return False
    
    def _is_simplify_conditional(self, removed: List[str], added: List[str]) -> bool:
        """Check if conditional logic was simplified"""
        removed_text = '\n'.join(removed)
        added_text = '\n'.join(added)
        
        # Count conditional keywords
        removed_conditionals = removed_text.count('if ') + removed_text.count('elif ')
        added_conditionals = added_text.count('if ') + added_text.count('elif ')
        
        # Check for early returns
        has_early_return = 'return' in added_text and added_conditionals < removed_conditionals
        
        return has_early_return or (added_conditionals < removed_conditionals and len(added) < len(removed))
    
    def _is_validation_pattern(self, added: List[str]) -> bool:
        """Check if validation was added"""
        added_text = '\n'.join(added)
        
        validation_indicators = ['assert ', 'raise ValueError', 'raise TypeError',
                               'if not ', 'is None', 'isinstance(', 'type(']
        
        return any(indicator in added_text for indicator in validation_indicators)
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for pattern extraction"""
        # Remove file paths and specific names
        normalized = re.sub(r'/[\w/\.\-]+', '[FILE]', prompt)
        normalized = re.sub(r'\b\w+\.(py|js|ts|java)\b', '[FILE]', normalized)
        
        # Remove numbers
        normalized = re.sub(r'\b\d+\b', '[NUM]', normalized)
        
        # Lowercase
        normalized = normalized.lower()
        
        return normalized
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from normalized text"""
        # Action words
        action_words = ['implement', 'optimize', 'refactor', 'add', 'fix', 'improve',
                       'create', 'update', 'remove', 'test', 'debug']
        
        # Target words
        target_words = ['function', 'method', 'class', 'performance', 'coverage',
                       'complexity', 'todo', 'error', 'bug', 'test']
        
        phrases = []
        
        # Extract action phrases
        for action in action_words:
            if action in text:
                # Find what follows the action
                pattern = rf'{action}\s+(\w+)'
                matches = re.findall(pattern, text)
                for match in matches:
                    if match in target_words:
                        phrases.append(f"{action}_{match}")
        
        # Extract goal phrases
        goal_patterns = [
            r'for\s+better\s+(\w+)',
            r'to\s+improve\s+(\w+)',
            r'ensure\s+(\w+)',
            r'with\s+(\w+)\s+handling'
        ]
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        return list(set(phrases))[:5]  # Limit to top 5 phrases