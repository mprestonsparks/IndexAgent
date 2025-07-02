"""
Refactor Complexity Action
Identifies and refactors complex code to improve maintainability
"""

import re
import ast
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from .base import Action, ActionResult, ActionMetrics
from infra.services.optimization.dspy_optimizer import Task


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity"""
    
    def __init__(self):
        self.complexity = 1
        self.functions = {}
        self.current_function = None
        
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        old_complexity = self.complexity
        self.complexity = 1
        
        self.generic_visit(node)
        
        self.functions[node.name] = {
            'complexity': self.complexity,
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'end_lineno': getattr(node, 'end_lineno', node.lineno + 10)
        }
        
        self.complexity = old_complexity
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
        
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_AsyncWith(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Assert(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            self.complexity += len(node.values) - 1
        self.generic_visit(node)


class RefactorComplexityAction(Action):
    """Action to refactor complex code for better maintainability"""
    
    def __init__(self, complexity_threshold: int = 10):
        super().__init__()
        self.complexity_threshold = complexity_threshold
    
    def get_description(self) -> str:
        return "Identify and refactor complex code to improve maintainability"
    
    def get_action_type(self) -> str:
        return "refactor_complexity"
    
    def analyze_complexity(self, worktree_path: str) -> Dict[str, Any]:
        """Analyze code complexity in the worktree"""
        complexity_data = {
            'total_functions': 0,
            'complex_functions': [],
            'average_complexity': 0.0,
            'max_complexity': 0,
            'files_analyzed': 0
        }
        
        all_functions = []
        
        for py_file in Path(worktree_path).rglob("*.py"):
            # Skip test files and hidden directories
            if (py_file.name.startswith('test_') or 
                any(part.startswith('.') for part in py_file.parts) or
                'test' in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                analyzer = ComplexityAnalyzer()
                analyzer.visit(tree)
                
                complexity_data['files_analyzed'] += 1
                
                for func_name, func_data in analyzer.functions.items():
                    complexity = func_data['complexity']
                    all_functions.append(complexity)
                    
                    if complexity >= self.complexity_threshold:
                        # Get the actual function code
                        lines = source.split('\n')
                        start_line = func_data['lineno'] - 1
                        end_line = func_data['end_lineno']
                        
                        func_code = '\n'.join(lines[start_line:end_line])
                        
                        complexity_data['complex_functions'].append({
                            'file': str(py_file.relative_to(worktree_path)),
                            'function': func_name,
                            'complexity': complexity,
                            'line': func_data['lineno'],
                            'code': func_code[:500] + '...' if len(func_code) > 500 else func_code,
                            'issues': self._identify_complexity_issues(func_code, complexity)
                        })
                        
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        # Calculate statistics
        if all_functions:
            complexity_data['total_functions'] = len(all_functions)
            complexity_data['average_complexity'] = sum(all_functions) / len(all_functions)
            complexity_data['max_complexity'] = max(all_functions)
        
        # Sort by complexity
        complexity_data['complex_functions'].sort(key=lambda x: x['complexity'], reverse=True)
        
        return complexity_data
    
    def _identify_complexity_issues(self, code: str, complexity: int) -> List[str]:
        """Identify specific complexity issues in code"""
        issues = []
        
        # Count various complexity indicators
        if_count = code.count('if ')
        for_count = code.count('for ')
        while_count = code.count('while ')
        try_count = code.count('try:')
        
        # Nested loops
        if re.search(r'for\s+.*:\s*\n\s*for\s+', code) or re.search(r'while\s+.*:\s*\n\s*while\s+', code):
            issues.append("Nested loops detected")
        
        # Long conditional chains
        if if_count > 5:
            issues.append(f"Many conditional statements ({if_count} if statements)")
        
        # Multiple try blocks
        if try_count > 2:
            issues.append(f"Multiple exception handlers ({try_count} try blocks)")
        
        # Long functions
        line_count = len(code.split('\n'))
        if line_count > 50:
            issues.append(f"Long function ({line_count} lines)")
        
        # Deep nesting
        max_indent = max((len(line) - len(line.lstrip()) for line in code.split('\n') if line.strip()), default=0)
        if max_indent > 16:  # 4 levels of nesting
            issues.append(f"Deep nesting (max indent: {max_indent} spaces)")
        
        return issues
    
    def generate_prompt(self, worktree_path: str, context: Dict[str, Any]) -> str:
        """Generate prompt for refactoring complex code"""
        complexity_data = self.analyze_complexity(worktree_path)
        
        if not complexity_data['complex_functions']:
            return f"No functions with complexity >= {self.complexity_threshold} found. Code complexity is good!"
        
        # Focus on top 3 most complex functions
        target_functions = complexity_data['complex_functions'][:3]
        
        prompt = f"Refactor the following complex functions to improve maintainability:\n\n"
        prompt += f"Current average complexity: {complexity_data['average_complexity']:.1f}\n"
        prompt += f"Functions above threshold: {len(complexity_data['complex_functions'])}\n\n"
        
        for i, func_data in enumerate(target_functions, 1):
            prompt += f"{i}. {func_data['file']} - {func_data['function']}()\n"
            prompt += f"   Complexity: {func_data['complexity']} (threshold: {self.complexity_threshold})\n"
            prompt += f"   Line: {func_data['line']}\n"
            
            if func_data['issues']:
                prompt += "   Issues:\n"
                for issue in func_data['issues']:
                    prompt += f"   - {issue}\n"
            
            prompt += f"\n   Current code:\n```python\n{func_data['code']}\n```\n\n"
        
        prompt += "Refactoring guidelines:\n"
        prompt += "1. Extract methods for complex logic blocks\n"
        prompt += "2. Replace nested conditionals with early returns or guard clauses\n"
        prompt += "3. Use dictionary dispatch instead of long if-elif chains\n"
        prompt += "4. Split long functions into smaller, focused functions\n"
        prompt += "5. Extract complex boolean expressions into well-named variables\n"
        prompt += "6. Consider using strategy pattern for complex branching logic\n"
        prompt += "7. Preserve all functionality - refactoring should not change behavior\n"
        prompt += "8. Add docstrings to new functions\n"
        prompt += "9. Ensure all tests still pass after refactoring\n"
        
        return prompt
    
    def execute(self, worktree_path: str, claude_cli, dspy_optimizer=None) -> ActionResult:
        """Execute complexity refactoring using Claude CLI"""
        start_time = datetime.now()
        
        # Analyze initial complexity
        initial_analysis = self.analyze_complexity(worktree_path)
        initial_complex_count = len(initial_analysis['complex_functions'])
        initial_avg_complexity = initial_analysis['average_complexity']
        
        # Generate prompt
        prompt = self.generate_prompt(worktree_path, {})
        
        if "No functions with complexity" in prompt:
            return ActionResult(
                action_type=self.get_action_type(),
                success=True,
                metrics=ActionMetrics(
                    compilation_success=True,
                    tests_passing=True,
                    quality_score=1.0,
                    task_specific_score=1.0,
                    token_cost=0,
                    execution_time_seconds=0,
                    files_modified=[]
                ),
                prompt_used=prompt,
                optimized_prompt=prompt,
                git_diff="",
                metadata={"message": "No complex functions to refactor"}
            )
        
        # Optimize prompt if DSPy optimizer available
        optimized_prompt = prompt
        if dspy_optimizer:
            target_files = list(set(f['file'] for f in initial_analysis['complex_functions'][:3]))
            
            task = Task(
                task_id=self.action_id,
                description=self.get_description(),
                target_files=target_files,
                constraints={"preserve_functionality": True, "maintain_tests": True},
                performance_target={"complexity_reduction": 30.0, "token_budget": 3000}
            )
            
            optimization_result = dspy_optimizer.forward(task, prompt)
            optimized_prompt = optimization_result.optimized_prompt
        
        # Execute with Claude CLI
        modification_result = claude_cli.execute_modification(optimized_prompt)
        
        # Get git diff
        git_diff = self._get_git_diff(worktree_path)
        
        # Re-analyze complexity
        final_analysis = self.analyze_complexity(worktree_path)
        final_complex_count = len(final_analysis['complex_functions'])
        final_avg_complexity = final_analysis['average_complexity']
        
        # Calculate improvements
        functions_improved = initial_complex_count - final_complex_count
        complexity_reduction = initial_avg_complexity - final_avg_complexity
        
        # Check compilation
        compilation_success = self._check_compilation(worktree_path, ['.py'])
        
        # Run tests
        tests_passing, tests_passed, tests_total = self._run_tests(worktree_path)
        
        # Calculate quality score
        quality_score = self._calculate_refactoring_quality(git_diff, tests_passing)
        
        # Calculate task-specific score
        if initial_complex_count > 0:
            # Score based on how many complex functions were improved
            improvement_rate = functions_improved / min(3, initial_complex_count)
            complexity_reduction_rate = complexity_reduction / initial_avg_complexity if initial_avg_complexity > 0 else 0
            
            task_specific_score = (0.6 * improvement_rate) + (0.4 * min(1.0, complexity_reduction_rate * 10))
        else:
            task_specific_score = 1.0
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = ActionMetrics(
            compilation_success=compilation_success,
            tests_passing=tests_passing,
            quality_score=quality_score,
            task_specific_score=task_specific_score,
            token_cost=modification_result.tokens_used,
            execution_time_seconds=execution_time,
            files_modified=modification_result.files_modified,
            error_messages=modification_result.error_messages,
            warnings=[]
        )
        
        return ActionResult(
            action_type=self.get_action_type(),
            success=modification_result.success and compilation_success and tests_passing,
            metrics=metrics,
            prompt_used=prompt,
            optimized_prompt=optimized_prompt,
            git_diff=git_diff,
            metadata={
                "initial_complex_functions": initial_complex_count,
                "final_complex_functions": final_complex_count,
                "functions_improved": functions_improved,
                "initial_avg_complexity": initial_avg_complexity,
                "final_avg_complexity": final_avg_complexity,
                "complexity_reduction": complexity_reduction,
                "refactored_functions": self._extract_refactored_functions(git_diff)
            }
        )
    
    def measure_success(self, worktree_path: str, result: ActionResult) -> float:
        """Measure success based on complexity reduction"""
        return result.metrics.overall_success_score()
    
    def _calculate_refactoring_quality(self, git_diff: str, tests_passing: bool) -> float:
        """Calculate quality score for refactoring"""
        score = 1.0 if tests_passing else 0.5
        
        if git_diff:
            # Check for good refactoring patterns
            good_patterns = [
                r'def\s+\w+_helper\s*\(',  # Helper function extraction
                r'def\s+_\w+\s*\(',  # Private method extraction
                r'return\s+\w+\s+if\s+',  # Early returns
                r'"""[\s\S]+?"""',  # Docstrings added
                r'@property',  # Property decorators
                r'@staticmethod',  # Static method extraction
            ]
            
            for pattern in good_patterns:
                if re.search(pattern, git_diff):
                    score = min(1.0, score + 0.05)
            
            # Check for concerning patterns
            bad_patterns = [
                r'global\s+',  # Global variables
                r'exec\s*\(',  # Dynamic execution
                r'eval\s*\(',  # Eval usage
            ]
            
            for pattern in bad_patterns:
                if re.search(pattern, git_diff):
                    score = max(0.0, score - 0.2)
        
        return score
    
    def _extract_refactored_functions(self, git_diff: str) -> List[str]:
        """Extract names of refactored functions from diff"""
        refactored = []
        
        # Look for function definitions in removed/added sections
        removed_funcs = set()
        added_funcs = set()
        
        for line in git_diff.split('\n'):
            if line.startswith('-') and 'def ' in line:
                match = re.search(r'def\s+(\w+)\s*\(', line)
                if match:
                    removed_funcs.add(match.group(1))
            elif line.startswith('+') and 'def ' in line:
                match = re.search(r'def\s+(\w+)\s*\(', line)
                if match:
                    added_funcs.add(match.group(1))
        
        # Functions that were removed and re-added (modified)
        refactored = list(removed_funcs & added_funcs)
        
        # New helper functions (likely extracted)
        new_helpers = [f for f in added_funcs - removed_funcs if '_' in f or 'helper' in f.lower()]
        refactored.extend(new_helpers)
        
        return refactored
    
    def _get_git_diff(self, worktree_path: str) -> str:
        """Get git diff for the changes"""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )
            
            return result.stdout if result.returncode == 0 else ""
            
        except Exception:
            return ""