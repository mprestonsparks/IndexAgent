"""
Implement TODOs Action
Finds and implements TODO comments in code
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .base import Action, ActionResult, ActionMetrics
from infra.services.optimization.dspy_optimizer import Task


class ImplementTodosAction(Action):
    """Action to find and implement TODO comments"""
    
    def get_description(self) -> str:
        return "Find and implement TODO comments in the codebase"
    
    def get_action_type(self) -> str:
        return "implement_todos"
    
    def find_todos(self, worktree_path: str) -> List[Dict[str, Any]]:
        """Find all TODO comments in the codebase"""
        todos = []
        
        # Common code file extensions
        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
        
        for ext in extensions:
            for file_path in Path(worktree_path).rglob(f"*{ext}"):
                # Skip hidden directories and common excludes
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(exclude in str(file_path) for exclude in ['node_modules', '__pycache__', 'venv']):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        # Match TODO, FIXME, HACK, XXX comments
                        todo_match = re.search(r'(?:#|//|/\*)\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.+?)(?:\*/)?$', line)
                        if todo_match:
                            todos.append({
                                'file': str(file_path.relative_to(worktree_path)),
                                'line': line_num,
                                'type': todo_match.group(1),
                                'description': todo_match.group(2).strip(),
                                'context': self._get_context(lines, line_num)
                            })
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        return todos
    
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 3) -> str:
        """Get surrounding context for a TODO"""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            context_lines.append(f"{prefix}{i+1}: {lines[i].rstrip()}")
        
        return '\n'.join(context_lines)
    
    def generate_prompt(self, worktree_path: str, context: Dict[str, Any]) -> str:
        """Generate prompt for implementing TODOs"""
        todos = self.find_todos(worktree_path)
        
        if not todos:
            return "No TODO comments found in the codebase."
        
        # Prioritize TODOs (limit to avoid token overflow)
        priority_todos = self._prioritize_todos(todos)[:5]
        
        prompt = f"Implement the following {len(priority_todos)} TODO items:\n\n"
        
        for i, todo in enumerate(priority_todos, 1):
            prompt += f"{i}. {todo['file']} (line {todo['line']})\n"
            prompt += f"   Type: {todo['type']}\n"
            prompt += f"   Description: {todo['description']}\n"
            prompt += f"   Context:\n{todo['context']}\n\n"
        
        prompt += "\nFor each TODO:\n"
        prompt += "1. Implement the requested functionality\n"
        prompt += "2. Remove the TODO comment after implementation\n"
        prompt += "3. Add appropriate error handling\n"
        prompt += "4. Add tests if the file has a corresponding test file\n"
        prompt += "5. Ensure the code follows the project's style conventions\n"
        
        return prompt
    
    def _prioritize_todos(self, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize TODOs based on type and location"""
        # Priority order: FIXME > TODO > HACK > XXX
        priority_map = {'FIXME': 0, 'TODO': 1, 'HACK': 2, 'XXX': 3}
        
        # Sort by priority and then by file (to group related changes)
        return sorted(todos, key=lambda t: (priority_map.get(t['type'], 4), t['file'], t['line']))
    
    def execute(self, worktree_path: str, claude_cli, dspy_optimizer=None) -> ActionResult:
        """Execute TODO implementation using Claude CLI"""
        start_time = datetime.now()
        
        # Count initial TODOs
        initial_todos = self.find_todos(worktree_path)
        initial_count = len(initial_todos)
        
        # Generate prompt
        prompt = self.generate_prompt(worktree_path, {})
        
        if "No TODO comments found" in prompt:
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
                metadata={"message": "No TODOs to implement"}
            )
        
        # Optimize prompt if DSPy optimizer available
        optimized_prompt = prompt
        if dspy_optimizer:
            task = Task(
                task_id=self.action_id,
                description=self.get_description(),
                target_files=[t['file'] for t in initial_todos[:5]],
                constraints={"preserve_functionality": True},
                performance_target={"todo_completion_rate": 0.8, "token_budget": 2000}
            )
            
            optimization_result = dspy_optimizer.forward(task, prompt)
            optimized_prompt = optimization_result.optimized_prompt
        
        # Execute with Claude CLI
        modification_result = claude_cli.execute_modification(optimized_prompt)
        
        # Get git diff
        git_diff = self._get_git_diff(worktree_path)
        
        # Count remaining TODOs
        remaining_todos = self.find_todos(worktree_path)
        remaining_count = len(remaining_todos)
        todos_implemented = initial_count - remaining_count
        
        # Check compilation
        compilation_success = self._check_compilation(worktree_path, ['.py', '.js', '.ts'])
        
        # Run tests
        tests_passing, tests_passed, tests_total = self._run_tests(worktree_path)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(worktree_path)
        
        # Calculate task-specific score
        if initial_count > 0:
            task_specific_score = todos_implemented / min(5, initial_count)  # We asked for up to 5
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
            success=modification_result.success and compilation_success,
            metrics=metrics,
            prompt_used=prompt,
            optimized_prompt=optimized_prompt,
            git_diff=git_diff,
            metadata={
                "initial_todo_count": initial_count,
                "remaining_todo_count": remaining_count,
                "todos_implemented": todos_implemented,
                "implementation_rate": todos_implemented / min(5, initial_count) if initial_count > 0 else 1.0
            }
        )
    
    def measure_success(self, worktree_path: str, result: ActionResult) -> float:
        """Measure success based on TODOs implemented"""
        return result.metrics.overall_success_score()
    
    def _get_git_diff(self, worktree_path: str) -> str:
        """Get git diff for the changes"""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "HEAD"],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout
            
            # Try unstaged changes
            result = subprocess.run(
                ["git", "diff"],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )
            
            return result.stdout if result.returncode == 0 else ""
            
        except Exception:
            return ""