"""
Base classes for IndexAgent actions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class ActionMetrics:
    """Metrics for measuring action success"""
    compilation_success: bool
    tests_passing: bool
    quality_score: float  # 0.0-1.0 based on linting, complexity
    task_specific_score: float  # 0.0-1.0 based on action goals
    token_cost: int
    execution_time_seconds: float
    files_modified: List[str]
    error_messages: List[str] = None
    warnings: List[str] = None
    
    def overall_success_score(self) -> float:
        """Calculate overall success score (0.0-1.0)"""
        # Compilation and tests are mandatory
        if not self.compilation_success or not self.tests_passing:
            return 0.0
        
        # Weighted average of quality and task-specific scores
        return (0.3 * self.quality_score) + (0.7 * self.task_specific_score)


@dataclass
class ActionResult:
    """Result of executing an action"""
    action_type: str
    success: bool
    metrics: ActionMetrics
    prompt_used: str
    optimized_prompt: str
    git_diff: str
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Action(ABC):
    """Base class for all IndexAgent actions"""
    
    def __init__(self, action_id: str = None):
        self.action_id = action_id or self._generate_id()
        self.description = self.get_description()
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the action"""
        pass
    
    @abstractmethod
    def get_action_type(self) -> str:
        """Get action type identifier"""
        pass
    
    @abstractmethod
    def generate_prompt(self, worktree_path: str, context: Dict[str, Any]) -> str:
        """Generate the initial prompt for Claude CLI"""
        pass
    
    @abstractmethod
    def execute(self, worktree_path: str, claude_cli, dspy_optimizer=None) -> ActionResult:
        """
        Execute the action using Claude CLI
        
        Args:
            worktree_path: Path to git worktree
            claude_cli: ClaudeCodeCLI instance
            dspy_optimizer: Optional DSPy optimizer for prompt optimization
            
        Returns:
            ActionResult with metrics and outcomes
        """
        pass
    
    @abstractmethod
    def measure_success(self, worktree_path: str, result: ActionResult) -> float:
        """
        Measure success of the action (0.0-1.0)
        
        Args:
            worktree_path: Path to git worktree
            result: ActionResult from execution
            
        Returns:
            Success score between 0.0 and 1.0
        """
        pass
    
    def _generate_id(self) -> str:
        """Generate unique action ID"""
        import uuid
        return f"{self.get_action_type()}_{uuid.uuid4().hex[:8]}"
    
    def _run_command(self, command: str, cwd: str) -> tuple[bool, str, str]:
        """Run a shell command and return (success, stdout, stderr)"""
        import subprocess
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def _check_compilation(self, worktree_path: str, file_extensions: List[str]) -> bool:
        """Check if code compiles/parses correctly"""
        import ast
        from pathlib import Path
        
        errors = []
        
        for ext in file_extensions:
            for file_path in Path(worktree_path).rglob(f"*{ext}"):
                if ext == ".py":
                    try:
                        with open(file_path, 'r') as f:
                            ast.parse(f.read())
                    except SyntaxError as e:
                        errors.append(f"{file_path}: {e}")
                # Add other language checks as needed
        
        return len(errors) == 0
    
    def _run_tests(self, worktree_path: str) -> tuple[bool, int, int]:
        """Run tests and return (success, passed, total)"""
        # Try pytest first
        success, stdout, stderr = self._run_command(
            "python -m pytest -q --tb=short",
            worktree_path
        )
        
        if "pytest" in stderr and "No module named" in stderr:
            # Fallback to unittest
            success, stdout, stderr = self._run_command(
                "python -m unittest discover -q",
                worktree_path
            )
        
        # Parse test results
        if success:
            # Simple parsing - can be enhanced
            import re
            passed_match = re.search(r'(\d+) passed', stdout)
            total_match = re.search(r'(\d+) total', stdout)
            
            passed = int(passed_match.group(1)) if passed_match else 0
            total = int(total_match.group(1)) if total_match else passed
            
            return True, passed, total
        
        return False, 0, 0
    
    def _calculate_quality_score(self, worktree_path: str) -> float:
        """Calculate code quality score based on linting and complexity"""
        score = 1.0
        
        # Run linter (ruff)
        success, stdout, stderr = self._run_command(
            "ruff check --quiet .",
            worktree_path
        )
        
        if not success and stdout:
            # Count issues
            issue_count = len(stdout.strip().split('\n'))
            score -= min(0.5, issue_count * 0.05)  # Max 0.5 penalty
        
        # Check complexity (simplified)
        # In production, use radon or similar tools
        
        return max(0.0, score)