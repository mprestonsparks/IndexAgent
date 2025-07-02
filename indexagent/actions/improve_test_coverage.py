"""
Improve Test Coverage Action
Analyzes code coverage and adds tests to improve it
"""

import re
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .base import Action, ActionResult, ActionMetrics
from infra.services.optimization.dspy_optimizer import Task


class ImproveTestCoverageAction(Action):
    """Action to improve test coverage by adding missing tests"""
    
    def get_description(self) -> str:
        return "Analyze code coverage and add tests to improve coverage percentage"
    
    def get_action_type(self) -> str:
        return "improve_test_coverage"
    
    def analyze_coverage(self, worktree_path: str) -> Dict[str, Any]:
        """Run coverage analysis and return results"""
        coverage_data = {
            'total_coverage': 0.0,
            'files': [],
            'uncovered_lines': {},
            'missing_tests': []
        }
        
        # Run coverage
        cmd = "python -m pytest --cov=. --cov-report=json --quiet"
        success, stdout, stderr = self._run_command(cmd, worktree_path)
        
        if not success:
            # Try with coverage directly
            self._run_command("coverage run -m pytest", worktree_path)
            success, stdout, stderr = self._run_command("coverage json", worktree_path)
        
        # Parse coverage report
        coverage_file = Path(worktree_path) / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    cov_json = json.load(f)
                
                coverage_data['total_coverage'] = cov_json.get('totals', {}).get('percent_covered', 0.0)
                
                # Find files with low coverage
                for file_path, file_data in cov_json.get('files', {}).items():
                    if file_path.endswith('.py') and not file_path.startswith('test_'):
                        coverage_percent = file_data.get('summary', {}).get('percent_covered', 0)
                        missing_lines = file_data.get('missing_lines', [])
                        
                        if coverage_percent < 80:  # Target 80% coverage
                            coverage_data['files'].append({
                                'path': file_path,
                                'coverage': coverage_percent,
                                'missing_lines': missing_lines,
                                'missing_count': len(missing_lines)
                            })
                            
                            if missing_lines:
                                coverage_data['uncovered_lines'][file_path] = missing_lines
                
                # Sort by lowest coverage first
                coverage_data['files'].sort(key=lambda x: x['coverage'])
                
            except Exception as e:
                pass
        
        # Find files without test files
        coverage_data['missing_tests'] = self._find_files_without_tests(worktree_path)
        
        return coverage_data
    
    def _find_files_without_tests(self, worktree_path: str) -> List[str]:
        """Find Python files that don't have corresponding test files"""
        missing_tests = []
        
        for py_file in Path(worktree_path).rglob("*.py"):
            # Skip test files, __init__, and hidden directories
            if (py_file.name.startswith('test_') or 
                py_file.name == '__init__.py' or
                any(part.startswith('.') for part in py_file.parts)):
                continue
            
            # Look for corresponding test file
            test_name = f"test_{py_file.stem}.py"
            test_file = py_file.parent / test_name
            
            # Also check in tests directory
            if 'tests' in py_file.parts:
                continue
                
            tests_dir = py_file.parent / 'tests' / test_name
            
            if not test_file.exists() and not tests_dir.exists():
                # Check if there's a tests directory at the project root
                root_tests = Path(worktree_path) / 'tests' / test_name
                if not root_tests.exists():
                    missing_tests.append(str(py_file.relative_to(worktree_path)))
        
        return missing_tests
    
    def _get_uncovered_code(self, file_path: str, missing_lines: List[int]) -> List[Dict[str, Any]]:
        """Get the actual uncovered code snippets"""
        uncovered_sections = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Group consecutive lines
            if missing_lines:
                sections = []
                current_section = [missing_lines[0]]
                
                for line in missing_lines[1:]:
                    if line == current_section[-1] + 1:
                        current_section.append(line)
                    else:
                        sections.append(current_section)
                        current_section = [line]
                
                if current_section:
                    sections.append(current_section)
                
                # Extract code for each section
                for section in sections:
                    start_line = max(0, section[0] - 3)
                    end_line = min(len(lines), section[-1] + 2)
                    
                    code_snippet = []
                    for i in range(start_line, end_line):
                        prefix = "!" if i + 1 in section else " "
                        code_snippet.append(f"{prefix} {i+1:4d}: {lines[i].rstrip()}")
                    
                    uncovered_sections.append({
                        'lines': section,
                        'code': '\n'.join(code_snippet)
                    })
        
        except Exception:
            pass
        
        return uncovered_sections
    
    def generate_prompt(self, worktree_path: str, context: Dict[str, Any]) -> str:
        """Generate prompt for improving test coverage"""
        coverage_data = self.analyze_coverage(worktree_path)
        
        prompt = f"Improve test coverage from {coverage_data['total_coverage']:.1f}% by adding tests.\n\n"
        
        # Prioritize files with lowest coverage
        target_files = coverage_data['files'][:3]  # Focus on top 3 files
        
        if target_files:
            prompt += "Files needing coverage improvement:\n\n"
            
            for file_data in target_files:
                prompt += f"File: {file_data['path']}\n"
                prompt += f"Current Coverage: {file_data['coverage']:.1f}%\n"
                prompt += f"Missing Lines: {file_data['missing_count']} lines\n"
                
                # Show some uncovered code
                if file_data['path'] in coverage_data['uncovered_lines']:
                    uncovered = self._get_uncovered_code(
                        Path(worktree_path) / file_data['path'],
                        coverage_data['uncovered_lines'][file_data['path']][:20]  # Limit lines
                    )
                    
                    if uncovered:
                        prompt += "Uncovered sections:\n"
                        for section in uncovered[:2]:  # Show first 2 sections
                            prompt += f"```python\n{section['code']}\n```\n"
                
                prompt += "\n"
        
        # Add files without tests
        if coverage_data['missing_tests']:
            prompt += f"\nFiles without test files ({len(coverage_data['missing_tests'])} total):\n"
            for file_path in coverage_data['missing_tests'][:3]:
                prompt += f"- {file_path}\n"
        
        prompt += "\nInstructions:\n"
        prompt += "1. Add comprehensive tests for uncovered code sections\n"
        prompt += "2. Create test files for modules that don't have them\n"
        prompt += "3. Focus on testing edge cases, error conditions, and main functionality\n"
        prompt += "4. Use appropriate test frameworks (pytest preferred)\n"
        prompt += "5. Ensure tests are meaningful, not just for coverage\n"
        prompt += "6. Follow existing test patterns in the codebase\n"
        
        return prompt
    
    def execute(self, worktree_path: str, claude_cli, dspy_optimizer=None) -> ActionResult:
        """Execute test coverage improvement using Claude CLI"""
        start_time = datetime.now()
        
        # Get initial coverage
        initial_coverage = self.analyze_coverage(worktree_path)
        initial_percent = initial_coverage['total_coverage']
        
        # Generate prompt
        prompt = self.generate_prompt(worktree_path, {})
        
        # Optimize prompt if DSPy optimizer available
        optimized_prompt = prompt
        if dspy_optimizer:
            target_files = [f['path'] for f in initial_coverage['files'][:3]]
            target_files.extend(initial_coverage['missing_tests'][:2])
            
            task = Task(
                task_id=self.action_id,
                description=self.get_description(),
                target_files=target_files,
                constraints={"maintain_functionality": True, "meaningful_tests": True},
                performance_target={"coverage_increase": 10.0, "token_budget": 3000}
            )
            
            optimization_result = dspy_optimizer.forward(task, prompt)
            optimized_prompt = optimization_result.optimized_prompt
        
        # Execute with Claude CLI
        modification_result = claude_cli.execute_modification(optimized_prompt)
        
        # Get git diff
        git_diff = self._get_git_diff(worktree_path)
        
        # Re-run coverage analysis
        final_coverage = self.analyze_coverage(worktree_path)
        final_percent = final_coverage['total_coverage']
        coverage_delta = final_percent - initial_percent
        
        # Check compilation
        compilation_success = self._check_compilation(worktree_path, ['.py'])
        
        # Run tests
        tests_passing, tests_passed, tests_total = self._run_tests(worktree_path)
        
        # Calculate quality score
        quality_score = self._calculate_test_quality(worktree_path, git_diff)
        
        # Calculate task-specific score based on coverage improvement
        if coverage_delta > 0:
            # Score based on how much we improved (target 10% improvement)
            task_specific_score = min(1.0, coverage_delta / 10.0)
        else:
            task_specific_score = 0.0
        
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
            success=modification_result.success and tests_passing,
            metrics=metrics,
            prompt_used=prompt,
            optimized_prompt=optimized_prompt,
            git_diff=git_diff,
            metadata={
                "initial_coverage": initial_percent,
                "final_coverage": final_percent,
                "coverage_delta": coverage_delta,
                "tests_added": tests_total - tests_passed if tests_total > tests_passed else 0,
                "files_covered": len([f for f in final_coverage['files'] if f['coverage'] > 80])
            }
        )
    
    def measure_success(self, worktree_path: str, result: ActionResult) -> float:
        """Measure success based on coverage improvement"""
        # Primary metric is coverage delta
        coverage_delta = result.metadata.get('coverage_delta', 0.0)
        
        # Bonus for reaching coverage milestones
        final_coverage = result.metadata.get('final_coverage', 0.0)
        milestone_bonus = 0.0
        
        if final_coverage >= 90:
            milestone_bonus = 0.2
        elif final_coverage >= 80:
            milestone_bonus = 0.1
        elif final_coverage >= 70:
            milestone_bonus = 0.05
        
        # Base score on coverage improvement + milestone bonus
        base_score = min(1.0, (coverage_delta / 10.0) + milestone_bonus)
        
        # Apply test quality factor
        quality_factor = result.metrics.quality_score
        
        return base_score * quality_factor
    
    def _calculate_test_quality(self, worktree_path: str, git_diff: str) -> float:
        """Calculate quality score for added tests"""
        score = 1.0
        
        # Check for test patterns in diff
        if git_diff:
            # Penalize tests without assertions
            if "def test_" in git_diff and "assert" not in git_diff:
                score -= 0.3
            
            # Reward comprehensive tests
            assertion_count = git_diff.count("assert")
            test_count = git_diff.count("def test_")
            
            if test_count > 0:
                avg_assertions = assertion_count / test_count
                if avg_assertions < 1:
                    score -= 0.2
                elif avg_assertions > 3:
                    score = min(1.0, score + 0.1)
            
            # Check for edge case testing
            edge_indicators = ['None', 'empty', 'invalid', 'error', 'exception', 'boundary']
            edge_case_count = sum(1 for indicator in edge_indicators if indicator in git_diff.lower())
            
            if edge_case_count > 0:
                score = min(1.0, score + 0.05 * edge_case_count)
        
        return max(0.0, score)
    
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