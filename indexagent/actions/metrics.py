"""
Action Metrics Module
Provides comprehensive metrics for action execution and success measurement
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ExecutionMetrics:
    """Detailed execution metrics for an action"""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    disk_io_mb: float = 0.0


@dataclass
class CodeQualityMetrics:
    """Code quality metrics after action execution"""
    syntax_errors: int = 0
    linting_issues: int = 0
    type_errors: int = 0
    security_issues: int = 0
    complexity_average: float = 0.0
    complexity_max: int = 0
    duplication_ratio: float = 0.0
    maintainability_index: float = 100.0


@dataclass
class TestMetrics:
    """Testing metrics after action execution"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    coverage_percent: float = 0.0
    coverage_delta: float = 0.0
    new_tests_added: int = 0
    test_execution_time: float = 0.0


@dataclass
class TaskSpecificMetrics:
    """Task-specific metrics based on action type"""
    action_type: str
    primary_metric: float  # Main success metric (0.0-1.0)
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)
    
    def add_achievement(self, achievement: str):
        """Add an achievement"""
        if achievement not in self.achievements:
            self.achievements.append(achievement)
    
    def add_metric(self, name: str, value: float):
        """Add a secondary metric"""
        self.secondary_metrics[name] = value


class MetricsCollector:
    """Collects and aggregates metrics across action executions"""
    
    def __init__(self, metrics_dir: str = "logs/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = []
    
    def record_action(self, action_type: str, result: 'ActionResult'):
        """Record metrics for an action execution"""
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'success': result.success,
            'overall_score': result.metrics.overall_success_score(),
            'token_cost': result.metrics.token_cost,
            'execution_time': result.metrics.execution_time_seconds,
            'files_modified': len(result.metrics.files_modified),
            'compilation_success': result.metrics.compilation_success,
            'tests_passing': result.metrics.tests_passing,
            'quality_score': result.metrics.quality_score,
            'task_specific_score': result.metrics.task_specific_score,
            'metadata': result.metadata
        }
        
        self.current_session.append(metrics_entry)
        self._save_metrics(metrics_entry)
    
    def get_action_statistics(self, action_type: str = None) -> Dict[str, Any]:
        """Get statistics for action executions"""
        # Load all metrics files
        all_metrics = []
        
        for metrics_file in self.metrics_dir.glob("metrics_*.json"):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if isinstance(metrics, list):
                        all_metrics.extend(metrics)
                    else:
                        all_metrics.append(metrics)
            except Exception:
                continue
        
        # Filter by action type if specified
        if action_type:
            all_metrics = [m for m in all_metrics if m.get('action_type') == action_type]
        
        if not all_metrics:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_score': 0.0,
                'average_token_cost': 0.0,
                'average_execution_time': 0.0
            }
        
        # Calculate statistics
        total = len(all_metrics)
        successful = sum(1 for m in all_metrics if m.get('success', False))
        
        return {
            'total_executions': total,
            'success_rate': successful / total,
            'average_score': sum(m.get('overall_score', 0) for m in all_metrics) / total,
            'average_token_cost': sum(m.get('token_cost', 0) for m in all_metrics) / total,
            'average_execution_time': sum(m.get('execution_time', 0) for m in all_metrics) / total,
            'total_tokens_used': sum(m.get('token_cost', 0) for m in all_metrics),
            'total_files_modified': sum(m.get('files_modified', 0) for m in all_metrics)
        }
    
    def get_improvement_trends(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get improvement trends over time"""
        trends = {
            'success_scores': [],
            'token_efficiency': [],
            'execution_speed': []
        }
        
        # Use current session if available
        metrics = self.current_session[-window_size:] if self.current_session else []
        
        for m in metrics:
            trends['success_scores'].append(m.get('overall_score', 0))
            
            # Token efficiency = score / tokens (higher is better)
            tokens = m.get('token_cost', 1)
            score = m.get('overall_score', 0)
            trends['token_efficiency'].append(score / (tokens / 1000) if tokens > 0 else 0)
            
            # Execution speed = score / time (higher is better)
            time = m.get('execution_time', 1)
            trends['execution_speed'].append(score / time if time > 0 else 0)
        
        return trends
    
    def _save_metrics(self, metrics_entry: Dict[str, Any]):
        """Save metrics to file"""
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"metrics_{date_str}.json"
        filepath = self.metrics_dir / filename
        
        # Load existing metrics for the day
        existing_metrics = []
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    existing_metrics = json.load(f)
            except Exception:
                pass
        
        # Append new metrics
        existing_metrics.append(metrics_entry)
        
        # Save updated metrics
        with open(filepath, 'w') as f:
            json.dump(existing_metrics, f, indent=2)


class SuccessCalculator:
    """Calculates success scores based on multiple factors"""
    
    @staticmethod
    def calculate_todo_success(todos_implemented: int, todos_target: int,
                             quality_score: float, tests_passing: bool) -> float:
        """Calculate success score for TODO implementation"""
        if not tests_passing:
            return 0.0
        
        completion_rate = todos_implemented / todos_target if todos_target > 0 else 1.0
        
        # Weight completion heavily but also consider quality
        return (0.7 * completion_rate) + (0.3 * quality_score)
    
    @staticmethod
    def calculate_coverage_success(coverage_delta: float, final_coverage: float,
                                 tests_quality: float, tests_passing: bool) -> float:
        """Calculate success score for coverage improvement"""
        if not tests_passing:
            return 0.0
        
        # Base score on improvement
        improvement_score = min(1.0, coverage_delta / 10.0)  # 10% improvement = perfect score
        
        # Bonus for reaching milestones
        milestone_bonus = 0.0
        if final_coverage >= 90:
            milestone_bonus = 0.2
        elif final_coverage >= 80:
            milestone_bonus = 0.1
        elif final_coverage >= 70:
            milestone_bonus = 0.05
        
        # Factor in test quality
        return (0.6 * improvement_score + 0.2 * milestone_bonus + 0.2 * tests_quality)
    
    @staticmethod
    def calculate_complexity_success(functions_improved: int, complexity_reduction: float,
                                   quality_score: float, tests_passing: bool) -> float:
        """Calculate success score for complexity refactoring"""
        if not tests_passing:
            return 0.0
        
        # Score based on number of functions improved and complexity reduction
        improvement_score = min(1.0, functions_improved / 3.0)  # Target 3 functions
        reduction_score = min(1.0, complexity_reduction / 0.3)  # Target 30% reduction
        
        return (0.5 * improvement_score + 0.3 * reduction_score + 0.2 * quality_score)